# =============================================================================
#  LAYER 3 — REVENUE FORECASTING
#  Author  : Jatin Gupta | Roll No: 24144 | MCA Final Year
#  Uni     : Himachal Pradesh University, Shimla
#  Input   : outputs/customer_churn_scored.csv  (from Layer 2)
#  Output  : outputs/revenue_forecast.csv
#            outputs/monthly_revenue_forecast.csv
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#F8F9FA",
    "axes.grid"        : True,
    "grid.alpha"       : 0.4,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 13,
    "axes.titleweight" : "bold",
})

PALETTE    = ["#1F3864", "#2E75B6", "#70AD47", "#ED7D31", "#FFC000", "#C00000"]
OUTPUT_DIR = "outputs"
VIZ_DIR    = "bank_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR,    exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/{name}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Saved → {VIZ_DIR}/{name}")

# ==============================================================================
#  PART 1 — LOAD LAYER 2 OUTPUT
# ==============================================================================
print("\n" + "="*65)
print("  PART 1 — LOAD CHURN SCORED DATA")
print("="*65)

df = pd.read_csv("outputs/customer_churn_scored.csv")
print(f"\n  Rows loaded       : {len(df):,}")
print(f"  Columns           : {list(df.columns)}")

# Annual Revenue per customer = 2% of Balance (bank net interest margin assumption)
REVENUE_RATE = 0.02
df["Annual_Revenue"] = df["Balance"] * REVENUE_RATE

total_customers   = len(df)
total_revenue     = df["Annual_Revenue"].sum()
avg_rev_customer  = df["Annual_Revenue"].mean()

print(f"\n  Total Customers       : {total_customers:,}")
print(f"  Total Annual Revenue  : €{total_revenue:,.0f}")
print(f"  Avg Revenue/Customer  : €{avg_rev_customer:,.2f}")

# ==============================================================================
#  PART 2 — SCENARIO PARAMETERS
# ==============================================================================
print("\n" + "="*65)
print("  PART 2 — SCENARIO PARAMETERS")
print("="*65)

# Base churn rate = model predicted (actual from Layer 2)
base_churn_rate   = df["Churn_Probability"].mean()

# Scenario multipliers
scenarios = {
    "Pessimistic" : base_churn_rate * 1.25,   # 25% worse than predicted
    "Base"        : base_churn_rate,            # exactly as predicted
    "Optimistic"  : base_churn_rate * 0.75,    # 25% better (retention works)
}

print(f"\n  Base Churn Rate (model avg)  : {base_churn_rate*100:.2f}%")
print(f"\n  Scenario Churn Rates:")
for s, r in scenarios.items():
    print(f"    {s:<15}: {r*100:.2f}%")

# ==============================================================================
#  PART 3 — 12-MONTH REVENUE FORECAST
# ==============================================================================
print("\n" + "="*65)
print("  PART 3 — 12-MONTH REVENUE FORECAST")
print("="*65)

months     = list(range(1, 13))
month_names= ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

forecast_results = {}

for scenario, annual_churn_rate in scenarios.items():
    monthly_churn_rate = annual_churn_rate / 12   # spread churn across 12 months

    remaining_customers = total_customers
    monthly_data        = []

    for m in months:
        # Customers who churn this month
        churned_this_month = int(remaining_customers * monthly_churn_rate)
        remaining_customers -= churned_this_month

        # Revenue this month from remaining customers
        monthly_revenue = remaining_customers * avg_rev_customer / 12
        revenue_lost    = churned_this_month  * avg_rev_customer / 12

        monthly_data.append({
            "Month"              : m,
            "Month_Name"         : month_names[m-1],
            "Scenario"           : scenario,
            "Remaining_Customers": remaining_customers,
            "Churned_This_Month" : churned_this_month,
            "Monthly_Revenue"    : round(monthly_revenue, 2),
            "Revenue_Lost"       : round(revenue_lost, 2),
            "Cumulative_Loss"    : 0   # filled below
        })

    # Calculate cumulative loss
    df_scenario = pd.DataFrame(monthly_data)
    df_scenario["Cumulative_Loss"] = df_scenario["Revenue_Lost"].cumsum().round(2)
    forecast_results[scenario] = df_scenario

    total_loss = df_scenario["Revenue_Lost"].sum()
    final_cust = df_scenario["Remaining_Customers"].iloc[-1]
    print(f"\n  {scenario}:")
    print(f"    Customers remaining after 12 months : {final_cust:,}")
    print(f"    Total Revenue Lost (12 months)      : €{total_loss:,.0f}")
    print(f"    Monthly Revenue (Month 12)          : €{df_scenario['Monthly_Revenue'].iloc[-1]:,.0f}")

# Combine all scenarios
df_forecast = pd.concat(forecast_results.values(), ignore_index=True)

# ==============================================================================
#  PART 4 — SEGMENT-LEVEL REVENUE FORECAST
# ==============================================================================
print("\n" + "="*65)
print("  PART 4 — SEGMENT-LEVEL REVENUE BREAKDOWN")
print("="*65)

# Revenue at risk by Geography
geo_rev = df.groupby("Geography").agg(
    Customers        = ("CustomerId",       "count"),
    Avg_Churn_Prob   = ("Churn_Probability","mean"),
    Total_Revenue    = ("Annual_Revenue",   "sum"),
    Avg_Balance      = ("Balance",          "mean"),
).round(2)
geo_rev["Revenue_at_Risk"] = (geo_rev["Total_Revenue"] * geo_rev["Avg_Churn_Prob"]).round(2)
print("\n  Revenue at Risk by Geography:")
print(geo_rev.to_string())

# Revenue at risk by CLV Tier
clv_rev = df.groupby("CLV_Tier").agg(
    Customers        = ("CustomerId",       "count"),
    Avg_Churn_Prob   = ("Churn_Probability","mean"),
    Total_Revenue    = ("Annual_Revenue",   "sum"),
).round(2)
clv_rev["Revenue_at_Risk"] = (clv_rev["Total_Revenue"] * clv_rev["Avg_Churn_Prob"]).round(2)
print("\n  Revenue at Risk by CLV Tier:")
print(clv_rev.to_string())

# Revenue at risk by Risk Category
risk_rev = df.groupby("Risk_Category").agg(
    Customers        = ("CustomerId",       "count"),
    Avg_Churn_Prob   = ("Churn_Probability","mean"),
    Total_Revenue    = ("Annual_Revenue",   "sum"),
).round(2)
risk_rev["Revenue_at_Risk"] = (risk_rev["Total_Revenue"] * risk_rev["Avg_Churn_Prob"]).round(2)
print("\n  Revenue at Risk by Risk Category:")
print(risk_rev.to_string())

# ==============================================================================
#  PART 5 — VISUALIZATIONS
# ==============================================================================
print("\n" + "="*65)
print("  PART 5 — GENERATING VISUALIZATIONS")
print("="*65)

colors_scenario = {
    "Pessimistic" : "#C00000",
    "Base"        : "#2E75B6",
    "Optimistic"  : "#70AD47",
}

# ── FIG 1 — 12-Month Revenue Forecast Lines ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("12-Month Revenue Forecast — 3 Scenarios",
             fontsize=15, fontweight="bold", color="#1F3864")

for scenario, df_s in forecast_results.items():
    color = colors_scenario[scenario]
    axes[0].plot(df_s["Month_Name"], df_s["Monthly_Revenue"],
                 color=color, linewidth=2.5, marker="o", markersize=5, label=scenario)
    axes[1].plot(df_s["Month_Name"], df_s["Cumulative_Loss"],
                 color=color, linewidth=2.5, marker="o", markersize=5, label=scenario)

axes[0].set_title("Monthly Revenue Retained (€)")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Revenue (€)")
axes[0].legend()
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
axes[0].tick_params(axis="x", rotation=45)

axes[1].set_title("Cumulative Revenue Loss (€)")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Cumulative Loss (€)")
axes[1].legend()
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
axes[1].tick_params(axis="x", rotation=45)

save("08_revenue_forecast_lines.png")
print("  ✔  FIG 8 — Revenue Forecast Lines")

# ── FIG 2 — Revenue at Risk by Segment ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Revenue at Risk — Segment Breakdown",
             fontsize=14, fontweight="bold", color="#1F3864")

# By Geography
bars = axes[0].bar(geo_rev.index, geo_rev["Revenue_at_Risk"],
                   color=PALETTE[:3], edgecolor="white")
axes[0].set_title("Revenue at Risk by Country (€)")
axes[0].set_ylabel("Revenue at Risk (€)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
for bar in bars:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                 f"€{bar.get_height():,.0f}", ha="center", fontsize=9, fontweight="bold")

# By CLV Tier
clv_order  = ["Premium","High","Medium","Low"]
clv_vals   = [clv_rev.loc[t,"Revenue_at_Risk"] if t in clv_rev.index else 0 for t in clv_order]
bars2 = axes[1].bar(clv_order, clv_vals, color=PALETTE[:4], edgecolor="white")
axes[1].set_title("Revenue at Risk by CLV Tier (€)")
axes[1].set_ylabel("Revenue at Risk (€)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
for bar in bars2:
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                 f"€{bar.get_height():,.0f}", ha="center", fontsize=9, fontweight="bold")

# By Risk Category
risk_order = ["Very High Risk","High Risk","Medium Risk","Low Risk"]
risk_vals  = [risk_rev.loc[r,"Revenue_at_Risk"] if r in risk_rev.index else 0 for r in risk_order]
bars3 = axes[2].bar(risk_order, risk_vals,
                    color=["#C00000","#ED7D31","#FFC000","#70AD47"], edgecolor="white")
axes[2].set_title("Revenue at Risk by Risk Category (€)")
axes[2].set_ylabel("Revenue at Risk (€)")
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
axes[2].tick_params(axis="x", rotation=15)
for bar in bars3:
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                 f"€{bar.get_height():,.0f}", ha="center", fontsize=9, fontweight="bold")

save("09_revenue_at_risk_segments.png")
print("  ✔  FIG 9 — Revenue at Risk by Segment")

# ── FIG 3 — Scenario Comparison Bar ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
scenario_names  = list(forecast_results.keys())
total_losses    = [forecast_results[s]["Revenue_Lost"].sum() for s in scenario_names]
bar_colors      = [colors_scenario[s] for s in scenario_names]


bars = ax.bar(scenario_names, total_losses, color=bar_colors, edgecolor="white", width=0.5)
ax.set_title("Total 12-Month Revenue Loss by Scenario", pad=12)
ax.set_ylabel("Total Revenue Lost (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
            f"€{bar.get_height():,.0f}", ha="center", fontsize=11, fontweight="bold")

save("10_scenario_comparison.png")
print("  ✔  FIG 10 — Scenario Comparison")

# ==============================================================================
#  PART 6 — SAVE OUTPUT FILES
# ==============================================================================
print("\n" + "="*65)
print("  PART 6 — SAVING OUTPUT FILES")
print("="*65)

# Monthly forecast — all 3 scenarios
df_forecast.to_csv("outputs/monthly_revenue_forecast.csv", index=False)
print(f"  ✔  outputs/monthly_revenue_forecast.csv ({len(df_forecast)} rows)")

# Segment revenue summary
geo_rev.reset_index().to_csv("outputs/revenue_by_geography.csv", index=False)
clv_rev.reset_index().to_csv("outputs/revenue_by_clv_tier.csv",  index=False)
risk_rev.reset_index().to_csv("outputs/revenue_by_risk.csv",     index=False)
print(f"  ✔  outputs/revenue_by_geography.csv")
print(f"  ✔  outputs/revenue_by_clv_tier.csv")
print(f"  ✔  outputs/revenue_by_risk.csv")

# ==============================================================================
#  PART 7 — FORECAST SUMMARY REPORT
# ==============================================================================
summary_report = f"""
================================================================================
  LAYER 3 — REVENUE FORECASTING SUMMARY
  Author  : Jatin Gupta | Roll No: 24144 | MCA Final Year
  Uni     : Himachal Pradesh University, Shimla
================================================================================

INPUT DATA
----------
  Customers Scored      : {total_customers:,}
  Total Annual Revenue  : €{total_revenue:,.0f}
  Avg Revenue/Customer  : €{avg_rev_customer:,.2f}
  Revenue Rate Assumed  : {REVENUE_RATE*100:.0f}% of Balance (net interest margin)

CHURN SCENARIOS
---------------
  Base Churn Rate       : {base_churn_rate*100:.2f}%
  Pessimistic Rate      : {scenarios['Pessimistic']*100:.2f}%  (+25% worse)
  Optimistic Rate       : {scenarios['Optimistic']*100:.2f}%   (-25% better)

12-MONTH REVENUE LOSS FORECAST
-------------------------------
  Pessimistic           : €{forecast_results['Pessimistic']['Revenue_Lost'].sum():,.0f}
  Base                  : €{forecast_results['Base']['Revenue_Lost'].sum():,.0f}
  Optimistic            : €{forecast_results['Optimistic']['Revenue_Lost'].sum():,.0f}

TOP REVENUE AT RISK SEGMENTS
-----------------------------
  Highest Risk Country  : {geo_rev['Revenue_at_Risk'].idxmax()} (€{geo_rev['Revenue_at_Risk'].max():,.0f})
  Highest Risk CLV Tier : {clv_rev['Revenue_at_Risk'].idxmax()} (€{clv_rev['Revenue_at_Risk'].max():,.0f})

================================================================================
"""

with open("outputs/revenue_forecast_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_report)
print(summary_report)

print("="*65)
print("  ✅  LAYER 3 COMPLETE — REVENUE FORECASTING DONE")
print("="*65)
print(f"\n  📄 outputs/monthly_revenue_forecast.csv")
print(f"  📄 outputs/revenue_by_geography.csv")
print(f"  📄 outputs/revenue_by_clv_tier.csv")
print(f"  📄 outputs/revenue_by_risk.csv")
print(f"  📊 bank_viz/08_revenue_forecast_lines.png")
print(f"  📊 bank_viz/09_revenue_at_risk_segments.png")
print(f"  📊 bank_viz/10_scenario_comparison.png\n")