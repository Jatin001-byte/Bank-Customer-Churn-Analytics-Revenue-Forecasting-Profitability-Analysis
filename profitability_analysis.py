# =============================================================================
#  LAYER 4 — PROFITABILITY ANALYSIS
#  Author  : Jatin Gupta | Roll No: 24144 | MCA Final Year
#  Uni     : Himachal Pradesh University, Shimla
#  Input   : outputs/customer_churn_scored.csv  (from Layer 2)
#            outputs/monthly_revenue_forecast.csv (from Layer 3)
#  Output  : outputs/profitability_analysis.csv
#            outputs/profitability_summary.csv
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

PALETTE    = ["#1F3864","#2E75B6","#70AD47","#ED7D31","#FFC000","#C00000"]
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
#  PART 1 — LOAD DATA
# ==============================================================================
print("\n" + "="*65)
print("  PART 1 — LOAD DATA")
print("="*65)

df = pd.read_csv("outputs/customer_churn_scored.csv")
print(f"\n  Customers loaded  : {len(df):,}")

# ==============================================================================
#  PART 2 — CALCULATE PROFITABILITY METRICS PER CUSTOMER
# ==============================================================================
print("\n" + "="*65)
print("  PART 2 — CUSTOMER PROFITABILITY METRICS")
print("="*65)

REVENUE_RATE    = 0.02    # 2% of balance = annual revenue
COST_TO_SERVE   = 150     # €150 flat annual cost per customer (operations)
RETENTION_COST  = 200     # €200 max budget to retain one at-risk customer

# Annual Revenue per customer
df["Annual_Revenue"] = (df["Balance"] * REVENUE_RATE).round(2)

# Cost to serve (base operational cost)
df["Cost_to_Serve"] = COST_TO_SERVE

# Net Profit per customer
df["Net_Profit"] = (df["Annual_Revenue"] - df["Cost_to_Serve"]).round(2)

# Revenue at Risk = Annual Revenue × Churn Probability
df["Revenue_at_Risk"] = (df["Annual_Revenue"] * df["Churn_Probability"]).round(2)

# Retention ROI = Revenue at Risk - Retention Cost
# If positive → worth spending money to retain this customer
df["Retention_ROI"] = (df["Revenue_at_Risk"] - RETENTION_COST).round(2)

# Retention Priority Flag
df["Retention_Priority"] = df["Retention_ROI"].apply(
    lambda x: "High Priority" if x > 300
    else ("Medium Priority" if x > 0
    else "Not Worth Retaining"))

print(f"\n  Revenue Rate      : {REVENUE_RATE*100:.0f}% of Balance")
print(f"  Cost to Serve     : €{COST_TO_SERVE}/year per customer")
print(f"  Max Retention Cost: €{RETENTION_COST}/customer")
print(f"\n  Sample profitability data:")
print(df[["CustomerId","Balance","Annual_Revenue","Net_Profit",
          "Revenue_at_Risk","Retention_ROI","Retention_Priority"]].head(8).to_string(index=False))

# ==============================================================================
#  PART 3 — PROFITABILITY SCORING & TIERS
# ==============================================================================
print("\n" + "="*65)
print("  PART 3 — PROFITABILITY SCORING & TIERS")
print("="*65)

# Profitability Score (0–100) combining 3 factors:
# 1. Net Profit contribution   (40% weight)
# 2. CLV score                 (35% weight)
# 3. Churn Risk penalty        (25% weight — higher churn = lower score)

# Normalize Net Profit to 0–100
net_min = df["Net_Profit"].min()
net_max = df["Net_Profit"].max()
df["NetProfit_Score"] = ((df["Net_Profit"] - net_min) / (net_max - net_min) * 100).round(2)

# Normalize CLV to 0–100
clv_min = df["Est_CLV"].min()
clv_max = df["Est_CLV"].max()
df["CLV_Score"] = ((df["Est_CLV"] - clv_min) / (clv_max - clv_min) * 100).round(2)

# Churn Risk Penalty (higher churn prob = lower score)
df["ChurnRisk_Penalty"] = ((1 - df["Churn_Probability"]) * 100).round(2)

# Final Profitability Score
df["Profitability_Score"] = (
    df["NetProfit_Score"]  * 0.40 +
    df["CLV_Score"]        * 0.35 +
    df["ChurnRisk_Penalty"]* 0.25
).round(2)

# Profitability Tier
def profitability_tier(score):
    if score >= 75:  return "Premium"
    elif score >= 50: return "Standard"
    elif score >= 25: return "Below Average"
    else:             return "Unprofitable"

df["Profitability_Tier"] = df["Profitability_Score"].apply(profitability_tier)

# Summary by tier
tier_summary = df.groupby("Profitability_Tier").agg(
    Customers         = ("CustomerId",          "count"),
    Avg_Score         = ("Profitability_Score",  "mean"),
    Avg_Balance       = ("Balance",              "mean"),
    Avg_CLV           = ("Est_CLV",              "mean"),
    Avg_Net_Profit    = ("Net_Profit",           "mean"),
    Total_Revenue     = ("Annual_Revenue",       "sum"),
    Avg_Churn_Prob    = ("Churn_Probability",    "mean"),
    Total_Rev_at_Risk = ("Revenue_at_Risk",      "sum"),
).round(2)

print("\n─── Profitability Tier Summary ───")
print(tier_summary.to_string())

# ==============================================================================
#  PART 4 — SEGMENT PROFITABILITY ANALYSIS
# ==============================================================================
print("\n" + "="*65)
print("  PART 4 — SEGMENT PROFITABILITY")
print("="*65)

# By Geography
geo_profit = df.groupby("Geography").agg(
    Customers          = ("CustomerId",         "count"),
    Avg_Net_Profit     = ("Net_Profit",          "mean"),
    Total_Revenue      = ("Annual_Revenue",      "sum"),
    Total_Rev_at_Risk  = ("Revenue_at_Risk",     "sum"),
    Avg_Profit_Score   = ("Profitability_Score", "mean"),
    High_Priority_Count= ("Retention_Priority",
                          lambda x: (x=="High Priority").sum()),
).round(2)
print("\n  Profitability by Geography:")
print(geo_profit.to_string())

# By CLV Tier
clv_profit = df.groupby("CLV_Tier").agg(
    Customers          = ("CustomerId",          "count"),
    Avg_Net_Profit     = ("Net_Profit",           "mean"),
    Avg_Profit_Score   = ("Profitability_Score",  "mean"),
    Total_Rev_at_Risk  = ("Revenue_at_Risk",      "sum"),
    High_Priority_Count= ("Retention_Priority",
                          lambda x: (x=="High Priority").sum()),
).round(2)
print("\n  Profitability by CLV Tier:")
print(clv_profit.to_string())

# By Age Group
age_profit = df.groupby("AgeGroup").agg(
    Customers       = ("CustomerId",          "count"),
    Avg_Net_Profit  = ("Net_Profit",           "mean"),
    Avg_Profit_Score= ("Profitability_Score",  "mean"),
    Avg_Churn_Prob  = ("Churn_Probability",    "mean"),
).round(2)
print("\n  Profitability by Age Group:")
print(age_profit.to_string())

# ==============================================================================
#  PART 5 — TOP CUSTOMERS TO RETAIN
# ==============================================================================
print("\n" + "="*65)
print("  PART 5 — TOP CUSTOMERS TO RETAIN")
print("="*65)

# High priority + high churn risk = act NOW
top_retain = df[
    (df["Retention_Priority"] == "High Priority") &
    (df["Churn_Probability"]  >= 0.55)
].sort_values("Revenue_at_Risk", ascending=False).head(20)

print(f"\n  Top 20 customers to retain immediately:")
print(top_retain[["CustomerId","Geography","Age","Balance",
                  "Churn_Probability","Revenue_at_Risk",
                  "Profitability_Score","Profitability_Tier"]].to_string(index=False))

total_high_priority = len(df[df["Retention_Priority"]=="High Priority"])
total_not_worth     = len(df[df["Retention_Priority"]=="Not Worth Retaining"])
print(f"\n  High Priority customers  : {total_high_priority:,}")
print(f"  Not Worth Retaining      : {total_not_worth:,}")
print(f"  Medium Priority          : {len(df) - total_high_priority - total_not_worth:,}")

# ==============================================================================
#  PART 6 — VISUALIZATIONS
# ==============================================================================
print("\n" + "="*65)
print("  PART 6 — GENERATING VISUALIZATIONS")
print("="*65)

tier_order  = ["Premium","Standard","Below Average","Unprofitable"]
tier_colors = ["#1F3864","#2E75B6","#FFC000","#C00000"]

# ── FIG 1 — Profitability Overview ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Customer Profitability Analysis Dashboard",
             fontsize=15, fontweight="bold", color="#1F3864")

# 1a. Tier Distribution Pie
tier_counts = df["Profitability_Tier"].value_counts()
tier_vals   = [tier_counts.get(t,0) for t in tier_order]
axes[0,0].pie(tier_vals, labels=tier_order, autopct="%1.1f%%",
              colors=tier_colors,
              wedgeprops={"edgecolor":"white","linewidth":2}, startangle=90)
axes[0,0].set_title("Profitability Tier Distribution")

# 1b. Avg Net Profit by Tier
avg_profits = [tier_summary.loc[t,"Avg_Net_Profit"] if t in tier_summary.index else 0
               for t in tier_order]
bars = axes[0,1].bar(tier_order, avg_profits, color=tier_colors, edgecolor="white")
axes[0,1].set_title("Avg Net Profit by Tier (€)")
axes[0,1].set_ylabel("Avg Net Profit (€)")
axes[0,1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"€{x:,.0f}"))
for bar in bars:
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                   f"€{bar.get_height():,.0f}", ha="center", fontsize=9, fontweight="bold")

# 1c. Revenue at Risk by Tier
rev_risk = [tier_summary.loc[t,"Total_Rev_at_Risk"] if t in tier_summary.index else 0
            for t in tier_order]
bars2 = axes[0,2].bar(tier_order, rev_risk, color=tier_colors, edgecolor="white")
axes[0,2].set_title("Total Revenue at Risk by Tier (€)")
axes[0,2].set_ylabel("Revenue at Risk (€)")
axes[0,2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"€{x:,.0f}"))
for bar in bars2:
    axes[0,2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
                   f"€{bar.get_height():,.0f}", ha="center", fontsize=9, fontweight="bold")

# 1d. Profitability Score Distribution
axes[1,0].hist(df["Profitability_Score"], bins=40,
               color="#2E75B6", edgecolor="white", alpha=0.8)
axes[1,0].axvline(25, color="#FFC000", linestyle="--", linewidth=1.5, label="25 (Below Avg)")
axes[1,0].axvline(50, color="#ED7D31", linestyle="--", linewidth=1.5, label="50 (Standard)")
axes[1,0].axvline(75, color="#C00000", linestyle="--", linewidth=1.5, label="75 (Premium)")
axes[1,0].set_title("Profitability Score Distribution")
axes[1,0].set_xlabel("Profitability Score")
axes[1,0].set_ylabel("Count")
axes[1,0].legend(fontsize=8)

# 1e. Retention Priority Breakdown
priority_order  = ["High Priority","Medium Priority","Not Worth Retaining"]
priority_colors = ["#C00000","#FFC000","#70AD47"]
priority_counts = [len(df[df["Retention_Priority"]==p]) for p in priority_order]
bars3 = axes[1,1].bar(priority_order, priority_counts,
                      color=priority_colors, edgecolor="white")
axes[1,1].set_title("Retention Priority Breakdown")
axes[1,1].set_ylabel("Number of Customers")
axes[1,1].tick_params(axis="x", rotation=10)
for bar in bars3:
    axes[1,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                   f"{bar.get_height():,}", ha="center", fontweight="bold")

# 1f. Profitability Score vs Churn Probability scatter
sample = df.sample(min(2000, len(df)), random_state=42)
scatter_colors = [tier_colors[tier_order.index(t)] if t in tier_order else "#888888"
                  for t in sample["Profitability_Tier"]]
axes[1,2].scatter(sample["Churn_Probability"], sample["Profitability_Score"],
                  c=scatter_colors, alpha=0.4, s=15, edgecolors="none")
axes[1,2].set_title("Profitability Score vs Churn Probability")
axes[1,2].set_xlabel("Churn Probability")
axes[1,2].set_ylabel("Profitability Score")
axes[1,2].axvline(0.55, color="black", linestyle="--", linewidth=1, label="High Risk threshold")
axes[1,2].legend(fontsize=8)

save("11_profitability_dashboard.png")
print("  ✔  FIG 11 — Profitability Dashboard")

# ── FIG 2 — Geography & CLV Profitability ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Segment Profitability Breakdown",
             fontsize=14, fontweight="bold", color="#1F3864")

# By Geography
bars = axes[0].bar(geo_profit.index, geo_profit["Avg_Profit_Score"],
                   color=PALETTE[:3], edgecolor="white")
axes[0].set_title("Avg Profitability Score by Country")
axes[0].set_ylabel("Avg Profitability Score")
for bar in bars:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{bar.get_height():.1f}", ha="center", fontweight="bold")

# By CLV Tier
clv_order_plot = ["Premium","High","Medium","Low"]
clv_scores     = [clv_profit.loc[t,"Avg_Profit_Score"] if t in clv_profit.index else 0
                  for t in clv_order_plot]
bars2 = axes[1].bar(clv_order_plot, clv_scores, color=PALETTE[:4], edgecolor="white")
axes[1].set_title("Avg Profitability Score by CLV Tier")
axes[1].set_ylabel("Avg Profitability Score")
for bar in bars2:
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{bar.get_height():.1f}", ha="center", fontweight="bold")

save("12_segment_profitability.png")
print("  ✔  FIG 12 — Segment Profitability")

# ==============================================================================
#  PART 7 — SAVE OUTPUT FILES
# ==============================================================================
print("\n" + "="*65)
print("  PART 7 — SAVING OUTPUT FILES")
print("="*65)

# Full customer profitability file — main output for Power BI
output_cols = ["CustomerId","Surname","Geography","Gender","Age","Balance",
               "Tenure","NumOfProducts","CLV_Tier","Est_CLV","Churn_Probability",
               "Risk_Category","Annual_Revenue","Net_Profit","Revenue_at_Risk",
               "Retention_ROI","Retention_Priority","Profitability_Score","Profitability_Tier"]

df[[c for c in output_cols if c in df.columns]].to_csv(
    "outputs/profitability_analysis.csv", index=False)
print(f"  ✔  outputs/profitability_analysis.csv ({len(df):,} rows)")

# Summary by tier
tier_summary.reset_index().to_csv("outputs/profitability_summary.csv", index=False)
print(f"  ✔  outputs/profitability_summary.csv")

# Segment summaries
geo_profit.reset_index().to_csv("outputs/profitability_by_geography.csv", index=False)
clv_profit.reset_index().to_csv("outputs/profitability_by_clv.csv",       index=False)
age_profit.reset_index().to_csv("outputs/profitability_by_age.csv",        index=False)
print(f"  ✔  outputs/profitability_by_geography.csv")
print(f"  ✔  outputs/profitability_by_clv.csv")
print(f"  ✔  outputs/profitability_by_age.csv")

# ==============================================================================
#  PART 8 — SUMMARY REPORT
# ==============================================================================
summary = f"""
================================================================================
  LAYER 4 — PROFITABILITY ANALYSIS SUMMARY
  Author  : Jatin Gupta | Roll No: 24144 | MCA Final Year
  Uni     : Himachal Pradesh University, Shimla
================================================================================

PROFITABILITY ASSUMPTIONS
--------------------------
  Revenue Rate      : 2% of Balance (annual net interest margin)
  Cost to Serve     : €150 per customer per year
  Max Retention Cost: €200 per customer

PROFITABILITY TIER BREAKDOWN
-----------------------------
  Premium       : {len(df[df['Profitability_Tier']=='Premium']):,} customers
  Standard      : {len(df[df['Profitability_Tier']=='Standard']):,} customers
  Below Average : {len(df[df['Profitability_Tier']=='Below Average']):,} customers
  Unprofitable  : {len(df[df['Profitability_Tier']=='Unprofitable']):,} customers

RETENTION PRIORITY
------------------
  High Priority         : {total_high_priority:,} customers
  Medium Priority       : {len(df) - total_high_priority - total_not_worth:,} customers
  Not Worth Retaining   : {total_not_worth:,} customers

TOP SEGMENTS
------------
  Highest Profit Country: {geo_profit['Avg_Net_Profit'].idxmax()} (€{geo_profit['Avg_Net_Profit'].max():,.0f} avg)
  Highest Risk Country  : Germany
  Best CLV Tier         : {clv_profit['Avg_Profit_Score'].idxmax()} (score: {clv_profit['Avg_Profit_Score'].max():.1f})

================================================================================
"""

with open("outputs/profitability_report.txt", "w", encoding="utf-8") as f:
    f.write(summary)
print(summary)

print("="*65)
print("  ✅  LAYER 4 COMPLETE — PROFITABILITY ANALYSIS DONE")
print("="*65)
print(f"\n  📄 outputs/profitability_analysis.csv   — {len(df):,} customers scored")
print(f"  📄 outputs/profitability_summary.csv    — tier summary")
print(f"  📄 outputs/profitability_by_geography.csv")
print(f"  📄 outputs/profitability_by_clv.csv")
print(f"  📄 outputs/profitability_by_age.csv")
print(f"  📊 bank_viz/11_profitability_dashboard.png")
print(f"  📊 bank_viz/12_segment_profitability.png\n")