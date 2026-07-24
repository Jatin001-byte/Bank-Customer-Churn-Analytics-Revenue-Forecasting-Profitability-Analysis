# Bank Customer Churn Analytics, Revenue Forecasting & Profitability Analysis

**Tech Stack:** Python · scikit-learn · XGBoost · MySQL · Power BI · Excel VBA
**License:** MIT

An end-to-end data analytics project that predicts customer churn for a retail bank and translates those predictions into financial impact — going beyond "who will leave" to answer "what will it cost us" and "where should we focus retention spend."

---

## 📌 Overview

This project combines **machine learning, financial forecasting, and business intelligence** into a single pipeline built on 10,000 bank customer records sourced from Kaggle. It consists of three connected modules:

1. **Customer Churn Prediction** — a tuned Random Forest classifier (ROC-AUC 0.8571) that scores every customer with a calibrated churn probability
2. **Revenue Forecasting** — a 12-month revenue projection under optimistic, base-case, and pessimistic churn scenarios
3. **Profitability Analysis** — segmentation of at-risk customers into tiers by combining churn probability with customer lifetime value, so retention effort goes where it's actually worth it

Results are delivered through an interactive **Power BI dashboard** for executives and an **Excel VBA tracker** that gives relationship managers a ready-to-use action list.

---

## 🎯 Problem Statement

Banks lose significant recurring revenue to customer churn, but most retention efforts are reactive and untargeted — treating all at-risk customers the same regardless of their actual value to the business. This project addresses three questions:

- **Who** is likely to churn, and how confident can we be in that prediction?
- **What** will it cost the bank in lost revenue over the next 12 months?
- **Where** should retention budget go to get the best return, rather than being spread evenly across everyone flagged as "at risk"?

---

## 🧠 Methodology

### Data & Feature Engineering
- 10,000 customer records, 13 original attributes (CreditScore, Geography, Age, Balance, EstimatedSalary, etc.)
- 7 engineered features added (17 total) to capture behavioral and demographic patterns
- Outlier treatment applied prior to modeling
- ~500 rows removed / cleaned during preprocessing, leaving 9,499 customers scored downstream

### Model Selection
| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.8015 |
| K-Nearest Neighbors | 0.7827 |
| Decision Tree | 0.7815 |
| Random Forest | 0.8349 |
| AdaBoost | 0.8439 |
| Gradient Boosting | 0.8583 |
| XGBoost | 0.8667 |
| **Tuned Random Forest (selected)** | **0.8571** |

The tuned Random Forest was selected as the final model after `GridSearchCV` hyperparameter tuning, balancing strong discriminative power with interpretability (feature importance) for business stakeholders.

### Revenue Forecasting
12-month forward revenue was modeled under three churn scenarios:

| Scenario | 12-Month Revenue Impact |
|---|---|
| Optimistic (early intervention) | €281,621 |
| Base Case | €361,152 |
| Pessimistic (no intervention) | €434,919 |

### Profitability Segmentation
At-risk customers were split into four tiers using churn probability × customer lifetime value:

| Tier | Customers | Notes |
|---|---|---|
| Premium | 3 | Highest priority for retention |
| Standard | 1,525 | Strong retention ROI |
| Below Average | 4,442 | Selective, lower-cost intervention |
| Unprofitable | 3,529 | Not worth active retention spend |

**Total annual revenue at risk identified: €3,484,976**

---

## 📊 Deliverables

- **Power BI Dashboard** — executive KPI summary, churn rate by geography and age band, revenue-at-risk by CLV tier, interactive slicers
- **Excel VBA Tracker** (`ChurnTracker.xlsm`) — a prioritized, ready-to-action list of at-risk, high-value customers for relationship managers
- **Python Pipeline** — reproducible, script-based ETL, feature engineering, model training and scoring
- **MySQL Database** — relational storage (`bank_churn` DB) for the customer, transaction, and scoring data

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.x |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| Database | MySQL |
| Visualization / BI | Power BI |
| Operational Tooling | Excel (VBA) |
| Environment | Jupyter Notebook |

---

## 📁 Repository Structure

```
├── data/               # Sample dataset (churn.csv) — full raw data not committed
├── notebooks/           # EDA, feature engineering, model training & tuning
├── sql/                  # MySQL schema and queries (bank_churn DB)
├── dashboard/             # Power BI (.pbix) file
├── excel-tracker/          # ChurnTracker.xlsm (VBA)
├── requirements.txt          # Python dependencies
├── LICENSE
└── README.md
```

---

## ⚙️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Jatin001-byte/Bank-Customer-Churn-Analytics-Revenue-Forecasting-Profitability-Analysis.git
cd Bank-Customer-Churn-Analytics-Revenue-Forecasting-Profitability-Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up the MySQL database
mysql -u root -p < sql/schema.sql

# 4. Run the pipeline
jupyter notebook notebooks/
```

> ⚠️ Update database credentials in your local `.env` file before running — never commit credentials directly.

---

## 📈 Key Results at a Glance

- **20.2%** overall churn rate
- **0.8571** ROC-AUC (tuned Random Forest)
- **€3.48M** annual revenue at risk identified
- **1,006** very-high-risk customers flagged
- 12-month scenario-based revenue forecast (€281K–€435K range)
- 4-tier profitability segmentation for targeted retention

---

## 🔭 Future Enhancements

- Real-time scoring pipeline instead of batch scoring
- Deep learning (LSTM) for time-series revenue forecasting
- Integration with a live CRM instead of static Excel tracker
- Automated model retraining schedule

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙋 Author

**Jatin** — MCA, Department of Computer Science, Himachal Pradesh University, Shimla
Data Analyst Intern | Python · Power BI · SQL · Machine Learning

[LinkedIn](#) · [Email](#)
