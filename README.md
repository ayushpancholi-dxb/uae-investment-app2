# UAE Personal Finance & Micro-Investment App
### MBA Data Analytics — Individual PBL

A complete, end-to-end data analytics project that validates the business
case for a UAE-focused personal finance and micro-investment mobile application.
The project generates a synthetic survey dataset, then applies four analytical
algorithms to answer a single core business question:

> **"Which users are most likely to adopt micro-investment features,
> and what feature combinations drive higher monthly investment?"**

---

## Repository Structure

```
.
├── run_all.py                     # ← Run the full pipeline in one command
│
├── step1_generate_dataset.py      # Synthetic dataset generation
├── step2_eda.py                   # Exploratory Data Analysis
├── step3_classification.py        # Classification  (Random Forest)
├── step4_clustering.py            # Clustering      (K-Means)
├── step5_association_rules.py     # Association Rule Mining (Apriori)
├── step6_regression.py            # Regression      (Linear + RF)
│
├── requirements.txt
│
├── data/                          # Created on first run
│   └── uae_finapp_dataset.csv     # 500 rows × 20 columns
│
└── outputs/                       # Created on first run
    ├── classification_report.txt
    ├── regression_report.txt
    ├── cluster_profiles.csv
    ├── association_rules.csv
    └── charts/                    # 14 PNG charts
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python run_all.py
```

Or run each step individually:

```bash
python step1_generate_dataset.py   # must run first
python step2_eda.py
python step3_classification.py
python step4_clustering.py
python step5_association_rules.py
python step6_regression.py
```

---

## Business Idea

A mobile-first personal finance app designed for the UAE market, targeting
young professionals, students, and expatriates. Core features include:

- Real-time spending categorisation
- Goal-based savings automation
- Micro-investment into diversified portfolios starting from AED 50
- Sharia-compliant investment toggle
- AI-driven financial insights

**Market context:** 72 % of UAE banking customers prefer mobile apps as their
primary channel; fewer than 30 % of UAE residents under 35 hold any formal
investment product — a significant gap the app aims to fill.

---

## Dataset (Step 1)

500 synthetic respondents with 20 columns:

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | 21–44 |
| `gender` | str | Male / Female |
| `nationality` | str | Emirati, Expat-Arab, Expat-South-Asian, Expat-Western, Other |
| `employment_status` | str | Student, Early-Career, Mid-Career, Freelancer |
| `monthly_income_aed` | int | Stratified by employment status |
| `financial_literacy_score` | int | 1 (low) – 5 (high) |
| `current_savings_level` | str | None / <5k / 5k-20k / >20k AED |
| `has_existing_investments` | int | Binary flag |
| `sharia_compliant_preference` | int | Binary flag |
| `uses_*` (6 columns) | int | Binary feature-usage flags |
| `app_sessions_per_week` | int | Engagement metric |
| `satisfaction_score` | float | 1.0 – 5.0 |
| `monthly_investment_aed` | int | **Regression target** |
| `will_adopt_microinvestment` | int | **Classification target** (0/1) |

---

## Algorithms

### Step 3 — Classification (Random Forest)
Predicts whether a user will adopt micro-investment features.
- **Top predictors:** monthly income, financial literacy, existing investments
- **Test accuracy:** ~67 % | **ROC-AUC:** ~0.69

### Step 4 — Clustering (K-Means, k=4)
Segments users into four actionable personas:

| Persona | Profile | Strategy |
|---------|---------|----------|
| Cautious Savers | Young, low literacy, low income | Education-led onboarding |
| Aspiring Investors | Moderate income, high literacy, high motivation | Investment challenges + gamification |
| Passive Professionals | Higher income, moderate engagement | Auto-invest nudges |
| Engaged High-Earners | High income, active users | Premium tier upsell |

### Step 5 — Association Rule Mining (Apriori, from scratch)
Discovers co-adopted feature bundles.
- **Strongest bundle:** Savings Goals ↔ Financial Insights (confidence ≈ 71 %)
- **Core budgeting pair:** Savings Goals ↔ Spending Tracker (support ≈ 52 %)

### Step 6 — Regression (Linear Regression + Random Forest)
Forecasts monthly investment amount per user.
- **Linear Regression R²:** 0.884 | **MAE:** AED ~159
- **Top levers:** Having existing investments (+AED 458), Auto-Invest feature (+AED 298), Portfolio View (+AED 266)

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computation & synthetic data generation |
| `pandas` | Data manipulation and CSV I/O |
| `scikit-learn` | Classification, clustering, regression, preprocessing |
| `matplotlib` | All chart rendering |
| `seaborn` | Heatmaps and statistical plots |

> No external ARM library (e.g., mlxtend) is required — Apriori is implemented from scratch in `step5_association_rules.py`.

---

## Output Charts

| File | Description |
|------|-------------|
| `eda_01_adoption_by_employment_and_literacy.png` | Adoption rate by employment & literacy |
| `eda_02_income_distribution_by_adoption.png` | Income distribution split by adoption |
| `eda_03_feature_usage_heatmap.png` | Feature usage: adopters vs non-adopters |
| `eda_04_nationality_savings_breakdown.png` | Savings level by nationality group |
| `eda_05_correlation_matrix.png` | Correlation matrix of numeric variables |
| `clf_01_feature_importance.png` | Random Forest feature importances |
| `clf_02_confusion_matrix.png` | Confusion matrix |
| `clust_01_elbow.png` | Elbow method (optimal k) |
| `clust_02_scatter.png` | Income vs investment coloured by cluster |
| `clust_03_radar.png` | Radar chart of cluster profiles |
| `clust_04_heatmap.png` | Cluster attribute heatmap |
| `arm_01_rules_bar.png` | Top rules — support, confidence, lift |
| `arm_02_scatter.png` | Support vs confidence bubble chart |
| `reg_01_actual_vs_predicted.png` | Actual vs predicted (LR vs RF) |
| `reg_02_coefficients.png` | Linear regression coefficients |
| `reg_03_residuals.png` | Residual distribution |

---

*MBA Data Analytics — Individual PBL | Dr. Anshul Gupta*
