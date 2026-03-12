"""
Step 2 — Exploratory Data Analysis (EDA)
==========================================
UAE Personal Finance & Micro-Investment App
MBA Data Analytics — Individual PBL

Loads the synthetic dataset and produces five publication-quality charts
that describe the data before modelling begins.

Charts saved
------------
outputs/charts/eda_01_adoption_by_employment_and_literacy.png
outputs/charts/eda_02_income_distribution_by_adoption.png
outputs/charts/eda_03_feature_usage_heatmap.png
outputs/charts/eda_04_nationality_savings_breakdown.png
outputs/charts/eda_05_correlation_matrix.png
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "uae_finapp_dataset.csv")
CHARTS_DIR  = os.path.join("outputs", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

BLUE1  = "#1B4F72"
BLUE2  = "#2E86C1"
BLUE3  = "#85C1E9"
ACCENT = "#F39C12"
GRAY   = "#BDC3C7"

FEATURE_COLS = [
    "uses_spending_tracker",
    "uses_savings_goals",
    "uses_auto_invest",
    "uses_portfolio_view",
    "uses_sharia_filter",
    "uses_financial_insights",
]
FEATURE_LABELS = [
    "Spending\nTracker",
    "Savings\nGoals",
    "Auto\nInvest",
    "Portfolio\nView",
    "Sharia\nFilter",
    "Financial\nInsights",
]

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows × {len(df.columns)} columns from {DATA_PATH}")


# ──────────────────────────────────────────────────────────────────────────
# Chart 1: Adoption rate by Employment Status AND Financial Literacy
# ──────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Micro-Investment Adoption Rate — Employment & Literacy",
    fontsize=14, fontweight="bold", y=1.01,
)

# Left: Employment
adopt_emp = (
    df.groupby("employment_status")["will_adopt_microinvestment"]
    .mean()
    .sort_values(ascending=False)
)
bars = axes[0].bar(
    adopt_emp.index, adopt_emp.values,
    color=[BLUE1, BLUE2, BLUE3, GRAY], edgecolor="white", width=0.55,
)
axes[0].set_title("By Employment Status", fontsize=12)
axes[0].set_ylabel("Adoption Rate")
axes[0].set_ylim(0, 1.0)
axes[0].tick_params(axis="x", rotation=20)
for bar in bars:
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{bar.get_height():.0%}",
        ha="center", fontsize=10, fontweight="bold",
    )

# Right: Financial Literacy
lit_adopt = (
    df.groupby("financial_literacy_score")["will_adopt_microinvestment"]
    .mean()
)
bars2 = axes[1].bar(
    lit_adopt.index, lit_adopt.values,
    color=[BLUE1, BLUE2, BLUE3, BLUE2, BLUE1], edgecolor="white", width=0.6,
)
axes[1].set_title("By Financial Literacy Score (1 = Low, 5 = High)", fontsize=12)
axes[1].set_xlabel("Financial Literacy Score")
axes[1].set_ylabel("Adoption Rate")
axes[1].set_ylim(0, 1.0)
for bar in bars2:
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{bar.get_height():.0%}",
        ha="center", fontsize=10, fontweight="bold",
    )

plt.tight_layout()
path1 = os.path.join(CHARTS_DIR, "eda_01_adoption_by_employment_and_literacy.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path1}")


# ──────────────────────────────────────────────────────────────────────────
# Chart 2: Income distribution by adoption status
# ──────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
adopters     = df[df["will_adopt_microinvestment"] == 1]["monthly_income_aed"]
non_adopters = df[df["will_adopt_microinvestment"] == 0]["monthly_income_aed"]

ax.hist(non_adopters, bins=35, alpha=0.65, color=BLUE3,
        label="Will NOT Adopt", edgecolor="white")
ax.hist(adopters,     bins=35, alpha=0.80, color=BLUE1,
        label="Will Adopt",     edgecolor="white")

med = df["monthly_income_aed"].median()
ax.axvline(med, color=ACCENT, linestyle="--", linewidth=2.0,
           label=f"Median = AED {med:,.0f}")

ax.set_title(
    "Monthly Income Distribution by Micro-Investment Adoption",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("Monthly Income (AED)", fontsize=11)
ax.set_ylabel("Number of Respondents", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
path2 = os.path.join(CHARTS_DIR, "eda_02_income_distribution_by_adoption.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path2}")


# ──────────────────────────────────────────────────────────────────────────
# Chart 3: Feature usage heatmap — Adopters vs Non-Adopters
# ──────────────────────────────────────────────────────────────────────────
feat_rates = df.groupby("will_adopt_microinvestment")[FEATURE_COLS].mean()
feat_rates.index = ["Non-Adopter", "Adopter"]
feat_rates.columns = FEATURE_LABELS

fig, ax = plt.subplots(figsize=(11, 4))
sns.heatmap(
    feat_rates,
    annot=True, fmt=".0%",
    cmap="Blues",
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Usage Rate"},
    ax=ax,
    vmin=0, vmax=1,
)
ax.set_title(
    "Feature Usage Rate: Adopters vs Non-Adopters",
    fontsize=13, fontweight="bold",
)
ax.set_ylabel("")
plt.tight_layout()
path3 = os.path.join(CHARTS_DIR, "eda_03_feature_usage_heatmap.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path3}")


# ──────────────────────────────────────────────────────────────────────────
# Chart 4: Nationality × savings level stacked bar
# ──────────────────────────────────────────────────────────────────────────
savings_order = ["None", "<5k AED", "5k-20k AED", ">20k AED"]
nat_sav = (
    df.groupby(["nationality", "current_savings_level"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=savings_order, fill_value=0)
)
nat_sav_pct = nat_sav.div(nat_sav.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(11, 5))
palette = [BLUE1, BLUE2, BLUE3, ACCENT]
nat_sav_pct.plot(
    kind="bar", stacked=True, ax=ax,
    color=palette, edgecolor="white", width=0.65,
)
ax.set_title(
    "Savings Level Distribution by Nationality Group",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("")
ax.set_ylabel("Proportion of Respondents")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=20)
ax.legend(title="Savings Level", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
path4 = os.path.join(CHARTS_DIR, "eda_04_nationality_savings_breakdown.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path4}")


# ──────────────────────────────────────────────────────────────────────────
# Chart 5: Correlation matrix (numeric columns)
# ──────────────────────────────────────────────────────────────────────────
num_cols = [
    "age", "monthly_income_aed", "financial_literacy_score",
    "has_existing_investments", "app_sessions_per_week",
    "satisfaction_score", "monthly_investment_aed",
    "will_adopt_microinvestment",
]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask,
    annot=True, fmt=".2f",
    cmap="coolwarm", center=0,
    linewidths=0.4, linecolor="white",
    cbar_kws={"shrink": 0.8},
    ax=ax,
    vmin=-1, vmax=1,
)
ax.set_title(
    "Correlation Matrix — Key Numeric Variables",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
path5 = os.path.join(CHARTS_DIR, "eda_05_correlation_matrix.png")
plt.savefig(path5, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path5}")


print("\nEDA complete — all 5 charts saved to", CHARTS_DIR)
