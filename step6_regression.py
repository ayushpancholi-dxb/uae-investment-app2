"""
Step 6 — Regression: Forecasting Monthly Investment Amount
===========================================================
UAE Personal Finance & Micro-Investment App
MBA Data Analytics — Individual PBL

Trains a Linear Regression model to forecast each user's expected monthly
investment amount (AED). Results are compared against a Random Forest
Regressor to demonstrate the trade-off between interpretability and
predictive power.

Outputs
-------
outputs/charts/reg_01_actual_vs_predicted.png
outputs/charts/reg_02_coefficients.png
outputs/charts/reg_03_residuals.png
outputs/regression_report.txt
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "uae_finapp_dataset.csv")
CHARTS_DIR  = os.path.join("outputs", "charts")
OUT_DIR     = "outputs"
os.makedirs(CHARTS_DIR, exist_ok=True)

SEED   = 42
BLUE1  = "#1B4F72"
BLUE2  = "#2E86C1"
ACCENT = "#F39C12"

# ── Load & encode ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
le = LabelEncoder()
df["employment_enc"] = le.fit_transform(df["employment_status"])
df["savings_enc"]    = le.fit_transform(df["current_savings_level"])

FEATURE_COLS = [
    "monthly_income_aed",
    "financial_literacy_score",
    "has_existing_investments",
    "uses_auto_invest",
    "uses_portfolio_view",
    "uses_savings_goals",
    "uses_financial_insights",
    "app_sessions_per_week",
    "satisfaction_score",
    "employment_enc",
]
FEATURE_LABELS = [
    "Monthly Income",
    "Financial Literacy",
    "Has Existing Investments",
    "Uses Auto-Invest",
    "Uses Portfolio View",
    "Uses Savings Goals",
    "Uses Financial Insights",
    "App Sessions / Week",
    "Satisfaction Score",
    "Employment Status",
]
TARGET = "monthly_investment_aed"

X = df[FEATURE_COLS]
y = df[TARGET]

# ── Train / Test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED
)

# ── Linear Regression ─────────────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

r2_lr   = r2_score(y_test, y_pred_lr)
mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

cv_r2_lr = cross_val_score(lr, X, y, cv=5, scoring="r2").mean()

# ── Random Forest Regressor (for comparison) ───────────────────────────────
rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

r2_rf   = r2_score(y_test, y_pred_rf)
mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# ── Console & file report ──────────────────────────────────────────────────
report_lines = [
    "REGRESSION REPORT",
    "=" * 55,
    "",
    "Linear Regression",
    "-" * 30,
    f"  R²               : {r2_lr:.4f}",
    f"  MAE              : AED {mae_lr:.1f}",
    f"  RMSE             : AED {rmse_lr:.1f}",
    f"  CV R² (5-fold)   : {cv_r2_lr:.4f}",
    "",
    "Random Forest Regressor",
    "-" * 30,
    f"  R²               : {r2_rf:.4f}",
    f"  MAE              : AED {mae_rf:.1f}",
    f"  RMSE             : AED {rmse_rf:.1f}",
    "",
    "Linear Regression Coefficients",
    "-" * 30,
]
coef_df = pd.DataFrame({
    "Feature":     FEATURE_LABELS,
    "Coefficient": lr.coef_,
}).sort_values("Coefficient", ascending=False)
report_lines.append(coef_df.to_string(index=False))
report_lines.append(f"\n  Intercept: {lr.intercept_:.2f}")

report_str = "\n".join(report_lines)
print(report_str)

report_path = os.path.join(OUT_DIR, "regression_report.txt")
with open(report_path, "w") as f:
    f.write(report_str)
print(f"\nReport saved: {report_path}")


# ── Chart 1: Actual vs Predicted (Linear vs RF side-by-side) ──────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_pred, title, color in [
    (axes[0], y_pred_lr, f"Linear Regression  (R²={r2_lr:.3f})", BLUE2),
    (axes[1], y_pred_rf, f"Random Forest Reg  (R²={r2_rf:.3f})", BLUE1),
]:
    ax.scatter(y_test, y_pred, alpha=0.45, color=color, edgecolors="white",
               s=30, linewidths=0.4)
    lim_lo = min(y_test.min(), y_pred.min()) - 100
    lim_hi = max(y_test.max(), y_pred.max()) + 100
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", linewidth=1.8,
            label="Perfect fit")
    ax.set_xlabel("Actual Monthly Investment (AED)", fontsize=10)
    ax.set_ylabel("Predicted Monthly Investment (AED)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

fig.suptitle(
    "Actual vs Predicted Monthly Investment — Model Comparison",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
path1 = os.path.join(CHARTS_DIR, "reg_01_actual_vs_predicted.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path1}")


# ── Chart 2: Linear Regression Coefficients ───────────────────────────────
coef_sorted = coef_df.sort_values("Coefficient")
colors = [BLUE1 if v >= 0 else ACCENT for v in coef_sorted["Coefficient"]]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(coef_sorted["Feature"], coef_sorted["Coefficient"],
        color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title(
    "Linear Regression Coefficients\n(effect on Monthly Investment AED)",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("Coefficient Value")
for i, (_, row) in enumerate(coef_sorted.iterrows()):
    offset = 5 if row["Coefficient"] >= 0 else -5
    ha     = "left" if row["Coefficient"] >= 0 else "right"
    ax.text(row["Coefficient"] + offset, i, f"{row['Coefficient']:+.1f}",
            va="center", fontsize=9, ha=ha)
plt.tight_layout()
path2 = os.path.join(CHARTS_DIR, "reg_02_coefficients.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path2}")


# ── Chart 3: Residual plot ─────────────────────────────────────────────────
residuals = y_test - y_pred_lr

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Residuals vs fitted
axes[0].scatter(y_pred_lr, residuals, alpha=0.45, color=BLUE2,
                edgecolors="white", s=28)
axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[0].set_xlabel("Fitted Values (AED)", fontsize=10)
axes[0].set_ylabel("Residuals (AED)", fontsize=10)
axes[0].set_title("Residuals vs Fitted Values", fontsize=12, fontweight="bold")

# Residual histogram
axes[1].hist(residuals, bins=35, color=BLUE1, edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Residual (AED)", fontsize=10)
axes[1].set_ylabel("Frequency", fontsize=10)
axes[1].set_title(
    f"Residual Distribution\n(mean={residuals.mean():.1f}, std={residuals.std():.1f})",
    fontsize=12, fontweight="bold",
)

plt.tight_layout()
path3 = os.path.join(CHARTS_DIR, "reg_03_residuals.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path3}")

print("\nRegression complete.")
