"""
Step 3 — Classification: Predicting Micro-Investment Adoption
==============================================================
UAE Personal Finance & Micro-Investment App
MBA Data Analytics — Individual PBL

Trains a Random Forest Classifier to predict whether a user will adopt
the app's micro-investment features (binary target: will_adopt_microinvestment).

Outputs
-------
outputs/charts/clf_01_feature_importance.png
outputs/charts/clf_02_confusion_matrix.png
outputs/classification_report.txt
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "uae_finapp_dataset.csv")
CHARTS_DIR  = os.path.join("outputs", "charts")
REPORT_DIR  = "outputs"
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

SEED   = 42
BLUE1  = "#1B4F72"
BLUE2  = "#2E86C1"
ACCENT = "#F39C12"

# ── Load & encode ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

le = LabelEncoder()
df["employment_enc"]    = le.fit_transform(df["employment_status"])
df["savings_enc"]       = le.fit_transform(df["current_savings_level"])

FEATURE_COLS = [
    "age",
    "monthly_income_aed",
    "financial_literacy_score",
    "has_existing_investments",
    "sharia_compliant_preference",
    "uses_spending_tracker",
    "uses_savings_goals",
    "uses_auto_invest",
    "uses_portfolio_view",
    "uses_financial_insights",
    "app_sessions_per_week",
    "employment_enc",
    "savings_enc",
]
FEATURE_LABELS = [
    "Age",
    "Monthly Income",
    "Financial Literacy",
    "Has Investments",
    "Sharia Preference",
    "Uses Spending Tracker",
    "Uses Savings Goals",
    "Uses Auto-Invest",
    "Uses Portfolio View",
    "Uses Financial Insights",
    "App Sessions / Week",
    "Employment Status",
    "Savings Level",
]
TARGET = "will_adopt_microinvestment"

X = df[FEATURE_COLS]
y = df[TARGET]

# ── Train / Test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED
)

# ── Model ──────────────────────────────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    random_state=SEED,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# ── Cross-validation ───────────────────────────────────────────────────────
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

# ── Console summary ────────────────────────────────────────────────────────
report_str = classification_report(y_test, y_pred,
                                   target_names=["Non-Adopter", "Adopter"])
auc = roc_auc_score(y_test, y_proba)

print("=" * 55)
print("  CLASSIFICATION — Random Forest")
print("=" * 55)
print(f"  Train size  : {len(X_train)}")
print(f"  Test size   : {len(X_test)}")
print(f"  Test accuracy  : {(y_pred == y_test).mean():.3f}")
print(f"  ROC-AUC        : {auc:.3f}")
print(f"  CV accuracy    : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
print("-" * 55)
print(report_str)

# ── Save text report ───────────────────────────────────────────────────────
report_path = os.path.join(REPORT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("CLASSIFICATION REPORT — Random Forest\n")
    f.write("=" * 55 + "\n")
    f.write(f"Train size      : {len(X_train)}\n")
    f.write(f"Test size       : {len(X_test)}\n")
    f.write(f"Test accuracy   : {(y_pred == y_test).mean():.3f}\n")
    f.write(f"ROC-AUC         : {auc:.3f}\n")
    f.write(f"CV accuracy (5-fold): {cv_acc.mean():.3f} ± {cv_acc.std():.3f}\n\n")
    f.write(report_str)
print(f"Report saved: {report_path}")


# ── Chart 1: Feature Importance ───────────────────────────────────────────
importances = pd.Series(clf.feature_importances_, index=FEATURE_LABELS)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = [BLUE1 if v >= importances.median() else BLUE2 for v in importances.values]
bars = ax.barh(importances.index, importances.values, color=colors, edgecolor="white")
ax.set_title(
    "Feature Importances — Random Forest Classifier",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("Importance Score (Mean Decrease in Impurity)")
for bar in bars:
    ax.text(
        bar.get_width() + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width():.3f}",
        va="center", fontsize=9,
    )
plt.tight_layout()
path1 = os.path.join(CHARTS_DIR, "clf_01_feature_importance.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path1}")


# ── Chart 2: Confusion Matrix ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Adopter", "Adopter"],
)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(
    f"Confusion Matrix\nRandom Forest  |  Accuracy = {(y_pred == y_test).mean():.2%}",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
path2 = os.path.join(CHARTS_DIR, "clf_02_confusion_matrix.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path2}")

print("\nClassification complete.")
