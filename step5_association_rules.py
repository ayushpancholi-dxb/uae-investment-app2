"""
Step 5 — Association Rule Mining: Feature Bundle Analysis
==========================================================
UAE Personal Finance & Micro-Investment App
MBA Data Analytics — Individual PBL

Implements the Apriori algorithm from scratch (no external ARM library
required) on the binary feature-usage matrix to discover which app
features are co-adopted by users.

Outputs
-------
outputs/charts/arm_01_rules_bar.png
outputs/charts/arm_02_scatter.png
outputs/association_rules.csv
"""

import os
import warnings
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "uae_finapp_dataset.csv")
CHARTS_DIR  = os.path.join("outputs", "charts")
OUT_DIR     = "outputs"
os.makedirs(CHARTS_DIR, exist_ok=True)

BLUE1  = "#1B4F72"
BLUE2  = "#2E86C1"
ACCENT = "#F39C12"
GREEN  = "#27AE60"

MIN_SUPPORT    = 0.30   # 30 % of users must co-use both features
MIN_CONFIDENCE = 0.50   # at least 50 % conditional probability

ITEM_COLS = [
    "uses_spending_tracker",
    "uses_savings_goals",
    "uses_auto_invest",
    "uses_portfolio_view",
    "uses_sharia_filter",
    "uses_financial_insights",
]
ITEM_NAMES = {
    "uses_spending_tracker":   "SpendTrack",
    "uses_savings_goals":      "SavingsGoal",
    "uses_auto_invest":        "AutoInvest",
    "uses_portfolio_view":     "Portfolio",
    "uses_sharia_filter":      "ShariaFilter",
    "uses_financial_insights": "Insights",
}

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from {DATA_PATH}")

# ── Build transaction list ─────────────────────────────────────────────────
transactions = []
for _, row in df[ITEM_COLS].iterrows():
    basket = frozenset(ITEM_NAMES[c] for c in ITEM_COLS if row[c] == 1)
    if basket:
        transactions.append(basket)

n_trans = len(transactions)
print(f"Transactions (non-empty baskets): {n_trans}")


# ── Apriori helper ─────────────────────────────────────────────────────────
def support(itemset: frozenset, txs: list) -> float:
    """Fraction of transactions containing all items in itemset."""
    return sum(1 for t in txs if itemset.issubset(t)) / len(txs)


# ── Frequent 1-itemsets ────────────────────────────────────────────────────
all_items = sorted({item for t in transactions for item in t})
freq1 = {
    frozenset([item]): support(frozenset([item]), transactions)
    for item in all_items
}
freq1 = {k: v for k, v in freq1.items() if v >= MIN_SUPPORT}

print(f"\nFrequent 1-itemsets (support ≥ {MIN_SUPPORT:.0%}):")
for k, v in sorted(freq1.items(), key=lambda x: -x[1]):
    print(f"  {list(k)[0]:<15} support = {v:.3f}")

# ── Frequent 2-itemsets ────────────────────────────────────────────────────
freq2 = {}
for a, b in combinations(freq1.keys(), 2):
    pair = a | b
    s = support(pair, transactions)
    if s >= MIN_SUPPORT:
        freq2[pair] = s

print(f"\nFrequent 2-itemsets (support ≥ {MIN_SUPPORT:.0%}): {len(freq2)} found")

# ── Generate association rules ─────────────────────────────────────────────
rules = []
for itemset, sup in freq2.items():
    items = sorted(itemset)
    for i in range(len(items)):
        ant  = frozenset([items[i]])
        cons = frozenset([items[1 - i]])
        ant_sup  = support(ant,  transactions)
        cons_sup = support(cons, transactions)
        conf = sup / ant_sup
        lift = conf / cons_sup
        if conf >= MIN_CONFIDENCE:
            rules.append({
                "Antecedent":  list(ant)[0],
                "Consequent":  list(cons)[0],
                "Support":     round(sup, 4),
                "Confidence":  round(conf, 4),
                "Lift":        round(lift, 4),
            })

rules_df = (
    pd.DataFrame(rules)
    .drop_duplicates()
    .sort_values("Lift", ascending=False)
    .reset_index(drop=True)
)

print(f"\nAssociation rules (confidence ≥ {MIN_CONFIDENCE:.0%}): {len(rules_df)} found")
print(rules_df.to_string(index=False))

rules_df.to_csv(os.path.join(OUT_DIR, "association_rules.csv"), index=False)
print(f"\n  Saved: {os.path.join(OUT_DIR, 'association_rules.csv')}")


# ── Chart 1: Grouped bar — Support / Confidence / Lift ────────────────────
top = rules_df.head(8).copy()
top["Rule"] = top["Antecedent"] + " → " + top["Consequent"]

x  = np.arange(len(top))
w  = 0.26
fig, ax = plt.subplots(figsize=(13, 6))

ax.bar(x - w,  top["Support"],    width=w, label="Support",    color=BLUE1,  edgecolor="white")
ax.bar(x,      top["Confidence"], width=w, label="Confidence", color=BLUE2,  edgecolor="white")
ax.bar(x + w,  top["Lift"],       width=w, label="Lift",       color=ACCENT, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(top["Rule"], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Metric Value")
ax.set_title(
    "Top Association Rules — Support, Confidence & Lift",
    fontsize=13, fontweight="bold",
)
ax.legend(fontsize=10)
ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
plt.tight_layout()
path1 = os.path.join(CHARTS_DIR, "arm_01_rules_bar.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path1}")


# ── Chart 2: Support vs Confidence bubble (size = Lift) ──────────────────
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(
    rules_df["Support"],
    rules_df["Confidence"],
    s=rules_df["Lift"] * 300,
    c=rules_df["Lift"],
    cmap="Blues",
    alpha=0.80,
    edgecolors=BLUE1,
    linewidths=0.8,
)
plt.colorbar(sc, ax=ax, label="Lift")

# Annotate each point with its rule
for _, row in rules_df.iterrows():
    label = f"{row['Antecedent'][:6]}→{row['Consequent'][:6]}"
    ax.annotate(
        label,
        (row["Support"], row["Confidence"]),
        textcoords="offset points", xytext=(6, 4),
        fontsize=7.5, color="#333333",
    )

ax.set_xlabel("Support", fontsize=11)
ax.set_ylabel("Confidence", fontsize=11)
ax.set_title(
    "Association Rules — Support vs Confidence\n(bubble size ∝ Lift)",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
path2 = os.path.join(CHARTS_DIR, "arm_02_scatter.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path2}")

print("\nAssociation Rule Mining complete.")
