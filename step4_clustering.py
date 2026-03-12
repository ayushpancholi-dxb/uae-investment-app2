"""
Step 4 — Clustering: Identifying User Personas
===============================================
UAE Personal Finance & Micro-Investment App
MBA Data Analytics — Individual PBL

Applies K-Means clustering to segment users into distinct personas.
The Elbow Method is used to select the optimal number of clusters (k).

Outputs
-------
outputs/charts/clust_01_elbow.png
outputs/charts/clust_02_scatter.png
outputs/charts/clust_03_radar.png
outputs/charts/clust_04_heatmap.png
outputs/cluster_profiles.csv
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join("data", "uae_finapp_dataset.csv")
CHARTS_DIR = os.path.join("outputs", "charts")
OUT_DIR    = "outputs"
os.makedirs(CHARTS_DIR, exist_ok=True)

SEED = 42
K    = 4

CLUSTER_COLORS = ["#1B4F72", "#F39C12", "#27AE60", "#E74C3C"]
CLUSTER_NAMES  = [
    "Cautious Savers",
    "Aspiring Investors",
    "Passive Professionals",
    "Engaged High-Earners",
]

CLUSTER_FEATURES = [
    "age",
    "monthly_income_aed",
    "financial_literacy_score",
    "app_sessions_per_week",
    "monthly_investment_aed",
    "satisfaction_score",
]

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from {DATA_PATH}")

# ── Scale ──────────────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])


# ── Chart 1: Elbow Method ─────────────────────────────────────────────────
inertias = []
k_range  = range(2, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_range, inertias, "o-", color="#2E86C1", linewidth=2.5, markersize=8)
ax.axvline(x=K, color="#F39C12", linestyle="--", linewidth=1.8,
           label=f"Chosen k = {K}")
ax.set_title("Elbow Method — Optimal k for K-Means", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia (Within-Cluster SSE)")
ax.legend(fontsize=10)
plt.tight_layout()
path1 = os.path.join(CHARTS_DIR, "clust_01_elbow.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path1}")


# ── Fit final model ────────────────────────────────────────────────────────
km_final    = KMeans(n_clusters=K, random_state=SEED, n_init=10)
df["cluster"] = km_final.fit_predict(X_scaled)

# Map cluster labels to meaningful names based on income/investment ordering
# Sort clusters by average income so names stay consistent
cluster_income = df.groupby("cluster")["monthly_income_aed"].mean().sort_values()
cluster_order  = cluster_income.index.tolist()   # low → high income

name_map = {}
preset   = [
    "Cautious Savers",
    "Aspiring Investors",
    "Passive Professionals",
    "Engaged High-Earners",
]
for new_id, old_id in enumerate(cluster_order):
    name_map[old_id] = preset[new_id]

df["persona"] = df["cluster"].map(name_map)

print("\nCluster distribution:")
print(df["persona"].value_counts().to_string())


# ── Cluster Profiles ───────────────────────────────────────────────────────
profile_cols = {
    "respondent_id": "count",
    "age":                       "mean",
    "monthly_income_aed":        "mean",
    "financial_literacy_score":  "mean",
    "app_sessions_per_week":     "mean",
    "monthly_investment_aed":    "mean",
    "satisfaction_score":        "mean",
    "will_adopt_microinvestment":"mean",
}
profiles = (
    df.groupby("persona")
    .agg(profile_cols)
    .rename(columns={"respondent_id": "n"})
    .round(2)
)
profiles.columns = [
    "n", "avg_age", "avg_income_aed", "avg_literacy",
    "avg_sessions_pw", "avg_investment_aed", "avg_satisfaction", "adoption_rate",
]
print("\nCluster profiles:")
print(profiles.to_string())

profiles.to_csv(os.path.join(OUT_DIR, "cluster_profiles.csv"))
print(f"\n  Saved: {os.path.join(OUT_DIR, 'cluster_profiles.csv')}")

# Assign consistent colours by persona name for all charts
persona_color = {
    "Cautious Savers":       "#1B4F72",
    "Aspiring Investors":    "#F39C12",
    "Passive Professionals": "#27AE60",
    "Engaged High-Earners":  "#E74C3C",
}


# ── Chart 2: Scatter — Income vs Investment ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for persona, grp in df.groupby("persona"):
    ax.scatter(
        grp["monthly_income_aed"], grp["monthly_investment_aed"],
        c=persona_color[persona], label=persona,
        alpha=0.65, s=40, edgecolors="white", linewidths=0.3,
    )
ax.set_xlabel("Monthly Income (AED)", fontsize=11)
ax.set_ylabel("Monthly Investment Amount (AED)", fontsize=11)
ax.set_title(
    "User Segments: Income vs Monthly Investment\n(K-Means, k=4)",
    fontsize=13, fontweight="bold",
)
ax.legend(title="Persona", fontsize=9, framealpha=0.9)
plt.tight_layout()
path2 = os.path.join(CHARTS_DIR, "clust_02_scatter.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path2}")


# ── Chart 3: Radar — Normalised Cluster Attributes ────────────────────────
radar_cols   = ["avg_age", "avg_income_aed", "avg_literacy",
                "avg_sessions_pw", "avg_investment_aed", "adoption_rate"]
radar_labels = ["Age", "Income\n(norm)", "Literacy",
                "Sessions/wk", "Investment", "Adoption Rate"]

# Normalise 0-1
norm = profiles[radar_cols].copy()
for col in radar_cols:
    rng = norm[col].max() - norm[col].min()
    norm[col] = (norm[col] - norm[col].min()) / (rng if rng > 0 else 1)

N_ax = len(radar_labels)
angles = [n / N_ax * 2 * np.pi for n in range(N_ax)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for persona, row in norm.iterrows():
    vals = row[radar_cols].tolist() + [row[radar_cols[0]]]
    ax.plot(angles, vals, color=persona_color[persona], linewidth=2.2)
    ax.fill(angles, vals, color=persona_color[persona], alpha=0.10)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=11)
ax.set_title(
    "Cluster Profile Comparison\n(Normalised Attributes)",
    fontsize=13, fontweight="bold", y=1.10,
)
patches = [
    mpatches.Patch(color=persona_color[p], label=p)
    for p in list(persona_color.keys())
]
ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.40, 1.15),
          fontsize=9)
plt.tight_layout()
path3 = os.path.join(CHARTS_DIR, "clust_03_radar.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path3}")


# ── Chart 4: Heatmap of cluster profiles ──────────────────────────────────
heat_data = profiles[radar_cols].copy()
heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(
    heat_norm,
    annot=profiles[radar_cols].round(1),
    fmt="g",
    cmap="Blues",
    linewidths=0.4, linecolor="white",
    ax=ax,
    cbar_kws={"label": "Normalised Value"},
    xticklabels=radar_labels,
)
ax.set_title("Cluster Attribute Heatmap (raw values annotated)", fontsize=13,
             fontweight="bold")
ax.set_ylabel("")
plt.tight_layout()
path4 = os.path.join(CHARTS_DIR, "clust_04_heatmap.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path4}")

print("\nClustering complete.")
