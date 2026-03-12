"""
run_all.py — Full Pipeline Runner
===================================
UAE Personal Finance & Micro-Investment App
MBA Data Analytics — Individual PBL

Runs all six steps in sequence:
  1. Generate synthetic dataset
  2. Exploratory Data Analysis (EDA)
  3. Classification  (Random Forest)
  4. Clustering      (K-Means)
  5. Association Rule Mining (Apriori)
  6. Regression      (Linear Regression)

Usage
-----
    python run_all.py

All outputs are written to:
    data/          ← CSV dataset
    outputs/       ← text reports + charts/
"""

import subprocess
import sys
import time

STEPS = [
    ("Step 1 — Generate Dataset",        "step1_generate_dataset.py"),
    ("Step 2 — Exploratory Data Analysis","step2_eda.py"),
    ("Step 3 — Classification",          "step3_classification.py"),
    ("Step 4 — Clustering",              "step4_clustering.py"),
    ("Step 5 — Association Rule Mining", "step5_association_rules.py"),
    ("Step 6 — Regression",              "step6_regression.py"),
]

def run_step(label: str, script: str) -> bool:
    print("\n" + "=" * 62)
    print(f"  {label}")
    print("  Script:", script)
    print("=" * 62)
    t0     = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ERROR: {script} exited with code {result.returncode}")
        return False
    print(f"\n  ✓ Completed in {elapsed:.1f}s")
    return True


if __name__ == "__main__":
    total_start = time.time()
    failed      = []

    for label, script in STEPS:
        ok = run_step(label, script)
        if not ok:
            failed.append(script)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 62)
    if failed:
        print(f"  Pipeline finished with ERRORS in: {failed}")
    else:
        print("  ✓ All steps completed successfully")
    print(f"  Total time: {total_elapsed:.1f}s")
    print("=" * 62)
    print("\nOutputs written to:")
    print("  data/uae_finapp_dataset.csv")
    print("  outputs/cluster_profiles.csv")
    print("  outputs/association_rules.csv")
    print("  outputs/classification_report.txt")
    print("  outputs/regression_report.txt")
    print("  outputs/charts/  (14 PNG charts)")
    sys.exit(1 if failed else 0)
