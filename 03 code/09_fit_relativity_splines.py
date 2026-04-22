"""
09_fit_relativity_splines.py
==============================
Compares model-predicted qx to CDC population qx to compute
mortality relativities. Analyzes how relativities evolve across
duration years 1-5 by risk group, revealing the convergence
pattern needed for projection beyond year 5.

Relativity = qx_xgb / qx_population
  >1 means higher mortality than population
  =1 means population-average mortality
  <1 means lower mortality than population

Input:  02 processed data - nhanes_predictions_annual.csv
        02 processed data - cdc_life_table.csv
Output: 05 artifacts - relativity_by_decile.csv
        05 artifacts - relativity_convergence.png
        05 artifacts - relativity_by_year_scatter.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR  = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
PRED_PATH    = os.path.join(PROJECT_DIR, "02 processed data",
                            "nhanes_predictions_annual.csv")
CDC_PATH     = os.path.join(PROJECT_DIR, "02 processed data",
                            "adjusted_cdc_life_table.csv")
ARTIFACT_DIR = os.path.join(PROJECT_DIR, "05 artifacts")

sns.set_theme(style="whitegrid", font_scale=1.1)

# =====================================================================
# STEP 1: Load data
# =====================================================================
print("=" * 65)
print("  09 - Fit Relativity Splines")
print("=" * 65)

df = pd.read_csv(PRED_PATH)
cdc = pd.read_csv(CDC_PATH)
print(f"\n  Predictions: {len(df):,} rows")
print(f"  CDC table: {len(cdc)} rows")

# Use test set only
df = df[df["SPLIT"] == "TEST"].copy()
print(f"  Test set: {len(df):,} rows")

# =====================================================================
# STEP 2: Compute individual relativities for each year
# =====================================================================
# For year k, attained age = exam_age + k - 1
# Relativity = qx_xgb / qx_population(attained_age, sex)
print("\nComputing relativities...")

df["exam_age"] = df["RIDAGEYR"].astype(int)
df["sex"] = df["IS_MALE"].map({1: "Male", 0: "Female"})

# Build a lookup dict from CDC table: (age, sex) -> qx
cdc_lookup = {}
for _, row in cdc.iterrows():
    cdc_lookup[(int(row["age"]), row["sex"])] = row["qx_adjusted"]

for yr in range(1, 6):
    attained = (df["exam_age"] + yr - 1).clip(upper=100)
    # Look up population qx at attained age
    df[f"qx_pop_yr{yr}"] = [
        cdc_lookup.get((a, s), np.nan)
        for a, s in zip(attained, df["sex"])
    ]
    # Relativity: model qx / population qx
    df[f"rel_yr{yr}"] = df[f"q{yr}_xgb"] / df[f"qx_pop_yr{yr}"]

# =====================================================================
# STEP 3: Assign risk deciles based on year-1 relativity
# =====================================================================
# Decile 1 = lowest risk, Decile 10 = highest risk
print("Assigning risk deciles based on year-1 relativity...")
df["risk_decile"] = pd.qcut(df["rel_yr1"], 10, labels=False,
                             duplicates="drop") + 1

# =====================================================================
# STEP 4: Compute average relativity by decile and year
# =====================================================================
print("\nComputing relativity by decile and duration year...")
rel_cols = [f"rel_yr{k}" for k in range(1, 6)]

decile_summary = []
for dec in sorted(df["risk_decile"].dropna().unique()):
    subset = df[df["risk_decile"] == dec]
    row = {"decile": int(dec), "n": len(subset)}
    for yr in range(1, 6):
        row[f"rel_yr{yr}"] = subset[f"rel_yr{yr}"].median()
    decile_summary.append(row)

ds = pd.DataFrame(decile_summary)

# Print summary
print(f"\n  {'Decile':>7}  {'n':>6}  " +
      "  ".join([f"Yr{k}" for k in range(1, 6)]))
print(f"  {'-----':>7}  {'---':>6}  " +
      "  ".join(["------"] * 5))
for _, r in ds.iterrows():
    vals = "  ".join([f"{r[f'rel_yr{k}']:6.3f}" for k in range(1, 6)])
    print(f"  {int(r['decile']):>7}  {int(r['n']):>6}  {vals}")

# =====================================================================
# STEP 5: Convergence chart - relativities over time by decile
# =====================================================================
print("\nGenerating convergence chart...")
fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 10))

for i, (_, r) in enumerate(ds.iterrows()):
    yrs = range(1, 6)
    vals = [r[f"rel_yr{k}"] for k in yrs]
    ax.plot(yrs, vals, "o-", color=colors[i], linewidth=2,
            markersize=6, label=f"D{int(r['decile'])} (n={int(r['n'])})")

ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5,
           label="Population (1.0)")
ax.set_xlabel("Duration Year Since Exam", fontsize=12)
ax.set_ylabel("Median Relativity (qx_model / qx_population)", fontsize=12)
ax.set_title("Mortality Relativity Convergence by Risk Decile",
             fontsize=14, fontweight="bold")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
ax.set_xticks(range(1, 6))
ax.set_ylim(bottom=0)
fig.tight_layout()
path = os.path.join(ARTIFACT_DIR, "relativity_convergence.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: relativity_convergence.png")

# =====================================================================
# STEP 6: Year-over-year relativity stability scatter
# =====================================================================
# How does year-1 relativity predict later-year relativities?
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for i, yr in enumerate([2, 3, 4, 5]):
    ax = axes[i]
    x = df["rel_yr1"].clip(upper=10)
    y = df[f"rel_yr{yr}"].clip(upper=10)
    ax.scatter(x, y, alpha=0.05, s=3, color="#2E5090")
    ax.plot([0, 10], [0, 10], "r--", linewidth=1)
    corr = x.corr(y)
    ax.set_title(f"Year 1 vs Year {yr}\n(r={corr:.3f})", fontsize=11)
    ax.set_xlabel("Year 1 Relativity")
    ax.set_ylabel(f"Year {yr} Relativity")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

fig.suptitle("Relativity Persistence: Year 1 vs Later Years",
             fontsize=14, fontweight="bold")
fig.tight_layout()
path = os.path.join(ARTIFACT_DIR, "relativity_by_year_scatter.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: relativity_by_year_scatter.png")

# =====================================================================
# STEP 7: Save summary and compute decay rates
# =====================================================================
# For each decile, compute how fast relativity moves toward 1.0
# Decay factor per year: (rel_yr5 - 1) / (rel_yr1 - 1)
# If rel_yr1 == 1.0, ratio is undefined (already at baseline)
ds["rel_change_yr1_to_yr5"] = (ds["rel_yr5"] - 1) / (ds["rel_yr1"] - 1)

# Compute annual decay rate (geometric)
# If rel_yr1 = r1 and rel_yr5 = r5, and both > 1 or both < 1:
# annual_decay = ((r5 - 1) / (r1 - 1))^(1/4) over 4 intervals
ds["annual_decay"] = np.nan
for idx, r in ds.iterrows():
    r1 = r["rel_yr1"] - 1
    r5 = r["rel_yr5"] - 1
    if abs(r1) > 0.01:
        ratio = r5 / r1
        if ratio > 0:
            ds.loc[idx, "annual_decay"] = ratio ** (1/4)

# Save
summary_path = os.path.join(ARTIFACT_DIR, "relativity_by_decile.csv")
ds.to_csv(summary_path, index=False)
print(f"\n  Saved: relativity_by_decile.csv")

# Print decay rates
print(f"\n  Decay Analysis (how fast relativities move toward 1.0):")
print(f"  {'Decile':>7}  {'Yr1 Rel':>8}  {'Yr5 Rel':>8}  "
      f"{'Yr1->5 Chg':>11}  {'Ann Decay':>10}")
print(f"  {'-----':>7}  {'------':>8}  {'------':>8}  "
      f"{'----------':>11}  {'---------':>10}")
for _, r in ds.iterrows():
    chg = f"{r['rel_change_yr1_to_yr5']:.3f}" if pd.notna(r['rel_change_yr1_to_yr5']) else "n/a"
    decay = f"{r['annual_decay']:.3f}" if pd.notna(r['annual_decay']) else "n/a"
    print(f"  {int(r['decile']):>7}  {r['rel_yr1']:>8.3f}  "
          f"{r['rel_yr5']:>8.3f}  {chg:>11}  {decay:>10}")

print(f"\n  Notes:")
print(f"    Yr1->5 Chg: ratio of (rel_yr5 - 1)/(rel_yr1 - 1)")
print(f"      Values < 1 mean relativity is moving toward 1.0")
print(f"      Values > 1 mean relativity is moving away from 1.0")
print(f"    Ann Decay: geometric annual rate of that change")

print(f"\n\nAll outputs saved to: {ARTIFACT_DIR}")
print("Done!")
