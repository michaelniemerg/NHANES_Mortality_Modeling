"""
10_project_and_grade.py
========================
Uses the observed relativity convergence pattern from script 09
to project mortality relativities beyond year 5 and grade them
back toward population (1.0) over time.

Approach: Exponential decay of the deviation from 1.0.
  r(t) = 1 + (r_5 - 1) * decay_rate^(t - 5)     for t > 5

The decay_rate is derived from the observed annual decay in
script 09 (~0.90 for most risk groups).

Input:  02 processed data - nhanes_predictions_annual.csv
        02 processed data - cdc_life_table.csv
        05 artifacts - relativity_by_decile.csv
Output: 02 processed data - grading_parameters.csv
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR  = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
PRED_PATH    = os.path.join(PROJECT_DIR, "02 processed data",
                            "nhanes_predictions_annual.csv")
CDC_PATH     = os.path.join(PROJECT_DIR, "02 processed data",
                            "adjusted_cdc_life_table.csv")
DECILE_PATH  = os.path.join(PROJECT_DIR, "05 artifacts",
                            "relativity_by_decile.csv")
OUTPUT_PATH  = os.path.join(PROJECT_DIR, "02 processed data",
                            "grading_parameters.csv")

# =====================================================================
# STEP 1: Load data
# =====================================================================
print("=" * 65)
print("  10 - Project and Grade Relativities")
print("=" * 65)

df = pd.read_csv(PRED_PATH)
cdc = pd.read_csv(CDC_PATH)
decile_df = pd.read_csv(DECILE_PATH)
print(f"\n  Predictions: {len(df):,} rows")

# =====================================================================
# STEP 2: Determine grading parameters from observed decay
# =====================================================================
# From script 09, the median annual decay rate across deciles is
# approximately 0.90, meaning each year the deviation from 1.0
# shrinks by ~10%. We use the observed decile-level rates.
print("\nObserved annual decay rates by decile:")

# Compute overall median decay (excluding decile 9 which is near 1.0)
valid_decays = decile_df[decile_df["annual_decay"].notna() &
                          (decile_df["decile"] != 9)]["annual_decay"]
median_decay = valid_decays.median()
print(f"  Median annual decay: {median_decay:.3f}")
print(f"  Range: {valid_decays.min():.3f} - {valid_decays.max():.3f}")

# =====================================================================
# STEP 3: Compute individual year-5 relativity as grading anchor
# =====================================================================
# Each person's relativity at year 5 is the starting point for
# projection. We compute it the same way as script 09.
print("\nComputing individual year-5 relativities...")

df["exam_age"] = df["RIDAGEYR"].astype(int)
df["sex"] = df["IS_MALE"].map({1: "Male", 0: "Female"})

cdc_lookup = {}
for _, row in cdc.iterrows():
    cdc_lookup[(int(row["age"]), row["sex"])] = row["qx_adjusted"]

# Year 5 attained age and population qx
attained_5 = (df["exam_age"] + 4).clip(upper=100)
df["qx_pop_yr5"] = [cdc_lookup.get((a, s), np.nan)
                     for a, s in zip(attained_5, df["sex"])]
df["rel_yr5"] = df["q5_xgb"] / df["qx_pop_yr5"]

# =====================================================================
# STEP 4: Build grading parameter table
# =====================================================================
# For each person, we store the parameters needed by script 11
# to build a full mortality table:
#   - Year 1-5 qx values (from XGBoost models)
#   - Year 5 relativity (anchor for grading)
#   - Decay rate to use for projection
#
# For simplicity and robustness, we use the overall median decay
# rate for everyone. A future enhancement could vary decay by
# risk level using the decile-specific rates.

print("\nBuilding grading parameters...")
grading = df[["SEQN", "CYCLE", "RIDAGEYR", "IS_MALE", "SPLIT"]].copy()

# Annual qx from XGBoost (years 1-5)
for yr in range(1, 6):
    grading[f"q{yr}_xgb"] = df[f"q{yr}_xgb"]

# Relativity at year 5 (anchor for projection)
grading["rel_yr5"] = df["rel_yr5"]

# Decay rate (same for all; could be decile-specific in future)
grading["decay_rate"] = median_decay

# =====================================================================
# STEP 5: Demonstrate projection for sample individuals
# =====================================================================
print("\nSample projection (3 individuals):")
samples = grading.head(3)
for _, p in samples.iterrows():
    age = int(p["RIDAGEYR"])
    sex = "Male" if p["IS_MALE"] == 1 else "Female"
    r5 = p["rel_yr5"]
    decay = p["decay_rate"]

    print(f"\n  Person: age={age}, sex={sex}, rel_yr5={r5:.3f}")
    print(f"  {'Year':>6} {'Age':>5} {'Relativity':>11} "
          f"{'qx_pop':>8} {'qx_adj':>8}")
    for t in range(1, 16):
        att_age = min(age + t - 1, 100)
        qx_pop = cdc_lookup.get((att_age, sex), np.nan)
        if t <= 5:
            qx_model = p[f"q{t}_xgb"]
            rel = qx_model / qx_pop if qx_pop > 0 else np.nan
            src = "model"
        else:
            rel = 1 + (r5 - 1) * (decay ** (t - 5))
            qx_model = rel * qx_pop if pd.notna(qx_pop) else np.nan
            src = "graded"
        print(f"  {t:>6} {att_age:>5} {rel:>11.3f} "
              f"{qx_pop:>8.5f} {qx_model:>8.5f}  ({src})")

# =====================================================================
# STEP 6: Save grading parameters
# =====================================================================
grading.to_csv(OUTPUT_PATH, index=False)
print(f"\n  Saved: {OUTPUT_PATH}")
print(f"  Rows: {len(grading):,}  Cols: {grading.shape[1]}")
print(f"  Size: {os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB")
print("\nDone!")
