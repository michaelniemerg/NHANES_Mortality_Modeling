"""
04_baseline_gam_annual.py
==========================
Fits GAM-smoothed baseline mortality curves by age and gender
for each annual duration (years 1-5). Also tests whether a
single pooled GAM per gender is sufficient vs 5 separate GAMs.

Input:  02 processed data - nhanes_modeling_ready.csv
Output: 02 processed data - baseline_annual_gam.csv
"""
import pandas as pd
import numpy as np
from pygam import LogisticGAM, s
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
INPUT_PATH  = os.path.join(PROJECT_DIR, "02 processed data",
                           "nhanes_modeling_ready.csv")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                           "baseline_annual_gam.csv")

# =====================================================================
# STEP 1: Load data
# =====================================================================
print("=" * 65)
print("  04 - Annual GAM Baselines")
print("=" * 65)

df = pd.read_csv(INPUT_PATH)
print(f"\n  Loaded: {len(df):,} rows")

# =====================================================================
# STEP 2: Fit separate GAMs per (gender, year)
# =====================================================================
print("\nFitting GAMs (10 models: 2 genders x 5 years)...")

ages = np.arange(18, 86)
rows = []
gam_stats = []

for gender_code, gender_label in [(1, "Male"), (0, "Female")]:
    for yr in range(1, 6):
        # Filter to at-risk population for this year
        subset = df[(df["IS_MALE"] == gender_code) &
                     (df[f"AT_RISK_YR{yr}"] == 1)].copy()

        X = subset[["RIDAGEYR"]].values
        y = subset[f"DIED_YR{yr}"].values

        gam = LogisticGAM(s(0, n_splines=20, lam=0.6))
        gam.fit(X, y)
        r2 = gam.statistics_["pseudo_r2"]["explained_deviance"]

        probs = gam.predict_proba(ages.reshape(-1, 1))

        for age, prob in zip(ages, probs):
            rows.append({
                "age": int(age),
                "gender": gender_label,
                "year": yr,
                "gam_qx": round(float(prob), 8),
            })

        gam_stats.append({
            "gender": gender_label, "year": yr,
            "n": len(subset), "deaths": int(y.sum()),
            "pseudo_r2": round(r2, 4),
        })
        print(f"  {gender_label} Year {yr}: "
              f"n={len(subset):,}  deaths={int(y.sum()):,}  "
              f"R2={r2:.4f}")

# =====================================================================
# STEP 3: Test pooled GAM (1 per gender, all years combined)
# =====================================================================
print("\nTesting pooled GAM (1 per gender, all years)...")
for gender_code, gender_label in [(1, "Male"), (0, "Female")]:
    # Pool all at-risk records across years
    all_X, all_y = [], []
    for yr in range(1, 6):
        sub = df[(df["IS_MALE"] == gender_code) &
                  (df[f"AT_RISK_YR{yr}"] == 1)]
        all_X.append(sub[["RIDAGEYR"]].values)
        all_y.append(sub[f"DIED_YR{yr}"].values)
    X_pool = np.vstack(all_X)
    y_pool = np.concatenate(all_y)

    gam_pooled = LogisticGAM(s(0, n_splines=20, lam=0.6))
    gam_pooled.fit(X_pool, y_pool)
    r2_pooled = gam_pooled.statistics_["pseudo_r2"]["explained_deviance"]

    # Compare: average R2 of separate GAMs vs pooled
    sep_r2s = [g["pseudo_r2"] for g in gam_stats
               if g["gender"] == gender_label]
    avg_sep = np.mean(sep_r2s)
    print(f"  {gender_label}: pooled R2={r2_pooled:.4f}  "
          f"avg separate R2={avg_sep:.4f}  "
          f"-> {'Separate wins' if avg_sep > r2_pooled else 'Pooled sufficient'}")

# =====================================================================
# STEP 4: Save baseline table
# =====================================================================
baseline = pd.DataFrame(rows)
baseline = baseline.sort_values(["gender", "year", "age"]).reset_index(drop=True)

baseline.to_csv(OUTPUT_PATH, index=False)
print(f"\n  Saved: {OUTPUT_PATH}")
print(f"  Rows: {len(baseline)} (68 ages x 2 genders x 5 years)")

# Preview
print("\nPreview (age 50 and 70, both genders, all years):")
prev = baseline[baseline["age"].isin([50, 70])]
print(prev.to_string(index=False))
print("\nDone!")
