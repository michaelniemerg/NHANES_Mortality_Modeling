"""
04_baseline_mortality_table.py
================================
Creates a baseline table showing the probability of death within
5 years by age and gender, using both raw rates and GAM-smoothed
estimates.

Input:  02 processed data\nhanes_modeling_ready.csv
Output: 02 processed data\baseline_mortality_by_age_gender.csv
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
                           "baseline_mortality_by_age_gender.csv")

# =====================================================================
# STEP 1: Load data
# =====================================================================
print("Loading modeling-ready data...")
df = pd.read_csv(INPUT_PATH)
print(f"  Loaded: {len(df):,} rows")

# =====================================================================
# STEP 2: Calculate raw mortality rates by age and gender
# =====================================================================
print("\nCalculating raw mortality rates by age and gender...")

adults = df[(df["RIDAGEYR"] >= 18)].copy()
adults["RIDAGEYR"] = adults["RIDAGEYR"].astype(int)

raw = (adults
       .groupby(["RIDAGEYR", "IS_MALE"])
       .agg(n_total=("DIED_5YR", "size"),
            n_died=("DIED_5YR", "sum"))
       .reset_index())
raw["raw_prob_death_5yr"] = raw["n_died"] / raw["n_total"]
raw = raw.rename(columns={"RIDAGEYR": "age"})
raw["gender"] = raw["IS_MALE"].map({1: "Male", 0: "Female"})

print(f"  Age range: {raw['age'].min()} - {raw['age'].max()}")
print(f"  Total age-gender cells: {len(raw)}")

# =====================================================================
# STEP 3: Fit GAM-smoothed mortality curves
# =====================================================================
# A separate logistic GAM is fit for each gender.
# The GAM uses a smooth spline on age to capture the non-linear
# relationship between age and mortality probability, smoothing
# out noise in raw rates especially at ages with few observations.
print("\nFitting GAM-smoothed curves...")

smoothed_rows = []
for gender_code, gender_label in [(1, "Male"), (0, "Female")]:
    subset = adults[adults["IS_MALE"] == gender_code]
    X = subset[["RIDAGEYR"]].values
    y = subset["DIED_5YR"].values

    # Fit logistic GAM with smooth term on age
    gam = LogisticGAM(s(0, n_splines=20, lam=0.6))
    gam.fit(X, y)

    # Predict smoothed probabilities for each integer age
    ages = np.arange(18, 86)
    probs = gam.predict_proba(ages.reshape(-1, 1))

    for age, prob in zip(ages, probs):
        smoothed_rows.append({
            "age": int(age),
            "gender": gender_label,
            "gam_prob_death_5yr": round(float(prob), 6),
        })

    r2 = gam.statistics_["pseudo_r2"]["explained_deviance"]
    print(f"  {gender_label}: GAM fit (pseudo-R2: {r2:.3f})")

smoothed = pd.DataFrame(smoothed_rows)

# =====================================================================
# STEP 4: Merge raw and smoothed into final table
# =====================================================================
print("\nMerging raw and smoothed estimates...")
final = raw.merge(smoothed, on=["age", "gender"], how="outer")
final = final.drop(columns=["IS_MALE"], errors="ignore")

# Select and order columns
final = final[["age", "gender", "n_total", "n_died",
               "raw_prob_death_5yr", "gam_prob_death_5yr"]]
final = final.sort_values(["gender", "age"]).reset_index(drop=True)
final["raw_prob_death_5yr"] = final["raw_prob_death_5yr"].round(6)

# =====================================================================
# STEP 5: Save and preview
# =====================================================================
final.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
print(f"  Rows: {len(final)}")

print("\nPreview (selected ages):")
preview_ages = [20, 30, 40, 50, 60, 70, 80, 85]
preview = final[final["age"].isin(preview_ages)]
print(preview.to_string(index=False))
print("\nDone!")
