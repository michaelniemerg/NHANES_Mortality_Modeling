"""
06_predict_annual.py
=====================
Loads the 5 annual XGBoost models and scores all individuals.
Merges GAM baseline rates for each year, adjusting age by
duration (age + k - 1).

Input:  02 processed data - nhanes_modeling_ready.csv
        02 processed data - baseline_annual_gam.csv
        04 models - annual_yr1.json .. annual_yr5.json
Output: 02 processed data - nhanes_predictions_annual.csv
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR  = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
INPUT_PATH   = os.path.join(PROJECT_DIR, "02 processed data",
                            "nhanes_modeling_ready.csv")
BASELINE_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                             "baseline_annual_gam.csv")
MODEL_DIR    = os.path.join(PROJECT_DIR, "04 models")
OUTPUT_PATH  = os.path.join(PROJECT_DIR, "02 processed data",
                            "nhanes_predictions_annual.csv")

FEATURE_COLS = [
    "RIDAGEYR", "IS_MALE", "DMDEDUC2", "INDFMPIR",
    "RACE_MEXICAN_AMERICAN", "RACE_OTHER_HISPANIC",
    "RACE_NH_BLACK", "RACE_OTHER_MULTI",
    "BMXBMI", "BMXWAIST", "AVG_SBP", "AVG_DBP",
    "LBXTC", "LBXGH",
    "SMQ020", "SMQ040", "ALQ101", "ALQ120Q",
    "DIQ010", "MCQ160B", "MCQ160C", "MCQ160F", "MCQ220",
]

# =====================================================================
# STEP 1: Load data and baselines
# =====================================================================
print("=" * 65)
print("  06 - Score All Individuals (Annual Models)")
print("=" * 65)

df = pd.read_csv(INPUT_PATH)
baseline = pd.read_csv(BASELINE_PATH)
print(f"\n  Loaded: {len(df):,} rows")
print(f"  Baseline table: {len(baseline)} rows")

# Gender label for baseline merge
df["gender"] = df["IS_MALE"].map({1: "Male", 0: "Female"})
df["exam_age"] = df["RIDAGEYR"].astype(int)

# =====================================================================
# STEP 2: Score each year and merge baselines
# =====================================================================
print("\nScoring and merging baselines...")

for yr in range(1, 6):
    # Load model
    model_path = os.path.join(MODEL_DIR, f"annual_yr{yr}.json")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Score ALL individuals (not just at-risk) so we have predictions
    # available for the composition step. Predictions for people not
    # at risk for year k are hypothetical ("if they survived to year k").
    X = df[FEATURE_COLS]
    df[f"q{yr}_xgb"] = model.predict_proba(X)[:, 1]

    # Merge GAM baseline for this year
    # Attained age for year k = exam_age + k - 1
    attained_age = (df["exam_age"] + yr - 1).clip(upper=85)
    df["_merge_age"] = attained_age

    yr_baseline = baseline[baseline["year"] == yr][["age", "gender", "gam_qx"]]
    yr_baseline = yr_baseline.rename(columns={"age": "_merge_age",
                                               "gam_qx": f"q{yr}_gam"})

    df = df.merge(yr_baseline, on=["_merge_age", "gender"], how="left")

    # Calibration check on test set (at-risk only)
    test_risk = df[(df["SPLIT"] == "TEST") & (df[f"AT_RISK_YR{yr}"] == 1)]
    sum_pred = test_risk[f"q{yr}_xgb"].sum()
    sum_actual = test_risk[f"DIED_YR{yr}"].sum()
    ratio = sum_pred / sum_actual if sum_actual > 0 else 0
    print(f"  Year {yr}: scored {len(df):,}  |  "
          f"test calibration: {sum_pred:.1f}/{sum_actual:.0f} = {ratio:.3f}")

df = df.drop(columns=["_merge_age"])

# =====================================================================
# STEP 3: Select output columns and save
# =====================================================================
id_cols = ["SEQN", "CYCLE", "RIDAGEYR", "IS_MALE", "SPLIT"]
outcome_cols = [f"DIED_YR{k}" for k in range(1, 6)]
risk_cols = [f"AT_RISK_YR{k}" for k in range(1, 6)]
xgb_cols = [f"q{k}_xgb" for k in range(1, 6)]
gam_cols = [f"q{k}_gam" for k in range(1, 6)]

output = df[id_cols + outcome_cols + risk_cols + xgb_cols + gam_cols].copy()
output.to_csv(OUTPUT_PATH, index=False)

print(f"\n  Saved: {OUTPUT_PATH}")
print(f"  Rows: {len(output):,}  Cols: {output.shape[1]}")
print(f"  Size: {os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB")

# Preview
print("\nPreview (first 5 rows, XGBoost predictions):")
prev = output[xgb_cols].head(5)
print(prev.to_string(index=False))
print("\nDone!")
