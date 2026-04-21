"""
05_xgboost_model.py
====================
Trains an XGBoost model to predict 5-year mortality probability,
evaluates on the held-out test set, and produces an output CSV
with individual predicted probabilities alongside the GAM-smoothed
baseline rates.

Input:  02 processed data - nhanes_modeling_ready.csv
        02 processed data - baseline_mortality_by_age_gender.csv
Output: 02 processed data - nhanes_predictions.csv
        04 models - xgboost_mortality_5yr.json
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, classification_report,
                             confusion_matrix)
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR   = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
INPUT_PATH    = os.path.join(PROJECT_DIR, "02 processed data",
                             "nhanes_modeling_ready.csv")
BASELINE_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                             "baseline_mortality_by_age_gender.csv")
OUTPUT_PATH   = os.path.join(PROJECT_DIR, "02 processed data",
                             "nhanes_predictions.csv")
MODEL_DIR     = os.path.join(PROJECT_DIR, "04 models")
MODEL_PATH    = os.path.join(MODEL_DIR, "xgboost_mortality_5yr.json")
os.makedirs(MODEL_DIR, exist_ok=True)

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
# STEP 1: Load data
# =====================================================================
print("=" * 65)
print("  05 - XGBoost 5-Year Mortality Model")
print("=" * 65)

print("\nLoading data...")
df = pd.read_csv(INPUT_PATH)
baseline = pd.read_csv(BASELINE_PATH)
print(f"  Modeling data: {len(df):,} rows")
print(f"  Baseline table: {len(baseline)} rows")

# =====================================================================
# STEP 2: Split into train and test
# =====================================================================
train = df[df["SPLIT"] == "TRAIN"].copy()
test  = df[df["SPLIT"] == "TEST"].copy()

X_train = train[FEATURE_COLS]
y_train = train["DIED_5YR"]
X_test  = test[FEATURE_COLS]
y_test  = test["DIED_5YR"]

print(f"\n  Train: {len(train):,} rows  ({y_train.mean()*100:.1f}% positive)")
print(f"  Test:  {len(test):,} rows  ({y_test.mean()*100:.1f}% positive)")

# =====================================================================
# STEP 3: Train XGBoost model
# =====================================================================
# Hyperparameters: Config F from 07_xgboost_tuning.py
# Selected for best calibration (ratio=0.982) with strong AUC (0.891).
#   - NO scale_pos_weight: avoids inflating predicted probabilities
#   - Shallow trees (depth=3) + slow LR + many trees: smooth probability
#     surface with good calibration
#   - No regularization: allows the model to learn true base rates
print("\nTraining XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=3,
    learning_rate=0.03,
    scale_pos_weight=1,
    subsample=1.0,
    colsample_bytree=1.0,
    min_child_weight=1,
    gamma=0,
    reg_alpha=0,
    reg_lambda=0,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print("  Training complete.")

# =====================================================================
# STEP 4: Evaluate on test set
# =====================================================================
print("\n" + "=" * 65)
print("  TEST SET EVALUATION")
print("=" * 65)

y_prob_test = model.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= 0.5).astype(int)

auc_roc = roc_auc_score(y_test, y_prob_test)
auc_pr  = average_precision_score(y_test, y_prob_test)
brier   = brier_score_loss(y_test, y_prob_test)

print(f"\n  AUC-ROC:          {auc_roc:.4f}")
print(f"  AUC-PR:           {auc_pr:.4f}")
print(f"  Brier Score:      {brier:.4f}")

print(f"\n  Classification Report (threshold=0.5):")
print(classification_report(y_test, y_pred_test,
                            target_names=["Survived", "Died"],
                            digits=3))

print("  Top 10 Features (gain):")
importance = model.get_booster().get_score(importance_type="gain")
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
for feat, gain in sorted_imp[:10]:
    print(f"    {feat:30s}  {gain:.1f}")

# =====================================================================
# STEP 5: Generate predictions for ALL individuals
# =====================================================================
print("\n" + "=" * 65)
print("  GENERATING PREDICTIONS")
print("=" * 65)

X_all = df[FEATURE_COLS]
df["xgb_prob_death_5yr"] = model.predict_proba(X_all)[:, 1]

print(f"\n  Predictions generated for {len(df):,} individuals")
print(f"  Predicted probability range:"
      f" {df['xgb_prob_death_5yr'].min():.4f}"
      f" - {df['xgb_prob_death_5yr'].max():.4f}")

# =====================================================================
# STEP 6: Merge GAM baseline probability
# =====================================================================
# Join each individual's age-gender baseline from the GAM-smoothed
# table so the output has both the XGBoost and baseline predictions.

df["gender"] = df["IS_MALE"].map({1: "Male", 0: "Female"})
df["age_int"] = df["RIDAGEYR"].astype(int)

baseline_slim = baseline[["age", "gender", "gam_prob_death_5yr"]].copy()
baseline_slim = baseline_slim.rename(columns={"age": "age_int"})

df = df.merge(baseline_slim, on=["age_int", "gender"], how="left")

print(f"  GAM baseline merged for {df['gam_prob_death_5yr'].notna().sum():,}"
      f" of {len(df):,} rows")

# =====================================================================
# STEP 7: Select output columns and save
# =====================================================================
output = df[[
    "SEQN", "CYCLE", "RIDAGEYR", "IS_MALE",
    "DIED_5YR", "SPLIT",
    "gam_prob_death_5yr",
    "xgb_prob_death_5yr",
]].copy()

output = output.rename(columns={
    "RIDAGEYR": "age",
    "IS_MALE": "is_male",
    "DIED_5YR": "died_within_5yr",
    "SPLIT": "split",
    "gam_prob_death_5yr": "baseline_prob_gam",
    "xgb_prob_death_5yr": "xgb_prob",
})

output.to_csv(OUTPUT_PATH, index=False)
print(f"\n  Saved predictions: {OUTPUT_PATH}")
print(f"  Size: {os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB")

# =====================================================================
# STEP 8: Save model
# =====================================================================
model.save_model(MODEL_PATH)
print(f"  Saved model: {MODEL_PATH}")

# =====================================================================
# STEP 9: Preview
# =====================================================================
print("\nSample predictions (first 10 test set rows):")
sample = output[output["split"] == "TEST"].head(10)
print(sample.to_string(index=False))

print("\nAvg predicted probability by actual outcome (test set):")
test_out = output[output["split"] == "TEST"]
for outcome in [0, 1]:
    subset = test_out[test_out["died_within_5yr"] == outcome]
    label = "Survived" if outcome == 0 else "Died"
    print(f"  {label:10s}  "
          f"baseline(GAM): {subset['baseline_prob_gam'].mean():.3f}  "
          f"XGBoost: {subset['xgb_prob'].mean():.3f}")

print("\nDone!")
