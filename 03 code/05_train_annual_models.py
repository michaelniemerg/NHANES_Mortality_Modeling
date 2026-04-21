"""
05_train_annual_models.py
==========================
Trains 5 XGBoost models, one for each annual duration (years 1-5).
Each model predicts conditional probability of death in year k,
given survival through years 1..k-1.

Uses Config F parameters from Phase 1 tuning (depth=3, lr=0.03,
800 trees, no scale_pos_weight, no regularization).

Input:  02 processed data - nhanes_modeling_ready.csv
Output: 04 models - annual_yr1.json .. annual_yr5.json
        05 artifacts - annual_model_summary.csv
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR  = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
INPUT_PATH   = os.path.join(PROJECT_DIR, "02 processed data",
                            "nhanes_modeling_ready.csv")
MODEL_DIR    = os.path.join(PROJECT_DIR, "04 models")
ARTIFACT_DIR = os.path.join(PROJECT_DIR, "05 artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

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
print("  05 - Train Annual XGBoost Models (Years 1-5)")
print("=" * 65)

df = pd.read_csv(INPUT_PATH)
print(f"\n  Loaded: {len(df):,} rows")

# =====================================================================
# STEP 2: Train one model per year
# =====================================================================
results = []

for yr in range(1, 6):
    print(f"\n  --- Year {yr} ---")

    # Filter to at-risk population for this year
    at_risk = df[df[f"AT_RISK_YR{yr}"] == 1].copy()
    target = f"DIED_YR{yr}"

    train = at_risk[at_risk["SPLIT"] == "TRAIN"]
    test  = at_risk[at_risk["SPLIT"] == "TEST"]

    X_train, y_train = train[FEATURE_COLS], train[target]
    X_test,  y_test  = test[FEATURE_COLS],  test[target]

    n_deaths_train = int(y_train.sum())
    n_deaths_test  = int(y_test.sum())
    print(f"    Train: {len(train):,} rows, {n_deaths_train:,} deaths "
          f"({y_train.mean()*100:.2f}%)")
    print(f"    Test:  {len(test):,} rows, {n_deaths_test:,} deaths "
          f"({y_test.mean()*100:.2f}%)")

    # Train with Config F parameters
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

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    # Evaluate
    prob_test  = model.predict_proba(X_test)[:, 1]
    prob_train = model.predict_proba(X_train)[:, 1]

    auc = roc_auc_score(y_test, prob_test)
    brier = brier_score_loss(y_test, prob_test)
    sum_pred_test  = prob_test.sum()
    sum_pred_train = prob_train.sum()
    ratio_test  = sum_pred_test / n_deaths_test
    ratio_train = sum_pred_train / n_deaths_train

    print(f"    AUC-ROC: {auc:.4f}  |  Brier: {brier:.4f}")
    print(f"    Calibration (test):  sum(pred)={sum_pred_test:.1f}  "
          f"deaths={n_deaths_test}  ratio={ratio_test:.3f}")
    print(f"    Calibration (train): sum(pred)={sum_pred_train:.1f}  "
          f"deaths={n_deaths_train}  ratio={ratio_train:.3f}")

    # Top 5 features
    importance = model.get_booster().get_score(importance_type="gain")
    top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_str = ", ".join([f"{f}({g:.0f})" for f, g in top5])
    print(f"    Top 5: {top5_str}")

    # Save model
    model_path = os.path.join(MODEL_DIR, f"annual_yr{yr}.json")
    model.save_model(model_path)
    print(f"    Saved: {model_path}")

    results.append({
        "year": yr,
        "n_train": len(train), "n_test": len(test),
        "deaths_train": n_deaths_train, "deaths_test": n_deaths_test,
        "auc_roc": round(auc, 4),
        "brier_score": round(brier, 4),
        "sum_pred_test": round(sum_pred_test, 1),
        "ratio_test": round(ratio_test, 3),
        "sum_pred_train": round(sum_pred_train, 1),
        "ratio_train": round(ratio_train, 3),
    })

# =====================================================================
# STEP 3: Save summary
# =====================================================================
summary = pd.DataFrame(results)
summary_path = os.path.join(ARTIFACT_DIR, "annual_model_summary.csv")
summary.to_csv(summary_path, index=False)

print(f"\n{'='*65}")
print("  SUMMARY")
print(f"{'='*65}")
print(f"{'Year':>5} {'AUC':>8} {'Brier':>8} "
      f"{'Deaths':>8} {'sum(pred)':>10} {'Ratio':>7}")
print(f"{'-'*5:>5} {'-'*8:>8} {'-'*8:>8} "
      f"{'-'*8:>8} {'-'*10:>10} {'-'*7:>7}")
for _, r in summary.iterrows():
    print(f"{int(r['year']):>5} {r['auc_roc']:>8.4f} "
          f"{r['brier_score']:>8.4f} {r['deaths_test']:>8.0f} "
          f"{r['sum_pred_test']:>10.1f} {r['ratio_test']:>7.3f}")

print(f"\n  Summary saved: {summary_path}")
print(f"  Models saved to: {MODEL_DIR}")
print("\nDone!")
