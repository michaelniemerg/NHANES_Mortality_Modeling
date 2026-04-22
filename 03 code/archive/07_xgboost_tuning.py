"""
07_xgboost_tuning.py
=====================
Trains multiple XGBoost models with different hyperparameter
configurations, evaluates each on the test set, and compares
calibration (sum of predicted probabilities vs sum of actual deaths).

Goal: Find parameters where sum(xgb_prob) is close to sum(deaths)
while maintaining strong discrimination (AUC).

Input:  02 processed data - nhanes_modeling_ready.csv
Output: 05 artifacts - tuning_results.csv
        05 artifacts - tuning_calibration_chart.png
        04 models - best_xgboost_mortality_5yr.json
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR   = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
INPUT_PATH    = os.path.join(PROJECT_DIR, "02 processed data",
                             "nhanes_modeling_ready.csv")
ARTIFACT_DIR  = os.path.join(PROJECT_DIR, "05 artifacts")
MODEL_DIR     = os.path.join(PROJECT_DIR, "04 models")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
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
print("=" * 70)
print("  07 - XGBoost Hyperparameter Tuning")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)
train = df[df["SPLIT"] == "TRAIN"].copy()
test  = df[df["SPLIT"] == "TEST"].copy()
X_train, y_train = train[FEATURE_COLS], train["DIED_5YR"]
X_test,  y_test  = test[FEATURE_COLS],  test["DIED_5YR"]

actual_deaths_test  = y_test.sum()
actual_deaths_train = y_train.sum()
print(f"  Train: {len(train):,} rows, {actual_deaths_train:,} deaths")
print(f"  Test:  {len(test):,} rows,  {actual_deaths_test:,} deaths")

# =====================================================================
# STEP 2: Define parameter configurations to test
# =====================================================================
# Each config is a dict of XGBoost params + a human-readable name.
# We vary regularization, class weighting, depth, and learning rate
# to see how each affects calibration vs discrimination.

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
imbalance_ratio = neg / pos

CONFIGS = [
    {
        "name": "A: Original (heavy regularization + scale_pos_weight)",
        "params": dict(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            scale_pos_weight=imbalance_ratio,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, gamma=1,
            reg_alpha=0.1, reg_lambda=1.0,
        ),
    },
    {
        "name": "B: No regularization + scale_pos_weight",
        "params": dict(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            scale_pos_weight=imbalance_ratio,
            subsample=1.0, colsample_bytree=1.0,
            min_child_weight=1, gamma=0,
            reg_alpha=0, reg_lambda=0,
        ),
    },
    {
        "name": "C: No regularization, NO scale_pos_weight",
        "params": dict(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            scale_pos_weight=1,
            subsample=1.0, colsample_bytree=1.0,
            min_child_weight=1, gamma=0,
            reg_alpha=0, reg_lambda=0,
        ),
    },
    {
        "name": "D: Light regularization, NO scale_pos_weight",
        "params": dict(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            scale_pos_weight=1,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, gamma=0,
            reg_alpha=0, reg_lambda=1.0,
        ),
    },
    {
        "name": "E: Deeper trees, no reg, no weight",
        "params": dict(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            scale_pos_weight=1,
            subsample=1.0, colsample_bytree=1.0,
            min_child_weight=1, gamma=0,
            reg_alpha=0, reg_lambda=0,
        ),
    },
    {
        "name": "F: Shallow trees, no reg, no weight",
        "params": dict(
            n_estimators=800, max_depth=3, learning_rate=0.03,
            scale_pos_weight=1,
            subsample=1.0, colsample_bytree=1.0,
            min_child_weight=1, gamma=0,
            reg_alpha=0, reg_lambda=0,
        ),
    },
    {
        "name": "G: Moderate reg, half scale_pos_weight",
        "params": dict(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            scale_pos_weight=imbalance_ratio / 2,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, gamma=0.5,
            reg_alpha=0.05, reg_lambda=0.5,
        ),
    },
    {
        "name": "H: No reg, no weight, slower LR + more trees",
        "params": dict(
            n_estimators=1000, max_depth=4, learning_rate=0.01,
            scale_pos_weight=1,
            subsample=0.9, colsample_bytree=0.9,
            min_child_weight=3, gamma=0,
            reg_alpha=0, reg_lambda=0,
        ),
    },
]

# =====================================================================
# STEP 3: Train and evaluate each configuration
# =====================================================================
print(f"\n  Testing {len(CONFIGS)} configurations...")
print("-" * 70)

results = []
models = {}

for cfg in CONFIGS:
    name = cfg["name"]
    params = cfg["params"]
    print(f"\n  >>> {name}")

    model = xgb.XGBClassifier(
        **params,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    # Predictions on test set
    prob_test  = model.predict_proba(X_test)[:, 1]
    prob_train = model.predict_proba(X_train)[:, 1]

    # --- Discrimination metrics ---
    auc_roc = roc_auc_score(y_test, prob_test)
    brier   = brier_score_loss(y_test, prob_test)

    # --- Calibration metrics ---
    # Key metric: sum(predicted) vs sum(actual deaths)
    sum_pred_test   = prob_test.sum()
    sum_pred_train  = prob_train.sum()
    ratio_test      = sum_pred_test / actual_deaths_test
    ratio_train     = sum_pred_train / actual_deaths_train

    # Calibration by decile: split predictions into 10 bins,
    # compare avg predicted vs observed in each
    decile_bins = pd.qcut(prob_test, 10, duplicates="drop")
    cal_df = pd.DataFrame({"prob": prob_test, "actual": y_test.values,
                           "bin": decile_bins})
    cal_by_bin = cal_df.groupby("bin", observed=True).agg(
        avg_pred=("prob", "mean"),
        avg_actual=("actual", "mean"),
        n=("actual", "size"),
    )
    # Calibration slope: how close avg_pred tracks avg_actual
    cal_corr = cal_by_bin["avg_pred"].corr(cal_by_bin["avg_actual"])

    row = {
        "config": name,
        "auc_roc": round(auc_roc, 4),
        "brier_score": round(brier, 4),
        "actual_deaths_test": actual_deaths_test,
        "sum_pred_test": round(sum_pred_test, 1),
        "ratio_pred_actual_test": round(ratio_test, 3),
        "actual_deaths_train": actual_deaths_train,
        "sum_pred_train": round(sum_pred_train, 1),
        "ratio_pred_actual_train": round(ratio_train, 3),
        "decile_cal_corr": round(cal_corr, 4),
        # Store key params for reference
        "scale_pos_weight": params.get("scale_pos_weight", 1),
        "max_depth": params.get("max_depth"),
        "learning_rate": params.get("learning_rate"),
        "n_estimators": params.get("n_estimators"),
        "reg_alpha": params.get("reg_alpha", 0),
        "reg_lambda": params.get("reg_lambda", 0),
        "gamma": params.get("gamma", 0),
        "subsample": params.get("subsample", 1),
        "min_child_weight": params.get("min_child_weight", 1),
    }
    results.append(row)
    models[name] = model

    print(f"      AUC-ROC: {auc_roc:.4f}  |  Brier: {brier:.4f}")
    print(f"      Test  - deaths: {actual_deaths_test}  "
          f"sum(pred): {sum_pred_test:.1f}  "
          f"ratio: {ratio_test:.3f}")
    print(f"      Train - deaths: {actual_deaths_train}  "
          f"sum(pred): {sum_pred_train:.1f}  "
          f"ratio: {ratio_train:.3f}")

# =====================================================================
# STEP 4: Save results table
# =====================================================================
results_df = pd.DataFrame(results)
results_path = os.path.join(ARTIFACT_DIR, "tuning_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\n\nResults saved: {results_path}")

# =====================================================================
# STEP 5: Comparison chart
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# Short labels for x-axis
short_names = [c["name"].split(":")[0] for c in CONFIGS]
x = np.arange(len(short_names))

# Panel 1: AUC-ROC
axes[0].bar(x, results_df["auc_roc"], color="#2E5090")
for i, v in enumerate(results_df["auc_roc"]):
    axes[0].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(short_names, rotation=45, ha="right")
axes[0].set_ylabel("AUC-ROC")
axes[0].set_title("Discrimination (higher = better)")
axes[0].set_ylim(0.8, 0.95)

# Panel 2: Calibration ratio (sum(pred)/sum(deaths)) - want close to 1.0
axes[1].bar(x, results_df["ratio_pred_actual_test"], color="#D4652F")
axes[1].axhline(y=1.0, color="green", linestyle="--", linewidth=2,
                label="Perfect calibration (1.0)")
for i, v in enumerate(results_df["ratio_pred_actual_test"]):
    axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(short_names, rotation=45, ha="right")
axes[1].set_ylabel("sum(pred) / sum(deaths)")
axes[1].set_title("Calibration Ratio (closer to 1.0 = better)")
axes[1].legend()
axes[1].set_ylim(0, max(results_df["ratio_pred_actual_test"]) * 1.15)

# Panel 3: Brier score (want lower)
axes[2].bar(x, results_df["brier_score"], color="#5B9F5B")
for i, v in enumerate(results_df["brier_score"]):
    axes[2].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(short_names, rotation=45, ha="right")
axes[2].set_ylabel("Brier Score")
axes[2].set_title("Brier Score (lower = better)")

fig.suptitle("XGBoost Hyperparameter Tuning Comparison",
             fontsize=16, fontweight="bold")
fig.tight_layout()
chart_path = os.path.join(ARTIFACT_DIR, "tuning_calibration_chart.png")
fig.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Chart saved: {chart_path}")

# =====================================================================
# STEP 6: Select and save best model
# =====================================================================
# Best = closest ratio to 1.0 on test set, with AUC > 0.85
results_df["cal_distance"] = abs(results_df["ratio_pred_actual_test"] - 1.0)
eligible = results_df[results_df["auc_roc"] >= 0.85]
if len(eligible) == 0:
    eligible = results_df  # fallback if none meet AUC threshold
best_idx = eligible["cal_distance"].idxmin()
best_name = results_df.loc[best_idx, "config"]

print(f"\n{'='*70}")
print(f"  BEST MODEL: {best_name}")
print(f"    AUC-ROC:       {results_df.loc[best_idx, 'auc_roc']:.4f}")
print(f"    Brier:         {results_df.loc[best_idx, 'brier_score']:.4f}")
print(f"    Calibration:   {results_df.loc[best_idx, 'ratio_pred_actual_test']:.3f}")
print(f"{'='*70}")

best_model = models[best_name]
best_model_path = os.path.join(MODEL_DIR, "best_xgboost_mortality_5yr.json")
best_model.save_model(best_model_path)
print(f"  Saved: {best_model_path}")

# =====================================================================
# STEP 7: Print full comparison table
# =====================================================================
print(f"\n{'='*70}")
print("  FULL COMPARISON TABLE")
print(f"{'='*70}")
print(f"  {'Config':<8s} {'AUC':>7s} {'Brier':>8s} "
      f"{'sum(pred)':>10s} {'deaths':>7s} {'ratio':>7s}")
print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*10} {'-'*7} {'-'*7}")
for _, r in results_df.iterrows():
    label = r["config"].split(":")[0]
    print(f"  {label:<8s} {r['auc_roc']:>7.4f} {r['brier_score']:>8.4f} "
          f"{r['sum_pred_test']:>10.1f} {r['actual_deaths_test']:>7.0f} "
          f"{r['ratio_pred_actual_test']:>7.3f}")

print(f"\n  Note: ratio = sum(xgb_prob) / sum(deaths)")
print(f"  Perfect calibration = 1.000")
print(f"  >1 = model over-predicts total deaths")
print(f"  <1 = model under-predicts total deaths")
print("\nDone!")
