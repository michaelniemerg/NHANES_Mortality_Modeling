"""
07_compose_multiyear_survival.py
=================================
Combines the 5 annual qx predictions into multi-year survival
and death probabilities using the chain:
  S(t) = (1 - q1)(1 - q2)...(1 - qt)
  p_death_t = 1 - S(t)

Validates composed 5-year probability against actual outcomes.

Input:  02 processed data - nhanes_predictions_annual.csv
Output: 02 processed data - nhanes_predictions_composed.csv
"""
import pandas as pd
import numpy as np
import os

PROJECT_DIR = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
INPUT_PATH  = os.path.join(PROJECT_DIR, "02 processed data",
                           "nhanes_predictions_annual.csv")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                           "nhanes_predictions_composed.csv")

# =====================================================================
# STEP 1: Load data
# =====================================================================
print("=" * 65)
print("  07 - Compose Multi-Year Survival")
print("=" * 65)

df = pd.read_csv(INPUT_PATH)
print(f"\n  Loaded: {len(df):,} rows")

# =====================================================================
# STEP 2: Compose cumulative survival from annual qx
# =====================================================================
# S(t) = product of (1 - q_k) for k = 1..t
# Compute for both XGBoost and GAM predictions
print("\nComposing multi-year survival...")

for source in ["xgb", "gam"]:
    surv = np.ones(len(df))  # S(0) = 1.0

    for yr in range(1, 6):
        q_col = f"q{yr}_{source}"
        surv = surv * (1 - df[q_col].values)

        # Store cumulative survival and death probability at each year
        df[f"S{yr}_{source}"] = surv
        df[f"p_death_{yr}yr_{source}"] = 1 - surv

    print(f"  {source.upper()}: S(5) range = "
          f"{df[f'S5_{source}'].min():.4f} - "
          f"{df[f'S5_{source}'].max():.4f}")

# =====================================================================
# STEP 3: Validate against actual 5-year outcomes
# =====================================================================
# A person "died within 5 years" if ANY of DIED_YR1..YR5 == 1
# A person "survived 5 years" if AT_RISK_YR5 == 1 AND DIED_YR5 == 0
# Exclude: people censored before year 5 completes
print("\nValidating composed 5-year probability...")

died_any = (df[["DIED_YR1","DIED_YR2","DIED_YR3","DIED_YR4","DIED_YR5"]]
            .max(axis=1))
survived_5 = (df["AT_RISK_YR5"] == 1) & (df["DIED_YR5"] == 0)

# DIED_5YR: 1 if died in any of years 1-5, 0 if survived all 5
df["DIED_5YR"] = np.nan
df.loc[died_any == 1, "DIED_5YR"] = 1
df.loc[survived_5, "DIED_5YR"] = 0

# Filter to people with known 5-year outcome for validation
known = df[df["DIED_5YR"].notna()].copy()
known["DIED_5YR"] = known["DIED_5YR"].astype(int)

# Restrict to test set
test = known[known["SPLIT"] == "TEST"]
actual_deaths = test["DIED_5YR"].sum()

# Compare XGBoost and GAM composed probabilities
sum_xgb = test["p_death_5yr_xgb"].sum()
sum_gam = test["p_death_5yr_gam"].sum()
ratio_xgb = sum_xgb / actual_deaths
ratio_gam = sum_gam / actual_deaths

print(f"\n  Test set (known 5-year outcomes): {len(test):,}")
print(f"  Actual deaths within 5 years: {actual_deaths:,}")
print(f"\n  Composed XGBoost:  sum(pred)={sum_xgb:.1f}  "
      f"ratio={ratio_xgb:.3f}")
print(f"  Composed GAM:      sum(pred)={sum_gam:.1f}  "
      f"ratio={ratio_gam:.3f}")

# Discrimination: average predicted prob by outcome
for outcome in [0, 1]:
    sub = test[test["DIED_5YR"] == outcome]
    label = "Survived" if outcome == 0 else "Died"
    print(f"\n  {label} (n={len(sub):,}):")
    print(f"    Avg composed XGBoost: {sub['p_death_5yr_xgb'].mean():.4f}")
    print(f"    Avg composed GAM:     {sub['p_death_5yr_gam'].mean():.4f}")

# =====================================================================
# STEP 4: Save
# =====================================================================
output = df.copy()
output.to_csv(OUTPUT_PATH, index=False)
print(f"\n  Saved: {OUTPUT_PATH}")
print(f"  Rows: {len(output):,}  Cols: {output.shape[1]}")
print(f"  Size: {os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB")
print("\nDone!")
