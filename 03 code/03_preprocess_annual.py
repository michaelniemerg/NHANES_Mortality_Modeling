"""
03_preprocess_annual.py
========================
Reads the combined NHANES file and creates a modeling-ready dataset
with conditional annual mortality outcomes for years 1-5.

Year k outcome is CONDITIONAL: only people who survived years 1..k-1
and have sufficient follow-up are included in year k's model.

Input:  02 processed data - nhanes_all_cycles_with_mortality.csv
Output: 02 processed data - nhanes_modeling_ready.csv
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

PROJECT_DIR = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
INPUT_PATH  = os.path.join(PROJECT_DIR, "02 processed data",
                           "nhanes_all_cycles_with_mortality.csv")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                           "nhanes_modeling_ready.csv")

# =====================================================================
# STEP 1: Load and filter
# =====================================================================
print("=" * 65)
print("  03 - Preprocess for Annual Models")
print("=" * 65)

df = pd.read_csv(INPUT_PATH)
print(f"\n  Loaded: {len(df):,} rows")

# Keep mortality-eligible adults only
df = df[df["ELIGSTAT"] == 1].copy()
print(f"  Mortality-eligible adults: {len(df):,}")

# Drop missing follow-up time
df = df.dropna(subset=["PERMTH_EXM"])
print(f"  After dropping missing PERMTH_EXM: {len(df):,}")

# =====================================================================
# STEP 2: Create conditional annual outcomes
# =====================================================================
# For year k (months (k-1)*12 to k*12):
#   AT_RISK_YRk = 1 if person survived prior years AND we can
#     observe their full year k outcome:
#       - died during year k, OR
#       - survived past month k*12
#     Excludes: alive but censored during year k (unknown outcome)
#   DIED_YRk = 1 if died during year k, 0 if survived year k
print("\nCreating annual outcomes...")

for k in range(1, 6):
    lo = (k - 1) * 12   # start of year k (exclusive for k>1)
    hi = k * 12          # end of year k

    # Survived all prior years: follow-up extends past start of year k
    survived_prior = df["PERMTH_EXM"] > lo

    # Died during year k
    died_in_yr = ((df["MORTSTAT"] == 1) &
                  (df["PERMTH_EXM"] > lo) &
                  (df["PERMTH_EXM"] <= hi))

    # Survived through year k: follow-up extends past end of year k
    survived_yr = df["PERMTH_EXM"] > hi

    # Censored during year k (alive but follow-up ended mid-year)
    censored_yr = ((df["MORTSTAT"] == 0) &
                   (df["PERMTH_EXM"] > lo) &
                   (df["PERMTH_EXM"] <= hi))

    # At risk = survived prior AND (died in year k OR survived year k)
    # Excludes censored-during-year-k (unknown outcome)
    at_risk = survived_prior & (died_in_yr | survived_yr)

    df[f"AT_RISK_YR{k}"] = at_risk.astype(int)
    df[f"DIED_YR{k}"] = np.nan
    df.loc[died_in_yr, f"DIED_YR{k}"] = 1
    df.loc[survived_yr, f"DIED_YR{k}"] = 0
    # People not at risk (including censored) stay NaN

    n_risk = at_risk.sum()
    n_died = died_in_yr.sum()
    n_cens = censored_yr.sum()
    rate = n_died / n_risk * 100 if n_risk > 0 else 0
    print(f"  Year {k}: at_risk={n_risk:,}  "
          f"died={n_died:,} ({rate:.2f}%)  censored={n_cens:,}")

# =====================================================================
# STEP 3: Feature transformations (same as Phase 1)
# =====================================================================
print("\nApplying feature transformations...")

# Gender: binary IS_MALE
df["IS_MALE"] = (df["RIAGENDR"] == 1).astype(int)

# Education: ordinal 1-5, codes 7/9 to NaN
df["DMDEDUC2"] = df["DMDEDUC2"].replace({7: np.nan, 9: np.nan})

# Smoking
df["SMQ020"] = df["SMQ020"].replace({2: 0, 7: np.nan, 9: np.nan})
df["SMQ040"] = df["SMQ040"].replace({7: np.nan, 9: np.nan})
df.loc[df["SMQ020"] == 0, "SMQ040"] = 0
df["SMQ040"] = df["SMQ040"].replace({1: 2, 2: 1, 3: 0})

# Alcohol
df["ALQ101"] = df["ALQ101"].replace({2: 0, 7: np.nan, 9: np.nan})
df["ALQ120Q"] = df["ALQ120Q"].replace({777: np.nan, 999: np.nan})

# Diabetes: 0=No, 1=Borderline, 2=Yes
df["DIQ010"] = df["DIQ010"].replace({2: 0, 3: 1, 1: 2,
                                      7: np.nan, 9: np.nan})

# Medical conditions: 1=Yes, 2=No -> 1/0
for col in ["MCQ160B", "MCQ160C", "MCQ160F", "MCQ220"]:
    df[col] = df[col].replace({2: 0, 7: np.nan, 9: np.nan})

# Average blood pressure
df["AVG_SBP"] = df[["BPXSY1", "BPXSY2"]].mean(axis=1)
df["AVG_DBP"] = df[["BPXDI1", "BPXDI2"]].mean(axis=1)

# Race/ethnicity: one-hot encode, drop NH White as reference
race_dummies = pd.get_dummies(df["RIDRETH1"], prefix="RACE", dtype=int)
race_dummies = race_dummies.rename(columns={
    "RACE_1.0": "RACE_MEXICAN_AMERICAN",
    "RACE_2.0": "RACE_OTHER_HISPANIC",
    "RACE_3.0": "RACE_NH_WHITE",
    "RACE_4.0": "RACE_NH_BLACK",
    "RACE_5.0": "RACE_OTHER_MULTI",
})
race_dummies = race_dummies.drop(columns=["RACE_NH_WHITE"],
                                  errors="ignore")
df = pd.concat([df, race_dummies], axis=1)

# =====================================================================
# STEP 4: Select columns
# =====================================================================
FEATURE_COLS = [
    "RIDAGEYR", "IS_MALE", "DMDEDUC2", "INDFMPIR",
    "RACE_MEXICAN_AMERICAN", "RACE_OTHER_HISPANIC",
    "RACE_NH_BLACK", "RACE_OTHER_MULTI",
    "BMXBMI", "BMXWAIST", "AVG_SBP", "AVG_DBP",
    "LBXTC", "LBXGH",
    "SMQ020", "SMQ040", "ALQ101", "ALQ120Q",
    "DIQ010", "MCQ160B", "MCQ160C", "MCQ160F", "MCQ220",
]

ID_COLS = ["SEQN", "CYCLE"]
ANNUAL_COLS = ([f"DIED_YR{k}" for k in range(1, 6)] +
               [f"AT_RISK_YR{k}" for k in range(1, 6)])

# =====================================================================
# STEP 5: Train/test split (70/30)
# =====================================================================
# Split assigned to ALL individuals. Stratified on year-1 outcome
# among those at risk for year 1. Everyone else gets random assignment.
print("\nAssigning train/test split...")

yr1_at_risk = df[df["AT_RISK_YR1"] == 1]
yr1_other   = df[df["AT_RISK_YR1"] != 1]

train_idx, test_idx = train_test_split(
    yr1_at_risk.index, test_size=0.30, random_state=42,
    stratify=yr1_at_risk["DIED_YR1"],
)

# For people not at risk for year 1 (very short follow-up),
# assign randomly with same 70/30 ratio
if len(yr1_other) > 0:
    other_train, other_test = train_test_split(
        yr1_other.index, test_size=0.30, random_state=42,
    )
    train_idx = train_idx.append(other_train)
    test_idx  = test_idx.append(other_test)

df["SPLIT"] = "TEST"
df.loc[train_idx, "SPLIT"] = "TRAIN"

for k in range(1, 6):
    at_risk_k = df[(df[f"AT_RISK_YR{k}"] == 1)]
    tr = at_risk_k[at_risk_k["SPLIT"] == "TRAIN"]
    te = at_risk_k[at_risk_k["SPLIT"] == "TEST"]
    rate_tr = tr[f"DIED_YR{k}"].mean() * 100
    rate_te = te[f"DIED_YR{k}"].mean() * 100
    print(f"  Year {k}: train={len(tr):,} ({rate_tr:.2f}%)  "
          f"test={len(te):,} ({rate_te:.2f}%)")

# =====================================================================
# STEP 6: Save
# =====================================================================
all_cols = ID_COLS + FEATURE_COLS + ANNUAL_COLS + ["SPLIT"]
all_cols = [c for c in all_cols if c in df.columns]
output = df[all_cols].copy()

output.to_csv(OUTPUT_PATH, index=False)
print(f"\n  Saved: {OUTPUT_PATH}")
print(f"  Rows: {len(output):,}  Cols: {output.shape[1]}")
print(f"  Size: {os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB")
print("\nDone!")
