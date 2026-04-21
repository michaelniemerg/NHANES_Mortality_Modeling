"""
03_preprocess_for_xgboost.py
=============================
Reads the combined NHANES file, creates a binary 5-year mortality
outcome, cleans and recodes variables for XGBoost, adds a 70/30
train/test split column, and saves the modeling-ready dataset.

Input:  02 processed data\nhanes_all_cycles_with_mortality.csv
Output: 02 processed data\nhanes_modeling_ready.csv
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

PROJECT_DIR = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
INPUT_PATH  = os.path.join(PROJECT_DIR, "02 processed data",
                           "nhanes_all_cycles_with_mortality.csv")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                           "nhanes_modeling_ready.csv")

# =====================================================================
# STEP 1: Load data
# =====================================================================
print("Loading combined NHANES data...")
df = pd.read_csv(INPUT_PATH)
print(f"  Loaded: {len(df):,} rows x {df.shape[1]} cols")

# =====================================================================
# STEP 2: Filter to mortality-eligible adults
# =====================================================================
# ELIGSTAT == 1 means the participant was an adult (18+) with
# sufficient identifying data for mortality linkage.
# We exclude children (ELIGSTAT == 2) and ineligible records (== 3).
df = df[df["ELIGSTAT"] == 1].copy()
print(f"  After filtering to mortality-eligible adults: {len(df):,}")

# =====================================================================
# STEP 3: Create the 5-year (60-month) mortality outcome
# =====================================================================
# We use PERMTH_EXM (person-months from MEC exam) because most
# predictors (labs, BP, BMI) were measured at the MEC exam.
#
# Three groups:
#   DIED within 5 years:  MORTSTAT == 1 AND PERMTH_EXM <= 60
#   ALIVE past 5 years:   PERMTH_EXM > 60 (regardless of later death)
#   UNKNOWN (excluded):   MORTSTAT == 0 AND PERMTH_EXM <= 60
#       These people were still alive when follow-up ended, but
#       follow-up ended before the 5-year mark, so we don't know
#       their true 5-year outcome.

# Drop rows with missing follow-up time
df = df.dropna(subset=["PERMTH_EXM"])
print(f"  After dropping missing PERMTH_EXM: {len(df):,}")

# Assign outcome
df["DIED_5YR"] = np.nan  # start with unknown

# Died within 5 years of MEC exam
df.loc[(df["MORTSTAT"] == 1) & (df["PERMTH_EXM"] <= 60), "DIED_5YR"] = 1

# Survived past 5 years (even if they died later, they survived the window)
df.loc[df["PERMTH_EXM"] > 60, "DIED_5YR"] = 0

# Exclude unknowns: alive but censored before 5 years
n_excluded = df["DIED_5YR"].isna().sum()
df = df.dropna(subset=["DIED_5YR"])
df["DIED_5YR"] = df["DIED_5YR"].astype(int)

print(f"  Excluded {n_excluded:,} censored-before-5-years records")
print(f"  Final outcome cohort: {len(df):,}")
print(f"    Died within 5 years:     {(df['DIED_5YR'] == 1).sum():,}"
      f" ({(df['DIED_5YR'] == 1).mean()*100:.1f}%)")
print(f"    Survived past 5 years:   {(df['DIED_5YR'] == 0).sum():,}"
      f" ({(df['DIED_5YR'] == 0).mean()*100:.1f}%)")

# =====================================================================
# STEP 4: Recode variables for XGBoost
# =====================================================================
# XGBoost handles missing values (NaN) natively, so we convert
# "refused" (7, 77) and "don't know" (9, 99) codes to NaN rather
# than imputing.

# --- Gender: recode to binary IS_MALE (1=male, 0=female) ---
df["IS_MALE"] = (df["RIAGENDR"] == 1).astype(int)

# --- Education: keep ordinal 1-5, recode 7/9 to NaN ---
# 1=<9th grade, 2=9-11th, 3=HS/GED, 4=some college, 5=college grad
df["DMDEDUC2"] = df["DMDEDUC2"].replace({7: np.nan, 9: np.nan})

# --- Smoking ---
# SMQ020: Smoked 100+ cigarettes (1=Yes, 2=No -> 1/0)
df["SMQ020"] = df["SMQ020"].replace({2: 0, 7: np.nan, 9: np.nan})

# SMQ040: Current status (1=every day, 2=some days, 3=not at all)
# Recode to: 0=not at all/never, 1=some days, 2=every day
# If SMQ020 == 0 (never smoked 100+), SMQ040 is NaN -> set to 0
df["SMQ040"] = df["SMQ040"].replace({7: np.nan, 9: np.nan})
df.loc[df["SMQ020"] == 0, "SMQ040"] = 0
# Reverse the scale so higher = more smoking
df["SMQ040"] = df["SMQ040"].replace({1: 2, 2: 1, 3: 0})

# --- Alcohol ---
# ALQ101: Had 12+ drinks past year (1=Yes, 2=No -> 1/0)
df["ALQ101"] = df["ALQ101"].replace({2: 0, 7: np.nan, 9: np.nan})

# ALQ120Q: Frequency of drinking (numeric, leave as-is)
# Values 777/999 are refused/don't know
df["ALQ120Q"] = df["ALQ120Q"].replace({777: np.nan, 999: np.nan})

# --- Diabetes ---
# DIQ010: Doctor told you have diabetes (1=Yes, 2=No, 3=Borderline)
# Recode to ordinal: 0=No, 1=Borderline, 2=Yes
df["DIQ010"] = df["DIQ010"].replace({2: 0, 3: 1, 1: 2,
                                      7: np.nan, 9: np.nan})

# --- Medical conditions (all use 1=Yes, 2=No -> 1/0) ---
for col in ["MCQ160B", "MCQ160C", "MCQ160F", "MCQ220"]:
    df[col] = df[col].replace({2: 0, 7: np.nan, 9: np.nan})

# =====================================================================
# STEP 5: Create derived features
# =====================================================================

# --- Average blood pressure from 2 readings ---
# If only one reading is available, use that; if both, average them.
df["AVG_SBP"] = df[["BPXSY1", "BPXSY2"]].mean(axis=1)
df["AVG_DBP"] = df[["BPXDI1", "BPXDI2"]].mean(axis=1)

# --- Race/ethnicity: one-hot encode ---
# RIDRETH1: 1=Mexican American, 2=Other Hispanic, 3=NH White,
#           4=NH Black, 5=Other/Multi
# Create dummies; drop NH White as reference category
race_dummies = pd.get_dummies(df["RIDRETH1"], prefix="RACE",
                              dtype=int)
race_dummies = race_dummies.rename(columns={
    "RACE_1.0": "RACE_MEXICAN_AMERICAN",
    "RACE_2.0": "RACE_OTHER_HISPANIC",
    "RACE_3.0": "RACE_NH_WHITE",
    "RACE_4.0": "RACE_NH_BLACK",
    "RACE_5.0": "RACE_OTHER_MULTI",
})
# Drop NH White as reference category
race_dummies = race_dummies.drop(columns=["RACE_NH_WHITE"],
                                  errors="ignore")
df = pd.concat([df, race_dummies], axis=1)

# =====================================================================
# STEP 6: Select final columns for modeling
# =====================================================================
# We keep only the identifier, features, response, and split column.
# Raw source variables that were recoded are dropped.

FEATURE_COLS = [
    # Demographics
    "RIDAGEYR",              # Age
    "IS_MALE",               # Gender (binary)
    "DMDEDUC2",              # Education (ordinal 1-5)
    "INDFMPIR",              # Income-to-poverty ratio
    "RACE_MEXICAN_AMERICAN", # Race dummies (ref = NH White)
    "RACE_OTHER_HISPANIC",
    "RACE_NH_BLACK",
    "RACE_OTHER_MULTI",
    # Body measures
    "BMXBMI",                # BMI
    "BMXWAIST",              # Waist circumference
    # Blood pressure (averaged readings)
    "AVG_SBP",               # Avg systolic BP
    "AVG_DBP",               # Avg diastolic BP
    # Labs
    "LBXTC",                 # Total cholesterol
    "LBXGH",                 # HbA1c
    # Smoking
    "SMQ020",                # Ever smoked 100+ cigs (binary)
    "SMQ040",                # Current smoking (0/1/2)
    # Alcohol
    "ALQ101",                # Drank 12+ times past year (binary)
    "ALQ120Q",               # Drinking frequency
    # Diabetes
    "DIQ010",                # Diabetes status (0=No,1=Bord,2=Yes)
    # Medical history
    "MCQ160B",               # CHF (binary)
    "MCQ160C",               # CHD (binary)
    "MCQ160F",               # Stroke (binary)
    "MCQ220",                # Cancer (binary)
]

ID_COLS       = ["SEQN", "CYCLE"]
RESPONSE_COL  = "DIED_5YR"

# =====================================================================
# STEP 7: Create train/test split (70/30)
# =====================================================================
# Stratified on DIED_5YR to preserve class balance in both sets.
# Random state is fixed for reproducibility.

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.30,
    random_state=42,
    stratify=df[RESPONSE_COL],
)

df["SPLIT"] = "TEST"
df.loc[train_idx, "SPLIT"] = "TRAIN"

print(f"\n  Train/test split (70/30, stratified on DIED_5YR):")
print(f"    TRAIN: {(df['SPLIT'] == 'TRAIN').sum():,}"
      f"  (died: {df.loc[df['SPLIT']=='TRAIN','DIED_5YR'].mean()*100:.1f}%)")
print(f"    TEST:  {(df['SPLIT'] == 'TEST').sum():,}"
      f"  (died: {df.loc[df['SPLIT']=='TEST','DIED_5YR'].mean()*100:.1f}%)")

# =====================================================================
# STEP 8: Select final columns and save
# =====================================================================
final_cols = ID_COLS + FEATURE_COLS + [RESPONSE_COL, "SPLIT"]

# Keep only columns that exist (guard against any missing dummies)
final_cols = [c for c in final_cols if c in df.columns]
output = df[final_cols].copy()

print(f"\n  Final dataset: {len(output):,} rows x {output.shape[1]} cols")
print(f"  Features: {len(FEATURE_COLS)}")
print(f"\n  Feature columns:")
for f in FEATURE_COLS:
    miss = output[f].isna().mean() * 100
    print(f"    {f:30s}  missing: {miss:.1f}%")

output.to_csv(OUTPUT_PATH, index=False)
print(f"\n  Saved: {OUTPUT_PATH}")
print(f"  Size:  {os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB")
print("\nDone! To use in XGBoost:")
print("  df = pd.read_csv(r'" + OUTPUT_PATH + "')")
print("  train = df[df['SPLIT'] == 'TRAIN']")
print("  test  = df[df['SPLIT'] == 'TEST']")
print("  X_train = train.drop(columns=['SEQN','CYCLE','DIED_5YR','SPLIT'])")
print("  y_train = train['DIED_5YR']")
