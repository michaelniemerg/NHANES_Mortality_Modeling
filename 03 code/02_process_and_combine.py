"""
02_process_and_combine.py
=========================
Reads the raw NHANES files downloaded by 01_download_raw_files.py,
harmonizes variables across cycles, merges with linked mortality,
and produces a single analysis-ready CSV.

Input:  01 raw data\{cycle}\*.xpt and *.dat
Output: 02 processed data\nhanes_all_cycles_with_mortality.csv
"""
import pandas as pd
import io
import os
import sys

PROJECT_DIR = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
RAW_DIR     = os.path.join(PROJECT_DIR, "01 raw data")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "02 processed data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Cycle definitions ──
CYCLES = [
    (1999, 2000, "",   "1999-2000"),
    (2001, 2002, "_B", "2001-2002"),
    (2003, 2004, "_C", "2003-2004"),
    (2005, 2006, "_D", "2005-2006"),
    (2007, 2008, "_E", "2007-2008"),
    (2009, 2010, "_F", "2009-2010"),
    (2011, 2012, "_G", "2011-2012"),
    (2013, 2014, "_H", "2013-2014"),
    (2015, 2016, "_I", "2015-2016"),
    (2017, 2018, "_J", "2017-2018"),
]

def get_local_files(suffix, cycle_dir):
    """Return dict of component_name -> local file path."""
    files = {
        "DEMO":  os.path.join(cycle_dir, f"DEMO{suffix}.xpt"),
        "BMX":   os.path.join(cycle_dir, f"BMX{suffix}.xpt"),
        "BPX":   os.path.join(cycle_dir, f"BPX{suffix}.xpt"),
        "SMQ":   os.path.join(cycle_dir, f"SMQ{suffix}.xpt"),
        "ALQ":   os.path.join(cycle_dir, f"ALQ{suffix}.xpt"),
        "MCQ":   os.path.join(cycle_dir, f"MCQ{suffix}.xpt"),
        "DIQ":   os.path.join(cycle_dir, f"DIQ{suffix}.xpt"),
    }
    if suffix in ("", "_B", "_C"):
        chol = "LAB13" if suffix == "" else f"L13{suffix}"
        ghb  = "LAB10" if suffix == "" else f"L10{suffix}"
    else:
        chol = f"TCHOL{suffix}"
        ghb  = f"GHB{suffix}"
    files["TCHOL"] = os.path.join(cycle_dir, f"{chol}.xpt")
    files["GHB"]   = os.path.join(cycle_dir, f"{ghb}.xpt")
    return files

def read_xpt(path):
    """Read a local SAS transport file."""
    try:
        return pd.read_sas(path, format="xport", encoding="utf-8")
    except Exception as e:
        return None

def read_mortality(path):
    """Read a local fixed-width mortality file."""
    try:
        colspecs = [
            (0, 14), (14, 15), (15, 16), (16, 19),
            (19, 20), (20, 21), (42, 45), (45, 48),
        ]
        names = ["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING",
                 "DIABETES_MORT", "HYPERTEN_MORT", "PERMTH_INT", "PERMTH_EXM"]
        return pd.read_fwf(path, colspecs=colspecs, header=None,
                           names=names, na_values=[".", ""])
    except Exception as e:
        return None

# ── Harmonized variables to keep (consistent across all cycles) ──
KEEP_VARS = [
    "SEQN",
    # Demographics
    "RIDAGEYR",   # Age in years
    "RIAGENDR",   # Gender (1=M, 2=F)
    "RIDRETH1",   # Race/ethnicity (5 categories)
    "DMDEDUC2",   # Education (adults 20+)
    "INDFMPIR",   # Family income to poverty ratio
    # Body measures
    "BMXBMI",     # BMI
    "BMXWT",      # Weight (kg)
    "BMXHT",      # Height (cm)
    "BMXWAIST",   # Waist circumference (cm)
    # Blood pressure
    "BPXSY1",     # Systolic BP 1st
    "BPXDI1",     # Diastolic BP 1st
    "BPXSY2",     # Systolic BP 2nd
    "BPXDI2",     # Diastolic BP 2nd
    # Labs
    "LBXTC",      # Total cholesterol (mg/dL)
    "LBXGH",      # Glycohemoglobin HbA1c (%)
    # Smoking
    "SMQ020",     # Smoked 100+ cigarettes in life
    "SMQ040",     # Current smoking status
    # Alcohol
    "ALQ101",     # Had 12+ drinks in past year
    "ALQ120Q",    # How often drink
    # Diabetes
    "DIQ010",     # Doctor told you have diabetes
    # Medical conditions
    "MCQ160B",    # Ever told had CHF
    "MCQ160C",    # Ever told had CHD
    "MCQ160F",    # Ever told had stroke
    "MCQ220",     # Ever told had cancer
]

MORT_VARS = ["ELIGSTAT", "MORTSTAT", "UCOD_LEADING",
             "DIABETES_MORT", "HYPERTEN_MORT",
             "PERMTH_INT", "PERMTH_EXM"]

# ── Main ──
print("=" * 65)
print("  02 - Processing NHANES Raw Files into Combined Dataset")
print(f"  Reading from: {RAW_DIR}")
print(f"  Saving to:    {OUTPUT_DIR}")
print("=" * 65)

all_cycles = []

for start, end, suffix, label in CYCLES:
    cycle_dir = os.path.join(RAW_DIR, label)
    if not os.path.isdir(cycle_dir):
        print(f"\n--- {label} --- FOLDER NOT FOUND, skipping")
        continue
    print(f"\n--- Cycle {label} ---")

    # Read all component files for this cycle
    file_map = get_local_files(suffix, cycle_dir)
    frames = {}
    for comp, path in file_map.items():
        print(f"  [{comp}]", end=" ", flush=True)
        if not os.path.exists(path):
            print(f"NOT FOUND ({os.path.basename(path)})")
            continue
        df = read_xpt(path)
        if df is not None:
            frames[comp] = df
            print(f"OK ({len(df):,}r)")
        else:
            print("READ ERROR")

    if "DEMO" not in frames:
        print(f"  WARNING: No DEMO for {label}, skipping cycle")
        continue

    # Merge all components within this cycle
    merged = frames["DEMO"].copy()
    for comp, df in frames.items():
        if comp == "DEMO":
            continue
        merged = merged.merge(df, on="SEQN", how="left",
                              suffixes=("", f"_{comp}"))

    # Read and merge mortality file
    mort_name = f"NHANES_{start}_{end}_MORT_2019_PUBLIC.dat"
    mort_path = os.path.join(cycle_dir, mort_name)
    print(f"  [MORT]", end=" ", flush=True)
    if os.path.exists(mort_path):
        mort = read_mortality(mort_path)
        if mort is not None:
            merged = merged.merge(mort, on="SEQN", how="left")
            print(f"OK ({len(mort):,}r)")
        else:
            print("READ ERROR")
    else:
        print("NOT FOUND")

    # Add cycle identifier and select harmonized variables
    merged["CYCLE"] = label
    all_wanted = KEEP_VARS + MORT_VARS + ["CYCLE"]
    available = [v for v in all_wanted if v in merged.columns]
    cycle_df = merged[available].copy()
    print(f"  -> {len(cycle_df):,} rows, {len(available)} cols kept")
    all_cycles.append(cycle_df)

# ── Stack all cycles ──
print("\n" + "=" * 65)
print("  COMBINING ALL CYCLES")
print("=" * 65)

if not all_cycles:
    print("\nERROR: No cycles were processed. Run 01_download_raw_files.py first.")
    sys.exit(1)

combined = pd.concat(all_cycles, ignore_index=True)
print(f"\nTotal dataset: {len(combined):,} rows x {combined.shape[1]} cols")
print(f"Cycles included: {combined['CYCLE'].nunique()}")

print(f"\nRows per cycle:")
for cyc, count in combined["CYCLE"].value_counts().sort_index().items():
    print(f"  {cyc}: {count:,}")

# ── Summary ──
elig = combined[combined["ELIGSTAT"] == 1]
print(f"\nMortality-eligible adults: {len(elig):,}")
print(f"  Deceased: {(elig['MORTSTAT'] == 1).sum():,}")
print(f"  Alive:    {(elig['MORTSTAT'] == 0).sum():,}")
if "PERMTH_INT" in elig.columns:
    pm = elig["PERMTH_INT"].dropna()
    print(f"\nFollow-up (person-months):")
    print(f"  Min:    {pm.min():.0f} mo ({pm.min()/12:.1f} yr)")
    print(f"  Median: {pm.median():.0f} mo ({pm.median()/12:.1f} yr)")
    print(f"  Max:    {pm.max():.0f} mo ({pm.max()/12:.1f} yr)")
if "RIDAGEYR" in combined.columns:
    print(f"\nAge range: {combined['RIDAGEYR'].min():.0f} - {combined['RIDAGEYR'].max():.0f}")

# ── Save ──
out = os.path.join(OUTPUT_DIR, "nhanes_all_cycles_with_mortality.csv")
combined.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(f"Size:  {os.path.getsize(out)/1e6:.1f} MB")
print("\nDone! Load with:")
print(f'  df = pd.read_csv(r"{out}")')
