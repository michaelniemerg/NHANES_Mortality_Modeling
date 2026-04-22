"""
14_create_adjusted_cdc_table.py
================================
Creates an adjusted CDC life table that matches NHANES mortality
experience. Uses 10-year age bands to compute A/E factors by
age and gender, applies them to the CDC qx, then applies an
overall scalar so that total adjusted expected deaths = total
actual NHANES deaths across the 5-year horizon.

This adjusted table replaces the raw CDC table AND the GAM
baselines as the expected mortality basis throughout the pipeline.

Input:  02 processed data - nhanes_modeling_ready.csv
        02 processed data - cdc_life_table.csv
Output: 02 processed data - adjusted_cdc_life_table.csv
        05 artifacts - cdc_adjustment_detail.csv
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR  = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
INPUT_PATH   = os.path.join(PROJECT_DIR, "02 processed data",
                            "nhanes_modeling_ready.csv")
CDC_PATH     = os.path.join(PROJECT_DIR, "02 processed data",
                            "cdc_life_table.csv")
OUTPUT_PATH  = os.path.join(PROJECT_DIR, "02 processed data",
                            "adjusted_cdc_life_table.csv")
ARTIFACT_DIR = os.path.join(PROJECT_DIR, "05 artifacts")

# =====================================================================
# STEP 1: Load data
# =====================================================================
print("=" * 70)
print("  14 - Create Adjusted CDC Life Table")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)
cdc = pd.read_csv(CDC_PATH)
print(f"\n  NHANES: {len(df):,} rows")
print(f"  CDC table: {len(cdc)} rows")

df["exam_age"] = df["RIDAGEYR"].astype(int)
df["sex"] = df["IS_MALE"].map({1: "Male", 0: "Female"})

cdc_lookup = {}
for _, row in cdc.iterrows():
    cdc_lookup[(int(row["age"]), row["sex"])] = row["qx"]

# =====================================================================
# STEP 2: Build exposure records for all at-risk person-years
# =====================================================================
# For each person at risk in each duration year, compute their
# attained age and look up the CDC qx. This gives us one record
# per person per duration year of exposure.
print("\nBuilding exposure records...")

records = []
for k in range(1, 6):
    at_risk = df[df[f"AT_RISK_YR{k}"] == 1].copy()
    at_risk["duration"] = k
    at_risk["attained_age"] = (at_risk["exam_age"] + k - 1).clip(upper=100)
    at_risk["actual_death"] = at_risk[f"DIED_YR{k}"]

    at_risk["cdc_qx"] = [
        cdc_lookup.get((a, s), np.nan)
        for a, s in zip(at_risk["attained_age"], at_risk["sex"])
    ]

    # 10-year age bands based on attained age
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 120]
    labels = ["<20", "20-29", "30-39", "40-49", "50-59",
              "60-69", "70-79", "80-89", "90+"]
    at_risk["age_band_10"] = pd.cut(at_risk["attained_age"],
                                     bins=bins, labels=labels, right=False)

    records.append(at_risk[["sex", "attained_age", "age_band_10",
                             "duration", "actual_death", "cdc_qx"]])

exposure = pd.concat(records, ignore_index=True)
print(f"  Total exposure records: {len(exposure):,}")

# =====================================================================
# STEP 3: Compute A/E factors by 10-year age band and gender
# =====================================================================
print("\nComputing A/E factors by 10-year age band x gender...")

ae_factors = exposure.groupby(["age_band_10", "sex"]).agg(
    n_exposure=("actual_death", "size"),
    actual_deaths=("actual_death", "sum"),
    cdc_expected=("cdc_qx", "sum"),
).reset_index()
ae_factors["raw_AE"] = np.where(
    ae_factors["cdc_expected"] > 0,
    ae_factors["actual_deaths"] / ae_factors["cdc_expected"],
    np.nan)

print(f"\n  {'Age Band':>10} {'Sex':>7} {'Exposure':>9} {'Actual':>7} "
      f"{'Expected':>9} {'Raw A/E':>8}")
print(f"  {'-'*10:>10} {'-'*7:>7} {'-'*9:>9} {'-'*7:>7} "
      f"{'-'*9:>9} {'-'*8:>8}")
for _, r in ae_factors.iterrows():
    ae_str = f"{r['raw_AE']:.4f}" if pd.notna(r['raw_AE']) else "n/a"
    print(f"  {r['age_band_10']:>10} {r['sex']:>7} {r['n_exposure']:>9,} "
          f"{r['actual_deaths']:>7.0f} {r['cdc_expected']:>9.1f} "
          f"{ae_str:>8}")

# =====================================================================
# STEP 4: Apply age-band A/E factors to CDC qx
# =====================================================================
# For each (age, sex) row in the CDC table, find its 10-year age band,
# look up the A/E factor, and multiply: preliminary_qx = CDC_qx * A/E
print("\nApplying age-band A/E factors to CDC table...")

# Build factor lookup: (age_band, sex) -> raw_AE
factor_lookup = {}
for _, r in ae_factors.iterrows():
    if pd.notna(r["raw_AE"]):
        factor_lookup[(r["age_band_10"], r["sex"])] = r["raw_AE"]

# Assign age bands to CDC table
cdc_adj = cdc.copy()
bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 120]
labels = ["<20", "20-29", "30-39", "40-49", "50-59",
          "60-69", "70-79", "80-89", "90+"]
cdc_adj["age_band_10"] = pd.cut(cdc_adj["age"], bins=bins,
                                 labels=labels, right=False)

# Apply factor
cdc_adj["age_band_factor"] = [
    factor_lookup.get((ab, s), 1.0)
    for ab, s in zip(cdc_adj["age_band_10"], cdc_adj["sex"])
]
cdc_adj["qx_prelim"] = cdc_adj["qx"] * cdc_adj["age_band_factor"]

# =====================================================================
# STEP 5: Compute overall calibration scalar
# =====================================================================
# Using the preliminary adjusted qx, recompute expected deaths
# across all NHANES exposure records. Then find the scalar that
# makes total expected = total actual.
print("\nComputing overall calibration scalar...")

# Build preliminary adjusted lookup
prelim_lookup = {}
for _, row in cdc_adj.iterrows():
    prelim_lookup[(int(row["age"]), row["sex"])] = row["qx_prelim"]

# Look up preliminary expected for each exposure record
exposure["prelim_qx"] = [
    prelim_lookup.get((a, s), np.nan)
    for a, s in zip(exposure["attained_age"], exposure["sex"])
]

total_actual = exposure["actual_death"].sum()
total_prelim_expected = exposure["prelim_qx"].sum()
overall_scalar = total_actual / total_prelim_expected

print(f"  Total actual deaths:       {total_actual:.0f}")
print(f"  Total prelim expected:     {total_prelim_expected:.1f}")
print(f"  Overall calibration scalar: {overall_scalar:.6f}")
print(f"  (Applied on top of age-band factors)")

# =====================================================================
# STEP 6: Create final adjusted table
# =====================================================================
cdc_adj["overall_scalar"] = overall_scalar
cdc_adj["combined_factor"] = cdc_adj["age_band_factor"] * overall_scalar
cdc_adj["qx_adjusted"] = cdc_adj["qx"] * cdc_adj["combined_factor"]

# Cap at [0, 1]
cdc_adj["qx_adjusted"] = cdc_adj["qx_adjusted"].clip(0, 1.0)

# =====================================================================
# STEP 7: Verify calibration
# =====================================================================
print("\nVerifying calibration...")

adj_lookup = {}
for _, row in cdc_adj.iterrows():
    adj_lookup[(int(row["age"]), row["sex"])] = row["qx_adjusted"]

exposure["adj_qx"] = [
    adj_lookup.get((a, s), np.nan)
    for a, s in zip(exposure["attained_age"], exposure["sex"])
]

verify_total_exp = exposure["adj_qx"].sum()
verify_total_act = exposure["actual_death"].sum()
print(f"  Adjusted expected: {verify_total_exp:.1f}")
print(f"  Actual deaths:     {verify_total_act:.0f}")
print(f"  Ratio:             {verify_total_exp / verify_total_act:.6f}")

# Check by age band
print(f"\n  Verification by age band:")
print(f"  {'Age Band':>10} {'Actual':>7} {'Adj Exp':>8} {'Ratio':>7}")
print(f"  {'-'*10:>10} {'-'*7:>7} {'-'*8:>8} {'-'*7:>7}")
for ab in labels:
    sub = exposure[exposure["age_band_10"] == ab]
    if len(sub) == 0:
        continue
    act = sub["actual_death"].sum()
    exp = sub["adj_qx"].sum()
    ratio = act / exp if exp > 0 else np.nan
    r_str = f"{ratio:.4f}" if pd.notna(ratio) else "n/a"
    print(f"  {ab:>10} {act:>7.0f} {exp:>8.1f} {r_str:>7}")

# Check by duration
print(f"\n  Verification by duration:")
print(f"  {'Dur':>5} {'Actual':>7} {'Adj Exp':>8} {'Ratio':>7}")
print(f"  {'-'*5:>5} {'-'*7:>7} {'-'*8:>8} {'-'*7:>7}")
for k in range(1, 6):
    sub = exposure[exposure["duration"] == k]
    act = sub["actual_death"].sum()
    exp = sub["adj_qx"].sum()
    ratio = act / exp if exp > 0 else np.nan
    print(f"  {k:>5} {act:>7.0f} {exp:>8.1f} {ratio:>7.4f}")

# Check by gender
print(f"\n  Verification by gender:")
for sex in ["Female", "Male"]:
    sub = exposure[exposure["sex"] == sex]
    act = sub["actual_death"].sum()
    exp = sub["adj_qx"].sum()
    ratio = act / exp if exp > 0 else np.nan
    print(f"  {sex:>8}: actual={act:.0f}  expected={exp:.1f}  "
          f"ratio={ratio:.4f}")

# =====================================================================
# STEP 8: Save outputs
# =====================================================================
# Save the adjusted life table (this replaces cdc_life_table.csv
# as the expected basis throughout the pipeline)
output = cdc_adj[["age", "sex", "qx", "qx_adjusted",
                   "age_band_10", "age_band_factor",
                   "overall_scalar", "combined_factor"]].copy()
output = output.rename(columns={"qx": "qx_cdc_original"})
output = output.sort_values(["sex", "age"]).reset_index(drop=True)
output.to_csv(OUTPUT_PATH, index=False)
print(f"\n  Saved adjusted table: {OUTPUT_PATH}")
print(f"  Rows: {len(output)}")

# Save the A/E detail
ae_factors["overall_scalar"] = overall_scalar
ae_factors["combined_factor"] = ae_factors["raw_AE"] * overall_scalar
ae_factors.to_csv(os.path.join(ARTIFACT_DIR, "cdc_adjustment_detail.csv"),
                  index=False)
print(f"  Saved detail: cdc_adjustment_detail.csv")

# Preview
print("\n  Preview of adjusted table (selected ages):")
prev_ages = [18, 30, 40, 50, 60, 70, 80, 90, 100]
prev = output[output["age"].isin(prev_ages)]
print(prev[["age", "sex", "qx_cdc_original", "combined_factor",
            "qx_adjusted"]].to_string(index=False))

print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"\n  Adjusted CDC table created with:")
print(f"    - 10-year age band x gender A/E factors")
print(f"    - Overall calibration scalar: {overall_scalar:.6f}")
print(f"    - Total adjusted expected = total actual deaths")
print(f"\n  This table should now replace:")
print(f"    - cdc_life_table.csv as the population qx basis")
print(f"    - GAM baselines as the expected mortality basis")
print(f"    in scripts 06, 09, 10, 11, and 12.")

print("\nDone!")
