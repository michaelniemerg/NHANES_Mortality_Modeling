"""
13_cdc_vs_actual_comparison.py
===============================
Compares expected deaths from the raw CDC life table to actual
observed deaths in the NHANES data, by duration year and age band.

This answers: does the NHANES population die at the same rate as
the CDC population table predicts? If not, we need adjustment
factors before using the CDC table as a relativity basis.

Input:  02 processed data - nhanes_modeling_ready.csv
        02 processed data - cdc_life_table.csv
Output: 05 artifacts - cdc_vs_actual_aggregate.csv
        05 artifacts - cdc_vs_actual_by_duration.csv
        05 artifacts - cdc_vs_actual_by_age_band.csv
        05 artifacts - cdc_adjustment_factors.csv
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
ARTIFACT_DIR = os.path.join(PROJECT_DIR, "05 artifacts")

# =====================================================================
# STEP 1: Load data
# =====================================================================
print("=" * 70)
print("  13 - CDC Expected vs NHANES Actual Deaths")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)
cdc = pd.read_csv(CDC_PATH)
print(f"\n  NHANES: {len(df):,} rows")
print(f"  CDC table: {len(cdc)} rows")

# Build CDC lookup: (age, sex) -> qx
df["exam_age"] = df["RIDAGEYR"].astype(int)
df["sex"] = df["IS_MALE"].map({1: "Male", 0: "Female"})

cdc_lookup = {}
for _, row in cdc.iterrows():
    cdc_lookup[(int(row["age"]), row["sex"])] = row["qx"]

# =====================================================================
# STEP 2: For each duration, compute CDC expected vs actual
# =====================================================================
# For each person at risk in year k:
#   attained_age = exam_age + k - 1
#   cdc_expected_qx = CDC qx at (attained_age, sex)
# Sum these across all at-risk individuals = total expected deaths
# Compare to sum of actual deaths (DIED_YRk)
print("\nComputing expected vs actual by duration...")

detail_rows = []  # one row per person per duration

for k in range(1, 6):
    at_risk = df[df[f"AT_RISK_YR{k}"] == 1].copy()
    attained = (at_risk["exam_age"] + k - 1).clip(upper=100)
    at_risk["attained_age"] = attained
    at_risk["duration"] = k
    at_risk["actual_death"] = at_risk[f"DIED_YR{k}"]

    # Look up CDC qx for each person at their attained age and sex
    at_risk["cdc_qx"] = [
        cdc_lookup.get((a, s), np.nan)
        for a, s in zip(at_risk["attained_age"], at_risk["sex"])
    ]

    # Age band (5-year bands)
    bins = list(range(15, 91, 5)) + [120]
    labels = [f"{lo}-{lo+4}" for lo in range(15, 86, 5)] + ["85+"]
    at_risk["age_band"] = pd.cut(at_risk["attained_age"],
                                  bins=bins, labels=labels, right=False)

    detail_rows.append(at_risk[["SEQN", "duration", "sex",
                                "attained_age", "age_band",
                                "actual_death", "cdc_qx"]])

    actual = at_risk["actual_death"].sum()
    expected = at_risk["cdc_qx"].sum()
    factor = actual / expected if expected > 0 else np.nan
    print(f"  Duration {k}: at_risk={len(at_risk):,}  "
          f"actual={actual:.0f}  CDC_expected={expected:.1f}  "
          f"A/E={factor:.4f}")

# Stack all durations into one DataFrame
detail = pd.concat(detail_rows, ignore_index=True)
print(f"\n  Detail rows: {len(detail):,}")

# =====================================================================
# TABLE 1: Aggregate (all durations combined)
# =====================================================================
print("\n" + "=" * 70)
print("  TABLE 1: AGGREGATE")
print("=" * 70)

agg = detail.groupby("sex").agg(
    n_at_risk=("actual_death", "size"),
    actual_deaths=("actual_death", "sum"),
    cdc_expected=("cdc_qx", "sum"),
).reset_index()
agg["AoverE"] = agg["actual_deaths"] / agg["cdc_expected"]

# Add total row
total = pd.DataFrame([{
    "sex": "Total",
    "n_at_risk": agg["n_at_risk"].sum(),
    "actual_deaths": agg["actual_deaths"].sum(),
    "cdc_expected": agg["cdc_expected"].sum(),
    "AoverE": agg["actual_deaths"].sum() / agg["cdc_expected"].sum(),
}])
agg = pd.concat([agg, total], ignore_index=True)
agg["cdc_expected"] = agg["cdc_expected"].round(1)
agg["AoverE"] = agg["AoverE"].round(4)
print(agg.to_string(index=False))

agg.to_csv(os.path.join(ARTIFACT_DIR, "cdc_vs_actual_aggregate.csv"),
           index=False)
print("  Saved: cdc_vs_actual_aggregate.csv")

# =====================================================================
# TABLE 2: By Duration
# =====================================================================
print("\n" + "=" * 70)
print("  TABLE 2: BY DURATION")
print("=" * 70)

by_dur = detail.groupby("duration").agg(
    n_at_risk=("actual_death", "size"),
    actual_deaths=("actual_death", "sum"),
    cdc_expected=("cdc_qx", "sum"),
).reset_index()
by_dur["AoverE"] = by_dur["actual_deaths"] / by_dur["cdc_expected"]
by_dur["cdc_expected"] = by_dur["cdc_expected"].round(1)
by_dur["AoverE"] = by_dur["AoverE"].round(4)
print(by_dur.to_string(index=False))

# Also by duration x gender
by_dur_sex = detail.groupby(["duration", "sex"]).agg(
    n_at_risk=("actual_death", "size"),
    actual_deaths=("actual_death", "sum"),
    cdc_expected=("cdc_qx", "sum"),
).reset_index()
by_dur_sex["AoverE"] = by_dur_sex["actual_deaths"] / by_dur_sex["cdc_expected"]
by_dur_sex["cdc_expected"] = by_dur_sex["cdc_expected"].round(1)
by_dur_sex["AoverE"] = by_dur_sex["AoverE"].round(4)

print("\n  By Duration x Gender:")
print(by_dur_sex.to_string(index=False))

by_dur_all = pd.concat([by_dur.assign(sex="Total"), by_dur_sex],
                         ignore_index=True)
by_dur_all.to_csv(os.path.join(ARTIFACT_DIR, "cdc_vs_actual_by_duration.csv"),
                   index=False)
print("  Saved: cdc_vs_actual_by_duration.csv")

# =====================================================================
# TABLE 3: By Age Band x Duration
# =====================================================================
print("\n" + "=" * 70)
print("  TABLE 3: BY AGE BAND x DURATION")
print("=" * 70)

by_age = detail.groupby(["age_band", "duration"]).agg(
    n_at_risk=("actual_death", "size"),
    actual_deaths=("actual_death", "sum"),
    cdc_expected=("cdc_qx", "sum"),
).reset_index()
by_age["AoverE"] = np.where(
    by_age["cdc_expected"] > 0,
    by_age["actual_deaths"] / by_age["cdc_expected"],
    np.nan)
by_age["cdc_expected"] = by_age["cdc_expected"].round(1)
by_age["AoverE"] = by_age["AoverE"].round(4)

# Pivot for easier reading: rows = age band, columns = duration
pivot_ae = by_age.pivot_table(index="age_band", columns="duration",
                               values="AoverE", observed=True)
pivot_ae.columns = [f"Dur{c}_AE" for c in pivot_ae.columns]

pivot_n = by_age.pivot_table(index="age_band", columns="duration",
                              values="n_at_risk", aggfunc="sum",
                              observed=True)
pivot_n.columns = [f"Dur{c}_n" for c in pivot_n.columns]

pivot_actual = by_age.pivot_table(index="age_band", columns="duration",
                                   values="actual_deaths", aggfunc="sum",
                                   observed=True)
pivot_actual.columns = [f"Dur{c}_deaths" for c in pivot_actual.columns]

pivot_expected = by_age.pivot_table(index="age_band", columns="duration",
                                     values="cdc_expected", aggfunc="sum",
                                     observed=True)
pivot_expected.columns = [f"Dur{c}_expected" for c in pivot_expected.columns]

# Age band totals across all durations
age_totals = detail.groupby("age_band").agg(
    total_n=("actual_death", "size"),
    total_actual=("actual_death", "sum"),
    total_expected=("cdc_qx", "sum"),
).reset_index()
age_totals["total_AE"] = age_totals["total_actual"] / age_totals["total_expected"]
age_totals = age_totals.set_index("age_band")

# Combine everything
combined = pd.concat([age_totals, pivot_n, pivot_actual,
                       pivot_expected, pivot_ae], axis=1)
combined = combined.round(4)
print(combined.to_string())

# Save detailed version
by_age.to_csv(os.path.join(ARTIFACT_DIR, "cdc_vs_actual_by_age_band.csv"),
              index=False)
combined.to_csv(os.path.join(ARTIFACT_DIR, "cdc_vs_actual_by_age_band_pivot.csv"))
print("\n  Saved: cdc_vs_actual_by_age_band.csv")
print("  Saved: cdc_vs_actual_by_age_band_pivot.csv")

# =====================================================================
# TABLE 4: Adjustment Factors
# =====================================================================
# These factors would be applied to the CDC qx to create an
# "adjusted CDC table" that matches the NHANES population.
# adjusted_qx = CDC_qx * factor
print("\n" + "=" * 70)
print("  TABLE 4: CDC ADJUSTMENT FACTORS")
print("=" * 70)

# Factor by age band (pooled across durations and genders)
age_factors = age_totals[["total_n", "total_actual", "total_expected",
                           "total_AE"]].copy()
age_factors = age_factors.rename(columns={"total_AE": "factor"})
age_factors["factor"] = age_factors["factor"].round(4)
print("\n  By Age Band (all durations pooled):")
print(age_factors.to_string())

# Factor by age band x gender (pooled across durations)
age_sex_factors = detail.groupby(["age_band", "sex"]).agg(
    n_at_risk=("actual_death", "size"),
    actual_deaths=("actual_death", "sum"),
    cdc_expected=("cdc_qx", "sum"),
).reset_index()
age_sex_factors["factor"] = np.where(
    age_sex_factors["cdc_expected"] > 0,
    age_sex_factors["actual_deaths"] / age_sex_factors["cdc_expected"],
    np.nan)
age_sex_factors["factor"] = age_sex_factors["factor"].round(4)
age_sex_factors["cdc_expected"] = age_sex_factors["cdc_expected"].round(1)

print("\n  By Age Band x Gender (all durations pooled):")
print(age_sex_factors.to_string(index=False))

# Factor by age band x gender x duration (full granularity)
full_factors = detail.groupby(["age_band", "sex", "duration"]).agg(
    n_at_risk=("actual_death", "size"),
    actual_deaths=("actual_death", "sum"),
    cdc_expected=("cdc_qx", "sum"),
).reset_index()
full_factors["factor"] = np.where(
    full_factors["cdc_expected"] > 0,
    full_factors["actual_deaths"] / full_factors["cdc_expected"],
    np.nan)
full_factors["factor"] = full_factors["factor"].round(4)
full_factors["cdc_expected"] = full_factors["cdc_expected"].round(1)

# Save all factor tables
age_factors.to_csv(os.path.join(ARTIFACT_DIR,
                    "cdc_adjustment_factors_by_age.csv"))
age_sex_factors.to_csv(os.path.join(ARTIFACT_DIR,
                        "cdc_adjustment_factors_by_age_sex.csv"),
                        index=False)
full_factors.to_csv(os.path.join(ARTIFACT_DIR,
                     "cdc_adjustment_factors_full.csv"),
                     index=False)
print("\n  Saved: cdc_adjustment_factors_by_age.csv")
print("  Saved: cdc_adjustment_factors_by_age_sex.csv")
print("  Saved: cdc_adjustment_factors_full.csv")

# =====================================================================
# Summary
# =====================================================================
overall_actual = detail["actual_death"].sum()
overall_expected = detail["cdc_qx"].sum()
overall_factor = overall_actual / overall_expected

print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"\n  Overall A/E factor: {overall_factor:.4f}")
print(f"  ({overall_actual:.0f} actual deaths vs "
      f"{overall_expected:.1f} CDC expected)")
if overall_factor < 1:
    pct = (1 - overall_factor) * 100
    print(f"  -> NHANES population has {pct:.1f}% LOWER mortality "
          f"than CDC table")
else:
    pct = (overall_factor - 1) * 100
    print(f"  -> NHANES population has {pct:.1f}% HIGHER mortality "
          f"than CDC table")

print(f"\n  Interpretation:")
print(f"    An A/E < 1.0 is expected because NHANES excludes")
print(f"    institutionalized populations (nursing homes, prisons)")
print(f"    who have higher mortality than the general population.")
print(f"    The CDC table covers the full U.S. population.")

print(f"\n  BACKLOG - Theory 3:")
print(f"    Use these adjustment factors to create an adjusted CDC")
print(f"    table. Recalculate ALL relativities in the pipeline")
print(f"    against this adjusted table instead of the raw CDC table.")
print(f"    The adjusted CDC table could also replace the GAM baselines")
print(f"    as the expected mortality basis throughout the project.")
print(f"    This would simplify the pipeline (no GAMs needed) and")
print(f"    ground the relativities on a recognized mortality table.")

print("\nDone!")
