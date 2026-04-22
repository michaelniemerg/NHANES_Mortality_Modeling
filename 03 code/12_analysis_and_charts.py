"""
12_analysis_and_charts.py
==========================
Final analysis script. Produces validation charts and summary
tables for the annual mortality models, composed survival,
relativity convergence, and sample individual mortality tables.

Input:  02 processed data - multiple files
        04 models - annual models
Output: 05 artifacts - charts and summary CSVs
"""
import pandas as pd
import numpy as np
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Import the mortality table builder
sys.path.insert(0, os.path.dirname(__file__))

import importlib.util
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "mortality_builder",
    os.path.join(SCRIPT_DIR, "11_mortality_table_builder.py"))
builder_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder_mod)
build_mortality_table = builder_mod.build_mortality_table

PROJECT_DIR  = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
PRED_PATH    = os.path.join(PROJECT_DIR, "02 processed data",
                            "nhanes_predictions_annual.csv")
COMPOSED_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                             "nhanes_predictions_composed.csv")

GRADING_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                            "grading_parameters.csv")
CDC_PATH     = os.path.join(PROJECT_DIR, "02 processed data",
                            "adjusted_cdc_life_table.csv")
BASELINE_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                             "baseline_annual_gam.csv")
FEATURES_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                             "nhanes_modeling_ready.csv")
ARTIFACT_DIR = os.path.join(PROJECT_DIR, "05 artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {"actual": "#2E5090", "predicted": "#D4652F"}
BAR_WIDTH = 0.35

# =====================================================================
# STEP 1: Load all data
# =====================================================================
print("=" * 65)
print("  12 - Analysis and Charts")
print("=" * 65)

pred = pd.read_csv(PRED_PATH)
composed = pd.read_csv(COMPOSED_PATH)
grading = pd.read_csv(GRADING_PATH)
cdc = pd.read_csv(CDC_PATH)
baseline = pd.read_csv(BASELINE_PATH)
features = pd.read_csv(FEATURES_PATH)
print(f"\n  All data loaded.")

# Merge features with predictions for subgroup analysis
test = pred.merge(features, on=["SEQN", "CYCLE"], how="left",
                  suffixes=("", "_feat"))
test = test[test["SPLIT"] == "TEST"].copy()

# =====================================================================
# CHART 1: Annual model calibration by year
# =====================================================================
print("\nChart 1: Annual calibration...")
fig, ax = plt.subplots(figsize=(10, 6))
years = range(1, 6)
actual_deaths = []
pred_deaths = []

for yr in years:
    at_risk = test[test[f"AT_RISK_YR{yr}"] == 1]
    a = at_risk[f"DIED_YR{yr}"].sum()
    p = at_risk[f"q{yr}_xgb"].sum()
    actual_deaths.append(a)
    pred_deaths.append(p)

x = np.arange(5)
ax.bar(x - BAR_WIDTH/2, actual_deaths, BAR_WIDTH,
       label="Actual Deaths", color=COLORS["actual"])
ax.bar(x + BAR_WIDTH/2, pred_deaths, BAR_WIDTH,
       label="sum(Predicted)", color=COLORS["predicted"])
for i in range(5):
    ax.text(i - BAR_WIDTH/2, actual_deaths[i] + 3,
            f"{actual_deaths[i]:.0f}", ha="center", fontsize=9)
    ax.text(i + BAR_WIDTH/2, pred_deaths[i] + 3,
            f"{pred_deaths[i]:.0f}", ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([f"Year {k}" for k in years])
ax.set_ylabel("Deaths")
ax.set_title("Annual Model Calibration: Actual vs Predicted Deaths (Test Set)",
             fontsize=14, fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(ARTIFACT_DIR, "annual_calibration.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: annual_calibration.png")

# =====================================================================
# CHART 2: Composed 5-year calibration by subgroup
# =====================================================================
print("\nChart 2: Relative mortality by subgroup (composed 5yr)...")

# Merge composed predictions with features
comp_test = composed.merge(features, on=["SEQN", "CYCLE"], how="left",
                            suffixes=("", "_feat"))
comp_test = comp_test[comp_test["SPLIT"] == "TEST"].copy()
comp_test = comp_test[comp_test["DIED_5YR"].notna()].copy()

def rel_mort(data, group_col, label_map=None):
    results = []
    for name, grp in data.groupby(group_col, dropna=False):
        if len(grp) < 20:
            continue
        gam_sum = grp["p_death_5yr_gam"].sum()
        actual = grp["DIED_5YR"].sum() / gam_sum if gam_sum > 0 else np.nan
        predicted = grp["p_death_5yr_xgb"].sum() / gam_sum if gam_sum > 0 else np.nan
        label = label_map.get(name, str(name)) if label_map else str(name)
        results.append({"group": label, "n": len(grp),
                        "actual_rel": actual, "pred_rel": predicted})
    return pd.DataFrame(results)

def plot_rel(rm, title, fname, figsize=(10, 6), rot=0):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(rm))
    ax.bar(x - BAR_WIDTH/2, rm["actual_rel"], BAR_WIDTH,
           label="Actual", color=COLORS["actual"])
    ax.bar(x + BAR_WIDTH/2, rm["pred_rel"], BAR_WIDTH,
           label="Predicted", color=COLORS["predicted"])
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1,
               label="Baseline (1.0)")
    for bar_set in [list(zip(x - BAR_WIDTH/2, rm["actual_rel"])),
                     list(zip(x + BAR_WIDTH/2, rm["pred_rel"]))]:
        for bx, h in bar_set:
            ax.text(bx, h + 0.02, f"{h:.2f}", ha="center", fontsize=8)
    labels = [f"{r['group']}\n(n={r['n']:,})" for _, r in rm.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="center")
    ax.set_ylabel("Relative Mortality")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return rm

# Create grouping variables
ct = comp_test
ct["gender_label"] = ct["IS_MALE"].map({1: "Male", 0: "Female"})
ct["smoke_status"] = np.nan
ct.loc[ct["SMQ020"] == 0, "smoke_status"] = 0
ct.loc[ct["SMQ040"] == 0, "smoke_status"] = 1
ct.loc[ct["SMQ040"].isin([1, 2]), "smoke_status"] = 2
smoke_map = {0: "Never", 1: "Former", 2: "Current"}
diab_map = {0: "No Diabetes", 1: "Borderline", 2: "Diabetes"}
bins_bmi = [0, 18.5, 25, 30, 35, 100]
labels_bmi = ["UW\n<18.5", "Normal\n18.5-25", "OW\n25-30",
              "Obese I\n30-35", "Obese II+\n35+"]
ct["bmi_group"] = pd.cut(ct["BMXBMI"], bins=bins_bmi, labels=labels_bmi)

educ_map = {1: "<9th", 2: "Some HS", 3: "HS/GED",
            4: "Some College", 5: "College Grad"}
bins_inc = [0, 1.0, 2.0, 3.0, 5.01]
labels_inc = ["Poverty", "Low", "Middle", "Higher"]
ct["income_group"] = pd.cut(ct["INDFMPIR"], bins=bins_inc,
                            labels=labels_inc, right=False)

# Generate all subgroup charts
all_tables = []
cuts = [
    ("gender_label", None, "Composed Relative Mortality by Gender",
     "composed_rel_mort_gender.png", (7,6), 0),
    ("smoke_status", smoke_map, "Composed Relative Mortality by Smoking",
     "composed_rel_mort_smoking.png", (8,6), 0),
    ("DIQ010", diab_map, "Composed Relative Mortality by Diabetes",
     "composed_rel_mort_diabetes.png", (8,6), 0),
    ("bmi_group", None, "Composed Relative Mortality by BMI",
     "composed_rel_mort_bmi.png", (10,6), 0),
    ("DMDEDUC2", educ_map, "Composed Relative Mortality by Education",
     "composed_rel_mort_education.png", (10,6), 0),
    ("income_group", None, "Composed Relative Mortality by Income",
     "composed_rel_mort_income.png", (9,6), 0),
]

for col, lmap, title, fname, fsize, rot in cuts:
    rm = rel_mort(ct, col, lmap)
    t = plot_rel(rm, title, fname, figsize=fsize, rot=rot)
    t["cut"] = title.split("by ")[-1]
    all_tables.append(t)

# Medical conditions combined
cond_rows = []
for col, label in [("MCQ160B","CHF"), ("MCQ160C","CHD"),
                    ("MCQ160F","Stroke"), ("MCQ220","Cancer")]:
    for val, vl in [(1,"Yes"), (0,"No")]:
        sub = ct[ct[col] == val]
        if len(sub) < 20:
            continue
        gam_s = sub["p_death_5yr_gam"].sum()
        cond_rows.append({
            "group": f"{label}: {vl}", "n": len(sub),
            "actual_rel": sub["DIED_5YR"].sum() / gam_s,
            "pred_rel": sub["p_death_5yr_xgb"].sum() / gam_s,
        })

rm_c = pd.DataFrame(cond_rows)
t = plot_rel(rm_c, "Composed Relative Mortality by Medical Condition",
             "composed_rel_mort_conditions.png", figsize=(14, 7))
t["cut"] = "Medical Conditions"
all_tables.append(t)

# Save all subgroup summary
summary = pd.concat(all_tables, ignore_index=True)
summary.to_csv(os.path.join(ARTIFACT_DIR, "composed_rel_mort_summary.csv"),
               index=False)
print("  Saved: composed_rel_mort_summary.csv")

# =====================================================================
# CHART 3: Sample individual mortality table profiles
# =====================================================================
print("\nChart 3: Sample mortality table profiles...")
grading_sorted = grading.sort_values("q1_xgb")
n = len(grading_sorted)
profiles = {
    "Low Risk":  grading_sorted.iloc[n // 10],
    "Median":    grading_sorted.iloc[n // 2],
    "High Risk": grading_sorted.iloc[int(n * 0.95)],
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors_profile = ["#2E8B57", "#DAA520", "#CC3333"]

for i, (label, person) in enumerate(profiles.items()):
    age = int(person["RIDAGEYR"])
    sex = "Male" if person["IS_MALE"] == 1 else "Female"
    aqx = {k: person[f"q{k}_xgb"] for k in range(1, 6)}
    r5 = person["rel_yr5"]
    decay = person["decay_rate"]
    table = build_mortality_table(
        age=age, sex=sex, annual_qx=aqx, rel_yr5=r5,
        decay_rate=decay, cdc_life_table=cdc, max_age=100)
    le = table.iloc[0]["ex"]

    ax = axes[i]
    ax.fill_between(table["attained_age"], table["lx"],
                     alpha=0.3, color=colors_profile[i])
    ax.plot(table["attained_age"], table["lx"],
            color=colors_profile[i], linewidth=2)
    ax.axvline(x=age + 5, color="gray", linestyle=":", linewidth=1)
    ax.text(age + 5.5, 0.95, "Model\nends", fontsize=8, color="gray")
    ax.set_xlabel("Age")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"{label}\n{sex} age {age}, LE={le:.1f}yr",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(age, 100)

fig.suptitle("Individual Survival Curves: Low, Median, High Risk",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ARTIFACT_DIR, "sample_survival_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: sample_survival_curves.png")

# =====================================================================
# CHART 4: Life expectancy distribution by subgroup
# =====================================================================
print("\nChart 4: Life expectancy distributions...")

# Sample 2000 individuals for LE computation (full set is expensive)
np.random.seed(42)
sample_idx = np.random.choice(len(grading), size=2000, replace=False)
le_rows = []

for idx in sample_idx:
    p = grading.iloc[idx]
    age = int(p["RIDAGEYR"])
    sex = "Male" if p["IS_MALE"] == 1 else "Female"
    aqx = {k: p[f"q{k}_xgb"] for k in range(1, 6)}
    r5 = p["rel_yr5"]
    decay = p["decay_rate"]
    table = build_mortality_table(
        age=age, sex=sex, annual_qx=aqx, rel_yr5=r5,
        decay_rate=decay, cdc_life_table=cdc, max_age=100)
    le = table.iloc[0]["ex"]
    le_rows.append({"age": age, "sex": sex, "le": le,
                    "is_male": int(p["IS_MALE"])})

le_df = pd.DataFrame(le_rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# By gender
for sex, color in [("Male", "#2E5090"), ("Female", "#D4652F")]:
    sub = le_df[le_df["sex"] == sex]
    axes[0].hist(sub["le"], bins=30, alpha=0.5, color=color,
                 label=f"{sex} (mean={sub['le'].mean():.1f})")
axes[0].set_xlabel("Life Expectancy (years)")
axes[0].set_ylabel("Count")
axes[0].set_title("Life Expectancy Distribution by Gender")
axes[0].legend()

# By age group
bins_age = [17, 40, 55, 70, 86]
labels_age = ["18-40", "41-55", "56-70", "71-85"]
le_df["age_group"] = pd.cut(le_df["age"], bins=bins_age, labels=labels_age)
age_colors = ["#2E8B57", "#DAA520", "#CC6600", "#CC3333"]
for ag, col in zip(labels_age, age_colors):
    sub = le_df[le_df["age_group"] == ag]
    if len(sub) > 0:
        axes[1].hist(sub["le"], bins=20, alpha=0.5, color=col,
                     label=f"{ag} (mean={sub['le'].mean():.1f})")
axes[1].set_xlabel("Life Expectancy (years)")
axes[1].set_ylabel("Count")
axes[1].set_title("Life Expectancy Distribution by Age Group")
axes[1].legend()

fig.suptitle("Predicted Life Expectancy Distributions (n=2,000 sample)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ARTIFACT_DIR, "life_expectancy_distributions.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: life_expectancy_distributions.png")

# =====================================================================
# CHART 5: Annual relativity trends (actual vs predicted by year)
# =====================================================================
print("\nChart 5: Annual relative mortality by year...")

# Show how relative mortality changes year-over-year for key subgroups
# Using smoking as the example (strongest lifestyle predictor)
test["smoke_status"] = np.nan
test.loc[test["SMQ020"] == 0, "smoke_status"] = 0
test.loc[test["SMQ040"] == 0, "smoke_status"] = 1
test.loc[test["SMQ040"].isin([1, 2]), "smoke_status"] = 2
test["smoke_label"] = test["smoke_status"].map(
    {0: "Never", 1: "Former", 2: "Current"})

# Simpler approach: compute rel mort per year per smoking group
fig, ax = plt.subplots(figsize=(10, 6))
for smoke_val, label, color in [(0, "Never Smoked", "#2E8B57"),
                                 (1, "Former Smoker", "#DAA520"),
                                 (2, "Current Smoker", "#CC3333")]:
    sub = test[test["smoke_status"] == smoke_val]
    if len(sub) < 50:
        continue

    actual_rels = []
    pred_rels = []
    for yr in range(1, 6):
        at_risk = sub[sub[f"AT_RISK_YR{yr}"] == 1]
        gam_sum = at_risk[f"q{yr}_gam"].sum()
        if gam_sum > 0:
            actual_rels.append(at_risk[f"DIED_YR{yr}"].sum() / gam_sum)
            pred_rels.append(at_risk[f"q{yr}_xgb"].sum() / gam_sum)
        else:
            actual_rels.append(np.nan)
            pred_rels.append(np.nan)
    ax.plot(range(1, 6), actual_rels, "o-", color=color, linewidth=2,
            label=f"{label} (actual)")
    ax.plot(range(1, 6), pred_rels, "s--", color=color, linewidth=1.5,
            alpha=0.7, label=f"{label} (predicted)")

ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Duration Year")
ax.set_ylabel("Relative Mortality (vs GAM baseline)")
ax.set_title("Annual Relative Mortality by Smoking Status (Test Set)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xticks(range(1, 6))
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(os.path.join(ARTIFACT_DIR, "annual_rel_mort_smoking.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: annual_rel_mort_smoking.png")

# =====================================================================
# Final summary
# =====================================================================
print(f"\n{'='*65}")
print("  PIPELINE COMPLETE")
print(f"{'='*65}")
print(f"\n  Charts saved to: {ARTIFACT_DIR}")
print(f"  Files created:")
for f in sorted(os.listdir(ARTIFACT_DIR)):
    if f.endswith((".png", ".csv")):
        size = os.path.getsize(os.path.join(ARTIFACT_DIR, f))
        print(f"    {f:45s} {size/1024:>7.0f} KB")
print("\nDone!")
