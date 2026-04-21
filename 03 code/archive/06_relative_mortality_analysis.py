"""
06_relative_mortality_analysis.py
==================================
Computes actual and predicted relative mortality across key
subgroups and creates comparison visualizations.

Definitions:
  Actual Relative Mortality   = sum(died_within_5yr) / sum(baseline_prob_gam)
  Predicted Relative Mortality = sum(xgb_prob) / sum(baseline_prob_gam)

A value of 1.0 means mortality matches the age-gender baseline.
Values > 1 indicate excess mortality; < 1 indicate lower mortality.

All analysis uses the TEST set only for honest evaluation.

Input:  02 processed data - nhanes_predictions.csv
        02 processed data - nhanes_modeling_ready.csv
Output: 05 artifacts - relative_mortality_*.png (charts)
        05 artifacts - relative_mortality_summary.csv
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR   = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
PRED_PATH     = os.path.join(PROJECT_DIR, "02 processed data",
                             "nhanes_predictions.csv")
FEATURES_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                             "nhanes_modeling_ready.csv")
ARTIFACT_DIR  = os.path.join(PROJECT_DIR, "05 artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {"actual": "#2E5090", "predicted": "#D4652F"}
BAR_WIDTH = 0.35

# =====================================================================
# STEP 1: Load and merge data
# =====================================================================
print("Loading data...")
pred = pd.read_csv(PRED_PATH)
features = pd.read_csv(FEATURES_PATH)
df = pred.merge(features, left_on=["SEQN", "CYCLE"],
                right_on=["SEQN", "CYCLE"], how="left")
df = df[df["split"] == "TEST"].copy()
print(f"  Test set: {len(df):,} rows")

# =====================================================================
# STEP 2: Helper function for grouped relative mortality
# =====================================================================
def compute_relative_mortality(data, group_col, label_map=None):
    """
    For each group, compute:
      actual_rel_mort = sum(died) / sum(baseline_prob_gam)
      pred_rel_mort   = sum(xgb_prob) / sum(baseline_prob_gam)
      n               = count
    Both use sum/sum aggregation so they are on the same scale.
    """
    results = []
    for name, grp in data.groupby(group_col, dropna=False):
        if len(grp) < 20:
            continue
        baseline_sum = grp["baseline_prob_gam"].sum()
        actual = grp["died_within_5yr"].sum() / baseline_sum
        predicted = grp["xgb_prob"].sum() / baseline_sum
        label = label_map.get(name, str(name)) if label_map else str(name)
        results.append({
            "group": label,
            "n": len(grp),
            "deaths": grp["died_within_5yr"].sum(),
            "actual_rel_mort": actual,
            "pred_rel_mort": predicted,
        })
    return pd.DataFrame(results)

# =====================================================================
# STEP 3: Charting helper
# =====================================================================
def plot_relative_mortality(rm_df, title, filename, figsize=(10, 6),
                            rotate_labels=0):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(rm_df))
    bars1 = ax.bar(x - BAR_WIDTH/2, rm_df["actual_rel_mort"],
                   BAR_WIDTH, label="Actual", color=COLORS["actual"])
    bars2 = ax.bar(x + BAR_WIDTH/2, rm_df["pred_rel_mort"],
                   BAR_WIDTH, label="Predicted", color=COLORS["predicted"])
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1,
               label="Baseline (1.0)")
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    labels = [f"{row['group']}\n(n={row['n']:,})"
              for _, row in rm_df.iterrows()]
    ax.set_xticklabels(labels, rotation=rotate_labels, ha="center")
    ax.set_ylabel("Relative Mortality (vs age-gender baseline)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = os.path.join(ARTIFACT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")
    return rm_df

# =====================================================================
# STEP 4: Create grouping variables
# =====================================================================
print("\nCreating subgroup variables...")
bins_age = [17, 30, 40, 50, 60, 70, 86]
labels_age = ["18-30", "31-40", "41-50", "51-60", "61-70", "71-85"]
df["age_group"] = pd.cut(df["age"], bins=bins_age, labels=labels_age)
df["gender_label"] = df["is_male"].map({1: "Male", 0: "Female"})
df["age_gender"] = df["age_group"].astype(str) + " " + df["gender_label"]

bins_bmi = [0, 18.5, 25, 30, 35, 100]
labels_bmi = ["Underweight\n(<18.5)", "Normal\n(18.5-25)",
              "Overweight\n(25-30)", "Obese I\n(30-35)", "Obese II+\n(35+)"]
df["bmi_group"] = pd.cut(df["BMXBMI"], bins=bins_bmi, labels=labels_bmi)

smoke_map = {0: "Never\nSmoked", 1: "Former\nSmoker", 2: "Current\nSmoker"}
df["smoke_status"] = np.nan
df.loc[df["SMQ020"] == 0, "smoke_status"] = 0
df.loc[df["SMQ040"] == 0, "smoke_status"] = 1
df.loc[df["SMQ040"].isin([1, 2]), "smoke_status"] = 2

diab_map = {0: "No Diabetes", 1: "Borderline", 2: "Diabetes"}
educ_map = {1: "<9th Grade", 2: "Some HS", 3: "HS/GED",
            4: "Some College", 5: "College Grad"}
bins_inc = [0, 1.0, 2.0, 3.0, 5.01]
labels_inc = ["Poverty\n(<1.0)", "Low\n(1.0-2.0)",
              "Middle\n(2.0-3.0)", "Higher\n(3.0+)"]
df["income_group"] = pd.cut(df["INDFMPIR"], bins=bins_inc,
                            labels=labels_inc, right=False)
bins_a1c = [0, 5.7, 6.5, 100]
labels_a1c = ["Normal\n(<5.7%)", "Prediabetes\n(5.7-6.5%)",
              "Diabetes\n(>=6.5%)"]
df["a1c_group"] = pd.cut(df["LBXGH"], bins=bins_a1c, labels=labels_a1c)
bins_bp = [0, 120, 130, 140, 300]
labels_bp = ["Normal\n(<120)", "Elevated\n(120-130)",
             "Stage 1 HTN\n(130-140)", "Stage 2 HTN\n(>=140)"]
df["bp_group"] = pd.cut(df["AVG_SBP"], bins=bins_bp, labels=labels_bp)
bins_chol = [0, 200, 240, 600]
labels_chol = ["Desirable\n(<200)", "Borderline\n(200-240)", "High\n(>=240)"]
df["chol_group"] = pd.cut(df["LBXTC"], bins=bins_chol, labels=labels_chol)

race_map = {1: "Mexican\nAmerican", 2: "Other\nHispanic", 3: "NH White",
            4: "NH Black", 5: "Other/\nMulti"}
df["race_code"] = np.nan
df.loc[df["RACE_MEXICAN_AMERICAN"] == 1, "race_code"] = 1
df.loc[df["RACE_OTHER_HISPANIC"] == 1, "race_code"] = 2
df.loc[df["RACE_NH_BLACK"] == 1, "race_code"] = 4
df.loc[df["RACE_OTHER_MULTI"] == 1, "race_code"] = 5
mask_white = ((df["RACE_MEXICAN_AMERICAN"] == 0) &
              (df["RACE_OTHER_HISPANIC"] == 0) &
              (df["RACE_NH_BLACK"] == 0) &
              (df["RACE_OTHER_MULTI"] == 0))
df.loc[mask_white, "race_code"] = 3

# =====================================================================
# STEP 5: Generate all charts
# =====================================================================
print("\nGenerating charts...")
all_tables = []

cuts = [
    ("CYCLE",        None,      "Relative Mortality by NHANES Cycle",
     "relative_mortality_by_cycle.png", (12,6), 0, "Cycle"),
    ("gender_label", None,      "Relative Mortality by Gender",
     "relative_mortality_by_gender.png", (7,6), 0, "Gender"),
    ("age_group",    None,      "Relative Mortality by Age Group",
     "relative_mortality_by_age.png", (10,6), 0, "Age Group"),
    ("smoke_status", smoke_map, "Relative Mortality by Smoking Status",
     "relative_mortality_by_smoking.png", (8,6), 0, "Smoking"),
    ("DIQ010",       diab_map,  "Relative Mortality by Diabetes Status",
     "relative_mortality_by_diabetes.png", (8,6), 0, "Diabetes"),
    ("bmi_group",    None,      "Relative Mortality by BMI Group",
     "relative_mortality_by_bmi.png", (10,6), 0, "BMI"),
    ("DMDEDUC2",     educ_map,  "Relative Mortality by Education Level",
     "relative_mortality_by_education.png", (10,6), 0, "Education"),
    ("income_group", None,      "Relative Mortality by Income-to-Poverty Ratio",
     "relative_mortality_by_income.png", (9,6), 0, "Income"),
    ("a1c_group",    None,      "Relative Mortality by HbA1c Level",
     "relative_mortality_by_a1c.png", (8,6), 0, "HbA1c"),
    ("bp_group",     None,      "Relative Mortality by Systolic Blood Pressure",
     "relative_mortality_by_bp.png", (9,6), 0, "Blood Pressure"),
    ("chol_group",   None,      "Relative Mortality by Total Cholesterol",
     "relative_mortality_by_cholesterol.png", (8,6), 0, "Cholesterol"),
    ("race_code",    race_map,  "Relative Mortality by Race/Ethnicity",
     "relative_mortality_by_race.png", (10,6), 0, "Race/Ethnicity"),
]

for col, lmap, title, fname, fsize, rot, cut_label in cuts:
    rm = compute_relative_mortality(df, col, lmap)
    if col == "CYCLE":
        rm = rm.sort_values("group")
    t = plot_relative_mortality(rm, title, fname, figsize=fsize,
                                rotate_labels=rot)
    t["cut"] = cut_label
    all_tables.append(t)

# Age x Gender (special sort order)
rm = compute_relative_mortality(df, "age_gender")
order = [f"{a} {g}" for a in labels_age for g in ["Female", "Male"]]
rm["sort_key"] = rm["group"].map({v: i for i, v in enumerate(order)})
rm = rm.sort_values("sort_key").drop(columns="sort_key")
t = plot_relative_mortality(rm, "Relative Mortality by Age Group x Gender",
                            "relative_mortality_by_age_gender.png",
                            figsize=(16, 7), rotate_labels=45)
t["cut"] = "Age x Gender"
all_tables.append(t)

# Medical conditions (Yes/No for each, combined chart)
conditions = {"MCQ160B": "CHF", "MCQ160C": "CHD",
              "MCQ160F": "Stroke", "MCQ220": "Cancer"}
cond_rows = []
for col, label in conditions.items():
    for val, val_label in [(1, "Yes"), (0, "No")]:
        subset = df[df[col] == val]
        if len(subset) < 20:
            continue
        baseline_sum = subset["baseline_prob_gam"].sum()
        actual = subset["died_within_5yr"].sum() / baseline_sum
        predicted = subset["xgb_prob"].sum() / baseline_sum
        cond_rows.append({
            "group": f"{label}: {val_label}",
            "n": len(subset),
            "deaths": subset["died_within_5yr"].sum(),
            "actual_rel_mort": actual,
            "pred_rel_mort": predicted,
        })
rm_cond = pd.DataFrame(cond_rows)
t = plot_relative_mortality(rm_cond, "Relative Mortality by Medical Condition",
                            "relative_mortality_by_conditions.png",
                            figsize=(14, 7), rotate_labels=0)
t["cut"] = "Medical Conditions"
all_tables.append(t)

# =====================================================================
# STEP 6: Save combined summary table
# =====================================================================
summary = pd.concat(all_tables, ignore_index=True)
summary = summary[["cut", "group", "n", "deaths",
                   "actual_rel_mort", "pred_rel_mort"]]
summary["actual_rel_mort"] = summary["actual_rel_mort"].round(4)
summary["pred_rel_mort"] = summary["pred_rel_mort"].round(4)
summary_path = os.path.join(ARTIFACT_DIR, "relative_mortality_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"\n  Summary table saved: {summary_path}")

# =====================================================================
# STEP 7: Print summary
# =====================================================================
print("\n" + "=" * 65)
print("  RELATIVE MORTALITY SUMMARY (all cuts)")
print("=" * 65)
for cut_name in summary["cut"].unique():
    cut_data = summary[summary["cut"] == cut_name]
    print(f"\n  --- {cut_name} ---")
    for _, row in cut_data.iterrows():
        g = row["group"].replace("\n", " ")
        print(f"    {g:25s}  n={row['n']:>6,}  "
              f"actual={row['actual_rel_mort']:.2f}  "
              f"pred={row['pred_rel_mort']:.2f}")

print(f"\n\nAll charts saved to: {ARTIFACT_DIR}")
print("Done!")
