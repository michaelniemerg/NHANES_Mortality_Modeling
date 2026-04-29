"""
Durational Durability Analysis — Relativity-Based
==================================================
Normalizes model predictions by CDC expected mortality (age/gender),
assigns quartiles on the RELATIVE score, then tracks A/E ratios
(actual deaths / CDC expected) by quartile across 15 duration years.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

PROJECT = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
ARTIFACT_DIR = os.path.join(PROJECT, "05 artifacts", "durational_durability")
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Load data ──
pred = pd.read_csv(os.path.join(PROJECT, "02 processed data",
                                "nhanes_predictions_annual.csv"))
combined = pd.read_csv(os.path.join(PROJECT, "02 processed data",
                                    "nhanes_all_cycles_with_mortality.csv"))
cdc = pd.read_csv(os.path.join(PROJECT, "02 processed data",
                                "cdc_life_table.csv"))

# ── Build CDC qx lookup: (age, sex) -> qx ──
cdc_lookup = {}
for _, row in cdc.iterrows():
    cdc_lookup[(int(row["age"]), row["sex"])] = row["qx"]

# ── Merge predictions with full follow-up ──
df = pred.merge(
    combined[["SEQN","CYCLE","MORTSTAT","PERMTH_EXM","RIDAGEYR"]],
    on=["SEQN","CYCLE"], how="inner", suffixes=("","_raw"))
df["exam_age"] = df["RIDAGEYR"].astype(int)
df["sex"] = df["IS_MALE"].map({1:"Male", 0:"Female"})

# ── Compute CDC expected qx for each year (attained age) ──
for yr in range(1, 6):
    att_age = (df["exam_age"] + yr - 1).clip(upper=100)
    df[f"q{yr}_cdc"] = [cdc_lookup.get((a, s), np.nan)
                        for a, s in zip(att_age, df["sex"])]

# ── Compose 5-year probabilities ──
# Model: P(die within 5yr) = 1 - prod(1 - q_k_xgb)
# CDC:   P(die within 5yr) = 1 - prod(1 - q_k_cdc)
model_surv = np.ones(len(df))
cdc_surv   = np.ones(len(df))
for yr in range(1, 6):
    model_surv *= (1 - df[f"q{yr}_xgb"])
    cdc_surv   *= (1 - df[f"q{yr}_cdc"])
df["p5_model"] = 1 - model_surv
df["p5_cdc"]   = 1 - cdc_surv

# ── Normalized relativity score ──
# Ratio of model 5-year mortality to CDC 5-year mortality
# Values > 1 = worse than population; < 1 = better than population
df["relativity_5yr"] = df["p5_model"] / df["p5_cdc"]

# Test set only
df = df[df["SPLIT"]=="TEST"].copy()

# Assign quartiles on the RELATIVITY score (age/gender normalized)
df["quartile"] = pd.qcut(df["relativity_5yr"], 4,
                         labels=["Q1 (Low)","Q2","Q3","Q4 (High)"])

print("="*72)
print("DURATIONAL DURABILITY — RELATIVITY-BASED ANALYSIS")
print("="*72)
print(f"\nTest set: {len(df):,} individuals, "
      f"{df['MORTSTAT'].sum():.0f} deaths")
print(f"Follow-up: up to {df['PERMTH_EXM'].max()/12:.1f} years\n")
print(f"Quartiles assigned on 5-year RELATIVITY "
      f"(model qx / CDC qx, age-gender normalized):\n")
fmt = "  {:>12}  n={:>5,}  rel={:.3f}-{:.3f}  "
fmt += "deaths={:.0f}  avg_age={:.1f}"
for q in ["Q1 (Low)","Q2","Q3","Q4 (High)"]:
    s = df[df["quartile"]==q]
    print(fmt.format(q, len(s),
          s["relativity_5yr"].min(), s["relativity_5yr"].max(),
          s["MORTSTAT"].sum(), s["exam_age"].mean()))

# ═══════════════════════════════════════════════════════════════════
# DURATIONAL A/E ANALYSIS
# For each duration year k and each quartile:
#   actual deaths / sum of CDC expected qx for at-risk individuals
# ═══════════════════════════════════════════════════════════════════
MAX_DUR = 15
results = []

for k in range(1, MAX_DUR+1):
    m0, m1 = (k-1)*12, k*12
    for q in ["Q1 (Low)","Q2","Q3","Q4 (High)"]:
        sub = df[df["quartile"]==q]

        # At risk: survived past start of year k
        at_risk = sub[sub["PERMTH_EXM"] > m0].copy()
        if len(at_risk) < 20:
            continue

        # Died during year k
        died = at_risk[(at_risk["MORTSTAT"]==1) &
                       (at_risk["PERMTH_EXM"]>m0) &
                       (at_risk["PERMTH_EXM"]<=m1)]
        # Survived past year k
        survived = at_risk[at_risk["PERMTH_EXM"] > m1]
        # Censored during year k (exclude from denom)
        censored = at_risk[(at_risk["MORTSTAT"]==0) &
                           (at_risk["PERMTH_EXM"]>m0) &
                           (at_risk["PERMTH_EXM"]<=m1)]

        # Denominator: died + survived (exclude censored)
        known = pd.concat([died, survived])
        n_known = len(known)
        n_died  = len(died)
        qx_actual = n_died / n_known if n_known > 0 else np.nan

        # CDC expected: sum of CDC qx for each at-risk person
        # at their attained age in duration year k
        att_age = (known["exam_age"] + k - 1).clip(upper=100)
        cdc_expected = sum(cdc_lookup.get((a, s), 0)
                          for a, s in zip(att_age, known["sex"]))
        ae_ratio = n_died / cdc_expected if cdc_expected > 0 else np.nan

        results.append(dict(
            duration=k, quartile=q,
            n_at_risk=n_known, n_died=n_died,
            cdc_expected=round(cdc_expected, 2),
            qx_actual=qx_actual,
            ae_ratio=ae_ratio,
            in_model=k<=5))

res = pd.DataFrame(results)

# ── Print results table ──
print(f"\n{'='*80}")
print("A/E RATIO BY QUARTILE AND DURATION")
print("(Actual Deaths / CDC Expected Deaths based on age+gender)")
print(f"{'='*80}")
print(f"{'Dur':>4} {'Quartile':>12} {'AtRisk':>7} {'Died':>5} "
      f"{'CDCExp':>8} {'A/E':>7} {'Window':>8}")
print("-"*62)
for _,r in res.iterrows():
    w = "MODEL" if r["in_model"] else "BEYOND"
    print(f"{r['duration']:>4.0f} {r['quartile']:>12} "
          f"{r['n_at_risk']:>7.0f} {r['n_died']:>5.0f} "
          f"{r['cdc_expected']:>8.1f} {r['ae_ratio']:>7.3f} {w:>8}")

# ── Color palette ──
COLORS = {"Q1 (Low)":"#2E8B57", "Q2":"#4682B4",
          "Q3":"#CC6600", "Q4 (High)":"#CC3333"}

# ═══════════════════════════════════════════════════════════════════
# CHART 1 — A/E Ratio by Quartile Over Time
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
for q in ["Q1 (Low)","Q2","Q3","Q4 (High)"]:
    s = res[res["quartile"]==q]
    ax.plot(s["duration"], s["ae_ratio"], "o-",
            color=COLORS[q], lw=2.5, ms=6, label=q)

ax.axhline(1.0, color="black", ls="-", lw=1, alpha=.4)
ax.axvline(5.5, color="grey", ls="--", lw=1.5, alpha=.6)
ax.text(3, ax.get_ylim()[1]*.92, "Model Window\n(Years 1–5)",
        ha="center", fontsize=10, color="grey", style="italic")
ax.text(10.25, ax.get_ylim()[1]*.92, "Beyond Model\n(Years 6–15)",
        ha="center", fontsize=10, color="grey", style="italic")
ax.set_xlabel("Duration Year Since Exam", fontsize=12)
ax.set_ylabel("A/E Ratio  (Actual Deaths / CDC Expected)", fontsize=12)
ax.set_title("Durability of Risk Stratification (Age-Gender Normalized)\n"
             "A/E Ratio by Relativity-Based Quartile",
             fontweight="bold", fontsize=14)
ax.legend(title="Relativity Quartile\n(age-gender normalized)", fontsize=10)
ax.set_xticks(range(1, MAX_DUR+1))
ax.set_xlim(.5, MAX_DUR+.5)
fig.tight_layout()
fig.savefig(os.path.join(ARTIFACT_DIR, "durational_ae_by_quartile.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: durational_ae_by_quartile.png")

# ═══════════════════════════════════════════════════════════════════
# CHART 2 — Cumulative Survival by Relativity Quartile
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 7))
for q in ["Q1 (Low)","Q2","Q3","Q4 (High)"]:
    s = res[res["quartile"]==q].sort_values("duration")
    lx = [1.0]
    for _,r in s.iterrows():
        lx.append(lx[-1]*(1-r["qx_actual"]))
    durs = [0]+list(s["duration"])
    ax.plot(durs, lx, "o-", color=COLORS[q], lw=2.5, ms=4, label=q)
    ax.annotate(f"  {lx[-1]:.1%}", (durs[-1], lx[-1]),
                fontsize=10, color=COLORS[q], fontweight="bold")

ax.axvline(5.5, color="grey", ls="--", lw=1.5, alpha=.6)
ax.set_xlabel("Duration Year Since Exam")
ax.set_ylabel("Cumulative Survival Probability")
ax.set_title("Survival by Relativity Quartile (Age-Gender Normalized)\n"
             "Test set, observed mortality",
             fontweight="bold", fontsize=14)
ax.legend(title="Relativity Quartile", fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_xticks(range(0, MAX_DUR+1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
fig.tight_layout()
fig.savefig(os.path.join(ARTIFACT_DIR, "durational_survival_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: durational_survival_curves.png")

# ═══════════════════════════════════════════════════════════════════
# CHART 3 — A/E Convergence Toward Population (Select & Ultimate)
# Shows how each quartile's A/E drifts toward 1.0 over time
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
pivot = res.pivot_table(index="duration", columns="quartile",
                        values="ae_ratio")

# Shade the population band around 1.0
ax.axhspan(0.8, 1.2, color="#EEEEEE", zorder=0)
ax.axhline(1.0, color="black", ls="-", lw=1.2, alpha=.5,
           label="CDC Population (A/E = 1.0)")

for q in ["Q1 (Low)","Q2","Q3","Q4 (High)"]:
    if q in pivot.columns:
        ax.plot(pivot.index, pivot[q], "o-", color=COLORS[q],
                lw=2.5, ms=6, label=q)

ax.axvline(5.5, color="grey", ls="--", lw=1.5, alpha=.6)
ax.set_xlabel("Duration Year Since Exam", fontsize=12)
ax.set_ylabel("A/E Ratio vs CDC Population", fontsize=12)
ax.set_title("Select-and-Ultimate Pattern:\n"
             "Convergence of A/E Ratios Toward Population Over Time",
             fontweight="bold", fontsize=14)
ax.legend(fontsize=10, loc="upper right")
ax.set_xticks(range(1, MAX_DUR+1))
ax.set_xlim(.5, MAX_DUR+.5)
fig.tight_layout()
fig.savefig(os.path.join(ARTIFACT_DIR,
            "durational_ae_convergence.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: durational_ae_convergence.png")

# ═══════════════════════════════════════════════════════════════════
# CHART 4 — Discrimination Ratio (Q4 A/E ÷ Q1 A/E)
# ═══════════════════════════════════════════════════════════════════
pivot["Q4/Q1_AE"] = pivot["Q4 (High)"] / pivot["Q1 (Low)"]
pivot["Q4/Q2_AE"] = pivot["Q4 (High)"] / pivot["Q2"]

fig, ax = plt.subplots(figsize=(13, 6))
w = .35
x = pivot.index.values
mask = ~pivot["Q4/Q1_AE"].isna()
ax.bar(x[mask]-w/2, pivot.loc[mask,"Q4/Q1_AE"], w,
       color="#CC3333", alpha=.75, label="Q4/Q1 A/E ratio", edgecolor="white")
ax.bar(x[mask]+w/2, pivot.loc[mask,"Q4/Q2_AE"], w,
       color="#4682B4", alpha=.75, label="Q4/Q2 A/E ratio", edgecolor="white")

for i in x[mask]:
    v = pivot.loc[i,"Q4/Q1_AE"]
    if not np.isnan(v):
        ax.text(i-w/2, v+.15, f"{v:.1f}x", ha="center",
                fontsize=7, color="#CC3333", fontweight="bold")

ax.axvline(5.5, color="grey", ls="--", lw=1.5, alpha=.6)
ax.axhline(1, color="grey", ls=":", lw=1, alpha=.5)
ax.set_xlabel("Duration Year Since Exam")
ax.set_ylabel("A/E Ratio of Q4 ÷ A/E Ratio of Q1 or Q2")
ax.set_title("Discrimination Power Over Time (Age-Gender Normalized)\n"
             "How Much Worse is Q4's A/E vs Q1's A/E?",
             fontweight="bold", fontsize=14)
ax.legend(fontsize=10)
ax.set_xticks(range(1, MAX_DUR+1))
fig.tight_layout()
fig.savefig(os.path.join(ARTIFACT_DIR,
            "durational_discrimination_ratio.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: durational_discrimination_ratio.png")

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("DURABILITY SUMMARY (AGE-GENDER NORMALIZED)")
print(f"{'='*72}")

print(f"\nA/E ratio by quartile and duration:")
print(f"  {'Dur':>4}  {'Q1 (Low)':>10}  {'Q2':>10}  {'Q3':>10}  "
      f"{'Q4 (High)':>10}  {'Q4/Q1':>7}")
print(f"  {'---':>4}  {'------':>10}  {'------':>10}  {'------':>10}  "
      f"{'------':>10}  {'-----':>7}")
for dur in range(1, MAX_DUR+1):
    if dur not in pivot.index:
        continue
    row = pivot.loc[dur]
    q41 = row.get("Q4/Q1_AE", np.nan)
    q41s = f"{q41:.1f}x" if not np.isnan(q41) else "n/a"
    print(f"  {dur:>4}  {row.get('Q1 (Low)',np.nan):>10.3f}  "
          f"{row.get('Q2',np.nan):>10.3f}  "
          f"{row.get('Q3',np.nan):>10.3f}  "
          f"{row.get('Q4 (High)',np.nan):>10.3f}  {q41s:>7}"
          f"  {'MODEL' if dur<=5 else 'BEYOND':>7}")

# Average A/E by quartile, in-window vs beyond
print(f"\nAverage A/E by quartile:")
for q in ["Q1 (Low)","Q2","Q3","Q4 (High)"]:
    s = res[res["quartile"]==q]
    in_w  = s[s["in_model"]]["ae_ratio"].mean()
    out_w = s[~s["in_model"]]["ae_ratio"].mean()
    print(f"  {q:>12}:  in-model={in_w:.3f}  beyond={out_w:.3f}")

# Q4/Q1 retention
q41_in  = pivot.loc[pivot.index<=5, "Q4/Q1_AE"].mean()
q41_out = pivot.loc[(pivot.index>5)&(pivot.index<=15), "Q4/Q1_AE"].mean()
print(f"\n  Avg Q4/Q1 A/E ratio, years 1-5:   {q41_in:.1f}x")
print(f"  Avg Q4/Q1 A/E ratio, years 6-15:  {q41_out:.1f}x")
if q41_in > 0:
    print(f"  Discrimination retained: {q41_out/q41_in:.0%}")

# 15-year survival by quartile
print(f"\n15-year cumulative survival by quartile:")
for q in ["Q1 (Low)","Q2","Q3","Q4 (High)"]:
    s = res[res["quartile"]==q].sort_values("duration")
    lx = 1.0
    for _,r in s.iterrows():
        lx *= (1 - r["qx_actual"])
    print(f"  {q:>12}: {lx:.1%} survival  "
          f"(avg age {df[df['quartile']==q]['exam_age'].mean():.0f})")

# Save
res.to_csv(os.path.join(ARTIFACT_DIR, "durational_analysis.csv"),
           index=False)
print(f"\nSaved: durational_analysis.csv")
print("Done!")
