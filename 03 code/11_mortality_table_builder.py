"""
11_mortality_table_builder.py
==============================
Reusable module providing functions to build a complete individual
mortality table from XGBoost predictions and CDC population rates.

This is a LIBRARY — import it from other scripts or notebooks.
It can also be run standalone to generate sample tables.

Usage:
    from 11_mortality_table_builder import build_mortality_table

    table = build_mortality_table(
        age=55, sex="Male",
        annual_qx={1: 0.015, 2: 0.018, 3: 0.020, 4: 0.022, 5: 0.025},
        rel_yr5=1.8, decay_rate=0.90,
        cdc_life_table=cdc_df,
        max_age=100,
    )
"""
import pandas as pd
import numpy as np
import os

def build_mortality_table(age, sex, annual_qx, rel_yr5, decay_rate,
                          cdc_life_table, max_age=100):
    """
    Build a complete individual mortality table.

    Parameters
    ----------
    age : int
        Age at exam (starting age).
    sex : str
        "Male" or "Female".
    annual_qx : dict
        {1: q1, 2: q2, ..., 5: q5} from XGBoost annual models.
    rel_yr5 : float
        Relativity at year 5 (qx_model / qx_pop). Anchor for grading.
    decay_rate : float
        Annual exponential decay rate for grading (e.g., 0.90).
    cdc_life_table : DataFrame
        CDC life table with columns: age, sex, qx.
    max_age : int
        Project through this age (default 100).

    Returns
    -------
    DataFrame with columns:
        duration, attained_age, qx_pop, relativity, qx_adj, px, lx, dx, ex
    """
    # Build CDC lookup
    cdc_lookup = {}
    for _, row in cdc_life_table.iterrows():
        cdc_lookup[(int(row["age"]), row["sex"])] = row["qx"]

    rows = []
    max_duration = max_age - age + 1

    for t in range(1, max_duration + 1):
        att_age = age + t - 1
        if att_age > max_age:
            break

        qx_pop = cdc_lookup.get((att_age, sex), np.nan)

        if t <= 5:
            # Years 1-5: use XGBoost model predictions directly
            qx_adj = annual_qx.get(t, np.nan)
            if pd.notna(qx_pop) and qx_pop > 0:
                relativity = qx_adj / qx_pop
            else:
                relativity = np.nan
        else:
            # Years 6+: grade relativity toward 1.0 using exponential decay
            relativity = 1 + (rel_yr5 - 1) * (decay_rate ** (t - 5))
            qx_adj = relativity * qx_pop if pd.notna(qx_pop) else np.nan

        # Cap qx at [0, 1]
        if pd.notna(qx_adj):
            qx_adj = max(0, min(1.0, qx_adj))

        rows.append({
            "duration": t,
            "attained_age": att_age,
            "qx_pop": qx_pop,
            "relativity": relativity,
            "qx_adj": qx_adj,
        })

    result = pd.DataFrame(rows)

    # Compute life table columns
    result["px"] = 1 - result["qx_adj"]
    result["lx"] = 0.0
    result["dx"] = 0.0

    # lx: number surviving to start of each year (starting at 1.0)
    result.loc[result.index[0], "lx"] = 1.0
    for i in range(1, len(result)):
        result.loc[result.index[i], "lx"] = (
            result.loc[result.index[i-1], "lx"] *
            result.loc[result.index[i-1], "px"]
        )

    # dx: deaths in each year
    result["dx"] = result["lx"] * result["qx_adj"]

    # ex: curtate life expectancy (sum of future lx / current lx)
    result["ex"] = 0.0
    for i in range(len(result)):
        future_lx = result.loc[result.index[i+1:], "lx"].sum()
        current_lx = result.loc[result.index[i], "lx"]
        if current_lx > 0:
            result.loc[result.index[i], "ex"] = future_lx / current_lx

    return result

# =====================================================================
# Standalone: generate sample tables when run directly
# =====================================================================
if __name__ == "__main__":
    PROJECT_DIR = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
    CDC_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                            "cdc_life_table.csv")
    GRADING_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                                "grading_parameters.csv")
    ARTIFACT_DIR = os.path.join(PROJECT_DIR, "05 artifacts")

    cdc = pd.read_csv(CDC_PATH)
    grading = pd.read_csv(GRADING_PATH)

    print("=" * 65)
    print("  11 - Mortality Table Builder (sample tables)")
    print("=" * 65)

    # Pick 3 sample individuals: young healthy, middle, old high-risk
    # Sort by year-1 qx to find examples at different risk levels
    grading_sorted = grading.sort_values("q1_xgb")
    n = len(grading_sorted)
    samples = [
        grading_sorted.iloc[n // 10],       # low risk (10th percentile)
        grading_sorted.iloc[n // 2],         # median risk
        grading_sorted.iloc[int(n * 0.95)],  # high risk (95th percentile)
    ]

    for label, person in zip(["Low Risk", "Median", "High Risk"], samples):
        age = int(person["RIDAGEYR"])
        sex = "Male" if person["IS_MALE"] == 1 else "Female"
        annual_qx = {k: person[f"q{k}_xgb"] for k in range(1, 6)}
        rel5 = person["rel_yr5"]
        decay = person["decay_rate"]

        table = build_mortality_table(
            age=age, sex=sex, annual_qx=annual_qx,
            rel_yr5=rel5, decay_rate=decay,
            cdc_life_table=cdc, max_age=100,
        )

        le = table.iloc[0]["ex"]
        print(f"\n  --- {label}: age={age}, {sex}, "
              f"rel_yr5={rel5:.3f}, LE={le:.1f} years ---")
        # Show first 10 and last 5 rows
        preview = pd.concat([table.head(10), table.tail(5)])
        cols = ["duration", "attained_age", "qx_pop", "relativity",
                "qx_adj", "lx", "ex"]
        print(preview[cols].to_string(index=False, float_format="{:.5f}".format))

    print("\nDone! Import build_mortality_table() to use in other scripts.")
