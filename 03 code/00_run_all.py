"""
00_run_all.py
==============
Master script that runs the entire NHANES mortality prediction
pipeline end to end. Execute this single file to reproduce
everything from raw data download through final analysis.

Usage:
    python "03 code/00_run_all.py"

Each step prints a banner and its output. If any step fails,
the pipeline stops and reports which script failed.
"""
import subprocess
import sys
import os
import time

# Project root is one level up from this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Scripts to run, in order.
# Each entry: (filename, description, skip_if_output_exists)
# Set skip_if_output_exists to a file path to skip the step
# if that output already exists (useful for the slow download steps).
# Set to None to always run.

STEPS = [
    ("01_download_raw_files.py",
     "Download raw NHANES files from CDC",
     os.path.join(PROJECT_DIR, "01 raw data", "2017-2018",
                  "NHANES_2017_2018_MORT_2019_PUBLIC.dat")),

    ("02_process_and_combine.py",
     "Combine raw files into single CSV",
     None),

    ("03_preprocess_annual.py",
     "Create conditional annual outcomes and features",
     None),

    ("04_baseline_gam_annual.py",
     "Fit GAM baselines by age, gender, and year",
     None),

    ("05_train_annual_models.py",
     "Train 5 annual XGBoost models",
     None),

    ("08_download_cdc_life_table.py",
     "Download CDC 2021 period life table",
     os.path.join(PROJECT_DIR, "02 processed data",
                  "cdc_life_table.csv")),

    ("06_predict_annual.py",
     "Score all individuals with annual models",
     None),

    ("07_compose_multiyear_survival.py",
     "Compose annual qx into multi-year survival",
     None),

    ("09_fit_relativity_splines.py",
     "Compute relativities and analyze convergence",
     None),

    ("10_project_and_grade.py",
     "Project relativities beyond year 5",
     None),

    ("11_mortality_table_builder.py",
     "Generate sample mortality tables (library demo)",
     None),

    ("12_analysis_and_charts.py",
     "Final analysis, charts, and summary tables",
     None),

    ("13_cdc_vs_actual_comparison.py",
     "CDC vs actual deaths comparison and A/E factors",
     None),
]

def run_step(step_num, filename, description, skip_check):
    """Run a single pipeline step. Returns True on success."""
    script_path = os.path.join(SCRIPT_DIR, filename)

    # Check if step can be skipped
    if skip_check and os.path.exists(skip_check):
        print(f"\n{'='*70}")
        print(f"  STEP {step_num:02d}: {description}")
        print(f"  Script: {filename}")
        print(f"  SKIPPED (output already exists)")
        print(f"{'='*70}")
        return True

    print(f"\n{'='*70}")
    print(f"  STEP {step_num:02d}: {description}")
    print(f"  Script: {filename}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_DIR,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  *** FAILED: {filename} (exit code {result.returncode})")
        print(f"  *** Pipeline stopped at step {step_num:02d}.")
        return False

    print(f"\n  Completed in {elapsed:.1f}s")
    return True

def main():
    total_start = time.time()

    print("=" * 70)
    print("  NHANES MORTALITY PREDICTION PIPELINE")
    print(f"  Project: {PROJECT_DIR}")
    print(f"  Steps: {len(STEPS)}")
    print("=" * 70)

    for i, (filename, desc, skip) in enumerate(STEPS, start=1):
        success = run_step(i, filename, desc, skip)
        if not success:
            sys.exit(1)

    total_elapsed = time.time() - total_start
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  All {len(STEPS)} steps finished successfully")
    print(f"  Total time: {minutes}m {seconds}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
