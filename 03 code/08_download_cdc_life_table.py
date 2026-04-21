"""
08_download_cdc_life_table.py
==============================
Downloads the U.S. national period life tables from NVSS and
produces a clean CSV with qx by single year of age and sex.

Source: United States Life Tables, 2021 (NVSR Vol 72, No 12)
  Table 2: Males
  Table 3: Females

Input:  CDC FTP (xlsx files)
Output: 02 processed data - cdc_life_table.csv
"""
import pandas as pd
import requests
import io
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation"
OUTPUT_PATH = os.path.join(PROJECT_DIR, "02 processed data",
                           "cdc_life_table.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NHANESDownloader/1.0)"}

# NVSR 72-12: United States Life Tables, 2021
TABLES = {
    "Male":   "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR/72-12/Table02.xlsx",
    "Female": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR/72-12/Table03.xlsx",
}

print("=" * 65)
print("  08 - Download CDC Life Table (2021)")
print("=" * 65)

rows = []
for sex, url in TABLES.items():
    print(f"\n  Downloading {sex} table...")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    # Parse Excel: CDC life tables have header rows to skip
    # Columns: Age, qx, lx, dx, Lx, Tx, ex
    raw = pd.read_excel(io.BytesIO(r.content), header=None)

    # Parse rows: age column has ranges like "0–1", "1–2", etc.
    # Extract starting age from each range
    data_rows = []
    for i, row in raw.iterrows():
        val = str(row.iloc[0]).strip()
        # Extract first number from range (e.g. "0–1" -> 0, "65–66" -> 65)
        import re
        match = re.match(r"^(\d+)", val)
        if match:
            try:
                age = int(match.group(1))
                if 0 <= age <= 120:
                    qx = float(row.iloc[1])
                    lx = float(row.iloc[2])
                    dx = float(row.iloc[3])
                    ex = float(row.iloc[6])
                    data_rows.append({
                        "age": age, "sex": sex,
                        "qx": qx, "lx": lx, "dx": dx, "ex": ex,
                    })
            except (ValueError, TypeError):
                continue

    rows.extend(data_rows)
    print(f"    Parsed {len(data_rows)} age rows (ages 0-{data_rows[-1]['age']})")

# Combine and save
life_table = pd.DataFrame(rows)
life_table = life_table.sort_values(["sex", "age"]).reset_index(drop=True)
life_table.to_csv(OUTPUT_PATH, index=False)

print(f"\n  Saved: {OUTPUT_PATH}")
print(f"  Rows: {len(life_table)}")

# Preview
print("\nPreview (selected ages):")
prev = life_table[life_table["age"].isin([0, 20, 40, 60, 70, 80, 90, 100])]
print(prev.to_string(index=False))
print("\nDone!")
