"""
01_download_raw_files.py
========================
Downloads all raw NHANES source files (.xpt and .dat) from the
CDC website and saves them locally, organized by cycle folder.
Does NOT combine or merge anything.

Output: 01 raw data\{cycle}\{file}
"""
import requests
import os
import sys

OUTPUT_DIR = r"C:\Users\nieme\OneDrive\Desktop\PA\Mortality Presentation\01 raw data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NHANESDownloader/1.0)"}
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
MORT_BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality"

CYCLES = [
    (1999, 2000, "",   1999),
    (2001, 2002, "_B", 2001),
    (2003, 2004, "_C", 2003),
    (2005, 2006, "_D", 2005),
    (2007, 2008, "_E", 2007),
    (2009, 2010, "_F", 2009),
    (2011, 2012, "_G", 2011),
    (2013, 2014, "_H", 2013),
    (2015, 2016, "_I", 2015),
    (2017, 2018, "_J", 2017),
]

def get_files_for_cycle(suffix, url_year):
    url_base = f"{BASE}/{url_year}/DataFiles"
    files = {
        f"DEMO{suffix}":  f"{url_base}/DEMO{suffix}.xpt",
        f"BMX{suffix}":   f"{url_base}/BMX{suffix}.xpt",
        f"BPX{suffix}":   f"{url_base}/BPX{suffix}.xpt",
        f"SMQ{suffix}":   f"{url_base}/SMQ{suffix}.xpt",
        f"ALQ{suffix}":   f"{url_base}/ALQ{suffix}.xpt",
        f"MCQ{suffix}":   f"{url_base}/MCQ{suffix}.xpt",
        f"DIQ{suffix}":   f"{url_base}/DIQ{suffix}.xpt",
    }
    if suffix in ("", "_B", "_C"):
        chol = "LAB13" if suffix == "" else f"L13{suffix}"
        ghb  = "LAB10" if suffix == "" else f"L10{suffix}"
    else:
        chol = f"TCHOL{suffix}"
        ghb  = f"GHB{suffix}"
    files[chol] = f"{url_base}/{chol}.xpt"
    files[ghb]  = f"{url_base}/{ghb}.xpt"
    return files

def download_file(url, dest_path):
    try:
        r = requests.get(url, timeout=120, headers=HEADERS)
        ct = r.headers.get("content-type", "")
        if "html" in ct.lower() or r.status_code != 200:
            return False, 0
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return True, len(r.content)
    except Exception as e:
        return False, 0

print("=" * 60)
print("  01 - Downloading Raw NHANES Files (1999-2018)")
print(f"  Saving to: {OUTPUT_DIR}")
print("=" * 60)

total_files = 0
total_bytes = 0

for start, end, suffix, url_year in CYCLES:
    label = f"{start}-{end}"
    cycle_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(cycle_dir, exist_ok=True)
    print(f"\n--- {label} ---")

    file_map = get_files_for_cycle(suffix, url_year)
    for name, url in file_map.items():
        dest = os.path.join(cycle_dir, f"{name}.xpt")
        print(f"  {name}.xpt ...", end=" ", flush=True)
        ok, size = download_file(url, dest)
        if ok:
            print(f"OK ({size/1024:.0f} KB)")
            total_files += 1
            total_bytes += size
        else:
            print("FAILED")

    mort_name = f"NHANES_{start}_{end}_MORT_2019_PUBLIC.dat"
    mort_url = f"{MORT_BASE}/{mort_name}"
    dest = os.path.join(cycle_dir, mort_name)
    print(f"  {mort_name} ...", end=" ", flush=True)
    ok, size = download_file(mort_url, dest)
    if ok:
        print(f"OK ({size/1024:.0f} KB)")
        total_files += 1
        total_bytes += size
    else:
        print("FAILED")

print("\n" + "=" * 60)
print(f"  DONE")
print(f"  Files downloaded: {total_files}")
print(f"  Total size: {total_bytes/1e6:.1f} MB")
print(f"  Location: {OUTPUT_DIR}")
print("=" * 60)
