"""Inspect dataset statistics and sample records."""
import os
import sys
import io

# Force utf-8 for Windows console
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

files = [
    "data/train/train_source1.tsv",
    "data/train/train_source2.tsv",
    "data/train/train_source3.tsv",
    "data/train/train_ground_truth.tsv",
    "data/test/test_source1.tsv",
    "data/test/test_source2.tsv",
    "data/test/test_source3.tsv",
]

print("=" * 60)
print("DATASET QUICK INSPECTION")
print("=" * 60)

for fpath in files:
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        continue
    
    # Read first 3 rows
    df_sample = pd.read_csv(fpath, sep="\t", nrows=3)
    
    # Fast line count
    with open(fpath, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f) - 1  # minus header
        
    print(f"\n[{fpath}]")
    print(f"Total rows: {line_count:,}")
    print(f"Columns: {list(df_sample.columns)}")
    print(f"Sample row 1:")
    print(df_sample.iloc[0].to_dict())

print("\n" + "=" * 60)
