"""Analyze ground truth statistics."""
import sys
import io

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from collections import Counter
from tqdm import tqdm

print("Reading train_ground_truth.tsv in chunks...")
gt_path = "data/train/train_ground_truth.tsv"

total_s1 = 0
singletons = 0
match_counts = []
s2_count = 0
s3_count = 0

chunksize = 200000
for chunk in tqdm(pd.read_csv(gt_path, sep="\t", chunksize=chunksize, dtype=str)):
    for _, row in chunk.iterrows():
        total_s1 += 1
        raw = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
        if not raw:
            singletons += 1
            match_counts.append(0)
        else:
            ids = [x.strip() for x in raw.split(",") if x.strip()]
            match_counts.append(len(ids))
            for mid in ids:
                if mid.startswith("S2-"):
                    s2_count += 1
                elif mid.startswith("S3-"):
                    s3_count += 1

match_series = pd.Series(match_counts)

print("\n" + "=" * 60)
print("GROUND TRUTH ANALYSIS SUMMARY")
print("=" * 60)
print(f"Total S1 entities:       {total_s1:,}")
print(f"Total Singletons (0):    {singletons:,} ({singletons / total_s1 * 100:.2f}%)")
print(f"Entities with matches:   {total_s1 - singletons:,} ({(total_s1 - singletons) / total_s1 * 100:.2f}%)")
print(f"Total matched pairs:     {sum(match_counts):,}")
print(f"  - Matches from S2:     {s2_count:,} ({s2_count / sum(match_counts) * 100:.2f}%)")
print(f"  - Matches from S3:     {s3_count:,} ({s3_count / sum(match_counts) * 100:.2f}%)")
print("\nMatch count quantiles (for entities with matches):")
nonzero = match_series[match_series > 0]
print(nonzero.describe())
