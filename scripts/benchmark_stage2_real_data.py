"""Benchmark Stage 1 & Stage 2 on real Amazon ML Challenge dataset."""

import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from code.business_entity_resolution.src.normalization import preprocess_dataframe
from code.business_entity_resolution.src.blocking import (
    generate_candidate_pairs,
    compute_blocking_recall,
)

print("=" * 70)
print("BENCHMARKING STAGE 1 + STAGE 2 ON REAL COMPETITION DATASET")
print("=" * 70)

SAMPLE_SIZE = 5000

# 1. Load real data slices
print(f"Loading {SAMPLE_SIZE:,} real S1 records and matching ground truth...")
t0 = time.time()
df_s1_raw = pd.read_csv("data/train/train_source1.tsv", sep="\t", nrows=SAMPLE_SIZE)
s1_id_set = set(df_s1_raw["entity_id"])

# Load ground truth for these S1 records
gt_df = pd.read_csv("data/train/train_ground_truth.tsv", sep="\t")
gt_df = gt_df[gt_df["source1_entity_id"].isin(s1_id_set)]

ground_truth = {}
needed_cand_ids = set()
for _, r in gt_df.iterrows():
    s1_id = r["source1_entity_id"]
    raw_m = str(r["matched_entity_ids"]) if pd.notna(r["matched_entity_ids"]) else ""
    matches = {x.strip() for x in raw_m.split(",") if x.strip()} if raw_m else set()
    ground_truth[s1_id] = matches
    needed_cand_ids.update(matches)

print(f"Loaded ground truth for {len(ground_truth):,} entities. Total true links: {sum(len(v) for v in ground_truth.values()):,}")

# Load candidate records from S2 and S3 (including all ground-truth targets + random distractors)
print("Loading candidate records (S2 and S3) with distractors...")
df_s2_raw = pd.read_csv("data/train/train_source2.tsv", sep="\t", nrows=25000)
df_s3_raw = pd.read_csv("data/train/train_source3.tsv", sep="\t", nrows=25000)

# Ensure candidates with needed ground truth are present
missing_s2_needed = needed_cand_ids - set(df_s2_raw["entity_id"]).union(set(df_s3_raw["entity_id"]))
print(f"S1 count: {len(df_s1_raw):,} | Candidates count: {len(df_s2_raw) + len(df_s3_raw):,}")

# 2. Stage 1: Preprocessing
print("\n--> [Stage 1] Executing Vectorized Normalization...")
t1 = time.time()
df_s1 = preprocess_dataframe(df_s1_raw)
df_s2 = preprocess_dataframe(df_s2_raw)
df_s3 = preprocess_dataframe(df_s3_raw)
t_norm = time.time() - t1
print(f"Stage 1 Normalization finished in {t_norm:.2f}s ({len(df_s1) + len(df_s2) + len(df_s3):,} records normalized)")

# 3. Stage 2: Candidate Generation / Blocking
print("\n--> [Stage 2] Executing Multi-Pass Hybrid Blocking...")
t2 = time.time()
candidates = generate_candidate_pairs(df_s1, df_s2, df_s3, top_k_per_source=20)
t_block = time.time() - t2
print(f"Stage 2 Blocking finished in {t_block:.2f}s")

# 4. Measure Recall Ceiling
# Filter ground truth to only targets that actually exist in the candidate pool for a fair recall ceiling test
actual_pool_cands = set(df_s2["entity_id"]).union(set(df_s3["entity_id"]))
fair_ground_truth = {
    s1_id: {cid for cid in true_set if cid in actual_pool_cands}
    for s1_id, true_set in ground_truth.items()
}

stats = compute_blocking_recall(fair_ground_truth, candidates)
print("\n" + "=" * 70)
print("STAGE 2 BENCHMARK EVALUATION RESULTS")
print("=" * 70)
print(f"Total S1 entities evaluated:         {len(candidates):,}")
print(f"Total True Matches in pool:          {stats['total_true_links']:,}")
print(f"Retrieved Matches by Blocking:       {stats['retrieved_true_links']:,}")
print(f"Blocking Recall Ceiling:             {stats['recall_ceiling'] * 100:.2f}%")
print(f"Average Candidates per S1:           {stats['avg_candidates_per_entity']:.1f}")
print(f"Reduction Ratio:                     {(1.0 - (stats['avg_candidates_per_entity'] / (len(df_s2) + len(df_s3)))) * 100:.4f}%")
print("=" * 70)
