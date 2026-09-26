"""Benchmark Stage 3 feature extraction on real competition candidate pairs."""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from code.business_entity_resolution.src.normalization import preprocess_dataframe
from code.business_entity_resolution.src.blocking import generate_candidate_pairs
from code.business_entity_resolution.src.features import extract_pair_features, FEATURE_NAMES

print("=" * 70)
print("BENCHMARKING STAGE 3 FEATURE EXTRACTION ON REAL DATASET")
print("=" * 70)

# Load real slice
print("1. Loading real data...")
df_s1_raw = pd.read_csv("data/train/train_source1.tsv", sep="\t", nrows=1000)
df_s2_raw = pd.read_csv("data/train/train_source2.tsv", sep="\t", nrows=5000)
df_s3_raw = pd.read_csv("data/train/train_source3.tsv", sep="\t", nrows=5000)

# Stage 1: Preprocess
print("2. Stage 1: Normalizing...")
df_s1 = preprocess_dataframe(df_s1_raw).set_index("entity_id")
df_s2 = preprocess_dataframe(df_s2_raw)
df_s3 = preprocess_dataframe(df_s3_raw)
df_cands = pd.concat([df_s2, df_s3], ignore_index=True).set_index("entity_id")

# Stage 2: Block
print("3. Stage 2: Blocking...")
candidates = generate_candidate_pairs(df_s1.reset_index(), df_s2, df_s3, top_k_per_source=15)

# Stage 3: Extract Features
print("4. Stage 3: Extracting 36 Features for all generated candidate pairs...")
t0 = time.time()
feature_rows = []
pair_count = 0

for s1_id, c_list in candidates.items():
    if s1_id not in df_s1.index:
        continue
    s1_row = df_s1.loc[s1_id]

    for rank, cid in enumerate(c_list, start=1):
        if cid not in df_cands.index:
            continue
        cand_row = df_cands.loc[cid]
        feat = extract_pair_features(s1_row, cand_row, blocking_rank=rank)
        feature_rows.append(feat)
        pair_count += 1

elapsed = time.time() - t0
X_mat = np.array(feature_rows, dtype=np.float32)

print("\n" + "=" * 70)
print("STAGE 3 BENCHMARK RESULTS")
print("=" * 70)
print(f"Total Candidate Pairs Evaluated:   {pair_count:,}")
print(f"Feature Matrix Shape:             {X_mat.shape} (Expected: {pair_count}, 36)")
print(f"Total Extraction Time:            {elapsed:.3f}s")
print(f"Throughput:                       {pair_count / elapsed:,.1f} pairs/sec")
print(f"Any NaNs in Matrix:               {bool(np.isnan(X_mat).any())}")
print(f"Any Infs in Matrix:               {bool(np.isinf(X_mat).any())}")
print(f"Total Features per Pair:          {len(FEATURE_NAMES)}")
print("=" * 70)
