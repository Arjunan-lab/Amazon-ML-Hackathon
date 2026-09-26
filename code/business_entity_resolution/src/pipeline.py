"""End-to-End Pipeline for Business Entity Resolution.

Integrates Stages 1 to 6:
- Stage 1: Vectorized Normalization & Multilingual Transliteration.
- Stage 2: Multi-Pass Hybrid Blocking & Candidate Generation (outputs output/candidate_pairs.tsv).
- Stage 3: Deep Feature Engineering (36 rich features).
- Stage 4: Production GBDT Re-Ranking with Probability Calibration.
- Stage 5: Joint 2D F_0.5 Optimization & Singleton Protection Shield.
- Stage 6: Final Submission Generation (outputs output/matching_results.tsv).
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

from code.business_entity_resolution.src.normalization import preprocess_dataframe
from code.business_entity_resolution.src.blocking import generate_candidate_pairs, write_candidate_pairs
from code.business_entity_resolution.src.features import extract_pair_features, FEATURE_NAMES
from code.business_entity_resolution.src.ranker import (
    create_production_ranker,
    train_ranker_with_calibration,
    predict_pair_scores,
    get_feature_importances,
)
from code.business_entity_resolution.src.postprocess import (
    optimize_f05_threshold_2d,
    apply_threshold_and_singleton_filter,
    write_matching_results,
)
from utils.metrics import compute_macro_f05


def load_ground_truth(file_path: str) -> Dict[str, Set[str]]:
    """Loads ground truth tsv file into dictionary mapping s1_id -> set of matched_ids."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, sep="\t", dtype=str).fillna("")
    gt = {}
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw_matches = str(row["matched_entity_ids"]).strip()
        matches = {x.strip() for x in raw_matches.split(",") if x.strip()} if raw_matches else set()
        gt[s1_id] = matches
    return gt


def run_pipeline(
    train_dir: str,
    test_dir: str,
    output_dir: str,
    top_k: int = 20,
    max_train_samples: int = 150000,
):
    print("=" * 70)
    print("STARTING BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 70)

    # 1. Check directories
    train_s1_path = os.path.join(train_dir, "train_source1.tsv")
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")

    has_train = os.path.exists(train_s1_path)
    has_test = os.path.exists(test_s1_path)

    if not has_train and not has_test:
        print(f"[!] Warning: Neither train nor test data files found at {train_dir} and {test_dir}.")
        return

    best_match_thresh = 0.68
    best_singleton_thresh = 0.65
    model = None

    # -------------------------------------------------------------------------
    # Training Stage
    # -------------------------------------------------------------------------
    if has_train:
        print("\n--> [Phase 1] Loading & Preprocessing Training Data (Stage 1)...")
        t0 = time.time()
        df_train_s1_raw = pd.read_csv(train_s1_path, sep="\t", nrows=max_train_samples)
        df_train_s2_raw = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", nrows=max_train_samples * 2)
        df_train_s3_raw = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", nrows=max_train_samples * 2)
        gt_train = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

        # Preprocess with Stage 1 Engine
        df_train_s1 = preprocess_dataframe(df_train_s1_raw)
        df_train_s2 = preprocess_dataframe(df_train_s2_raw)
        df_train_s3 = preprocess_dataframe(df_train_s3_raw)
        print(f"Loaded and normalized S1 ({len(df_train_s1):,}), S2 ({len(df_train_s2):,}), S3 ({len(df_train_s3):,}) in {time.time()-t0:.2f}s")

        print("\n--> [Phase 2] Generating Candidate Pairs via Multi-Pass Blocking (Stage 2)...")
        t0 = time.time()
        train_candidates = generate_candidate_pairs(df_train_s1, df_train_s2, df_train_s3, top_k_per_source=top_k)

        # Measure blocking recall ceiling
        total_true = sum(len(v) for k, v in gt_train.items() if k in train_candidates)
        found_true = sum(len(gt_train.get(k, set()).intersection(set(train_candidates[k]))) for k in train_candidates)
        recall_pct = (found_true / total_true * 100) if total_true > 0 else 100.0
        print(f"Blocking complete in {time.time()-t0:.2f}s | Recall Ceiling: {recall_pct:.2f}% ({found_true:,} / {total_true:,} matches)")

        print("\n--> [Phase 3] Extracting 36 Deep Interaction Features (Stage 3)...")
        t0 = time.time()
        df_train_cand_all = pd.concat([df_train_s2, df_train_s3], ignore_index=True)
        s1_lookup = df_train_s1.set_index("entity_id")
        cand_lookup = df_train_cand_all.set_index("entity_id")

        X_rows = []
        y_labels = []

        for s1_id, c_list in tqdm(train_candidates.items(), desc="Extracting Train Features"):
            if s1_id not in s1_lookup.index:
                continue
            s1_row = s1_lookup.loc[s1_id]
            true_matches = gt_train.get(s1_id, set())

            for rank, cid in enumerate(c_list, start=1):
                if cid not in cand_lookup.index:
                    continue
                cand_row = cand_lookup.loc[cid]
                feat = extract_pair_features(s1_row, cand_row, blocking_rank=rank)
                label = 1 if cid in true_matches else 0

                X_rows.append(feat)
                y_labels.append(label)

        X_train = np.array(X_rows, dtype=np.float32)
        y_train = np.array(y_labels, dtype=np.int32)
        print(f"Constructed feature matrix: {X_train.shape} with {y_train.sum():,} positive pairs in {time.time()-t0:.2f}s")

        print("\n--> [Phase 4] Training & Calibrating LightGBM Re-Ranker (Stage 4)...")
        t0 = time.time()
        model = create_production_ranker()
        model, train_metrics = train_ranker_with_calibration(X_train, y_train, model=model)
        print(f"Training completed in {time.time()-t0:.2f}s | Train ROC-AUC: {train_metrics.get('train_roc_auc', 0.0):.4f}")

        # Show top 5 features
        top_features = get_feature_importances(model, FEATURE_NAMES)[:5]
        print("Top 5 Driving Features:")
        for feat_name, imp in top_features:
            print(f"  - {feat_name}: {imp*100:.2f}%")

        print("\n--> [Phase 5] 2D Macro F_0.5 Threshold & Singleton Optimization (Stage 5)...")
        t0 = time.time()
        train_probs = predict_pair_scores(model, X_train)
        idx = 0
        train_scores_per_s1 = {}
        for s1_id, c_list in train_candidates.items():
            s1_cand_scores = []
            for cid in c_list:
                if idx < len(train_probs):
                    s1_cand_scores.append((cid, float(train_probs[idx])))
                    idx += 1
            train_scores_per_s1[s1_id] = s1_cand_scores

        best_match_thresh, best_singleton_thresh, best_f05 = optimize_f05_threshold_2d(
            gt_train, train_scores_per_s1
        )
        print(f"Optimal Thresholds Found in {time.time()-t0:.2f}s:")
        print(f"  - Match Threshold (tau_match):       {best_match_thresh:.4f}")
        print(f"  - Singleton Cutoff (tau_singleton):   {best_singleton_thresh:.4f}")
        print(f"  - Achieved Macro F_0.5 Score:         {best_f05:.4f}")

    # -------------------------------------------------------------------------
    # Test Inference Stage
    # -------------------------------------------------------------------------
    if has_test:
        print("\n--> [Phase 6] Running Inference on Test Dataset (Stage 6)...")
        t0 = time.time()
        df_test_s1_raw = pd.read_csv(test_s1_path, sep="\t")
        df_test_s2_raw = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t")
        df_test_s3_raw = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t")

        df_test_s1 = preprocess_dataframe(df_test_s1_raw)
        df_test_s2 = preprocess_dataframe(df_test_s2_raw)
        df_test_s3 = preprocess_dataframe(df_test_s3_raw)
        print(f"Loaded & normalized test records: S1 ({len(df_test_s1):,}), S2 ({len(df_test_s2):,}), S3 ({len(df_test_s3):,})")

        print("--> Generating Test Candidate Pairs...")
        test_candidates = generate_candidate_pairs(df_test_s1, df_test_s2, df_test_s3, top_k_per_source=top_k)

        cand_output_path = os.path.join(output_dir, "candidate_pairs.tsv")
        write_candidate_pairs(test_candidates, cand_output_path)
        print(f"Saved: {cand_output_path}")

        print("--> Extracting Features & Scoring Test Pairs in Memory-Safe Batches...")
        df_test_cand_all = pd.concat([df_test_s2, df_test_s3], ignore_index=True)
        s1_test_lookup = df_test_s1.set_index("entity_id")
        cand_test_lookup = df_test_cand_all.set_index("entity_id")

        matching_output_path = os.path.join(output_dir, "matching_results.tsv")
        all_test_s1_ids = list(test_candidates.keys())
        batch_size = 5000

        with open(matching_output_path, "w", encoding="utf-8") as f_out:
            f_out.write("source1_entity_id\tmatched_entity_ids\n")

            for i in tqdm(range(0, len(all_test_s1_ids), batch_size), desc="Scoring Test Batches"):
                batch_s1_ids = all_test_s1_ids[i : i + batch_size]
                batch_rows = []
                batch_pairs = []

                for s1_id in batch_s1_ids:
                    if s1_id not in s1_test_lookup.index:
                        continue
                    s1_row = s1_test_lookup.loc[s1_id]
                    c_list = test_candidates.get(s1_id, [])

                    for rank, cid in enumerate(c_list, start=1):
                        if cid not in cand_test_lookup.index:
                            continue
                        cand_row = cand_test_lookup.loc[cid]
                        feat = extract_pair_features(s1_row, cand_row, blocking_rank=rank)
                        batch_rows.append(feat)
                        batch_pairs.append((s1_id, cid))

                batch_scores_per_s1 = {s1_id: [] for s1_id in batch_s1_ids}
                if model is not None and batch_rows:
                    X_batch = np.array(batch_rows, dtype=np.float32)
                    batch_probs = predict_pair_scores(model, X_batch)
                    for (s1_id, cid), prob in zip(batch_pairs, batch_probs):
                        batch_scores_per_s1[s1_id].append((cid, float(prob)))

                batch_matches = apply_threshold_and_singleton_filter(
                    batch_scores_per_s1,
                    all_s1_ids=batch_s1_ids,
                    match_threshold=best_match_thresh,
                    singleton_threshold=best_singleton_thresh,
                )

                for s1_id in batch_s1_ids:
                    matches = batch_matches.get(s1_id, [])
                    f_out.write(f"{s1_id}\t{','.join(matches)}\n")

        print(f"Saved: {matching_output_path} in {time.time()-t0:.2f}s")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--train-dir", default="data/train", help="Directory with train source and ground truth files")
    parser.add_argument("--test-dir", default="data/test", help="Directory with test source files")
    parser.add_argument("--output-dir", default="output", help="Directory to save submission files")
    parser.add_argument("--top-k", type=int, default=20, help="Number of candidates to generate per source")
    parser.add_argument("--max-train-samples", type=int, default=80000, help="Maximum S1 training samples")
    args = parser.parse_args()

    run_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        max_train_samples=args.max_train_samples,
    )

