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


def load_country_dataframe(file_path: str, country: str) -> pd.DataFrame:
    """Memory-efficiently loads only rows matching the specified country in chunks."""
    chunks = []
    for chunk in pd.read_csv(file_path, sep="\t", chunksize=250000, dtype=str):
        if "country" in chunk.columns:
            filtered = chunk[chunk["country"].astype(str).str.strip() == country]
            if not filtered.empty:
                chunks.append(filtered)
    if not chunks:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    return pd.concat(chunks, ignore_index=True)


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

    os.makedirs(output_dir, exist_ok=True)

    # 1. Check directories
    train_s1_path = os.path.join(train_dir, "train_source1.tsv")
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")

    has_train = os.path.exists(train_s1_path)
    has_test = os.path.exists(test_s1_path)

    if not has_train and not has_test:
        print(f"[!] Warning: Neither train nor test data files found at {train_dir} and {test_dir}.")
        return

    best_match_thresh = 0.86
    best_singleton_thresh = 0.90
    model = None

    model_save_path = os.path.join(output_dir, "ranker_model.joblib")
    thresh_save_path = os.path.join(output_dir, "thresholds.json")

    if os.path.exists(model_save_path) and os.path.exists(thresh_save_path):
        import joblib, json
        print(f"\n--> Found cached trained model at {model_save_path}. Loading...")
        model = joblib.load(model_save_path)
        with open(thresh_save_path, "r") as f_th:
            th_data = json.load(f_th)
            best_match_thresh = th_data["match_thresh"]
            best_singleton_thresh = th_data["singleton_thresh"]
        print(f"Loaded trained model! Thresholds: match={best_match_thresh:.4f}, singleton={best_singleton_thresh:.4f}")
        has_train = False

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
        s1_dict = df_train_s1.set_index("entity_id").to_dict("index")
        cand_dict = df_train_cand_all.set_index("entity_id").to_dict("index")
        del df_train_cand_all
        import gc
        gc.collect()

        X_rows = []
        y_labels = []

        for s1_id, c_list in tqdm(train_candidates.items(), desc="Extracting Train Features"):
            s1_row = s1_dict.get(s1_id)
            if s1_row is None:
                continue
            true_matches = gt_train.get(s1_id, set())

            for rank, cid in enumerate(c_list, start=1):
                cand_row = cand_dict.get(cid)
                if cand_row is None:
                    continue
                feat = extract_pair_features(s1_row, cand_row, blocking_rank=rank)
                label = 1 if cid in true_matches else 0

                X_rows.append(feat)
                y_labels.append(label)

        del s1_dict, cand_dict
        gc.collect()

        X_train = np.array(X_rows, dtype=np.float32)
        y_train = np.array(y_labels, dtype=np.int32)
        del X_rows, y_labels
        gc.collect()
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

        # Save model and thresholds for fast reuse
        try:
            import joblib, json
            joblib.dump(model, model_save_path)
            with open(thresh_save_path, "w") as f_th:
                json.dump({"match_thresh": float(best_match_thresh), "singleton_thresh": float(best_singleton_thresh)}, f_th)
            print(f"Saved trained model and thresholds to {output_dir}/")
        except Exception as e:
            print(f"Warning: Failed to save model: {e}")

        # Free all training memory before test phase
        del df_train_s1, df_train_s2, df_train_s3, gt_train, train_candidates, train_scores_per_s1, X_train, y_train, train_probs
        gc.collect()

    # -------------------------------------------------------------------------
    # Test Inference Stage (Country-Streamed for Bounded < 3.5 GB Memory)
    # -------------------------------------------------------------------------
    if has_test:
        print("\n--> [Phase 6] Running Inference on Test Dataset (Stage 6)...")
        t0 = time.time()
        import gc

        # Discover countries present in test_source1
        print("--> Discovering test countries...")
        test_s1_full = pd.read_csv(test_s1_path, sep="\t", usecols=["country"])
        countries_present = sorted([c for c in test_s1_full["country"].dropna().unique() if str(c).strip()])
        del test_s1_full
        gc.collect()
        print(f"Discovered {len(countries_present)} test partitions: {countries_present}")

        cand_output_path = os.path.join(output_dir, "candidate_pairs.tsv")
        matching_output_path = os.path.join(output_dir, "matching_results.tsv")

        # Initialize output files with headers
        with open(cand_output_path, "w", encoding="utf-8") as f_c:
            f_c.write("source1_entity_id\tcandidate_entity_ids\n")

        with open(matching_output_path, "w", encoding="utf-8") as f_m:
            f_m.write("source1_entity_id\tmatched_entity_ids\n")

        s2_test_path = os.path.join(test_dir, "test_source2.tsv")
        s3_test_path = os.path.join(test_dir, "test_source3.tsv")

        for country in countries_present:
            print(f"\n---> [Partition: {country}] Loading & Normalizing...")
            t_part = time.time()

            df_s1_c = load_country_dataframe(test_s1_path, country)
            df_s2_c = load_country_dataframe(s2_test_path, country)
            df_s3_c = load_country_dataframe(s3_test_path, country)
            print(f"  [{country}] Raw rows: S1={len(df_s1_c):,}, S2={len(df_s2_c):,}, S3={len(df_s3_c):,}")

            # Preprocess only this country partition
            df_s1_c = preprocess_dataframe(df_s1_c)
            df_s2_c = preprocess_dataframe(df_s2_c)
            df_s3_c = preprocess_dataframe(df_s3_c)

            print(f"  [{country}] Generating candidates...")
            candidates_c = generate_candidate_pairs(df_s1_c, df_s2_c, df_s3_c, top_k_per_source=top_k)

            # Append candidates to candidate_pairs.tsv
            with open(cand_output_path, "a", encoding="utf-8") as f_c:
                for s1_id, c_list in candidates_c.items():
                    f_c.write(f"{s1_id}\t{','.join(c_list)}\n")

            # Build hash tables for scoring this country only
            s1_dict_c = df_s1_c.set_index("entity_id").to_dict("index")
            del df_s1_c
            gc.collect()

            cand_dict_c = df_s2_c.set_index("entity_id").to_dict("index")
            del df_s2_c
            gc.collect()

            cand_dict_c.update(df_s3_c.set_index("entity_id").to_dict("index"))
            del df_s3_c
            gc.collect()

            print(f"  [{country}] Scoring candidates in memory-safe streaming batches...")
            c_s1_ids = list(candidates_c.keys())
            batch_size = 5000

            with open(matching_output_path, "a", encoding="utf-8") as f_m:
                for i in tqdm(range(0, len(c_s1_ids), batch_size), desc=f"Scoring [{country}] Batches"):
                    batch_s1_ids = c_s1_ids[i : i + batch_size]
                    batch_rows = []
                    batch_pairs = []

                    for s1_id in batch_s1_ids:
                        s1_row = s1_dict_c.get(s1_id)
                        if s1_row is None:
                            continue
                        c_list = candidates_c.get(s1_id, [])

                        for rank, cid in enumerate(c_list, start=1):
                            cand_row = cand_dict_c.get(cid)
                            if cand_row is None:
                                continue
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
                        f_m.write(f"{s1_id}\t{','.join(matches)}\n")

            # Completely free partition memory
            del candidates_c, s1_dict_c, cand_dict_c, c_s1_ids
            gc.collect()
            print(f"  [{country}] Partition completed in {time.time()-t_part:.2f}s and memory reclaimed!")

        print(f"\n--> Test inference complete in {time.time()-t0:.2f}s!")
        print(f"Saved: {cand_output_path}")
        print(f"Saved: {matching_output_path}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--train-dir", default="data/train", help="Directory with train source and ground truth files")
    parser.add_argument("--test-dir", default="data/test", help="Directory with test source files")
    parser.add_argument("--output-dir", default="output", help="Directory to save submission files")
    parser.add_argument("--top-k", type=int, default=15, help="Number of candidates to generate per source")
    parser.add_argument("--max-train-samples", type=int, default=40000, help="Maximum S1 training samples")
    args = parser.parse_args()

    run_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        max_train_samples=args.max_train_samples,
    )



