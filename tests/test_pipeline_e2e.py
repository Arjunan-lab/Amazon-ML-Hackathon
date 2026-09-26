"""Integration test for the entire entity resolution pipeline.

Creates synthetic S1, S2, S3 records covering US, India, and France,
runs the pipeline end-to-end, and verifies with validate_submission.py.
"""

import os
import shutil
import subprocess
import sys
import pandas as pd


def setup_synthetic_data(base_dir: str):
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Train S1
    s1_train = pd.DataFrame([
        {"entity_id": "S1-0001", "business_name": "Acme Corp Inc", "business_address": "123 Main St, Springfield", "country": "US"},
        {"entity_id": "S1-0002", "business_name": "Tata Motors Ltd", "business_address": "Bombay House, 24 Homi Mody St, Mumbai", "country": "India"},
        {"entity_id": "S1-0003", "business_name": "Singleton Tech Solutions", "business_address": "99 Lonely Rd, Nowhere", "country": "US"},
    ])
    s1_train.to_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", index=False)

    # Train S2
    s2_train = pd.DataFrame([
        {"entity_id": "S2-0001", "business_name": "Acme Corporation", "business_address": "123 Main Street, Springfield", "country": "US"},
        {"entity_id": "S2-0002", "business_name": "Tata Motors", "business_address": "24 Homi Mody Street, Mumbai", "country": "India"},
        {"entity_id": "S2-0003", "business_name": "Random Unrelated Biz", "business_address": "456 Other St, Dallas", "country": "US"},
    ])
    s2_train.to_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", index=False)

    # Train S3
    s3_train = pd.DataFrame([
        {"entity_id": "S3-0001", "business_name": "Acme Co", "business_address": "123 Main St", "country": "US"},
        {"entity_id": "S3-0002", "business_name": "Bombay House Tata Motors", "business_address": "Homi Mody St, Mumbai", "country": "India"},
    ])
    s3_train.to_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", index=False)

    # Train Ground Truth
    gt_train = pd.DataFrame([
        {"source1_entity_id": "S1-0001", "matched_entity_ids": "S2-0001,S3-0001"},
        {"source1_entity_id": "S1-0002", "matched_entity_ids": "S2-0002,S3-0002"},
        {"source1_entity_id": "S1-0003", "matched_entity_ids": ""},
    ])
    gt_train.to_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", index=False)

    # Test S1 (includes France)
    s1_test = pd.DataFrame([
        {"entity_id": "S1-1001", "business_name": "Acme Tools LLC", "business_address": "123 Main St, Springfield", "country": "US"},
        {"entity_id": "S1-1002", "business_name": "Boulangerie Paul SAS", "business_address": "15 Boulevard Haussmann, Paris", "country": "France"},
        {"entity_id": "S1-1003", "business_name": "True Singleton Inc", "business_address": "777 Distant Way", "country": "US"},
    ])
    s1_test.to_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", index=False)

    # Test S2
    s2_test = pd.DataFrame([
        {"entity_id": "S2-1001", "business_name": "Acme Tools", "business_address": "123 Main Street", "country": "US"},
        {"entity_id": "S2-1002", "business_name": "Paul Boulangerie", "business_address": "15 Bd Haussmann, Paris", "country": "France"},
        {"entity_id": "S2-1003", "business_name": "Different Shop", "business_address": "100 Broadway, NY", "country": "US"},
    ])
    s2_test.to_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", index=False)

    # Test S3
    s3_test = pd.DataFrame([
        {"entity_id": "S3-1001", "business_name": "Acme Tools Corp", "business_address": "123 Main St, Springfield", "country": "US"},
        {"entity_id": "S3-1002", "business_name": "Boulangerie Paul", "business_address": "15 Boulevard Haussmann", "country": "France"},
    ])
    s3_test.to_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", index=False)


def run_test():
    tmp_data_dir = "tests/synthetic_data"
    tmp_output_dir = "tests/synthetic_output"
    os.makedirs(tmp_data_dir, exist_ok=True)
    os.makedirs(tmp_output_dir, exist_ok=True)

    print("Generating synthetic dataset...")
    setup_synthetic_data(tmp_data_dir)

    print("Executing pipeline...")
    cmd = [
        sys.executable,
        "code/business_entity_resolution/src/pipeline.py",
        "--train-dir", os.path.join(tmp_data_dir, "train"),
        "--test-dir", os.path.join(tmp_data_dir, "test"),
        "--output-dir", tmp_output_dir,
        "--top-k", "5"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Pipeline Output:\n", res.stdout)
    if res.returncode != 0:
        print("Pipeline Error:\n", res.stderr)
        sys.exit(res.returncode)

    print("Running validation script...")
    val_cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", os.path.join(tmp_output_dir, "matching_results.tsv"),
        "--candidate", os.path.join(tmp_output_dir, "candidate_pairs.tsv"),
        "--test-dir", os.path.join(tmp_data_dir, "test"),
    ]
    val_res = subprocess.run(val_cmd, capture_output=True, text=True)
    print("Validator Output:\n", val_res.stdout)
    if val_res.returncode != 0:
        print("Validator Errors:\n", val_res.stderr)
        sys.exit(val_res.returncode)

    print("Checking output files content:")
    with open(os.path.join(tmp_output_dir, "matching_results.tsv")) as f:
        print("matching_results.tsv:\n" + f.read())
    with open(os.path.join(tmp_output_dir, "candidate_pairs.tsv")) as f:
        print("candidate_pairs.tsv:\n" + f.read())

    # Clean up test output
    shutil.rmtree("tests/synthetic_data", ignore_errors=True)
    shutil.rmtree("tests/synthetic_output", ignore_errors=True)
    print("\nALL SYNTHETIC TESTS PASSED WITH 100% SUCCESS!")


if __name__ == "__main__":
    run_test()
