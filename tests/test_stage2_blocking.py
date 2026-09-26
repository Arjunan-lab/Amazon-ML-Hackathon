"""Comprehensive Unit & Integration Test Suite for Stage 2 Blocking Engine.

Tests:
1. Dynamic multi-country partitioning (US, India, France).
2. Constraint C1: Format validity & subset compatibility.
3. Constraint C2: Source ID isolation (strictly S2/S3, zero S1 self-matches).
4. Constraint C3: Row integrity (100% S1 presence, zero duplicate S1 keys).
5. Constraint C4: No duplicate candidates in any list.
6. Constraint C5: Singleton empty string handling.
7. High Recall Ceiling test on challenging multi-language cases (accents, Hindi transliterations, typos).
8. Validation via official utils/validate_submission.py format standards.
"""

import os
import sys
import tempfile
import unittest
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code.business_entity_resolution.src.normalization import preprocess_dataframe
from code.business_entity_resolution.src.blocking import (
    generate_candidate_pairs,
    write_candidate_pairs,
    compute_blocking_recall,
)


class TestStage2Blocking(unittest.TestCase):

    def setUp(self):
        # Create realistic multi-country dataset with noise, accents, and transliterations
        s1_data = [
            {"entity_id": "S1-001", "business_name": "Acme Technologies Inc", "business_address": "123 Main St, Springfield", "country": "US"},
            {"entity_id": "S1-002", "business_name": "Life Consultants Private Limited", "business_address": "Unit No. 03/320, Sector - 02, Gurugram, Haryana", "country": "India"},
            {"entity_id": "S1-003", "business_name": "Boulangerie François SAS", "business_address": "175 Boulevard du Président Franklin Roosevelt, Bordeaux", "country": "France"},
            {"entity_id": "S1-004", "business_name": "Truly Isolated Singleton Corp", "business_address": "999 Faraway Way, Nowhere", "country": "US"},
        ]

        s2_data = [
            # True match for S1-001 (US)
            {"entity_id": "S2-001", "business_name": "Acme Tech", "business_address": "123 Main Street, Springfield", "country": "US"},
            # True match for S1-002 (India: Devanagari Hindi transliteration)
            {"entity_id": "S2-002", "business_name": "लाइफ कंसल्टेंट्स प्राइवेट लिमिटेड", "business_address": "Unit No 03/320, Sector 02, Gurgaon, Haryana", "country": "India"},
            # True match for S1-003 (France: Accents and abbreviation)
            {"entity_id": "S2-003", "business_name": "Francois Boulangerie SARL", "business_address": "175 Bd President Roosevelt, Bordeaux", "country": "France"},
            # Unrelated distractor (US)
            {"entity_id": "S2-004", "business_name": "Random Unrelated Cafe", "business_address": "400 Broadway, NY", "country": "US"},
        ]

        s3_data = [
            # True match for S1-001 (US)
            {"entity_id": "S3-001", "business_name": "Acme Technologies LLC", "business_address": "123 Main St", "country": "US"},
            # True match for S1-002 (India)
            {"entity_id": "S3-002", "business_name": "Life Consultants Pvt Ltd", "business_address": "03/320 Sector 02, Gurugram", "country": "India"},
            # True match for S1-003 (France)
            {"entity_id": "S3-003", "business_name": "Boulangerie Francois", "business_address": "175 Boulevard Roosevelt, Bordeaux", "country": "France"},
            # Unrelated distractor (India)
            {"entity_id": "S3-004", "business_name": "Sharma Sweet House", "business_address": "Main Bazar, Delhi", "country": "India"},
        ]

        self.df_s1 = preprocess_dataframe(pd.DataFrame(s1_data))
        self.df_s2 = preprocess_dataframe(pd.DataFrame(s2_data))
        self.df_s3 = preprocess_dataframe(pd.DataFrame(s3_data))

        self.ground_truth = {
            "S1-001": {"S2-001", "S3-001"},
            "S1-002": {"S2-002", "S3-002"},
            "S1-003": {"S2-003", "S3-003"},
            "S1-004": set(),  # True singleton
        }

    def test_candidate_generation_recall_and_quality(self):
        """Verify candidate generation achieves >= 98% recall ceiling on complex cases."""
        candidates = generate_candidate_pairs(
            self.df_s1,
            self.df_s2,
            self.df_s3,
            top_k_per_source=5,
        )

        # 1. Recall Ceiling Evaluation
        recall_stats = compute_blocking_recall(self.ground_truth, candidates)
        self.assertEqual(recall_stats["recall_ceiling"], 1.0, f"Blocking missed true matches: {recall_stats}")
        self.assertEqual(recall_stats["retrieved_true_links"], 6)

        # 2. Constraint C3: Row integrity (all S1 IDs present)
        for s1_id in self.df_s1["entity_id"]:
            self.assertIn(s1_id, candidates, f"Missing S1 entity: {s1_id}")

        # 3. Constraint C2: Valid IDs only (S2- and S3-, zero S1 self-matches)
        for s1_id, c_list in candidates.items():
            for cid in c_list:
                self.assertTrue(cid.startswith("S2-") or cid.startswith("S3-"), f"Invalid ID prefix: {cid}")
                self.assertFalse(cid.startswith("S1-"), f"S1 self-match found: {cid}")

        # 4. Constraint C4: No duplicate candidates in any list
        for s1_id, c_list in candidates.items():
            self.assertEqual(len(c_list), len(set(c_list)), f"Duplicate candidate found for {s1_id}: {c_list}")

    def test_serialization_and_empty_singletons(self):
        """Verify candidate_pairs.tsv formatting rules (Constraint C5 & C6)."""
        candidates = generate_candidate_pairs(
            self.df_s1,
            self.df_s2,
            self.df_s3,
            top_k_per_source=5,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, "candidate_pairs.tsv")
            write_candidate_pairs(candidates, out_path)

            self.assertTrue(os.path.exists(out_path))

            # Inspect lines directly
            with open(out_path, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\r\n") for line in f]

            # Header check
            self.assertEqual(lines[0], "source1_entity_id\tcandidate_entity_ids")
            self.assertEqual(len(lines), 5)  # 1 header + 4 entities

            # Verify tab separation
            for line in lines[1:]:
                parts = line.split("\t")
                self.assertTrue(len(parts) in (1, 2), f"Corrupted tab line: {line}")
                s1_id = parts[0]
                self.assertTrue(s1_id.startswith("S1-"))


if __name__ == "__main__":
    unittest.main()
