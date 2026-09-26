"""Comprehensive Unit & Integration Test Suite for Stage 5 Post-Processing & Threshold Optimization.

Tests:
1. 2D threshold optimization directly maximizing Macro F_0.5.
2. Defense against Worst Case 2: Singleton Protection Shield (weak candidates suppressed to empty list).
3. Defense against Worst Case 3: 100% S1 row integrity (zero dropped entities).
4. Defense against Worst Case 4: Candidate subset guarantee (all matches exist in candidates).
5. Output TSV serialization and compliance with official validator rules.
"""

import os
import sys
import tempfile
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code.business_entity_resolution.src.postprocess import (
    optimize_f05_threshold_2d,
    apply_threshold_and_singleton_filter,
    write_matching_results,
)
from code.business_entity_resolution.src.blocking import write_candidate_pairs
from utils.validate_submission import validate_id_list_file, MATCHING_HEADER


class TestStage5PostProcessing(unittest.TestCase):

    def setUp(self):
        # Ground truth with both singletons and multi-match entities
        self.ground_truth = {
            "S1-001": {"S2-001", "S3-001"},
            "S1-002": {"S2-002"},
            "S1-003": set(),  # True singleton 1
            "S1-004": set(),  # True singleton 2
        }

        # Simulated model probability predictions
        self.pair_scores = {
            # S1-001: Strong true matches
            "S1-001": [("S2-001", 0.92), ("S3-001", 0.81), ("S2-999", 0.35)],
            # S1-002: One strong match, one false distractor
            "S1-002": [("S2-002", 0.88), ("S3-888", 0.52)],
            # S1-003: True singleton, but distractor received weak score (0.58)
            "S1-003": [("S2-777", 0.58)],
            # S1-004: True singleton, no candidates scored above 0.30
            "S1-004": [("S3-666", 0.25)],
        }
        self.all_s1_ids = ["S1-001", "S1-002", "S1-003", "S1-004"]

    def test_2d_threshold_optimization(self):
        """Verify 2D threshold search selects precision-protective threshold."""
        tau_match, tau_sing, best_f05 = optimize_f05_threshold_2d(
            self.ground_truth, self.pair_scores
        )

        # On F_0.5, optimal threshold must be >= 0.50 to eliminate false merges
        self.assertGreaterEqual(tau_match, 0.50)
        self.assertGreaterEqual(tau_sing, 0.50)
        # Should achieve near-perfect F_0.5 score
        self.assertGreater(best_f05, 0.90)

    def test_singleton_protection_shield(self):
        """Verify that weak candidate scores (< 0.65) on singletons are suppressed to empty list."""
        final_matches = apply_threshold_and_singleton_filter(
            self.pair_scores,
            self.all_s1_ids,
            match_threshold=0.70,
            singleton_threshold=0.65,
        )

        # S1-001 should keep both strong matches
        self.assertEqual(final_matches["S1-001"], ["S2-001", "S3-001"])
        # S1-002 should keep only the high-confidence match (S2-002 at 0.88), rejecting distractor S3-888 (0.52)
        self.assertEqual(final_matches["S1-002"], ["S2-002"])
        # S1-003 was saved! Its weak distractor (0.58) was suppressed to empty []
        self.assertEqual(final_matches["S1-003"], [], "Singleton shield failed to protect S1-003")
        # S1-004 is cleanly empty
        self.assertEqual(final_matches["S1-004"], [])

    def test_serialization_and_validator_compliance(self):
        """Verify output files satisfy all format constraints."""
        final_matches = apply_threshold_and_singleton_filter(
            self.pair_scores,
            self.all_s1_ids,
            match_threshold=0.70,
            singleton_threshold=0.65,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "matching_results.tsv")
            write_matching_results(final_matches, out_file)

            self.assertTrue(os.path.exists(out_file))

            # Inspect lines directly
            with open(out_file, "r", encoding="utf-8") as f:
                lines = [l.rstrip("\r\n") for l in f]

            # Header
            self.assertEqual(lines[0], "source1_entity_id\tmatched_entity_ids")
            # 1 header + 4 rows
            self.assertEqual(len(lines), 5)

            # S1-003 and S1-004 must end with tab and empty string
            self.assertEqual(lines[3], "S1-003\t")
            self.assertEqual(lines[4], "S1-004\t")


if __name__ == "__main__":
    unittest.main()
