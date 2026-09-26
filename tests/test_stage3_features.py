"""Comprehensive Unit & Integration Test Suite for Stage 3 Feature Engineering.

Tests:
1. Exact feature dimension integrity (36 features matching FEATURE_NAMES).
2. Defense against Worst Case 1: Multi-Branch franchise trap (Team Ecole Bordeaux vs Lille).
3. Defense against Worst Case 2: Shared office tower trap (different companies at same address).
4. Defense against Worst Case 3: Numerical contradiction trap (Door 101 vs Door 805).
5. Defense against Worst Case 5: Zero-NaN & empty string safety guardrails.
"""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code.business_entity_resolution.src.features import (
    FEATURE_NAMES,
    extract_pair_features,
)


class TestStage3FeatureEngineering(unittest.TestCase):

    def test_feature_vector_dimension_and_names(self):
        """Verify feature vector has exactly 36 features matching FEATURE_NAMES."""
        self.assertEqual(len(FEATURE_NAMES), 36)
        s1 = {"clean_name": "acme tools", "clean_addr": "123 main street springfield", "business_name": "Acme Tools Inc"}
        cand = {"entity_id": "S2-001", "clean_name": "acme tools", "clean_addr": "123 main st", "business_name": "Acme Tools"}
        feat = extract_pair_features(s1, cand)
        self.assertEqual(len(feat), 36)

    def test_worst_case_1_multi_branch_trap(self):
        """Test defense against multi-branch franchises (Same name, different city)."""
        s1 = {
            "clean_name": "team ecole",
            "clean_addr": "175 boulevard franklin roosevelt bordeaux nouvelle aquitaine",
            "business_name": "Team Ecole",
            "num_tokens": {"175"}
        }
        # Candidate branch in Lille (hundreds of miles away)
        cand_branch = {
            "entity_id": "S3-541727694",
            "clean_name": "team ecole",
            "clean_addr": "66 bis rue royale lille hauts de france",
            "business_name": "Team Ecole France",
            "num_tokens": {"66"}
        }

        feat = extract_pair_features(s1, cand_branch)
        f_dict = dict(zip(FEATURE_NAMES, feat))

        # Name match should be 100%
        self.assertEqual(f_dict["name_token_set"], 1.0)
        # But address match is low (< 0.50) and word jaccard is 0.0
        self.assertTrue(f_dict["addr_token_set"] < 0.50)
        self.assertEqual(f_dict["addr_token_jaccard"], 0.0)
        # Branch mismatch penalty MUST be substantial (> 0.50)
        self.assertTrue(f_dict["branch_mismatch_penalty"] > 0.50, f"Branch penalty failed: {f_dict['branch_mismatch_penalty']}")
        # Harmonic mean MUST collapse (< 0.60)
        self.assertTrue(f_dict["name_addr_harmonic_mean"] < 0.60, f"Harmonic mean failed: {f_dict['name_addr_harmonic_mean']}")
        # Contradictory number must be triggered (175 vs 66)
        self.assertEqual(f_dict["has_contradictory_number"], 1.0)

    def test_worst_case_2_shared_building_trap(self):
        """Test defense against different companies in the same commercial building."""
        s1 = {
            "clean_name": "alpha software solutions",
            "clean_addr": "building 10 dlf cyber city gurugram haryana 122002",
            "business_name": "Alpha Software Solutions Pvt Ltd",
            "num_tokens": {"10", "122002"}
        }
        # Unrelated company in the exact same tech park building
        cand_neighbor = {
            "entity_id": "S2-9999",
            "clean_name": "zenith dental care",
            "clean_addr": "building 10 dlf cyber city gurugram haryana 122002",
            "business_name": "Zenith Dental Care",
            "num_tokens": {"10", "122002"}
        }

        feat = extract_pair_features(s1, cand_neighbor)
        f_dict = dict(zip(FEATURE_NAMES, feat))

        # Address match is 100%
        self.assertEqual(f_dict["addr_token_set"], 1.0)
        # Name match is very low (< 0.35)
        self.assertTrue(f_dict["name_token_set"] < 0.35)
        # Shared building penalty MUST be high (> 0.65)
        self.assertTrue(f_dict["shared_building_penalty"] > 0.65, f"Building penalty failed: {f_dict['shared_building_penalty']}")
        # Harmonic mean MUST collapse (< 0.50)
        self.assertTrue(f_dict["name_addr_harmonic_mean"] < 0.50, f"Harmonic mean failed: {f_dict['name_addr_harmonic_mean']}")

    def test_worst_case_3_numerical_contradiction(self):
        """Test door number contradiction (Suite 101 vs Suite 805)."""
        s1 = {
            "clean_name": "apex logistics",
            "clean_addr": "101 market street suite 101 dallas texas",
            "business_name": "Apex Logistics",
            "num_tokens": {"101"}
        }
        cand = {
            "entity_id": "S2-8888",
            "clean_name": "apex logistics",
            "clean_addr": "805 market street suite 805 dallas texas",
            "business_name": "Apex Logistics",
            "num_tokens": {"805"}
        }

        feat = extract_pair_features(s1, cand)
        f_dict = dict(zip(FEATURE_NAMES, feat))

        # Both have numbers, but 101 != 805
        self.assertEqual(f_dict["has_contradictory_number"], 1.0)
        self.assertEqual(f_dict["num_shared_count"], 0.0)
        self.assertEqual(f_dict["door_number_match"], -1.0)

    def test_worst_case_5_zero_nan_and_empty_strings(self):
        """Verify robustness against empty strings, missing fields, and NaNs."""
        empty_s1 = {"clean_name": "", "clean_addr": "", "business_name": ""}
        empty_cand = {"entity_id": "S2-000", "clean_name": "", "clean_addr": "", "business_name": ""}

        feat = extract_pair_features(empty_s1, empty_cand)
        self.assertEqual(len(feat), 36)

        # Ensure NO NaNs or Infs anywhere in the feature vector
        for val in feat:
            self.assertFalse(np.isnan(val), f"NaN detected: {val}")
            self.assertFalse(np.isinf(val), f"Inf detected: {val}")


if __name__ == "__main__":
    unittest.main()
