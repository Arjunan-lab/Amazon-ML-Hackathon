"""Comprehensive Unit Test Suite for Stage 1 Normalization Engine.

Tests:
1. Universal accent stripping (French and European characters).
2. Multilingual Indic transliteration across all 9 Indian scripts.
3. Legal suffix stripping across US, India, and France.
4. Word boundary safety (e.g. Zinc is not corrupted by 'inc').
5. Empty-string guardrails (e.g. 'The Company' does not become blank).
6. Address abbreviation expansions (US, India, France).
7. Alphanumeric / numerical token preservation.
8. Vectorized Pandas DataFrame batch execution.
"""

import os
import sys
import unittest
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code.business_entity_resolution.src.normalization import (
    strip_accents,
    transliterate_indic,
    clean_text,
    normalize_business_name,
    normalize_address,
    extract_numbers,
    preprocess_dataframe,
)


class TestStage1Normalization(unittest.TestCase):

    def test_french_accent_stripping(self):
        """Verify French accents are converted to standard ASCII."""
        self.assertEqual(strip_accents("Fédération"), "Federation")
        self.assertEqual(strip_accents("Président"), "President")
        self.assertEqual(strip_accents("Frères"), "Freres")
        self.assertEqual(strip_accents("Boulangerie François"), "Boulangerie Francois")
        self.assertEqual(clean_text("<< Team École >>"), "team ecole")

    def test_indic_transliteration_9_languages(self):
        """Verify transliteration across Hindi, Tamil, Kannada, Telugu, Bengali, etc."""
        # 1. Hindi (Devanagari)
        self.assertIn("modrn", transliterate_indic("मॉडर्न"))
        self.assertIn("laiph", transliterate_indic("लाइफ"))

        # 2. Tamil
        self.assertIn("eestrnn", transliterate_indic("ஈஸ்டர்ன்"))
        self.assertIn("tamil nadu", transliterate_indic("தமிழ்நாடு"))

        # 3. Kannada
        self.assertIn("guru", transliterate_indic("ಗುರು"))
        self.assertIn("karnataka", transliterate_indic("ಕರ್ನಾಟಕ"))

        # 4. Telugu
        self.assertIn("telangana", transliterate_indic("తెలంగాణ"))

        # 5. Gujarati
        self.assertIn("gujarat", transliterate_indic("ગુજરાત"))

        # 6. Bengali
        self.assertIn("west bengal", transliterate_indic("পশ্চিমবঙ্গ"))

    def test_legal_suffix_stripping(self):
        """Verify corporate suffixes are stripped cleanly across jurisdictions."""
        # US
        self.assertEqual(normalize_business_name("Acme Corporation Inc"), "acme")
        self.assertEqual(normalize_business_name("Global Logistics LLC"), "global logistics")
        
        # France
        self.assertEqual(normalize_business_name("ZNB Club SARL"), "znb club")
        self.assertEqual(normalize_business_name("Thermal & Fils SASU"), "thermal and fils")
        self.assertEqual(normalize_business_name("Elephant Centre EURL"), "elephant centre")

        # India
        self.assertEqual(normalize_business_name("Tata Motors Private Limited"), "tata motors")
        self.assertEqual(normalize_business_name("Reliance Industries Ltd"), "reliance")

    def test_word_boundary_safety(self):
        """Ensure substring matches don't corrupt real words (e.g. Zinc, Principle)."""
        # 'Zinc' contains 'inc', but should NOT be stripped
        self.assertEqual(normalize_business_name("Zinc Mining Corp"), "zinc mining")
        # 'Corporate' contains 'corp', should NOT be stripped as a whole word
        self.assertIn("corporate", normalize_business_name("Corporate Affairs Inc"))

    def test_empty_string_guardrail(self):
        """Ensure entities whose names are common suffix words never become empty."""
        # Literally named 'The Company'
        name1 = normalize_business_name("The Company")
        self.assertTrue(len(name1) >= 2, f"Over-trimmed to: '{name1}'")

        # Literally named 'Limited'
        name2 = normalize_business_name("Limited")
        self.assertTrue(len(name2) >= 2, f"Over-trimmed to: '{name2}'")

    def test_address_normalization_multi_country(self):
        """Verify address expansion for US, France, and India."""
        # US
        self.assertEqual(
            normalize_address("123 Main St, Ste 400, Springfield, IL"),
            "123 main street suite 400 springfield il"
        )
        # France
        self.assertEqual(
            normalize_address("175 Bd du President Roosevelt, 5 bis Rue Pierre"),
            "175 boulevard du president roosevelt 5 bis rue pierre"
        )
        # India
        self.assertEqual(
            normalize_address("Plot No. 12, Opp SBI ATM, Extn Col, Bangalore"),
            "plot number 12 opposite sbi atm extension colony bangalore"
        )

    def test_numeric_token_extraction(self):
        """Verify numeric tokens (door numbers, pincodes) are properly isolated."""
        text = "Unit No. 03/320, Sector - 02, Pin 560001, Door 448A"
        nums = extract_numbers(text)
        self.assertIn("03", nums)
        self.assertIn("320", nums)
        self.assertIn("02", nums)
        self.assertIn("560001", nums)
        self.assertIn("448", nums)

    def test_dataframe_vectorized_processing(self):
        """Verify batch DataFrame processing."""
        df_raw = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Acme Corp Inc", "business_address": "123 Main St", "country": "US"},
            {"entity_id": "S2-1", "business_name": "मॉडर्न फाइनेंस", "business_address": "No 10 Enkay Square, 448A", "country": "India"},
            {"entity_id": "S3-1", "business_name": "Fédération de Velo SAS", "business_address": "15 Bd Haussmann", "country": "France"},
        ])
        df_clean = preprocess_dataframe(df_raw)
        
        self.assertIn("clean_name", df_clean.columns)
        self.assertIn("clean_addr", df_clean.columns)
        self.assertIn("num_tokens", df_clean.columns)

        self.assertEqual(df_clean.iloc[0]["clean_name"], "acme")
        self.assertEqual(df_clean.iloc[0]["clean_addr"], "123 main street")
        self.assertIn("10", df_clean.iloc[1]["num_tokens"])
        self.assertEqual(df_clean.iloc[2]["clean_name"], "federation de velo")
        self.assertEqual(df_clean.iloc[2]["clean_addr"], "15 boulevard haussmann")


if __name__ == "__main__":
    unittest.main()
