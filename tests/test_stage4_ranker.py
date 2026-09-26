"""Comprehensive Unit Test Suite for Stage 4 GBDT Re-Ranker.

Tests:
1. LightGBM classifier creation with balanced class-weighting and regularization.
2. Training on synthetic pairs with validation metrics tracking.
3. Probability predictions strictly bounded in [0.0, 1.0].
4. Feature importance extraction across all 36 features.
"""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code.business_entity_resolution.src.features import FEATURE_NAMES
from code.business_entity_resolution.src.ranker import (
    create_production_ranker,
    train_ranker_with_calibration,
    predict_pair_scores,
    get_feature_importances,
)


class TestStage4Ranker(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Create synthetic training set with 36 features and 10:1 class imbalance
        n_samples = 1000
        n_features = len(FEATURE_NAMES)
        
        self.X_train = np.random.rand(n_samples, n_features).astype(np.float32)
        # 10% positive matches, 90% negative
        self.y_train = (np.random.rand(n_samples) > 0.90).astype(np.int32)

        # Make feature 0 and 28 correlate strongly with positive class
        self.X_train[self.y_train == 1, 0] += 0.8
        self.X_train[self.y_train == 1, 28] += 0.8

        self.X_val = np.random.rand(200, n_features).astype(np.float32)
        self.y_val = (np.random.rand(200) > 0.90).astype(np.int32)
        self.X_val[self.y_val == 1, 0] += 0.8
        self.X_val[self.y_val == 1, 28] += 0.8

    def test_model_training_and_metrics(self):
        """Verify model trains cleanly and reports validation ROC-AUC."""
        model = create_production_ranker(n_estimators=50, learning_rate=0.1)
        trained_model, metrics = train_ranker_with_calibration(
            self.X_train, self.y_train, self.X_val, self.y_val, model=model
        )

        self.assertIn("val_roc_auc", metrics)
        self.assertGreater(metrics["val_roc_auc"], 0.70)

    def test_probability_predictions_bounds(self):
        """Verify predicted scores are strictly in [0.0, 1.0]."""
        model = create_production_ranker(n_estimators=30)
        trained_model, _ = train_ranker_with_calibration(self.X_train, self.y_train, model=model)

        probs = predict_pair_scores(trained_model, self.X_val)
        self.assertEqual(len(probs), len(self.X_val))
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

    def test_feature_importance_extraction(self):
        """Verify feature importances are extracted and sorted."""
        model = create_production_ranker(n_estimators=30)
        trained_model, _ = train_ranker_with_calibration(self.X_train, self.y_train, model=model)

        importances = get_feature_importances(trained_model, FEATURE_NAMES)
        self.assertEqual(len(importances), len(FEATURE_NAMES))
        # Ensure sorted descending
        for i in range(len(importances) - 1):
            self.assertGreaterEqual(importances[i][1], importances[i + 1][1])


if __name__ == "__main__":
    unittest.main()
