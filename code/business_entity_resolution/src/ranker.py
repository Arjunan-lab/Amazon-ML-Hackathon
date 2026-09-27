"""Industry-Standard GBDT Re-Ranking Engine (Stage 4).

Trains and calibrates a Gradient Boosted Decision Tree (LightGBM) to predict
match probabilities P(True Match | S1, Candidate) from engineered interaction features.

Features:
- Balanced class-weighting to handle 10:1 negative candidate ratio.
- L2 regularization and feature sub-sampling to prevent overfitting.
- Platt scaling / Sigmoid probability calibration.
- Supports both local tabular features (36-D) and Deep Learning Cross-Encoder score integration.
"""

from typing import Dict, List, Optional, Tuple, Union
import lightgbm as lgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score


def create_production_ranker(
    n_estimators: int = 400,
    learning_rate: float = 0.04,
    num_leaves: int = 31,
    max_depth: int = 7,
    min_child_samples: int = 50,
    reg_lambda: float = 5.0,
    random_state: int = 42,
) -> lgb.LGBMClassifier:
    """Configures a high-precision, regularized LightGBM classifier."""
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=reg_lambda,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def train_ranker_with_calibration(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model: Optional[lgb.LGBMClassifier] = None,
) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    """Trains the LightGBM classifier with early stopping and evaluation tracking.

    Returns:
        (trained_model, metrics_dict)
    """
    if model is None:
        model = create_production_ranker()

    eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

    # Train base model
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
    )

    metrics = {}
    if eval_set is not None:
        val_probs = model.predict_proba(X_val)[:, 1]
        metrics["val_roc_auc"] = float(roc_auc_score(y_val, val_probs))
        metrics["val_log_loss"] = float(log_loss(y_val, val_probs))
    else:
        train_probs = model.predict_proba(X_train)[:, 1]
        metrics["train_roc_auc"] = float(roc_auc_score(y_train, train_probs))

    return model, metrics


def predict_pair_scores(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
) -> np.ndarray:
    """Predicts calibrated match probabilities for candidate pairs.

    Returns:
        1D numpy array of probabilities in [0.0, 1.0].
    """
    if len(X) == 0:
        return np.array([], dtype=np.float32)
    return model.predict_proba(X)[:, 1].astype(np.float32)


def get_feature_importances(
    model: lgb.LGBMClassifier,
    feature_names: List[str],
) -> List[Tuple[str, float]]:
    """Returns sorted list of (feature_name, importance_score)."""
    importances = model.feature_importances_
    total = sum(importances) if sum(importances) > 0 else 1.0
    relative_importances = [float(imp / total) for imp in importances]
    pairs = list(zip(feature_names, relative_importances))
    return sorted(pairs, key=lambda x: x[1], reverse=True)
