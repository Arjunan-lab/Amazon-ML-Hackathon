"""Evaluation metrics for Amazon ML Challenge: Business Entity Resolution.

Evaluates predictions using Macro-Averaged F_0.5 Score:
- F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
- Calculated per Source 1 entity, then averaged across all Source 1 entities.
- Singletons:
  - If ground truth is empty:
    - Prediction is empty -> Score = 1.0
    - Prediction is non-empty -> Score = 0.0
  - If ground truth is non-empty:
    - Prediction is empty -> Score = 0.0
    - Otherwise -> Standard F_0.5 based on Precision and Recall.
"""

from typing import Dict, List, Set, Union
import pandas as pd


def compute_entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> float:
    """Computes F_0.5 score for a single Source 1 entity."""
    is_true_empty = len(true_ids) == 0
    is_pred_empty = len(pred_ids) == 0

    # Singleton handling
    if is_true_empty:
        return 1.0 if is_pred_empty else 0.0

    # Non-singleton entity with empty prediction
    if is_pred_empty:
        return 0.0

    tp = len(true_ids.intersection(pred_ids))
    if tp == 0:
        return 0.0

    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)

    denom = (0.25 * precision) + recall
    if denom == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom


def compute_macro_f05(
    ground_truth: Union[pd.DataFrame, Dict[str, Set[str]]],
    predictions: Union[pd.DataFrame, Dict[str, Set[str]]],
) -> Dict[str, float]:
    """Computes Macro-Averaged F_0.5 score across all Source 1 entities.

    Args:
        ground_truth: DataFrame with ['source1_entity_id', 'matched_entity_ids']
                      or dict mapping s1_id -> set of matched_entity_ids
        predictions: DataFrame with ['source1_entity_id', 'matched_entity_ids']
                     or dict mapping s1_id -> set of matched_entity_ids

    Returns:
        Dict containing:
            - 'macro_f05': Macro-averaged F_0.5 score
            - 'total_entities': Number of S1 entities evaluated
            - 'singletons_count': Number of true singletons
            - 'singletons_accuracy': Accuracy on true singletons
            - 'non_singletons_f05': Average F_0.5 on entities with matches
    """
    if isinstance(ground_truth, pd.DataFrame):
        gt_dict = {}
        for _, row in ground_truth.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            raw_matches = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
            matches = {x.strip() for x in raw_matches.split(",") if x.strip()} if raw_matches else set()
            gt_dict[s1_id] = matches
    else:
        gt_dict = ground_truth

    if isinstance(predictions, pd.DataFrame):
        pred_dict = {}
        for _, row in predictions.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            raw_matches = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
            matches = {x.strip() for x in raw_matches.split(",") if x.strip()} if raw_matches else set()
            pred_dict[s1_id] = matches
    else:
        pred_dict = predictions

    scores = []
    singleton_scores = []
    non_singleton_scores = []

    for s1_id, true_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        score = compute_entity_f05(true_set, pred_set)
        scores.append(score)

        if len(true_set) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

    total_entities = len(scores)
    macro_f05 = sum(scores) / total_entities if total_entities > 0 else 0.0
    singleton_acc = sum(singleton_scores) / len(singleton_scores) if singleton_scores else 0.0
    non_singleton_f05 = sum(non_singleton_scores) / len(non_singleton_scores) if non_singleton_scores else 0.0

    return {
        "macro_f05": macro_f05,
        "total_entities": total_entities,
        "singletons_count": len(singleton_scores),
        "singletons_accuracy": singleton_acc,
        "non_singletons_f05": non_singleton_f05,
    }


if __name__ == "__main__":
    # Test with example given in the problem statement PDF:
    # Model predicts: [S2-00047, S2-00193, S3-00812]
    # Ground truth: [S2-00047, S3-00812]
    # Precision = 2/3, Recall = 1.0 -> F_0.5 = 0.714285...
    test_true = {"S2-00047", "S3-00812"}
    test_pred = {"S2-00047", "S2-00193", "S3-00812"}
    score = compute_entity_f05(test_true, test_pred)
    print(f"Sample entity F_0.5 score: {score:.4f} (Expected: ~0.714)")
    assert abs(score - 0.7142857) < 1e-4, f"Mismatch: {score}"
    print("Self-test passed successfully!")
