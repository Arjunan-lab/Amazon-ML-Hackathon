"""Industry-Standard Post-Processing & F_0.5 Threshold Optimization Engine (Stage 5).

Implements:
1. Joint 2D Threshold Optimization (Candidate Match Cutoff + Singleton Gatekeeper).
2. Direct Macro-Averaged F_0.5 metric maximization on out-of-fold validation sets.
3. Singleton Protection Shield: guarantees 1.0 credit by suppressing weak candidate guesses.
4. Strict Submission Format Serializer:
   - Exactly one row per S1 entity.
   - Clean empty tab string for singletons (no 'None', 'null', or NaNs).
   - Strict subset guarantee: matching_results.tsv is a subset of candidate_pairs.tsv.
"""

import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from utils.metrics import compute_macro_f05


def optimize_f05_threshold_2d(
    ground_truth: Dict[str, Set[str]],
    pair_scores_per_s1: Dict[str, List[Tuple[str, float]]],
    min_match_thresh: float = 0.50,
    max_match_thresh: float = 0.86,
    step: float = 0.02,
) -> Tuple[float, float, float]:
    """Finds the optimal match threshold and singleton cutoff maximizing Macro F_0.5.

    Args:
        ground_truth: Mapping s1_id -> set of true matched entity IDs.
        pair_scores_per_s1: Mapping s1_id -> list of (candidate_id, prob_score).
        min_match_thresh: Lower bound for match threshold search.
        max_match_thresh: Upper bound for match threshold search.
        step: Grid search step resolution.

    Returns:
        (best_match_thresh, best_singleton_thresh, best_macro_f05)
    """
    best_match_thresh = 0.65
    best_singleton_thresh = 0.65
    best_f05 = -1.0

    threshold_candidates = np.arange(min_match_thresh, max_match_thresh + 1e-5, step)

    # Pre-extract max score per entity for ultra-fast singleton filtering
    max_score_per_s1 = {
        s1_id: max((score for _, score in c_list), default=0.0)
        for s1_id, c_list in pair_scores_per_s1.items()
    }

    for tau_match in threshold_candidates:
        # Singleton threshold is searched around tau_match
        singleton_candidates = [tau_match - 0.04, tau_match, tau_match + 0.04]

        for tau_sing in singleton_candidates:
            preds: Dict[str, Set[str]] = {}

            for s1_id, c_list in pair_scores_per_s1.items():
                # Singleton Gatekeeper: if top candidate confidence is weak, declare singleton
                if max_score_per_s1.get(s1_id, 0.0) < tau_sing:
                    preds[s1_id] = set()
                else:
                    # Filter candidates meeting match threshold
                    qualified = {cid for cid, score in c_list if score >= tau_match}
                    preds[s1_id] = qualified

            eval_res = compute_macro_f05(ground_truth, preds)
            score = eval_res["macro_f05"]

            if score > best_f05:
                best_f05 = score
                best_match_thresh = float(tau_match)
                best_singleton_thresh = float(tau_sing)

    return best_match_thresh, best_singleton_thresh, float(best_f05)


def apply_threshold_and_singleton_filter(
    pair_scores_per_s1: Dict[str, List[Tuple[str, float]]],
    all_s1_ids: List[str],
    match_threshold: float = 0.65,
    singleton_threshold: float = 0.65,
) -> Dict[str, List[str]]:
    """Generates confirmed final matches adhering strictly to competition rules.

    - Every S1 entity in all_s1_ids is guaranteed to exist with an entry.
    - Entities where max(P) < singleton_threshold are suppressed to empty list [].
    - Candidates sorted by probability score descending.
    - No duplicate candidate IDs.
    """
    final_matches: Dict[str, List[str]] = {}

    for s1_id in all_s1_ids:
        c_list = pair_scores_per_s1.get(s1_id, [])

        if not c_list:
            final_matches[s1_id] = []
            continue

        # Find maximum candidate score for this entity
        max_score = max((score for _, score in c_list), default=0.0)

        # Singleton Shield: if top score is below singleton cutoff, output empty
        if max_score < singleton_threshold:
            final_matches[s1_id] = []
            continue

        # Filter by match threshold and sort by confidence descending
        qualified = [cid for cid, score in sorted(c_list, key=lambda x: x[1], reverse=True) if score >= match_threshold]

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for cid in qualified:
            if cid not in seen and not cid.startswith("S1-"):
                seen.add(cid)
                deduped.append(cid)

        final_matches[s1_id] = deduped

    return final_matches


def write_matching_results(
    matches_dict: Dict[str, List[str]],
    output_path: str,
):
    """Writes matching_results.tsv strictly matching competition format rules.

    Rules:
        - Tab-separated columns: source1_entity_id \t matched_entity_ids
        - No quotes
        - Exact one row per S1 entity
        - Empty string for singletons (e.g. 'S1-00003\t\n')
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, m_list in matches_dict.items():
            f.write(f"{s1_id}\t{','.join(m_list)}\n")
