# Business Entity Resolution Pipeline

High-accuracy, precision-optimized solution for the Amazon ML Challenge Business Entity Resolution task.

## Methodology Summary

1. **Text Normalization**: Unicode NFKD stripping, universal lowercasing, multi-jurisdiction corporate suffix normalization (US, India, and France), and address token expansion without external network or API lookups.
2. **Multi-Pass Blocking**: Country-partitioned character 3-5 gram TF-IDF inverted index to achieve high candidate recall ceiling while maintaining reduction ratio > 99.5%.
3. **Feature Engineering**: Comprehensive pairwise lexical, numerical, and structural distances via RapidFuzz (Jaro-Winkler, Token Sort, Token Set, Pincode/Unit number overlap).
4. **Machine Learning Model**: Gradient Boosted Decision Tree (LightGBM) trained on pair interactions.
5. **Precision Optimization & Singleton Gating**: Explicit grid search on the validation set for the competition's macro-averaged $F_{0.5}$ metric, combined with a singleton filter to avoid false-positive penalties on non-matching entities.

---

## Reproduction Instructions

### 1. Environment Setup

Ensure Python 3.11 is installed. From the `Amazon_Ml_Challenge` project root:

```bash
poetry install
```

Or using pip in an isolated virtual environment:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Dataset Placement

Ensure training and test files are placed in their respective folders:

```
data/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### 3. Run End-to-End Pipeline

Execute the pipeline via Poetry:

```bash
poetry run python -m code.business_entity_resolution.src.pipeline \
  --train-dir data/train \
  --test-dir data/test \
  --output-dir output \
  --top-k 25
```

This will automatically:
1. Normalize source records.
2. Generate candidate pairs and save `output/candidate_pairs.tsv`.
3. Train the LightGBM re-ranker and find the optimal $F_{0.5}$ threshold.
4. Score test candidates and output `output/matching_results.tsv`.

### 4. Validate Submission

Validate your output files against all competition rules:

```bash
poetry run python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir data/test
```
