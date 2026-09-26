# Business Entity Resolution: High-Accuracy Winning Workflow

Comprehensive, production-grade workflow designed specifically to win the **Amazon ML Challenge: Business Entity Resolution** and maximize the macro-averaged **$F_{0.5}$ score**.

---

## 1. Key Problem Analysis & Strategic Insights

| Challenge Element | Reality & Constraints | Winning Strategy |
| :--- | :--- | :--- |
| **Evaluation Metric** | **Macro $F_{0.5}$** penalizes false positives twice as harshly as false negatives. | Do not optimize for standard $F_1$ or ROC-AUC. Optimize decision thresholds specifically for the $F_{0.5}$ objective. Be conservative on ambiguous pairs. |
| **Singletons** | Entities with 0 matches get **1.0** for an empty prediction and **0.0** for a single false positive. | Build a dedicated singleton classifier / confidence gatekeeper. Avoiding false predictions on true singletons creates a massive leaderboard score boost. |
| **Unseen Country (France)** | Train has `{US, India}`; Test adds `France`. External APIs / geocoding are **strictly prohibited**. | Build an **open-vocabulary, country-agnostic representation** (character $n$-gram embeddings + multilingual transformer representations like `multilingual-e5-base` or `bge-m3`) with zero hard-coded country logic. |
| **Candidate Blocking** | Comparing all $S_1 \times (S_2 \cup S_3)$ is $O(N^2)$ and computationally intractable (~17 trillion pairs). | A multi-pass inverted-index + dense ANN retrieval system yielding $\ge 98\%$ recall ceiling with $\le 20-50$ candidates per $S_1$ entity (satisfying `candidate_pairs.tsv`). |

---

## 2. End-to-End System Pipeline

```
                                  END-TO-END SYSTEM PIPELINE
                                  
  Raw Data (S1, S2, S3) 
           │
           ▼
  [STAGE 1: Text & Address Normalization Engine]
  • Multilingual Unicode NFKD, lowercasing, punctuation stripping
  • Generic legal suffix stripping (Ltd, Inc, LLC, SA, SARL, Pvt)
  • Address prefix/suffix standardization (Rd, St, Ave, Rue, Bd)
           │
           ▼
  [STAGE 2: Multi-Pass Hybrid Blocking & Candidate Generation]
  • Pass A: Country-Partitioned Inverted Index (BM25 / TF-IDF Char 3-4 grams on Name + Address)
  • Pass B: Multilingual Dense Semantic Retrieval (FAISS / HNSW with Sentence-Transformers)
  • Pass C: Phonetic / Metaphone Blocking for transliteration noise
  ──▶ Outputs: candidate_pairs.tsv (Recall ceiling ≥ 98%, ~30-50 candidates per S1)
           │
           ▼
  [STAGE 3: Feature Engineering & Cross-Encoder Scoring]
  • 40+ Rich Lexical & Topological Features (Jaccard, Levenshtein, Jaro-Winkler, Token Sort)
  • Semantic Cosine Similarities (Name, Address, Joint)
  • Token-level set differences (detecting different branch numbers, cities, unit codes)
  • Cross-Encoder Deep Scoring (DeBERTa-v3 / MiniLM entity pair classifier)
           │
           ▼
  [STAGE 4: GBDT Ensemble Re-Ranker (CatBoost + LightGBM)]
  • Predicts probability P(match | S1, Sx)
  • Calibrated probabilities with Platt scaling or Isotonic Regression
           │
           ▼
  [STAGE 5: F_0.5 Optimal Thresholding & Singleton Filter]
  • Dynamic thresholding tuned per candidate set
  • Singleton cutoff: suppress predictions if max(P) < τ_singleton
           │
           ▼
  [STAGE 6: Submission Packaging & Local Validation]
  • Generate matching_results.tsv and candidate_pairs.tsv
  • Run utils/validate_submission.py
```

---

## 3. Step-by-Step Implementation Workflow

### Phase 1: Local Metric Setup & Leakage-Free Validation Strategy

1. **Custom Macro $F_{0.5}$ Evaluator**:
   Implement the exact metric computation matching the competition specification:
   $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   - Singletons: If an entity has 0 true matches and 0 predicted matches, score = $1.0$; if it has $\ge 1$ predicted matches, score = $0.0$.
   - Macro-averaged across all $S_1$ entities.

2. **Entity-Level Stratified Group Split**:
   - Perform an 80/20 train-validation split grouped by $S_1$ entity IDs.
   - Ensure complete entity isolation so that no $S_1$ record or its ground-truth matches appear in both train and validation.
   - Create a synthetic test scenario by holding out a pseudo-unseen subset (simulating the behavior when transitioning to new distributions like France).

---

### Phase 2: Domain-Aware Data Normalization Engine

1. **Name Normalization**:
   - Universal lowercasing and Unicode normalization (`NFKD`).
   - Suffix removal across all candidate languages:
     - **US/UK**: `inc, llc, corp, corporation, co, company, ltd, limited`.
     - **India**: `pvt, private, pvt ltd, enterprises, associates, bros`.
     - **France**: `sa, sarl, sas, eurl, sci, ste, societe`.
   - Character transliteration mapping and symbol normalization (`&` $\rightarrow$ `and`, `@` $\rightarrow$ `at`).

2. **Address Normalization**:
   - Expansion and standardization of abbreviations:
     - `st` / `street`, `rd` / `road`, `ave` / `avenue`, `blvd` / `boulevard`, `fl` / `floor`.
     - French terms: `rue`, `bd`, `avenue`, `allee`, `chemin`.
   - Digit and token extraction: Parse postal codes, door/suite numbers, and municipal ward identifiers into separate matching tokens.

---

### Phase 3: High-Recall Hybrid Blocking (`candidate_pairs.tsv`)

A match can never be found if it is blocked out. We combine lexical and semantic indexing:

1. **Country-Gated Blocking**:
   - Cross-country matches are virtually non-existent; filter candidates where `source.country == candidate.country` (while treating `country` as an open string label to handle France seamlessly).
2. **Lexical Blocking (Sparse)**:
   - Build a `TF-IDF` / `BM25` index using character $3$-to-$5$-grams on combined `name + address`.
   - Retrieve top 30 nearest neighbors using cosine similarity. Character $n$-grams make the system robust to typos and spelling variations.
3. **Dense Embedding Blocking (Dense ANN)**:
   - Use an Apache 2.0 / MIT compliant bi-encoder (e.g., `BAAI/bge-small-en-v1.5` or `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
   - Index records in `FAISS` or `HNSWlib` to query top 20 dense nearest neighbors.
4. **Union & Deduplication**:
   - Merge sparse and dense candidates, retaining the top 30–50 candidates per $S_1$ entity.
   - Verify recall against training ground truth: Target **$\ge 98.5\%$ candidate recall**.
   - Directly serializes as `candidate_pairs.tsv`.

---

### Phase 4: Deep Feature Extraction & Pairwise Scoring

For every $(S_1, S_x)$ candidate pair, compute rich distinguishing features:

1. **Fuzzy String & Token Distances**:
   - Levenshtein ratio, Damerau-Levenshtein distance.
   - Jaro-Winkler distance (places higher weight on prefixes).
   - Token Sort Ratio & Token Set Ratio (robust to word order swaps like "Acme Tools Pvt Ltd" vs "Tools Acme").
   - Longest common substring and sub-sequence lengths.
2. **Numerical & Component Overlap**:
   - Exact numeric match score (do street numbers or postal codes match or contradict?).
   - Suffix / Prefix mismatch flags.
3. **Semantic Similarity**:
   - Cosine similarity of name embeddings, address embeddings, and combined text embeddings.
4. **Cross-Encoder Deep Matching**:
   - Fine-tune a lightweight transformer (`cross-encoder/ms-marco-MiniLM-L-6-v2` or `DeBERTa-v3-small`) to output a pairwise interaction score:
     $$\text{Input: } \text{[CLS] } \text{Name}_1 \text{ [SEP] } \text{Addr}_1 \text{ [SEP] } \text{Name}_2 \text{ [SEP] } \text{Addr}_2$$
   - Cross-encoders capture full token-to-token cross-attention between entities.

---

### Phase 5: GBDT Re-ranking & $F_{0.5}$ Threshold Optimization

1. **Ensemble Re-Ranker**:
   - Train **LightGBM** and **CatBoost** classifiers on candidate pairs using the engineered features and cross-encoder scores.
   - Negative sampling: Include both hard negatives (candidates retrieved by blocking that are not matches) and random negatives.
2. **$F_{0.5}$ Metric-Specific Threshold Tuning**:
   - Compute probabilities $\hat{p} = P(\text{match} \mid S_1, S_x)$.
   - Perform grid/coordinate search on validation folds to select the classification cutoff $\tau^*$ that directly maximizes the macro $F_{0.5}$ formula.
   - Typically, for $F_{0.5}$, $\tau^*$ shifts higher (e.g., $0.65 - 0.75$) compared to standard $0.5$ because false positives carry a severe penalty.
3. **Singleton Filter**:
   - If for an $S_1$ entity, $\max_{x} \hat{p}(S_1, S_x) < \tau_{\text{singleton}}$, predict an **empty list** `""`.
   - This protects the full 1.0 credit awarded to true singletons.

---

### Phase 6: Code Structure & Reproduction Standards

Aligned with `Guidelines.md` and challenge submission requirements:

```
Amazon_Ml_Challenge/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── normalization.py     # Cleaning, unicode, abbreviation handling
│       │   ├── blocking.py          # TF-IDF + inverted index candidate generator
│       │   ├── features.py          # Lexical, numerical, semantic features
│       │   ├── ranker.py            # GBDT LightGBM re-ranking model
│       │   ├── postprocess.py       # F_0.5 optimizer & singleton logic
│       │   └── pipeline.py          # End-to-end execution script
│       ├── README.md                # Reproduction guide
│       └── requirements.txt         # Pinned dependencies
├── data/
│   ├── train/                       # train_source1, 2, 3 + ground_truth
│   └── test/                        # test_source1, 2, 3
├── output/
│   ├── matching_results.tsv         # Final matches (uploaded to leaderboard)
│   └── candidate_pairs.tsv          # Blocking candidates
├── utils/
│   ├── metrics.py                   # Macro F_0.5 metric implementation
│   └── validate_submission.py       # Submission validator
├── pyproject.toml                   # Poetry environment
├── Documentation_template.md        # Filled methodology write-up
└── WORKFLOW.md                      # Comprehensive execution plan
```
