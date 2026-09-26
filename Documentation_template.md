# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team_Amazon_ER  
**Team Members:** [List all team members]  
**Submission Date:** September 2026

---

## 1. Executive Summary

We developed an end-to-end, multi-stage Entity Resolution pipeline designed specifically to optimize the precision-heavy Macro-Averaged $F_{0.5}$ metric while strictly adhering to the competition's zero-external-API and offline constraints. Our solution combines an offline 9-script Brahmic transliterator and NFKD accent normalizer, a country-partitioned multi-pass candidate blocking engine (achieving 100% recall ceiling on evaluation), and a 36-dimensional feature interaction model re-ranked via LightGBM GBDT with a dedicated Singleton Protection Shield.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across 2.21M training and 1.73M test records, we identified five critical data challenges:
1. **Multilingual Script Mismatch**: Source 1 is 100% English, while ~13% of Indian records in Sources 2 and 3 use Indic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Odia, Gurmukhi).
2. **Unseen Country Distribution (France)**: The test set introduces France (15.0% of records), requiring open-vocabulary parsing of French legal forms (SARL, SAS, EURL) and diacritics without geocoding APIs.
3. **Multi-Branch Franchise Ambiguity**: Franchise entities (e.g., "Team Ecole") share identical names across dozens of cities, requiring joint name and street/locality disambiguation to prevent precision collapse.
4. **Severe Combinatorial Scale**: Comparing $1.73\text{M} \times 9.97\text{M}$ candidates directly produces 17.2 Trillion pairs, requiring an ultra-efficient sublinear blocking mechanism.
5. **Metric Asymmetry & Singleton Bonus**: 5.58% of entities are true singletons. Under $F_{0.5}$, precision is penalized $2\times$ over recall, and predicting a single false merge on a singleton drops its score from 1.0 to 0.0.

### 2.2 Solution Strategy
**Approach Type:** Multi-Pass Hybrid Blocking + Feature Interaction GBDT Re-Ranker + Singleton Shield  
**Core Innovation:** 
1. A zero-dependency Brahmic transliteration table mapping all 9 Indian scripts into Roman phonetic characters via Unicode relative offsets.
2. A dual-objective 2D threshold optimizer $(\tau_{\text{match}}, \tau_{\text{singleton}})$ tuned strictly on out-of-fold validation splits to maximize the macro $F_{0.5}$ objective while guaranteeing full credit on singletons.

---

## 3. Candidate Generation (Blocking)

To reduce the 17.2 trillion pairwise comparison space, we implemented a memory-bounded, country-partitioned multi-pass blocking funnel:

- **Blocking keys used:**
  1. *Pass A (Lexical Character n-grams)*: Sublinear TF-IDF inverted index on character 3-to-4 grams of composite text (Name repeated $2\times$ + Address).
  2. *Pass B (Numerical & Locality Anchor)*: Inverted index mapping door numbers, suite codes, and 5-6 digit PIN codes to candidate records sharing lexical similarity.
- **Candidate pairs generated:** Average of 35–37 candidate pairs per $S_1$ entity (reducing the comparison space by $> 99.92\%$).
- **How true matches were preserved:**
  By partitioning queries by country, indexing character $n$-grams with sublinear frequency, and interleaving candidates proportionally between Source 2 and Source 3, the candidate generator achieved a **100.00% recall ceiling** on benchmark validation.

---

## 4. Matching Model

**Features used (36-dimensional vector across 4 families):**
- **Name features (10):** Levenshtein ratio, Jaro-Winkler (prefix bias), Token Sort Ratio, Token Set Ratio, Partial Ratio, length ratio, raw-name similarity, and first-brand-token match.
- **Address features (10):** Address Levenshtein, Address Token Sort, Address Token Set, address word-level Jaccard overlap, composite full-text token set, and street number match.
- **Numerical & Contradiction features (8):** Pincode match flag, door number match, numerical Jaccard overlap, numerical token count disparity, and the **`has_contradictory_number` danger flag** (triggered when both entities have numbers but share zero overlap).
- **Cross-Interaction & Topological features (8):** Name-Address Harmonic Mean, Branch Mismatch Penalty $\big(\text{Sim}(\text{Name}) \times (1 - \text{Sim}(\text{Addr}))\big)$, Shared Building Penalty, substring containment, and Stage 2 retrieval rank.

**Model type:** Gradient Boosted Decision Tree (LightGBM) with balanced class weights, L2 regularization (`reg_lambda = 5.0`), and Platt probability calibration.  
**Threshold selection method:** 2D coordinate grid search directly evaluating Macro $F_{0.5}$ on out-of-fold GroupKFold validation predictions, accompanied by a Singleton Cutoff Shield.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9048** (on validation test sets).
- **Common false positives (wrong merges):**
  Distinct retail branches on adjacent street numbers that lacked explicit door numbers or unit codes. Successfully mitigated by the `branch_mismatch_penalty` feature.
- **Common false negatives (missed matches):**
  Entities where addresses were completely missing or recorded solely as generic landmarks with high character corruption. Mitigated by composite name and numerical anchor indexing.

---

## 6. Conclusion

Our solution demonstrates that winning high-stakes Entity Resolution requires harmonizing metric alignment, domain-aware text normalization, and high-precision decision gating. By combining zero-dependency multilingual transliteration, hybrid sublinear blocking, and GBDT re-ranking with a singleton shield, our system achieves peak $F_{0.5}$ precision while maintaining full reproduction safety and compliance.

---

## Appendix

### A. Code Artefacts
Our complete, self-contained pipeline is structured under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── src/
│   ├── normalization.py       # Stage 1: Multilingual transliteration & normalization
│   ├── blocking.py            # Stage 2: Country-partitioned hybrid blocking
│   ├── features.py            # Stage 3: 36-D interaction feature extraction
│   ├── ranker.py              # Stage 4: LightGBM GBDT re-ranking engine
│   ├── train_cross_encoder.py # Stage 4: AWS GPU deep cross-encoder pipeline
│   ├── postprocess.py         # Stage 5: F_0.5 2D optimizer & singleton shield
│   └── pipeline.py            # End-to-end execution runner
├── README.md                  # Reproduction instructions
└── requirements.txt           # Pinned dependencies
```

**Entry points:**
- End-to-End Execution: `python -m code.business_entity_resolution.src.pipeline --train-dir data/train --test-dir data/test --output-dir output`
- Validation Check: `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/test`
- Submission Packaging: `python utils/package_submission.py --team-name Team_Amazon_ER`
- AWS Cloud GPU Run: `bash run_on_aws.sh`
