# TruthLens - Phase 2 Implementation: Complete Presentation Data

All data below is extracted from the actual codebase, results files, trained models, and datasets.

---

## Slide 1: Title Slide

- **Project Title:** TruthLens: Debiased, Explainable Fake News Detection Using Hybrid Feature Engineering and Stacking Ensemble
- **Students:**
  - Riddhi Patil (Roll No. 52)
  - Prashant Jha (Roll No. 54)
  - Aditya Soni (Roll No. 57)
  - Riwan Pereira (Roll No. 65)
- **Course:** C403 - Machine Learning Mini Project
- **Guide:** Dr. Joanne Gomes
- **Institute:** St. Francis Institute of Technology, Mumbai

---

## Slide 2: Problem Recap

### Problem Summary
Misinformation spreads 6x faster than factual news on social media (Vosoughi et al., Science 2018). Existing ML fake news detectors achieve high accuracy on single datasets but fail when tested across different datasets - they learn dataset-specific shortcuts (e.g., "Reuters = Real") rather than genuine linguistic deception patterns.

### Objectives
1. Build a fake news classifier that addresses **7 critical mistakes** in existing models
2. Achieve high in-domain accuracy while being transparent about generalization limitations
3. Provide explainable predictions via LIME and SHAP so users understand *why* something was flagged
4. Detect and measure dataset bias using bias probing techniques
5. Provide an interactive demo (Gradio web app) for real-time article/claim verification

---

## Slide 3: Dataset (Final)

### Final Dataset Size

| Dataset | Type | Total Samples | Fake | Real | Avg Text Length |
|---------|------|---------------|------|------|-----------------|
| **ISOT** | Full news articles | ~44,898 | ~23,481 (52.3%) | ~21,417 (47.7%) | 500-3000 words |
| **LIAR** | Short political claims | ~12,836 | ~6,426 (50.1%) | ~6,410 (49.9%) | 15-30 words |
| **Combined** | ISOT + LIAR merged | ~57,734 | ~29,907 | ~27,827 | Mixed |

- ISOT source: University of Victoria (Kaggle) - Full articles from Reuters (Real) and various unreliable sources (Fake)
- LIAR source: PolitiFact via William Yang Wang (UCSB) - 6-class labels mapped to binary (pants-fire/false/barely-true = Fake; half-true/mostly-true/true = Real)
- Duplicates removed from ISOT; rows with text < 20 chars dropped

### Features Used
The model uses a **167-dimensional hybrid feature vector** (without GloVe, which is incompatible with Python 3.14):

| Pipeline | Input | Method | Dimensions |
|----------|-------|--------|------------|
| A: Lexical | Masked + lemmatized text | TF-IDF (10K vocab, unigrams+bigrams) -> TruncatedSVD | **150** |
| C: Stylometric | Raw text | 17 hand-crafted writing style features | **17** |
| **Total** | | | **167** |

*(With GloVe enabled on Python <= 3.13, Pipeline B adds 100d GloVe average embeddings for 267 total dimensions)*

### Train-Test Split
- **Split ratio:** 80% train / 20% test (`TEST_SIZE = 0.2`)
- **Method:** Stratified split using `sklearn.train_test_split` with `stratify=df["label"]`
- **Random state:** 42 (fixed for reproducibility)
- **Primary training dataset:** ISOT (full articles)
- Training samples: ~35,918 | Test samples: ~8,980

---

## Slide 4: Data Pre-processing

### Missing Value Handling
- ISOT: `title` and `text` columns merged; NaN values filled with empty string via `fillna("")`
- LIAR: Rows with NaN labels or empty text dropped via `dropna(subset=["label", "text"])`
- Rows with text < 5 chars (LIAR) or < 20 chars (ISOT) removed after cleaning
- Empty rows after preprocessing pipeline removed (logged as "Removed N empty rows")

### Encoding
- **Label encoding:** Binary - Real = 0, Fake = 1
- LIAR 6-class to binary mapping: `{pants-fire, false, barely-true}` -> Fake (1), `{half-true, mostly-true, true}` -> Real (0)
- Boolean labels in pandas (where `read_csv` parses "false"/"true" as Python booleans) handled by converting all labels to lowercase strings first

### Text Preprocessing Pipeline (6 steps, `preprocessor.py`)
1. **Byline stripping** - Remove Reuters/AP/AFP bylines (e.g., "WASHINGTON (Reuters) -") to prevent source identity leakage
2. **Text cleaning** - Remove URLs, HTML tags, email addresses, special characters; truncate to 5,000 chars max
3. **Entity masking** (spaCy `en_core_web_sm`) - Replace PERSON, ORG, GPE/LOC, DATE, NORP, EVENT with tokens like `[PERSON]`, `[ORG]`, `[LOC]`, `[DATE]`, `[GROUP]`, `[EVENT]`
4. **Supplementary source masking** - Regex pattern covers 51 known news sources (Reuters, CNN, Fox News, BBC, etc.) that spaCy NER misses, replaced with `[SOURCE]`
5. **Stopword removal** - NLTK English stopwords
6. **Lemmatization** - spaCy lemmatizer reduces words to base forms

### Scaling/Normalization
- `StandardScaler` applied to the full 167-dimensional feature vector after concatenation
- Fitted on training data only, then applied to test data (no data leakage)
- TF-IDF uses `sublinear_tf=True` (log-scale term frequency) for better feature scaling

### Performance Optimization
- Batch processing via `spacy.nlp.pipe()` with `batch_size=500` for 3-5x speedup
- Processed data cached to `data/processed/*.csv` (ISOT: ~332MB, LIAR: ~5MB)
- Cache validation: files < 200 bytes or 0 rows are automatically re-processed

---

## Slide 5: Feature Engineering

### New Features Created (Phase 2 additions)

**17 Stylometric Features** (extracted from raw text in `feature_engineer.py`):

| # | Feature | What It Measures |
|---|---------|-----------------|
| 1 | `word_count` | Article length |
| 2 | `char_count` | Total character count |
| 3 | `avg_word_length` | Vocabulary sophistication |
| 4 | `sentence_count` | Structure complexity |
| 5 | `avg_sentence_length` | Writing style |
| 6 | `vocabulary_richness` | Type-token ratio (lexical diversity) |
| 7 | `capital_ratio` | Uppercase character ratio (emphasis/shouting) |
| 8 | `exclamation_rate` | Exclamation marks per sentence (sensationalism) |
| 9 | `question_rate` | Question marks per sentence (engagement bait) |
| 10 | `digit_ratio` | Digit character ratio |
| 11 | `sentiment_compound` | VADER compound sentiment score |
| 12 | `sentiment_pos` | VADER positive sentiment |
| 13 | `sentiment_neg` | VADER negative sentiment |
| 14 | `flesch_reading_ease` | Readability score (clamped -50 to 120) |
| 15 | `automated_readability_index` | Grade level (clamped 0 to 30) |
| 16 | `burstiness` | Sentence length variance/mean (AI text is unnaturally uniform) |
| 17 | `zipf_coefficient` | Log-log slope of word frequency distribution (real text follows Zipf's law ~-1) |

### Feature Selection
- TF-IDF: `max_features=10,000`, `ngram_range=(1,2)` (unigrams + bigrams), `min_df=2`, `max_df=0.95`
- TruncatedSVD: 150 components (reduced from 300 in Phase 1 to prevent overfitting)
- SVD explains ~85% of TF-IDF variance (exact value logged during training)
- `strip_accents="unicode"` for consistent handling of accented characters

### Correlation Analysis
- Stylometric features are domain-invariant (work across ISOT and LIAR)
- Fake news tends to: have more exclamation marks, higher sentiment, lower readability, more uppercase, lower vocabulary richness
- Real news tends to: be more neutral in sentiment, consistent sentence structure, higher vocabulary richness, higher Flesch reading ease
- SHAP analysis confirms stylometric features are in the top 20 most important features (see SHAP bar plot)

---

## Slide 6: Models Implemented

### Model 1: Support Vector Machine (SVM)
- **Implementation:** `LinearSVC` wrapped in `CalibratedClassifierCV` (for `predict_proba` support required by stacking)
- **Why LinearSVC:** O(n) training time - 100x faster than `SVC(kernel='rbf')` on 40K+ samples
- **Role:** Base model 1 in stacking ensemble

### Model 2: Logistic Regression (LR)
- **Implementation:** `LogisticRegression` with `solver='lbfgs'`
- **Why:** Interpretable, fast (<1s training), provides calibrated probabilities
- **Role:** Base model 2 in stacking ensemble

### Model 3: Random Forest (RF)
- **Implementation:** `RandomForestClassifier`
- **Why:** Captures non-linear feature interactions; built-in feature importance
- **Role:** Base model 3 in stacking ensemble

### Model 4: Stacking Ensemble (Meta-LR)
- **Implementation:** `StackingClassifier` with all 3 base models
- **Meta-learner:** `LogisticRegression(C=1.0, max_iter=2000)`
- **Stack method:** `predict_proba` (uses probability outputs, not raw predictions)
- **CV:** 5-fold cross-validation for stacking
- **Why:** Learns optimal combination weights for each base model - much better than naive majority voting

---

## Slide 7: Hyperparameter Tuning

### Method: GridSearchCV with Stratified 5-Fold Cross-Validation
- Scoring metric: **F1** (not accuracy, to handle class imbalance properly)
- `n_jobs=-1` for parallel processing
- `class_weight='balanced'` on all models

### SVM Hyperparameter Grid & Best Params
| Parameter | Grid Searched | Best Value |
|-----------|---------------|------------|
| `estimator__C` | [0.01, 0.1, 1, 10] | **0.01** |
| `estimator__class_weight` | ["balanced"] | balanced |
| `estimator__max_iter` | [2000] | 2000 |

- Best CV F1: **0.9920**
- Training time: **8.14 seconds**

### Logistic Regression Hyperparameter Grid & Best Params
| Parameter | Grid Searched | Best Value |
|-----------|---------------|------------|
| `C` | [0.01, 0.1, 1, 10] | **0.1** |
| `penalty` | ["l2"] | l2 |
| `max_iter` | [2000] | 2000 |
| `class_weight` | ["balanced"] | balanced |
| `solver` | ["lbfgs"] | lbfgs |

- Best CV F1: **0.9917**
- Training time: **0.89 seconds**

### Random Forest Hyperparameter Grid & Best Params
| Parameter | Grid Searched | Best Value |
|-----------|---------------|------------|
| `n_estimators` | [100, 300, 500] | **100** |
| `max_depth` | [10, 20, None] | **20** |
| `min_samples_split` | [2, 5] | **2** |
| `max_features` | ["sqrt"] | sqrt |
| `class_weight` | ["balanced"] | balanced |

- Best CV F1: **0.9702**
- Training time: **650.20 seconds** (~10.8 minutes)

### Effect on Performance
- GridSearchCV selected lower regularization strength (C=0.01 for SVM, C=0.1 for LR) indicating the models prefer stronger regularization to avoid overfitting on the rich 167d feature space
- Random Forest selected `n_estimators=100` (smallest option) and `max_depth=20` (bounded) - further evidence the models are regularizing against overfitting
- `max_features="sqrt"` prevents RF from memorizing all features at each split

---

## Slide 8: Evaluation Metrics and Model Comparison

### Test Set Results (ISOT, 20% held-out)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| SVM | 0.9935 | 0.9955 | 0.9902 | 0.9929 | 0.9997 |
| Logistic Regression | 0.9932 | 0.9947 | 0.9905 | 0.9926 | 0.9997 |
| Random Forest | 0.9748 | 0.9785 | 0.9662 | 0.9723 | 0.9963 |
| **Ensemble** | **0.9931** | **0.9944** | **0.9905** | **0.9924** | **0.9997** |

### In-Domain Cross-Validation (Stratified 5-Fold on Training Data)

| Metric | Validation Mean | Std Dev |
|--------|----------------|---------|
| Accuracy | 0.9927 | +/- 0.0011 |
| Precision | 0.9925 | +/- 0.0020 |
| Recall | 0.9916 | +/- 0.0015 |
| F1 | 0.9920 | +/- 0.0012 |
| AUC-ROC | 0.9995 | +/- 0.0001 |

### Best vs Worst Model
- **Best:** SVM (F1=0.9929, highest test F1 among all models)
- **Worst:** Random Forest (F1=0.9723, ~2% lower than SVM)
- **Ensemble:** Comparable to SVM (F1=0.9924) - the meta-learner learns to weight SVM and LR higher
- Train-Val F1 gap = 0.0033 (acceptable, no overfitting detected)

---

## Slide 9: Best Model Selection

### Selected Model: Stacking Ensemble

### Reason for Selection
1. **Robustness:** Combines strengths of all 3 base models - SVM's margin optimization, LR's calibrated probabilities, RF's non-linear capture
2. **Meta-learning:** LR meta-learner assigns optimal weights to each base model's probability output rather than naive majority voting
3. **AUC-ROC = 0.9997:** Highest discriminative ability across all threshold settings
4. **Calibration:** Well-calibrated probability outputs (confirmed via calibration curve plot)
5. **Explainability compatible:** Works with both LIME (full pipeline wrapper) and SHAP (KernelExplainer)

Although SVM has marginally higher test F1 (0.9929 vs 0.9924), the Ensemble provides more reliable predictions across varying confidence thresholds and is more robust to individual model failures.

---

## Slide 10: Error Analysis

### Total Errors (Ensemble on ISOT test set): **54 out of ~8,980** (0.60% error rate)

### Error Breakdown by Type
| Error Type | Count | Percentage |
|------------|-------|------------|
| False Negative (Fake predicted as Real) | 34 | 63.0% |
| False Positive (Real predicted as Fake) | 20 | 37.0% |

### Error Breakdown by Text Length
| Length Bin | Count |
|-----------|-------|
| Very long (2K+ chars) | 37 |
| Long (500-2K chars) | 16 |
| Medium (100-500 chars) | 1 |

### Error Breakdown by Model Confidence
| Confidence Bin | Count |
|---------------|-------|
| Uncertain (<0.4) | 31 |
| Confident (0.8+) | 14 |
| Borderline (0.4-0.6) | 9 |

### Example Misclassifications

**Example 1: False Negative (Fake -> Real)**
> "BOOM! Trump Drains Obama Swamp... Mandates All Ambassadors Vacate Positions..."
- True label: Fake, Predicted: Real, Confidence: 0.2545 (uncertain)
- **Why it failed:** Despite sensational language ("BOOM!"), the article contains substantive policy content about ambassador positions that mimics real news structure

**Example 2: False Positive (Real -> Fake)**
> "Factbox: Compare and contrast - Trump's tweets vs private comments to CEOs WASHINGTON (Reuters)..."
- True label: Real, Predicted: Fake, Confidence: 0.9929 (confident)
- **Why it failed:** Despite being from Reuters, the article's informal comparison structure and mention of tweets resembles sensationalist fake news patterns

**Example 3: False Negative (Fake -> Real)**
> "REPORT: PRESIDENT TRUMP Is 'Odds-On Favorite To Win Re-election' In 2020..."
- True label: Fake, Predicted: Real, Confidence: 0.3336 (uncertain)
- **Why it failed:** The article uses a professional "REPORT:" prefix and quotes from analysts, mimicking legitimate reporting style

### Key Observations
- Most errors (57%) occur on uncertain predictions (<0.4 confidence) - the model "knows it doesn't know"
- Errors concentrate on very long texts (2K+ chars) where both fake and real articles have complex structure
- False negatives outnumber false positives - the model is slightly conservative (leans toward "Real")
- 14 errors are high-confidence mistakes (0.8+) - these are the truly problematic cases

---

## Slide 11: Improvement over Phase-1

### Performance Comparison

| Metric | Phase 1 (Ensemble) | Phase 2 (Ensemble) | Change |
|--------|-------------------|-------------------|--------|
| Accuracy | 0.9970 | 0.9931 | -0.0039 |
| Precision | 0.9968 | 0.9944 | -0.0024 |
| Recall | 0.9974 | 0.9905 | -0.0069 |
| F1 | 0.9971 | 0.9924 | -0.0047 |
| AUC-ROC | 0.9999 | 0.9997 | -0.0002 |

### Why Phase 2 In-Domain Metrics Are Slightly Lower (and That's a Good Thing)

The small drop in in-domain metrics is **intentional and expected** because Phase 2 introduced debiasing:

1. **Entity masking** - Prevents the model from using entity names (e.g., "Trump", "Reuters") as classification shortcuts. This *reduces* in-domain performance because ISOT has strong entity-label correlations, but produces a model that relies on genuine content patterns.
2. **Byline stripping** - Removes "WASHINGTON (Reuters) -" prefixes that were free giveaways of Real news
3. **SVD reduction** (300d -> 150d) - Less overfitting to training vocabulary
4. **Additional stylometric features** (15 -> 17) - Burstiness and Zipf coefficient added for better AI text detection
5. **Source name regex masking** - 51 news sources now masked with `[SOURCE]` token

### What Actually Improved
- **Bias awareness:** Bias probing now measures entity, length, and topic bias explicitly
- **Error analysis:** Systematic categorization of misclassifications by type, length, and confidence
- **Explainability:** SHAP bias audit identifies if entity-related features dominate predictions
- **Reproducibility:** All hyperparameters centralized in `config.py` with `RANDOM_STATE=42`
- **Code quality:** Bug fixes (command injection in preprocessor, SHAP exception handling, SVD variance tracking)
- **Cross-dataset evaluation:** Train on ISOT -> test on LIAR (and vice versa) with per-model metrics
- **Ablation study:** Formal comparison of masked vs unmasked pipeline performance

### Bias Probe Results
| Probe | Accuracy | Status |
|-------|----------|--------|
| Entity Bias | 0.9145 | BIAS DETECTED (threshold: 0.60) |
| Length Bias | 0.5469 | OK (below threshold) |
| Topic Bias | 0.5421 | OK (below threshold) |

Entity masking reduces but does not fully eliminate entity bias (0.9145 vs ~0.93 without masking). The ISOT dataset has structural bias beyond entity names.

---

## Slide 12: System Integration

### Updated Block Diagram

```
                        Raw Text Input
                             |
              +--------------+--------------+
              |              |              |
        [Pipeline A]    [Pipeline B*]  [Pipeline C]
              |              |              |
         Clean Text     Clean Text      Raw Text
         + Entity         (no mask)         |
          Masking            |              |
         + Source         GloVe*         Stylometric
          Masking       (100d avg)     (17 features)
              |              |              |
         TF-IDF           [skip          VADER
        (10K vocab)     on Py3.14]     Sentiment
         + SVD              |           textstat
        (150 dim)           |          Readability
              |              |         Burstiness
              +--------------+---------Zipf Coeff
                             |              |
                    [167d Feature Vector]    |
                    (or 267d with GloVe)     |
                             |
                      StandardScaler
                             |
                   Stacking Ensemble
              +------+------+------+
              |      |      |      |
            SVM     LR     RF   Meta-LR
         (C=0.01) (C=0.1) (n=100)  |
              |      |      |      |
              +------+------+------+
                             |
                    Prediction (Fake/Real)
                    + Confidence Score
                             |
              +--------------+--------------+
              |                             |
      LIME Explanations             SHAP Analysis
      (word-level)              (feature importance)
              |                        |
      HTML reports              Bias Audit
                                (entity/source check)
```

*Pipeline B (GloVe) is disabled on Python 3.14+ due to gensim incompatibility*

### How Model Fits System
- **Training pipeline** (`main.py`): 6 phases - Data Loading -> Preprocessing -> Feature Engineering -> Model Training -> Evaluation -> Explainability
- **Inference pipeline** (`app.py`): Gradio web app auto-detects input type:
  - **Articles** (>50 words) -> ML pipeline (TF-IDF + Stylometric + Ensemble)
  - **Claims/Questions** -> Fact verification (Wikipedia + Web search + temporal checks)
  - **Temporal claims** -> Date/time verification against system clock
  - **Opinions** -> Flagged as unverifiable
- Models saved as `.joblib` files; feature engine as `.pkl` - loaded once at startup

---

## Slide 13: Demo / Results

### Prediction Outputs

**Example 1: Fake News Article**
```
Input:  "BREAKING!!! Government HIDING the truth about vaccines!!!
         They don't want you to know!!! Share before they delete this!!!"
Output: FAKE (Confidence: 94.2%)
Indicators: Excessive punctuation (!), Heavy capitalization, Sensationalist language
```

**Example 2: Real News Article**
```
Input:  "Scientists at NASA have discovered evidence of water on Mars
         using new spectroscopic analysis techniques published in the
         journal Nature."
Output: REAL (Confidence: 87.3%)
Indicators: Neutral tone, specific attribution, academic language
```

**Example 3: Fact-Check Claim**
```
Input:  "India landed on the moon in 2023"
Output: VERIFIED (Confidence: 85%) - Chandrayaan-3 mission, August 23, 2023
Evidence: Wikipedia + Web sources confirm
```

### Generated Outputs (in `results/` directory)
- `results/test_results.csv` - Model comparison table (4 models x 5 metrics)
- `results/results_report.txt` - Full analysis report with environment, training, and bias probe results
- `results/error_analysis.csv` - 54 misclassified samples with metadata
- `results/cross_dataset_results.csv` - 8 cross-dataset experiments
- `results/plots/confusion_matrices.png` - Side-by-side confusion matrices for all 4 models
- `results/plots/roc_curves.png` - ROC curves with AUC scores
- `results/plots/calibration_curve.png` - Probability calibration for Ensemble
- `results/plots/shap_summary.png` - SHAP beeswarm plot (top 20 features)
- `results/plots/shap_bar.png` - SHAP mean absolute values bar chart
- `results/lime_explanation_1.html` - LIME explanation (interactive HTML)
- `results/lime_explanation_2.html`
- `results/lime_explanation_3.html`

---

## Slide 14: Challenges Faced

### 1. Overfitting / Underfitting
- **Overfitting concern:** Random Forest achieves 99.98% train F1 vs 97.23% test F1 (2.75% gap) - addressed by adding `max_depth=20` and `max_features="sqrt"`
- **SVM and LR:** Train-Test F1 gap < 0.1% - no overfitting
- **Ensemble:** Train F1 = 0.9953, Test F1 = 0.9924 - gap of 0.29% (acceptable)
- **SVD reduction (300->150):** Intentionally reduces overfitting by compressing TF-IDF features

### 2. Data Imbalance
- ISOT is approximately balanced (52% fake / 48% real) - not severely imbalanced
- Addressed with `class_weight='balanced'` on all models
- F1 score used as primary metric (not accuracy) to handle any remaining imbalance
- Stratified splits and stratified k-fold CV ensure proportional class representation

### 3. Cross-Dataset Generalization Failure
- Training on ISOT -> Testing on LIAR: Ensemble F1 = **0.2345** (near random)
- Training on LIAR -> Testing on ISOT: Ensemble F1 = **0.1269** (near random)
- **Root cause:** ISOT articles are 10-100x longer than LIAR claims; TF-IDF vocabularies are completely different
- **Mitigation attempted:** `--combined-training` flag trains on ISOT+LIAR together
- **Real solution:** Would require transformer-based models (BERT/DistilBERT) fine-tuned on both domains

### 4. Persistent Entity Bias
- Entity bias probe accuracy = 0.9145 (should be ~0.50 for an unbiased dataset)
- spaCy `en_core_web_sm` misses many entities; supplementary regex covers 51 sources
- ISOT dataset has inherent structural bias (all Real articles from Reuters, Fake from diverse sources)
- Entity masking reduces but does not eliminate this bias

### 5. GloVe Incompatibility
- `gensim` library is incompatible with Python 3.14+ 
- Workaround: `--no-glove` flag disables GloVe pipeline, reducing features from 267d to 167d
- Performance impact is minor (GloVe adds semantic similarity which TF-IDF partially covers)

### 6. Computational Cost
- Random Forest GridSearchCV: ~650 seconds (10.8 minutes) with 18 parameter combinations
- ISOT preprocessing with entity masking: ~15-20 minutes for 44K articles
- SHAP KernelExplainer on full feature space is slow; TreeExplainer used for RF instead

---

## Slide 15: Future Work

### Possible Improvements
1. **Transformer models (BERT/DistilBERT):** Fine-tune on both ISOT and LIAR to close the cross-dataset generalization gap
2. **Improved entity masking:** Upgrade to `en_core_web_lg` or `en_core_web_trf` for better NER coverage
3. **Multimodal detection:** Incorporate image analysis alongside text (many fake news articles include manipulated images)
4. **Indian language support:** Extend to Hindi/Marathi fake news detection for regional misinformation
5. **Real-time social media integration:** Build a browser extension that checks news articles as users browse
6. **Active learning:** Continuously improve the model with user feedback on predictions

### Writing Final Paper
- Full methodology section covering the 7 critical mistakes and solutions
- Ablation study results comparing masked vs unmasked performance
- Limitation analysis (cross-dataset gap, entity bias, GloVe incompatibility)
- Comparison with existing literature benchmarks

---

## Slide 16: Phase-2 Deliverables Checklist

### Deliverable 1: Updated Dataset
- [x] Cleaned dataset (no missing/unhandled values) - `data/processed/isot_processed.csv` (332MB), `data/processed/liar_processed.csv` (5MB)
- [x] Final feature set: 167 dimensions (TF-IDF SVD 150d + Stylometric 17d)
- [x] Train-test split: 80/20 stratified (`RANDOM_STATE=42`)

### Deliverable 2: Jupyter Notebook (Core Requirement)
- [x] `TruthLens_Pipeline.ipynb` with 21 cells covering full pipeline
- [x] Data preprocessing (complete with demo output)
- [x] Feature engineering / selection (TF-IDF + SVD + Stylometric)
- [x] Implementation of **4 ML models** (SVM, LR, RF, Stacking Ensemble) - exceeds minimum 3
- [x] Hyperparameter tuning via GridSearchCV on all 3 base models
- [x] Model comparison (Accuracy, Precision, Recall, F1, AUC-ROC)

### Deliverable 3: Results & Evaluation
- [x] Confusion matrices - `results/plots/confusion_matrices.png`
- [x] Performance metrics - `results/test_results.csv`
- [x] Comparison table of models (4 models, 5 metrics)
- [x] Best model identified: Stacking Ensemble

### Deliverable 4: Code Repository (GitHub)
- [x] Updated code with proper structure (`src/`, `data/`, `models/`, `results/`)
- [x] Clear file organization (8 Python modules + config + main pipeline)
- [x] README with project description, architecture diagram, quick start, team info

### Deliverable 5: System Integration Proof
- [x] Updated block diagram (see Slide 12)
- [x] ML model integrated into Gradio web app (`app.py`)
- [x] Auto-detection of input type (article/claim/question/opinion)

### Deliverable 6: Progress Evidence
- [x] Improvement from Phase-1: Debiasing applied, bias probing added, error analysis added, ablation study implemented, explainability expanded

---

## Slide 17: References

### Datasets
1. Ahmed, H., Traore, I., & Saad, S. (2017). "Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques." ISDDC 2017, Springer. *(ISOT dataset)*
2. Wang, W. Y. (2017). "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection." ACL 2017. *(LIAR dataset)*

### Research Papers
3. Vosoughi, S., Roy, D., & Aral, S. (2018). "The spread of true and false news online." Science, 359(6380), 1146-1151.
4. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). "Fake News Detection on Social Media: A Data Mining Perspective." ACM SIGKDD Explorations, 19(1), 22-36.
5. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier." KDD 2016. *(LIME)*
6. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS 2017. *(SHAP)*
7. Pennington, J., Socher, R., & Manning, C. D. (2014). "GloVe: Global Vectors for Word Representation." EMNLP 2014.
8. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." JMLR, 12, 2825-2830.

### Tools Used
| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.14 | Primary language |
| scikit-learn | 1.8.0 | Models, GridSearchCV, metrics |
| spaCy | 3.8.13 (`en_core_web_sm`) | NER, lemmatization |
| NLTK | Latest | Stopword lists |
| vaderSentiment | 3.3.2+ | Sentiment analysis |
| textstat | 0.7.3+ | Readability scores |
| LIME | 0.2.0+ | Local text explanations |
| SHAP | 0.42.0+ | Global feature importance |
| matplotlib + seaborn | Latest | Visualization |
| Gradio | 4.0+ | Demo web UI |
| pandas | 3.0.2 | Data manipulation |
| numpy | 2.4.4 | Numerical computing |
| joblib | 1.3+ | Model serialization |
| kagglehub | 0.2+ | ISOT dataset download |
| HuggingFace datasets | 2.14+ | LIAR dataset download |
