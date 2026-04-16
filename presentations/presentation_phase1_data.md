# TruthLens - Implementation Phase 1: Presentation Data

All data below is sourced from the actual codebase, results files, and datasets.

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

## Slide 2: Problem Statement (Revised)

- **Problem:** Misinformation spreads 6x faster than factual news on social media (MIT, 2018). Existing ML-based fake news detectors achieve high accuracy on single datasets but fail catastrophically when tested across different datasets, revealing they learn dataset-specific shortcuts rather than genuine linguistic deception patterns.
- **Input:** Raw news article text (or document: PDF, image, DOCX)
- **Output:** Binary classification (Fake / Real) with:
  - Confidence score (0-100%)
  - Explainability report (LIME word importance + SHAP feature importance)
  - Bias audit flag (whether prediction relies on entity names vs. content)
- **Real-World Use Case:** A journalist or social media user pastes a news article and receives an instant verdict with explanation of *why* the system flagged it, enabling informed decision-making rather than blind trust.
- **SDG Alignment:** UN SDG 16 - Peace, Justice and Strong Institutions (Target 16.10: Public access to information)

---

## Slide 3: Dataset Details

### ISOT Fake News Dataset
| Property | Value |
|----------|-------|
| Source | University of Victoria (Kaggle) |
| Total Samples | 44,898 articles |
| Fake Articles | ~23,481 (from various unreliable sources) |
| Real Articles | ~21,417 (from Reuters.com) |
| Columns | title, text, subject, date |
| Text Length | Full-length articles (avg ~500-3000 words) |
| Subjects | politicsNews, worldnews, politics, left-news, Government News, US_News, Middle-east |

### LIAR Dataset (PolitiFact)
| Property | Value |
|----------|-------|
| Source | PolitiFact.com via William Yang Wang (UCSB) |
| Total Samples | 12,836 statements |
| Original Labels | 6 classes: pants-fire, false, barely-true, half-true, mostly-true, true |
| Binary Mapping | pants-fire/false/barely-true = Fake; half-true/mostly-true/true = Real |
| Columns | id, label, statement, subject, speaker, job_title, state_info, party, + 5 credit history counts, context |
| Text Length | Short political claims (avg ~15-30 words) |

### Combined Dataset
| Property | Value |
|----------|-------|
| Total Samples | ~57,734 |
| Purpose | Cross-dataset generalization testing |

---

## Slide 4: Data Understanding

### Data Types
| Column | Type | Dataset |
|--------|------|---------|
| text | String (categorical/text) | Both |
| title | String | ISOT only |
| subject | String (categorical) | ISOT |
| date | String (date) | ISOT |
| label | String -> Binary (0/1) | Both |
| speaker, party | String (categorical) | LIAR only |

### Missing Values
- ISOT: Minimal missing values; handled via `fillna("")` in data_loader.py
- LIAR: Some statements have empty speaker/context fields
- All rows with text < 5 characters or empty labels are dropped

### Data Distribution Insights
- ISOT: Roughly balanced (~52% fake, ~48% real)
- LIAR: 6-class distribution is uneven; binary mapping creates ~48% fake, ~52% real
- Key difference: ISOT articles are 10-100x longer than LIAR statements
- ISOT real news has consistent Reuters formatting; ISOT fake has diverse sources

---

## Slide 5: Preprocessing Plan

### 6-Step Pipeline (implemented in `preprocessor.py`)
1. **Lowercasing** - Normalize all text to lowercase
2. **URL & HTML Removal** - Regex-based removal of hyperlinks, HTML tags, email addresses
3. **Entity Masking** - spaCy NER replaces PERSON, ORG, GPE, LOC, DATE entities with tokens like `[PERSON]`, `[ORG]` to prevent the model from memorizing entity names as shortcuts
4. **Special Character Removal** - Strip non-alphanumeric characters except spaces
5. **Stopword Removal** - NLTK English stopwords removed to reduce noise
6. **Lemmatization** - spaCy lemmatizer reduces words to base forms (e.g., "running" -> "run")

### Performance Optimization
- Batch processing via `spacy.nlp.pipe()` with `batch_size=200` for 3-5x speedup
- Processed data cached to `data/processed/*.csv` to avoid re-computation
- Text truncated to 5,000 characters max to prevent memory issues

---

## Slide 6: System Block Diagram

```
                        Raw Text Input
                             |
              +--------------+--------------+
              |              |              |
        [Pipeline 1]   [Pipeline 2]   [Pipeline 3]
              |              |              |
         Clean Text     Clean Text      Raw Text
         + Entity         (no mask)         |
          Masking            |              |
              |              |              |
         TF-IDF          GloVe         Stylometric
        (10K vocab)    (Wiki-100d)    (15 features)
              |           avg pool          |
         TruncatedSVD        |              |
          (300 dim)    (100 dim)       (15 dim)
              |              |              |
              +--------------+--------------+
                             |
                    [415-dim Feature Vector]
                             |
                   Stacking Ensemble
              +------+------+------+
              |      |      |      |
            SVM     LR     RF   Meta-LR
              |      |      |      |
              +------+------+------+
                             |
                    Prediction (Fake/Real)
                             |
                   LIME + SHAP Explainability
                             |
                    Bias Audit Report
```

---

## Slide 7: UML Diagrams

### Use Case Diagram
- **Actors:** User (Journalist/Researcher), System
- **Use Cases:**
  1. User submits text / uploads document (PDF, Image, DOCX)
  2. System preprocesses text (clean, mask entities, lemmatize)
  3. System extracts hybrid features (TF-IDF + GloVe + stylometric)
  4. System predicts Fake/Real via stacking ensemble
  5. System generates LIME/SHAP explanation
  6. System performs bias audit
  7. System returns verdict + confidence + explanation to user
  8. (Web App) System runs RAG fact-checking (Wikipedia, Web, Geo, Temporal)

### Activity Diagram Flow
```
Start -> Load Data -> Check Cache -> [Cached?]
  Yes -> Load Processed Data
  No  -> Clean Text -> Mask Entities -> Lemmatize -> Save Cache
-> Extract Features (TF-IDF | GloVe | Stylometric) -> Concatenate 415d Vector
-> Train Models (SVM, LR, RF) via GridSearchCV -> Train Stacking Ensemble
-> Evaluate (In-domain CV + Cross-dataset + Bias Probe)
-> Generate Explanations (LIME HTML + SHAP plots)
-> Output Results Report -> End
```

---

## Slide 8: Algorithm Selection

| Algorithm | Why Selected | Role | Key Strength |
|-----------|-------------|------|--------------|
| **SVM (RBF kernel)** | Best for high-dimensional text classification; maximum margin classifier | Base model 1 | Handles 415-dim space well, robust to overfitting with proper C |
| **Logistic Regression** | Interpretable, fast, provides calibrated probabilities | Base model 2 | Coefficients directly show feature importance; <1ms inference |
| **Random Forest** | Ensemble of decision trees, captures non-linear relationships | Base model 3 | Handles feature interactions; built-in feature importance |
| **Stacking Ensemble (Meta-LR)** | Combines all three base models with a Logistic Regression meta-learner | Final model | Learns optimal combination weights; outperforms naive voting |

### Why Not Deep Learning?
- Dataset size (~45K) doesn't justify BERT/transformer overhead
- Traditional ML with good features achieves comparable performance
- Explainability (LIME/SHAP) works better with sklearn models
- Faster training and inference (seconds vs. hours)

### Baseline Model
- Individual Logistic Regression with TF-IDF features only

---

## Slide 9: Feature Selection

### Pipeline 1: TF-IDF + SVD (300 dimensions)
- **TF-IDF:** 10,000 vocabulary, unigrams + bigrams, sublinear TF scaling, min_df=3, max_df=0.95
- **SVD:** TruncatedSVD reduces 10K sparse features to 300 dense dimensions
- **Captures:** Lexical patterns, word frequency distributions, term importance

### Pipeline 2: GloVe Embeddings (100 dimensions)
- **Model:** glove-wiki-gigaword-100 (pre-trained on Wikipedia + Gigaword)
- **Method:** Average word embedding across all tokens in the article
- **Captures:** Semantic meaning, word relationships, context that TF-IDF misses

### Pipeline 3: Stylometric Features (15 dimensions)
| Feature | What It Measures |
|---------|-----------------|
| word_count | Article length |
| avg_word_length | Vocabulary sophistication |
| sentence_count | Structure complexity |
| avg_sentence_length | Writing style |
| vocab_richness (TTR) | Lexical diversity |
| exclamation_ratio | Sensationalism |
| question_ratio | Engagement bait |
| uppercase_ratio | Emphasis/shouting |
| sentiment_compound | VADER sentiment polarity |
| sentiment_pos/neg/neu | Sentiment breakdown |
| flesch_reading_ease | Readability level |
| automated_readability | Grade level |
| hapax_legomena_ratio | Rare word usage |

### Domain Knowledge
- Fake news tends to: use more exclamation marks, have higher sentiment, lower readability, more uppercase
- Real news tends to: be more neutral in sentiment, have consistent sentence structure, higher vocabulary richness

---

## Slide 10: Evaluation Metrics

| Metric | Formula | Why Used |
|--------|---------|----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness; simple but can be misleading with class imbalance |
| **Precision** | TP/(TP+FP) | Of all "Fake" predictions, how many are actually fake? Reduces false accusations |
| **Recall** | TP/(TP+FN) | Of all actual fake news, how many did we catch? Reduces missed misinformation |
| **F1 Score** | 2*(P*R)/(P+R) | Harmonic mean of Precision & Recall; primary metric for balanced evaluation |
| **AUC-ROC** | Area under ROC curve | Threshold-independent performance; measures discriminative ability |

### Additional Evaluation Metrics (Unique to TruthLens)
| Metric | Purpose |
|--------|---------|
| **Cross-Dataset F1** | Train on ISOT, test on LIAR (and vice versa) - measures generalization |
| **Entity Bias Probe** | Accuracy of a simple model using only entity names - measures if model relies on shortcuts |
| **Confusion Matrix** | Visual breakdown of TP/TN/FP/FN per model |
| **ROC Curves** | Per-model ROC curves with AUC scores |

### Why F1 over Accuracy?
- Classes are approximately balanced but not perfectly
- In fake news detection, both false positives (censoring real news) and false negatives (missing fake news) have real-world consequences
- F1 penalizes models that sacrifice one for the other

---

## Slide 11: Tools & Technologies

| Category | Tool | Version | Purpose |
|----------|------|---------|---------|
| **Language** | Python | 3.13 | Primary development language |
| **ML Framework** | scikit-learn | Latest | Models, GridSearchCV, metrics, pipelines |
| **NLP** | spaCy | en_core_web_sm | NER, lemmatization, POS tagging |
| **NLP** | NLTK | Latest | Stopword lists |
| **Embeddings** | gensim | Latest | GloVe word vector loading |
| **Sentiment** | vaderSentiment | Latest | Compound/pos/neg/neu sentiment scores |
| **Readability** | textstat | Latest | Flesch reading ease, ARI |
| **Explainability** | LIME | Latest | Local word-level explanations |
| **Explainability** | SHAP | Latest | Global feature importance (TreeExplainer, LinearExplainer) |
| **Visualization** | matplotlib + seaborn | Latest | Confusion matrices, ROC curves, SHAP plots |
| **Backend** | FastAPI + Uvicorn | Latest | REST API for web app |
| **Frontend** | React 19 + Vite 8 | Latest | Web UI with animations |
| **OCR** | EasyOCR | Latest | Image text extraction (GPU accelerated) |
| **PDF** | PyPDF2 | Latest | PDF text extraction |
| **Fact-Checking** | Wikipedia API, DuckDuckGo Search, Nominatim OSM | Latest | RAG-based verification |
| **Platform** | Windows / macOS | - | Development |
| **IDE** | VS Code / Claude Code | - | Development environment |

---

## Slide 12: Initial Implementation

### Completed Phases
1. **Data Loading** - ISOT (44,898 articles) and LIAR (12,836 statements) loaded successfully with automatic download from Kaggle/HuggingFace
2. **Preprocessing** - Full 6-step pipeline executed with entity masking; processed data cached (ISOT: 371MB, LIAR: 4.9MB)
3. **Feature Engineering** - 415-dimensional hybrid vectors generated (TF-IDF 300d + GloVe 100d + Stylometric 15d)
4. **Model Training** - 4 models trained with GridSearchCV:
   - SVM: Best params C=10, kernel=rbf
   - LR: Best params C=1, solver=lbfgs
   - RF: Best params n_estimators=300, max_depth=20
   - Stacking Ensemble: LR meta-learner combining all three
5. **Evaluation** - In-domain stratified CV, cross-dataset transfer, bias probing all completed
6. **Explainability** - LIME HTML reports generated for 3 test samples; SHAP summary plot generated

### Saved Artifacts
- 6 model files in `models/` directory (ensemble.joblib = 43.8MB)
- Processed datasets cached in `data/processed/`
- Results report + plots in `results/`

---

## Slide 13: Initial Results

### In-Domain Performance (ISOT Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| SVM | 0.9962 | 0.9953 | 0.9974 | 0.9964 | 0.9999 |
| Logistic Regression | 0.9965 | 0.9955 | 0.9979 | 0.9967 | 0.9999 |
| Random Forest | 0.9759 | 0.9730 | 0.9813 | 0.9771 | 0.9969 |
| **Stacking Ensemble** | **0.9970** | **0.9968** | **0.9974** | **0.9971** | **0.9999** |

### Cross-Dataset Generalization

| Train Dataset | Test Dataset | Model | F1 Score | AUC-ROC |
|--------------|-------------|-------|----------|---------|
| ISOT | LIAR | Ensemble | 0.4960 | 0.5328 |
| LIAR | ISOT | Ensemble | 0.1980 | 0.5912 |

### Bias Probe Results
| Probe Type | Accuracy | Status |
|------------|----------|--------|
| Entity Bias | 0.8930 | BIAS DETECTED (threshold: 0.60) |
| Length Bias | 0.4953 | OK (below threshold) |

### Key Observations
- Ensemble outperforms all individual models in-domain
- **Critical gap:** 99.7% F1 in-domain vs 49.6% cross-dataset reveals overfitting to dataset-specific patterns
- Entity bias at 89.3% means entity masking is not aggressive enough - the model can still predict using entity-related features
- Length bias is not an issue (49.5% = random chance)

---

## Slide 14: Challenges Faced

1. **Cross-Dataset Generalization Failure**
   - Models trained on ISOT (full articles) fail on LIAR (short claims)
   - Root cause: fundamentally different text distributions (length, style, vocabulary)
   - The model learns "Reuters writing style = real" rather than content deception patterns

2. **Persistent Entity Bias Despite Masking**
   - spaCy's `en_core_web_sm` model misses many entities
   - ISOT real articles start with "CITY (Reuters) -" byline which is a dead giveaway
   - News source names embedded in article text leak through feature extraction

3. **Domain Mismatch Between Datasets**
   - ISOT: 500-3000 word full articles from 2016-2017
   - LIAR: 15-30 word political claims from PolitiFact
   - TF-IDF vocabularies are completely different between datasets

4. **Large Model & Data Sizes**
   - GloVe model download: 128MB on first run
   - Processed ISOT data: 371MB
   - Ensemble model: 43.8MB
   - SVM training time: ~190 seconds with GridSearchCV

5. **Explainability Complexity**
   - SHAP KernelExplainer is very slow on large feature spaces (415d)
   - LIME explanations require wrapping the full pipeline as a callable function

---

## Slide 15: Plan for Next Phase

### Model Improvement
- [ ] Improve entity masking (upgrade spaCy model, add regex supplementary masking)
- [ ] Strip Reuters byline patterns from ISOT data
- [ ] Train on combined ISOT+LIAR dataset for better generalization
- [ ] Add domain adaptation features (weight stylometric features higher for cross-dataset)
- [ ] Reduce feature dimensionality for faster training (SVD 300 -> 150)

### Web Application
- [ ] Fix all identified backend bugs (unreachable code, cache collisions, CORS)
- [ ] Improve RAG fact-checking pipeline reliability
- [ ] Add proper error handling and logging
- [ ] Add file upload size limits and input validation

### Documentation & Report
- [ ] Create Jupyter Notebook with full pipeline walkthrough
- [ ] Write project report with methodology, results, analysis
- [ ] Prepare final presentation (Phase 2)
- [ ] Push clean codebase to GitHub

---

## Slide 16: Deliverables

| # | Deliverable | Status | Location |
|---|------------|--------|----------|
| 1 | Dataset files (CSV/TSV) | Done | `data/raw/isot/`, `data/raw/liar/` |
| 2 | Jupyter Notebook / Python Pipeline | Done (Python scripts) | `src/*.py`, `main.py` |
|   | - Data loading | Done | `src/data_loader.py` |
|   | - Preprocessing | Done | `src/preprocessor.py` |
|   | - At least 1 model trained | Done (4 models) | `src/model_trainer.py` |
| 3 | GitHub Repository | In Progress | To be pushed |
| 4 | Block Diagram Image | Done | See Slide 6 architecture |

---

## Slide 17: References

1. Ahmed, H., Traore, I., & Saad, S. (2017). "Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques." ISDDC 2017, Springer. *(ISOT dataset source)*
2. Wang, W. Y. (2017). "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection." ACL 2017. *(LIAR dataset source)*
3. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). "Fake News Detection on Social Media: A Data Mining Perspective." ACM SIGKDD Explorations, 19(1), 22-36.
4. Vosoughi, S., Roy, D., & Aral, S. (2018). "The spread of true and false news online." Science, 359(6380), 1146-1151.
5. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier." KDD 2016. *(LIME)*
6. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS 2017. *(SHAP)*
7. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." JMLR, 12, 2825-2830.
8. Pennington, J., Socher, R., & Manning, C. D. (2014). "GloVe: Global Vectors for Word Representation." EMNLP 2014.
