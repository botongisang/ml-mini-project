# Gamma AI Prompts - TruthLens Implementation Phase 1

Copy-paste each prompt below into Gamma AI separately to generate the presentation.

---

## PROMPT 1: Slides 1-10

```
Create a professional academic presentation with exactly 10 slides for a Machine Learning college mini project. Use a clean, modern dark theme with blue accents. The project is called "TruthLens" - a fake news detection system.

SLIDE 1 - TITLE SLIDE:
Title: "TruthLens: Debiased, Explainable Fake News Detection Using Hybrid Feature Engineering and Stacking Ensemble"
Students: Riddhi Patil (52), Prashant Jha (54), Aditya Soni (57), Riwan Pereira (65)
Course: C403 - Machine Learning Mini Project
Guide: Dr. Joanne Gomes
Institute: St. Francis Institute of Technology, Mumbai
Add a subtle AI/news-related background image.

SLIDE 2 - PROBLEM STATEMENT (REVISED):
Title: "Problem Statement"
Content:
- Misinformation spreads 6x faster than real news online (MIT, 2018)
- Existing ML fake news detectors achieve high in-domain accuracy but FAIL when tested on different datasets
- Models learn dataset-specific shortcuts (like news source names) instead of genuine deception patterns
- Input: Raw news article text or document (PDF, image, DOCX)
- Output: Binary classification (Fake/Real) + confidence score + LIME/SHAP explanation + bias audit
- Real-world use: Journalist pastes an article, gets instant verdict with explanation of WHY it was flagged
- Aligned with UN SDG 16: Peace, Justice and Strong Institutions
Use an infographic style with icons for Input and Output.

SLIDE 3 - DATASET DETAILS:
Title: "Dataset Details"
Create a comparison table with two columns:
ISOT Fake News Dataset:
- Source: University of Victoria (Kaggle)
- 44,898 full-length news articles
- ~23K fake + ~21K real articles
- Columns: title, text, subject, date
- Average length: 500-3000 words
- Real news from Reuters.com

LIAR Dataset (PolitiFact):
- Source: William Yang Wang, UCSB
- 12,836 short political statements
- 6-class labels mapped to binary (Fake/Real)
- Columns: statement, speaker, party, context, + 5 credit history counts
- Average length: 15-30 words
- From PolitiFact fact-checking website

Combined total: 57,734 samples
Add a visual showing sample data snippets from each dataset.

SLIDE 4 - DATA UNDERSTANDING:
Title: "Data Understanding"
Content:
- Data types: All text data (string), labels converted to binary (0=Fake, 1=Real)
- ISOT columns: title, text, subject, date
- LIAR columns: id, label, statement, subject, speaker, job_title, state_info, party, + credit counts, context
- Missing values: Handled via fillna(""), rows with text < 5 chars dropped
- Class distribution: Roughly balanced (~52% fake, ~48% real in ISOT; ~48% fake, ~52% real in LIAR)
- Key insight: ISOT articles are 10-100x longer than LIAR claims - this creates a domain mismatch challenge
- LIAR label mapping: pants-fire/false/barely-true = Fake; half-true/mostly-true/true = Real
Use a data profiling card layout with stats and small charts.

SLIDE 5 - PREPROCESSING PLAN:
Title: "Preprocessing Pipeline"
Show a 6-step vertical flow diagram:
Step 1: Lowercasing - Normalize all text
Step 2: URL & HTML Removal - Regex removes hyperlinks, HTML tags, emails
Step 3: Entity Masking - spaCy NER replaces PERSON, ORG, GPE, LOC, DATE with tokens like [PERSON], [ORG] to prevent shortcut learning
Step 4: Special Character Removal - Strip non-alphanumeric characters
Step 5: Stopword Removal - NLTK English stopwords filtered out
Step 6: Lemmatization - spaCy reduces words to base forms ("running" -> "run")
Performance note: Batch processing via spacy.nlp.pipe() for 3-5x speedup. Processed data cached to avoid re-computation.
Use a pipeline/flowchart visual style.

SLIDE 6 - SYSTEM BLOCK DIAGRAM:
Title: "System Architecture"
Create a visual flow diagram showing:
Raw Text Input splits into 3 parallel pipelines:
Pipeline 1: Clean + Entity Mask -> TF-IDF (10K vocab) -> TruncatedSVD -> 300 dimensions
Pipeline 2: Clean (no mask) -> GloVe Average Embedding (Wiki-100d) -> 100 dimensions
Pipeline 3: Raw Text -> Stylometric Extraction (sentiment, readability, etc.) -> 15 dimensions
All 3 merge into: 415-dimensional Feature Vector
Feeds into: Stacking Ensemble (SVM + Logistic Regression + Random Forest + Meta-LR)
Output: Prediction (Fake/Real)
Then: LIME + SHAP Explainability -> Bias Audit Report
Use a clean architectural diagram with boxes and arrows. Color-code the 3 pipelines differently.

SLIDE 7 - UML DIAGRAMS:
Title: "UML Diagrams"
Show two diagrams side by side:

Use Case Diagram:
- Actor: User (Journalist/Researcher)
- Use cases: Submit Text, Upload Document, View Prediction, View Explanation, View Bias Audit
- System boundary: TruthLens System containing Preprocess, Extract Features, Predict, Explain, Audit Bias

Activity Diagram:
Start -> Load Data -> Check Cache -> [Cached?] -> Yes: Load Processed / No: Clean + Mask + Lemmatize -> Save Cache -> Extract Features (TF-IDF + GloVe + Stylometric) -> Concatenate 415d Vector -> Train Models (GridSearchCV) -> Train Stacking Ensemble -> Evaluate (In-domain + Cross-dataset + Bias Probe) -> Generate LIME/SHAP Explanations -> Output Report -> End
Use standard UML notation with proper shapes.

SLIDE 8 - ALGORITHM SELECTION:
Title: "Algorithm Selection"
Create a table/cards for 4 algorithms:

1. SVM (Support Vector Machine)
   - Role: Base Model 1
   - Why: Best for high-dimensional text; maximum margin classifier
   - Kernel: RBF, C=10

2. Logistic Regression
   - Role: Base Model 2
   - Why: Interpretable, fast, calibrated probabilities
   - Solver: L-BFGS, C=1

3. Random Forest
   - Role: Base Model 3
   - Why: Captures non-linear relationships; handles feature interactions
   - Trees: 300, max_depth=20

4. Stacking Ensemble (Meta-LR)
   - Role: Final Model
   - Why: Learns optimal combination of all 3 base models
   - Meta-learner: Logistic Regression on base model probabilities

Baseline: Individual LR with TF-IDF only
Why not deep learning: Dataset size (45K) doesn't justify transformer overhead; traditional ML + good features achieves comparable performance with better explainability.

SLIDE 9 - FEATURE SELECTION:
Title: "Feature Engineering & Selection"
Show 3 feature groups with details:

Group 1 - Lexical Features (TF-IDF + SVD): 300 dimensions
- 10,000 word vocabulary, unigrams + bigrams
- Sublinear TF scaling, min_df=3, max_df=0.95
- TruncatedSVD reduces to 300 dense dimensions
- Captures: word frequency patterns, term importance

Group 2 - Semantic Features (GloVe): 100 dimensions
- Pre-trained glove-wiki-gigaword-100
- Average word embedding per article
- Captures: word meaning, relationships, context

Group 3 - Stylometric Features: 15 dimensions
- word_count, avg_word_length, sentence_count, avg_sentence_length
- vocab_richness (type-token ratio), hapax_legomena_ratio
- exclamation_ratio, question_ratio, uppercase_ratio
- VADER sentiment (compound, positive, negative, neutral)
- Flesch reading ease, Automated Readability Index
- Captures: writing style, tone, complexity

Total: 415 features per article
Domain knowledge: Fake news uses more exclamation marks, higher sentiment, lower readability, more uppercase.

SLIDE 10 - EVALUATION METRICS:
Title: "Evaluation Metrics"
Create a table:

| Metric | Formula Concept | Why Used |
| Accuracy | Correct / Total | Overall correctness |
| Precision | TP / (TP+FP) | "Of predicted fakes, how many are actually fake?" |
| Recall | TP / (TP+FN) | "Of actual fakes, how many did we catch?" |
| F1 Score | Harmonic mean of P & R | Primary metric - balances precision and recall |
| AUC-ROC | Area under ROC curve | Threshold-independent discrimination |

Unique TruthLens metrics:
- Cross-Dataset F1: Train on Dataset A, test on Dataset B (measures generalization)
- Entity Bias Probe: Can a simple model predict using only entity names? (measures shortcut learning)
- Confusion Matrix & ROC Curves: Visual per-model comparison

Why F1 over Accuracy: Both false positives (censoring real news) and false negatives (missing fake news) have serious consequences. F1 penalizes imbalance between precision and recall.
```

---

## PROMPT 2: Slides 11-17

```
Create a professional academic presentation continuing from slide 10. This is slides 11 through 17 for the same ML Mini Project "TruthLens: Debiased, Explainable Fake News Detection." Use the same clean, modern dark theme with blue accents as the first batch.

SLIDE 11 - TOOLS & TECHNOLOGIES:
Title: "Tools & Technologies"
Create a categorized grid/table:

Programming: Python 3.13
ML Framework: scikit-learn (models, GridSearchCV, metrics, preprocessing)
NLP Processing: spaCy (NER + lemmatization), NLTK (stopwords)
Word Embeddings: gensim (GloVe vectors - glove-wiki-gigaword-100)
Sentiment Analysis: VADER Sentiment
Readability: textstat (Flesch, ARI)
Explainability: LIME (local word explanations), SHAP (global feature importance)
Visualization: matplotlib, seaborn
Web Backend: FastAPI + Uvicorn (REST API)
Web Frontend: React 19 + Vite 8 (SPA with Framer Motion animations)
OCR: EasyOCR (GPU-accelerated image text extraction)
Document Parsing: PyPDF2 (PDF), python-docx (DOCX)
Fact-Checking RAG: Wikipedia API, DuckDuckGo Search, Nominatim OSM
Platform: Windows / macOS, VS Code
Use technology logos/icons where possible. Group into ML, NLP, Web, and Utils categories.

SLIDE 12 - INITIAL IMPLEMENTATION:
Title: "Initial Implementation Status"
Show a checklist/progress view:

Phase 1 - Data Loading: DONE
- ISOT: 44,898 articles loaded (auto-download from Kaggle)
- LIAR: 12,836 statements loaded (auto-download from HuggingFace)
- Combined: 57,734 total samples

Phase 2 - Preprocessing: DONE
- Full 6-step pipeline executed with entity masking
- Processed data cached (ISOT: 371MB, LIAR: 4.9MB)
- Batch spaCy processing for speed

Phase 3 - Feature Engineering: DONE
- 415-dimensional hybrid vectors generated
- TF-IDF (300d) + GloVe (100d) + Stylometric (15d)

Phase 4 - Model Training: DONE
- 4 models trained via GridSearchCV
- Best SVM: C=10, kernel=rbf
- Best LR: C=1, solver=lbfgs
- Best RF: n_estimators=300, max_depth=20
- Stacking Ensemble: LR meta-learner

Phase 5 - Evaluation: DONE
- In-domain CV, cross-dataset transfer, bias probing complete

Phase 6 - Explainability: DONE
- 3 LIME HTML reports generated
- SHAP summary plot generated

Artifacts: 6 model files saved, results report + plots generated.
Use green checkmarks and progress bar visuals.

SLIDE 13 - INITIAL RESULTS:
Title: "Initial Results"
Create a clear results table:

In-Domain Performance (ISOT Test Set):
| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
| SVM | 99.62% | 99.53% | 99.74% | 99.64% | 99.99% |
| Logistic Regression | 99.65% | 99.55% | 99.79% | 99.67% | 99.99% |
| Random Forest | 97.59% | 97.30% | 98.13% | 97.71% | 99.69% |
| Stacking Ensemble | 99.70% | 99.68% | 99.74% | 99.71% | 99.99% |

Cross-Dataset Generalization:
| Train -> Test | F1 Score | AUC-ROC |
| ISOT -> LIAR | 49.60% | 53.28% |
| LIAR -> ISOT | 19.80% | 59.12% |

Bias Probe:
| Probe | Accuracy | Status |
| Entity Bias | 89.30% | BIAS DETECTED |
| Length Bias | 49.53% | OK |

Key Observation: Ensemble achieves 99.71% F1 in-domain but only 49.60% cross-dataset. This 50-point gap reveals the model overfits to dataset-specific patterns. Entity bias at 89.3% confirms the model can still predict using entity names despite masking.
Highlight the cross-dataset gap visually with a comparison bar chart or before/after visual. Use red/warning colors for the problematic numbers.

SLIDE 14 - CHALLENGES FACED:
Title: "Challenges Faced"
Show 5 challenge cards:

1. Cross-Dataset Generalization Failure
- 99.7% in-domain vs 49.6% cross-dataset F1
- Root cause: ISOT articles are 10-100x longer than LIAR claims
- TF-IDF vocabularies are completely different between datasets
- Model learns "Reuters style = real" not "deception patterns"

2. Persistent Entity Bias (89.3%)
- spaCy en_core_web_sm misses many entities
- ISOT real articles start with "CITY (Reuters) -" byline = dead giveaway
- News source names leak through feature extraction

3. Domain Mismatch
- ISOT: 500-3000 word news articles from 2016-2017
- LIAR: 15-30 word political claims from PolitiFact
- Fundamentally different text distributions

4. Computational Costs
- GloVe download: 128MB first run
- Processed ISOT: 371MB disk space
- SVM GridSearchCV: ~190 seconds training
- SHAP KernelExplainer: Very slow on 415-dimensional features

5. Explainability Integration
- LIME requires wrapping entire pipeline as single callable
- SHAP explainer selection varies by model type (Tree vs Linear vs Kernel)
- Balancing explanation quality vs computation time

Use warning/caution icon style. Make it visually distinct from success slides.

SLIDE 15 - PLAN FOR NEXT PHASE:
Title: "Plan for Next Phase"
Show a roadmap with 3 tracks:

Track 1 - Model Improvement:
- Improve entity masking (upgrade spaCy model + regex supplementary masking)
- Strip Reuters byline patterns from ISOT data
- Train on combined ISOT+LIAR dataset for better generalization
- Domain adaptation: weight stylometric features higher for cross-dataset
- Reduce feature dimensionality (SVD 300->150) for speed

Track 2 - Web Application:
- Fix backend bugs (unreachable code, cache collisions, CORS)
- Improve RAG fact-checking reliability
- Add proper error handling, logging, input validation
- File upload size limits and security hardening

Track 3 - Documentation & Submission:
- Create Jupyter Notebook with full pipeline walkthrough
- Write project report (methodology, results, analysis)
- Prepare Phase 2 final presentation
- Clean GitHub repository with proper structure

Use a timeline/roadmap visual with milestones.

SLIDE 16 - DELIVERABLES:
Title: "Deliverables"
Create a deliverable checklist:

1. Dataset Files (CSV/TSV) - COMPLETED
   - ISOT: Fake.csv + True.csv (raw) + isot_processed.csv
   - LIAR: train.tsv + test.tsv + valid.tsv + liar_processed.csv

2. Python Pipeline / Jupyter Notebook - COMPLETED
   - 6 source modules: data_loader, preprocessor, feature_engineer, model_trainer, evaluator, explainer
   - Main orchestrator: main.py with 6-phase pipeline
   - Contains: data loading, preprocessing, 4 trained models

3. GitHub Repository - IN PROGRESS
   - Full codebase with modular structure
   - README with architecture, setup instructions, team credits

4. Block Diagram Image - COMPLETED
   - 3-pipeline architecture (TF-IDF + GloVe + Stylometric -> Stacking Ensemble)
   - Included in README and presentation

Use status badges (green for done, yellow for in-progress).

SLIDE 17 - REFERENCES:
Title: "References"
List these references in IEEE format:

[1] H. Ahmed, I. Traore, and S. Saad, "Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques," in ISDDC 2017, Springer, 2017. (ISOT Dataset)
[2] W. Y. Wang, "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection," in Proc. ACL, 2017. (LIAR Dataset)
[3] K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, "Fake News Detection on Social Media: A Data Mining Perspective," ACM SIGKDD Explorations, vol. 19, no. 1, pp. 22-36, 2017.
[4] S. Vosoughi, D. Roy, and S. Aral, "The spread of true and false news online," Science, vol. 359, no. 6380, pp. 1146-1151, 2018.
[5] M. T. Ribeiro, S. Singh, and C. Guestrin, "Why Should I Trust You?: Explaining the Predictions of Any Classifier," in Proc. KDD, 2016. (LIME)
[6] S. M. Lundberg and S. I. Lee, "A Unified Approach to Interpreting Model Predictions," in Proc. NeurIPS, 2017. (SHAP)
[7] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, vol. 12, pp. 2825-2830, 2011.
[8] J. Pennington, R. Socher, and C. D. Manning, "GloVe: Global Vectors for Word Representation," in Proc. EMNLP, 2014.
Use a clean, compact reference list format. Add "Thank You" at the bottom of the slide.
```

---

## Instructions for Using These Prompts

1. Go to [gamma.app](https://gamma.app)
2. Click "Create new" -> "Paste in text"
3. Copy **PROMPT 1** (everything inside the first code block) and paste it
4. Let Gamma generate slides 1-10
5. Review and adjust as needed
6. Start a **new presentation** or continue
7. Copy **PROMPT 2** (everything inside the second code block) and paste it
8. Let Gamma generate slides 11-17
9. Merge both decks or present sequentially

**Theme suggestions for Gamma:** Dark mode, blue/purple accent colors, "Professional" or "Tech" template category.
