# Gamma AI Prompts — TruthLens Phase 2 Presentation

Use each prompt separately in Gamma AI. Together they produce a 17-slide professional presentation.

---

## PROMPT 1 OF 3 (Slides 1-6: Introduction, Data & Models)

```
Create a professional, modern presentation (6 slides) for a college ML Mini Project Phase-2 implementation review. Use a clean dark/blue theme. Keep slides concise with bullet points, tables, and diagrams — no walls of text.

Slide 1 — Title Slide
Title: "TruthLens: Debiased, Explainable Fake News Detection Using Hybrid Feature Engineering and Stacking Ensemble"
Subtitle: ML Mini Project — Phase 2 Implementation
Team: Riddhi Patil (52), Prashant Jha (54), Aditya Soni (57), Riwan Pereira (65)
Guide: Dr. Joanne Gomes
Institute: St. Francis Institute of Technology, Mumbai
Course: C403 — Machine Learning Mini Project

Slide 2 — Problem Recap
Heading: "Problem Recap"
Two sections:
Problem: Misinformation spreads 6x faster than factual news online. Existing ML detectors learn dataset shortcuts (e.g. "Reuters = Real") instead of genuine deception patterns, causing them to fail on unseen data.
Objectives (bullet list):
- Address 7 critical mistakes in existing fake news models
- Achieve 99%+ in-domain accuracy with transparent generalization reporting
- Provide explainable predictions via LIME and SHAP
- Detect and measure dataset bias via bias probing
- Interactive Gradio web demo for real-time verification

Slide 3 — Dataset (Final)
Heading: "Dataset Overview"
Table with columns: Dataset, Type, Samples, Fake, Real, Avg Length
Row 1: ISOT, Full articles, 44898, 23481 (52%), 21417 (48%), 500-3000 words
Row 2: LIAR, Short claims, 12836, 6426 (50%), 6410 (50%), 15-30 words
Row 3: Combined, Merged, 57734, 29907, 27827, Mixed
Below table:
- Feature vector: 167 dimensions (TF-IDF SVD 150d + Stylometric 17d)
- Train/Test split: 80/20 stratified, random_state=42
- Primary training data: ISOT (~35918 train, ~8980 test)

Slide 4 — Data Preprocessing
Heading: "Data Preprocessing Pipeline"
Show a numbered pipeline flow (6 steps):
1. Byline Stripping — Remove Reuters/AP/AFP bylines to prevent source leakage
2. Text Cleaning — Remove URLs, HTML, emails, special chars; truncate to 5000 chars
3. Entity Masking (spaCy NER) — Replace PERSON, ORG, LOC, DATE with tokens like [PERSON], [ORG]
4. Source Masking (Regex) — 51 news source names masked with [SOURCE]
5. Stopword Removal — NLTK English stopwords
6. Lemmatization — spaCy lemmatizer
Additional info as small bullets:
- Missing values: NaN filled or rows dropped; short texts removed
- Label encoding: Real=0, Fake=1; LIAR 6-class mapped to binary
- StandardScaler on full feature vector (fitted on train only)
- Batch processing with spacy.nlp.pipe() for 3-5x speedup

Slide 5 — Feature Engineering
Heading: "Hybrid Feature Engineering"
Show three parallel pipelines merging into one vector:
Pipeline A — Lexical: Masked text → TF-IDF (10K vocab, bigrams) → TruncatedSVD → 150 dims
Pipeline C — Stylometric: Raw text → 17 handcrafted features → 17 dims
Total: 167-dim dense vector
Then a compact table of key stylometric features (pick 8-10):
word_count, avg_word_length, vocabulary_richness, capital_ratio, exclamation_rate, sentiment_compound, flesch_reading_ease, burstiness, zipf_coefficient
Add a bullet: "Fake news: more exclamation marks, higher sentiment, lower readability. Real news: neutral tone, higher vocabulary richness."

Slide 6 — Models Implemented
Heading: "Models Implemented (4 Models)"
Show 4 cards or boxes:
1. SVM (LinearSVC + CalibratedClassifierCV) — Margin-based, O(n) training, handles high dimensions
2. Logistic Regression — Interpretable, fast (<1s), calibrated probabilities
3. Random Forest — Captures non-linear interactions, built-in feature importance
4. Stacking Ensemble (Meta-LR) — Combines all 3 via StackingClassifier with LR meta-learner using predict_proba; 5-fold CV stacking
Add a note: "Ensemble learns optimal weights per model — better than naive majority voting"
```

---

## PROMPT 2 OF 3 (Slides 7-12: Tuning, Evaluation & Analysis)

```
Create a professional, modern presentation (6 slides) continuing a college ML Mini Project Phase-2 review. Same clean dark/blue theme as Part 1. Keep slides concise — use tables, comparison charts, and bullet points.

Slide 7 — Hyperparameter Tuning
Heading: "Hyperparameter Tuning — GridSearchCV"
Method: GridSearchCV, Stratified 5-Fold CV, scoring=F1, class_weight='balanced'
Three mini-tables side by side or stacked:

SVM: C searched [0.01, 0.1, 1, 10] → Best C=0.01 | CV F1=0.9920 | Time: 8s
LR: C searched [0.01, 0.1, 1, 10], penalty=l2, solver=lbfgs → Best C=0.1 | CV F1=0.9917 | Time: <1s
RF: n_estimators [100,300,500], max_depth [10,20,None], min_samples_split [2,5] → Best: n=100, depth=20, split=2 | CV F1=0.9702 | Time: 650s

Insight bullet: "Models selected stronger regularization (low C, bounded depth) — evidence of active overfitting prevention on 167d feature space"

Slide 8 — Evaluation Metrics & Model Comparison
Heading: "Model Comparison — Test Set Results"
Main table:
| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
| SVM | 0.9935 | 0.9955 | 0.9902 | 0.9929 | 0.9997 |
| Logistic Regression | 0.9932 | 0.9947 | 0.9905 | 0.9926 | 0.9997 |
| Random Forest | 0.9748 | 0.9785 | 0.9662 | 0.9723 | 0.9963 |
| Ensemble | 0.9931 | 0.9944 | 0.9905 | 0.9924 | 0.9997 |

Below: "5-Fold CV: F1 = 0.9920 ± 0.0012 | AUC = 0.9995 ± 0.0001"
Highlight: Best individual = SVM (F1=0.9929), Worst = RF (F1=0.9723), Ensemble selected for robustness

Slide 9 — Best Model Selection
Heading: "Best Model: Stacking Ensemble"
Why selected (bullet list):
- Combines SVM's margin optimization + LR's calibration + RF's non-linearity
- Meta-LR learns optimal combination weights from probability outputs
- AUC-ROC = 0.9997 — highest discriminative ability
- Well-calibrated probabilities (verified via calibration curve)
- Compatible with LIME and SHAP explainability
Note: "SVM has marginally higher F1 (0.9929 vs 0.9924), but Ensemble is more robust across confidence thresholds and resistant to individual model failures."

Slide 10 — Error Analysis
Heading: "Error Analysis — Where the Model Fails"
Stats: 54 errors out of ~8980 test samples (0.60% error rate)
Two mini-tables:
By Type: False Negatives (Fake→Real): 34 (63%) | False Positives (Real→Fake): 20 (37%)
By Confidence: Uncertain (<0.4): 31 | Confident (0.8+): 14 | Borderline: 9

Show 2 example misclassifications:
Example 1 — False Negative: "BOOM! Trump Drains Obama Swamp..." → Predicted Real (conf: 0.25). Why: Policy content mimics real structure despite sensational headline.
Example 2 — False Positive: "Factbox: Trump's tweets vs CEO comments (Reuters)" → Predicted Fake (conf: 0.99). Why: Informal comparison style resembles sensationalist patterns.

Key insight: "57% of errors are on uncertain predictions — the model knows when it doesn't know."

Slide 11 — Improvement Over Phase 1
Heading: "Phase 1 vs Phase 2"
Comparison table:
| Metric | Phase 1 | Phase 2 | Delta |
| Accuracy | 0.9970 | 0.9931 | -0.004 |
| F1 | 0.9971 | 0.9924 | -0.005 |
| AUC-ROC | 0.9999 | 0.9997 | -0.0002 |

Explain the drop (key message — bold): "Slight metric drop is intentional — Phase 2 introduced debiasing (entity masking, byline stripping, SVD reduction 300→150) that removes shortcut features."
What actually improved (bullet list):
- Bias probing (entity, length, topic bias measurement)
- Systematic error analysis with categorization
- SHAP bias audit for feature transparency
- Ablation study (masked vs unmasked comparison)
- Bug fixes and code quality improvements

Slide 12 — System Integration
Heading: "System Architecture"
Show a clean block diagram:
Raw Text → 3 parallel pipelines:
  Pipeline A: Clean + Entity Mask + Source Mask → TF-IDF (10K) → SVD (150d)
  Pipeline C: Raw Text → 17 Stylometric Features
→ Concatenate → StandardScaler → 167d vector
→ Stacking Ensemble (SVM + LR + RF → Meta-LR) → Prediction + Confidence
→ LIME explanations + SHAP feature importance + Bias Audit

Below: "Deployed as Gradio web app — auto-detects input type: Articles → ML pipeline, Claims → Fact verification, Opinions → flagged as unverifiable"
```

---

## PROMPT 3 OF 3 (Slides 13-17: Demo, Challenges, Future & References)

```
Create a professional, modern presentation (5 slides) concluding a college ML Mini Project Phase-2 review. Same clean dark/blue theme as Parts 1 and 2. Keep slides professional and concise.

Slide 13 — Demo & Results
Heading: "Demo / Results"
Show 2-3 prediction examples as styled cards:
Example 1: Input: "BREAKING!!! Government HIDING the truth about vaccines!!! Share before they delete!!!" → Output: FAKE (94% confidence). Indicators: excessive punctuation, heavy caps, sensationalist tone.
Example 2: Input: "Scientists at NASA discovered evidence of water on Mars using spectroscopic analysis published in Nature." → Output: REAL (87% confidence). Indicators: neutral tone, academic language, specific attribution.
Example 3: Input: "India landed on the moon in 2023" → Output: VERIFIED (85%). Source: Wikipedia + web confirm Chandrayaan-3, Aug 23 2023.

Below: "Generated outputs: confusion matrices, ROC curves, calibration plot, SHAP summary/bar plots, LIME HTML explanations, error analysis CSV, cross-dataset results"

Slide 14 — Challenges Faced
Heading: "Challenges Faced"
Show as 5 challenge cards with brief descriptions:
1. Cross-Dataset Generalization: ISOT→LIAR F1=0.23, LIAR→ISOT F1=0.13. Root cause: 10-100x text length difference. Fix needs transformers (BERT).
2. Persistent Entity Bias: Bias probe accuracy=0.91 (should be ~0.50). spaCy misses entities. ISOT has structural source bias beyond names.
3. Overfitting in Random Forest: Train F1=99.98% vs Test F1=97.23%. Addressed with max_depth=20, max_features=sqrt, SVD compression.
4. Data Imbalance: Handled with class_weight='balanced', F1 as primary metric, stratified splits.
5. GloVe Incompatibility: gensim breaks on Python 3.14+. Workaround: --no-glove flag (167d vs 267d). Minor performance impact.

Slide 15 — Future Work
Heading: "Future Work"
Bullet list:
- Transformer models (BERT/DistilBERT) to close cross-dataset gap
- Improved NER with en_core_web_trf for better entity masking
- Multimodal detection (image + text analysis)
- Indian language support (Hindi/Marathi fake news)
- Browser extension for real-time news checking
- Active learning with user feedback loop
Final paper: methodology, ablation study, limitations, literature comparison

Slide 16 — Deliverables
Heading: "Phase-2 Deliverables"
Checklist style (all checked):
1. Cleaned Dataset — ISOT (332MB) + LIAR (5MB), no missing values, 80/20 stratified split
2. Jupyter Notebook — 21-cell pipeline: preprocessing, feature engineering, 4 ML models, GridSearchCV tuning, model comparison
3. Results & Evaluation — Confusion matrices, ROC curves, calibration plot, comparison table, best model: Stacking Ensemble
4. GitHub Repository — Organized structure (src/, data/, models/, results/), README with architecture diagram
5. System Integration — Gradio web app with auto input-type detection, block diagram
6. Progress Evidence — Debiasing, bias probing, error analysis, ablation study, SHAP audit

Slide 17 — References
Heading: "References"
Two columns:
Left — Datasets & Papers:
- Ahmed et al. (2017) — ISOT dataset, ISDDC Springer
- Wang (2017) — LIAR dataset, ACL 2017
- Vosoughi et al. (2018) — Spread of misinformation, Science
- Ribeiro et al. (2016) — LIME, KDD 2016
- Lundberg & Lee (2017) — SHAP, NeurIPS 2017
- Pennington et al. (2014) — GloVe, EMNLP 2014

Right — Tools:
Python 3.14, scikit-learn 1.8.0, spaCy 3.8.13, NLTK, VADER Sentiment, textstat, LIME, SHAP, matplotlib, seaborn, Gradio, pandas 3.0.2, numpy 2.4.4, joblib, kagglehub, HuggingFace datasets
```
