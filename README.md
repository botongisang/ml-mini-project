# TruthLens — Debiased, Explainable Fake News Detection

> **ML Mini Project** | St. Francis Institute of Technology  
> Guide: Dr. Joanne Gomes

---

## What is TruthLens?

TruthLens is a fake news detection system built to address 7 critical flaws found in existing models — dataset bias, generalization failure, black-box predictions, and more. It combines a hybrid feature engineering pipeline with a stacking ensemble classifier and provides human-readable explanations for every prediction.

---

## Team

| Name | Roll No |
|---|---|
| Riddhi Patil | 52 |
| Prashant Jha | 54 |
| Aditya Soni | 57 |
| Riwan Pereira | 65 |

---

## Repository Structure

```
ml-mini-project/
├── codebase/                        # Full source code
│   └── Ml_MP/
│       ├── TruthLens/               # Core ML pipeline
│       │   ├── src/                 # All modules
│       │   │   ├── feature_engineer.py   ← core algorithm
│       │   │   ├── model_trainer.py      ← stacking ensemble
│       │   │   ├── preprocessor.py
│       │   │   ├── data_loader.py
│       │   │   ├── evaluator.py
│       │   │   ├── explainer.py
│       │   │   ├── fact_checker.py
│       │   │   └── claim_detector.py
│       │   ├── main.py              # Pipeline entry point
│       │   ├── config.py            # All hyperparameters
│       │   └── results/             # Plots, reports, LIME HTMLs
│       └── Truth/                   # Web app
│           ├── backend/             # FastAPI server
│           └── frontend/            # React + Vite UI
│
├── datasets/                        # Raw datasets (gitignored CSVs)
│   ├── ISOT/                        # ~44K full news articles
│   └── LIAR/                        # ~12.8K short political claims
│
├── presentations/                   # Phase-wise submission PDFs
│   ├── ML_Literature_Survey_presentation.pdf
│   ├── ML_Project_Implementation_Phase-1.pdf
│   ├── ML_Project_Implementation_Phase-2.pdf
│   └── ML_Project_Implementation_Phase-3_Final.pdf
│
├── imp code and algo/
│   └── imp code for exp 10/
│       └── feature_engineer.py      # Core hybrid feature pipeline
│
├── report(research paper)/          # Research paper (coming soon)
│
└── research/                        # Plans, phase notes, issues & fixes
```

---

## Core Algorithm — Hybrid Feature Engineering + Stacking Ensemble

The project title says it all: the system works in two stages.

### Stage 1 — Hybrid Feature Engineering (`feature_engineer.py`)

Three complementary feature types are extracted and concatenated into a single 265-dimensional vector:

| Feature Type | Method | Dimensions |
|---|---|---|
| Lexical | TF-IDF (10K vocab, bigrams) → TruncatedSVD | 150d |
| Semantic | GloVe word embeddings (averaged) | 100d |
| Stylometric | Sentiment, readability, punctuation, caps ratio, vocabulary richness | 15d |

```
Raw Text
  ├── Entity Masking ([PERSON], [ORG], [LOC]) ──→ TF-IDF → SVD (150d) ─┐
  ├── Cleaned Text ──────────────────────────────→ GloVe avg (100d) ────┤→ [265d]
  └── Raw Text ─────────────────────────────────→ Stylometric (15d) ────┘
```

### Stage 2 — Stacking Ensemble (`model_trainer.py`)

Three base classifiers are trained in parallel, then a Logistic Regression meta-learner is trained on their out-of-fold predictions:

```
[265d vector] → SVM  ──┐
              → LR   ──┤→ Meta-LR → Final Prediction + Confidence
              → RF   ──┘
```

### Explainability

- **SHAP** — global feature importance across the dataset
- **LIME** — per-prediction word-level explanation (1000 perturbations)
- **SVD Backprojection** — identifies which words drove the TF-IDF component
- **Bias Audit** — source-only, length-only, and topic-only probes to detect shortcut learning

---

## Results

**In-Domain (ISOT 5-Fold CV):**

| Metric | Score |
|---|---|
| Accuracy | 99.27% ± 0.11% |
| F1-Score | 99.20% ± 0.12% |
| AUC-ROC | 99.95% ± 0.01% |

**Cross-Dataset (Train ISOT → Test LIAR):**

| Model | Accuracy | F1 |
|---|---|---|
| SVM | 54.98% | 19.98% |
| Logistic Regression | 55.00% | 21.33% |
| Random Forest | 49.32% | 53.13% |
| Ensemble | 54.82% | 23.45% |

> Cross-dataset drop is expected — ISOT contains 200–2000 word articles; LIAR contains 10–100 word political claims. Use `--combined-training` to improve generalization.

**Bias Probe:**

| Probe | Accuracy | Status |
|---|---|---|
| Entity-only (source names) | 91.45% | BIAS detected |
| Length-only | 54.69% | OK |
| Topic-only | 54.21% | OK |

---

## Quick Start

```bash
# 1. Install dependencies
cd codebase/Ml_MP/TruthLens
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run the full pipeline
python main.py

# 3. Common flags
python main.py --no-glove           # Required on Python 3.14+
python main.py --quick              # Faster training (smaller grids)
python main.py --combined-training --no-glove  # Better cross-dataset generalization
```

---

## Datasets

| Dataset | Description | Size |
|---|---|---|
| [ISOT](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) | Full news articles (Reuters real + scraped fake) | ~44K |
| [LIAR](https://huggingface.co/datasets/liar) | Short political claims with 6-class labels | ~12.8K |

> Dataset CSV files are gitignored due to size. Download ISOT from Kaggle and place `Fake.csv` / `True.csv` in `codebase/Ml_MP/TruthLens/data/raw/isot/`.

---

## Known Limitations

- **Cross-dataset gap** — ISOT and LIAR have fundamentally different text lengths. A transformer (BERT/DistilBERT) would close this gap.
- **GloVe on Python 3.14+** — `gensim` is incompatible; run with `--no-glove` (165d features instead of 265d).
- **Entity masking coverage** — `en_core_web_sm` misses some entities; supplementary regex covers 51 known news sources as fallback.
