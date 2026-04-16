"""
TruthLens Configuration
=======================
All hyperparameters, paths, and feature settings in one place.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Create all directories
for _dir in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ──────────────────────────────────────────────
# General
# ──────────────────────────────────────────────
RANDOM_STATE = 42
USE_COMBINED_TRAINING = False  # Set True via --combined-training CLI flag
TEST_SIZE = 0.2
CV_FOLDS = 5

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
ENABLE_ENTITY_MASKING = True       # Replace PERSON/ORG/LOC with tokens
SPACY_MODEL = "en_core_web_sm"     # Lightweight spaCy model
MAX_TEXT_LENGTH = 5000             # Truncate articles longer than this (chars)

# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────
# TF-IDF
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)        # Unigrams + bigrams
TFIDF_SUBLINEAR_TF = True         # Log-scale term frequency

# Dimensionality Reduction (TruncatedSVD on TF-IDF)
SVD_COMPONENTS = 150               # Reduce TF-IDF → 150 dims (faster, retains ~85% variance)

# GloVe Embeddings
ENABLE_GLOVE = True
GLOVE_MODEL_NAME = "glove-wiki-gigaword-100"  # 100-dim via gensim
GLOVE_DIM = 100

# Stylometric Features
ENABLE_STYLOMETRIC = True

# ──────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────
# Quick mode uses smaller grids for fast experimentation.
# Set via --quick CLI flag. Default is False (full grids).
QUICK_MODE = False


def get_svm_params():
    """SVM hyperparameter grid — evaluated at call time so --quick flag works."""
    if QUICK_MODE:
        return {
            "estimator__C": [0.1, 1],
            "estimator__class_weight": ["balanced"],
            "estimator__max_iter": [2000],
        }
    return {
        "estimator__C": [0.01, 0.1, 1, 10],
        "estimator__class_weight": ["balanced"],
        "estimator__max_iter": [2000],
    }


def get_lr_params():
    """Logistic Regression hyperparameter grid — evaluated at call time."""
    if QUICK_MODE:
        return {
            "C": [1],
            "penalty": ["l2"],
            "max_iter": [2000],
            "class_weight": ["balanced"],
            "solver": ["lbfgs"],
        }
    return {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "max_iter": [2000],
        "class_weight": ["balanced"],
        "solver": ["lbfgs"],
    }


def get_rf_params():
    """Random Forest hyperparameter grid — evaluated at call time."""
    if QUICK_MODE:
        return {
            "n_estimators": [200],
            "max_depth": [20],
            "max_features": ["sqrt"],
            "class_weight": ["balanced"],
        }
    return {
        "n_estimators": [100, 300, 500],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt"],
        "class_weight": ["balanced"],
    }

# ──────────────────────────────────────────────
# Explainability
# ──────────────────────────────────────────────
LIME_NUM_FEATURES = 15             # Top N words in LIME explanations
LIME_NUM_SAMPLES = 1000            # Perturbation samples for LIME
SHAP_MAX_SAMPLES = 100             # Background samples for SHAP KernelExplainer
SHAP_NUM_EXPLAIN = 50              # Number of instances to explain with SHAP

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
BIAS_PROBE_THRESHOLD = 0.60        # Source-only accuracy above this = bias alert
