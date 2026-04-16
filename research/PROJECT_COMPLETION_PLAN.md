# TruthLens - Project Completion Plan

## Current Status (After Phase 1 Implementation)
- Pipeline fully functional: data loading, preprocessing, feature engineering, training, evaluation, explainability
- In-domain F1: 99.71% (Ensemble on ISOT)
- Cross-dataset F1: 49.60% (ISOT -> LIAR) -- needs improvement
- Entity bias: 89.3% -- needs improvement
- Web app (Truth): FastAPI backend + React frontend -- has bugs (now fixed)
- Bugs fixed: 11 bugs across pipeline + backend

---

## Phase 1: Clear Cached Data (Force Re-preprocessing)
**Goal:** Ensure new preprocessing improvements (Reuters byline stripping, news source masking) take effect.

- [ ] Delete `Ml_MP/TruthLens/data/processed/isot_processed.csv`
- [ ] Delete `Ml_MP/TruthLens/data/processed/liar_processed.csv`
- [ ] Verify no stale model files by checking timestamps in `models/`

**Checkpoint:** Processed data directory is empty, forcing fresh preprocessing on next run.

---

## Phase 2: Re-run Full Pipeline with Fixes
**Goal:** Regenerate all models and results with the improved preprocessing + config.

- [ ] Run `python main.py` from the TruthLens directory
- [ ] Verify Phase 1 (data loading) completes -- expect ~57K samples
- [ ] Verify Phase 2 (preprocessing) runs fresh with Reuters byline stripping and source masking
- [ ] Verify Phase 3 (features) produces ~265d vectors (was 415d, now 150 SVD + 100 GloVe + 15 stylometric)
- [ ] Verify Phase 4 (training) completes with 5-fold CV (was 3-fold)
- [ ] Verify Phase 5 (evaluation) runs in-domain + cross-dataset + bias probe
- [ ] Verify Phase 6 (explainability) generates LIME + SHAP outputs

**Checkpoint:** New `test_results.csv` and `results_report.txt` generated. Compare with previous results.

---

## Phase 3: Evaluate Improvements
**Goal:** Measure the impact of all fixes.

- [ ] Record new in-domain F1 (expected: ~99.5%, slight drop due to reduced features is OK)
- [ ] Record new cross-dataset F1 (target: >55%, up from 49.6%)
- [ ] Record new entity bias probe (target: <75%, down from 89.3%)
- [ ] Compare confusion matrices before/after
- [ ] Document improvements in a before/after table

**Checkpoint:** Clear numerical evidence that fixes improved generalization.

---

## Phase 4: Train on Combined Dataset (Optional Experiment)
**Goal:** Test if training on ISOT+LIAR combined improves cross-dataset transfer.

- [ ] In `main.py` line 353, change `processed_datasets["isot"]` to `processed_datasets["combined"]`
- [ ] Re-run pipeline
- [ ] Record combined-trained model metrics
- [ ] Compare ISOT-only vs Combined training results
- [ ] Choose the better approach for final submission

**Checkpoint:** Decision made on training dataset strategy with data to back it up.

---

## Phase 5: Create Jupyter Notebook Deliverable
**Goal:** Produce the required notebook for submission.

- [ ] Create `Ml_MP/TruthLens/TruthLens_Pipeline.ipynb`
- [ ] Section 1: Imports & Configuration (show all dependencies)
- [ ] Section 2: Data Loading (show dataset shapes, .head(), .describe())
- [ ] Section 3: Preprocessing Demo (show before/after entity masking on 3 examples)
- [ ] Section 4: Feature Engineering (explain 3 pipelines, show feature dimensions)
- [ ] Section 5: Model Training (GridSearchCV output, best params)
- [ ] Section 6: Evaluation (inline confusion matrix, ROC curves, results table)
- [ ] Section 7: Explainability (LIME example, SHAP summary)
- [ ] Add markdown cells explaining the "7 critical mistakes" approach

**Checkpoint:** Notebook runs end-to-end without errors.

---

## Phase 6: Generate Presentation with Gamma AI
**Goal:** Create the Phase-1 Implementation presentation.

- [ ] Open `gamma_prompts.md`
- [ ] Copy Prompt 1 into Gamma AI -> Generate slides 1-10
- [ ] Review and adjust auto-generated content
- [ ] Copy Prompt 2 into Gamma AI -> Generate slides 11-17
- [ ] Review and adjust (especially the results table on slide 13)
- [ ] Export as PDF/PPTX
- [ ] Save to `Ml_MP/` directory

**Checkpoint:** 17-slide presentation ready with actual project data.

---

## Phase 7: System Block Diagram Image
**Goal:** Create a clean block diagram image (required deliverable).

- [ ] Use draw.io, Figma, or Canva to create the architecture diagram from Slide 6
- [ ] Show 3 parallel feature pipelines merging into 265d vector
- [ ] Show stacking ensemble with SVM + LR + RF + Meta-LR
- [ ] Show LIME/SHAP explainability output
- [ ] Export as PNG (high resolution)
- [ ] Save to `Ml_MP/TruthLens/` and include in README

**Checkpoint:** Block diagram image file exists and matches the code architecture.

---

## Phase 8: Frontend Polish
**Goal:** Ensure the web app demo works cleanly.

- [ ] Start backend: `cd Ml_MP && uvicorn Truth.backend.main:app --port 8000`
- [ ] Start frontend: `cd Ml_MP/Truth/frontend && npm run dev`
- [ ] Test with a real Reuters article (should predict "Real News")
- [ ] Test with a known fake article (should predict "Fake News")
- [ ] Test with a short LIAR-style claim (verify it handles short text)
- [ ] Test file upload (PDF, image)
- [ ] Verify RAG checks return reasonable results
- [ ] Screenshot the UI for the presentation

**Checkpoint:** End-to-end demo works for all test cases.

---

## Phase 9: GitHub Repository Setup
**Goal:** Create a clean, submittable GitHub repo.

- [ ] Initialize git repo if not exists: `cd Ml_MP && git init`
- [ ] Create `.gitignore` with:
  ```
  __pycache__/
  *.pyc
  .venv/
  venv/
  node_modules/
  data/raw/
  data/processed/
  models/*.joblib
  models/*.pkl
  *.zip
  .env
  ```
- [ ] Stage all source files (NOT data/models/venvs)
- [ ] Write meaningful commit messages
- [ ] Create GitHub repo and push
- [ ] Add README with: project overview, architecture, setup instructions, team credits
- [ ] Verify repo URL works

**Checkpoint:** GitHub link is live, README is presentable.

---

## Phase 10: Update README with Final Results
**Goal:** README reflects the actual final state of the project.

- [ ] Update the results table in README with new numbers
- [ ] Update architecture section if feature dimensions changed (415d -> 265d)
- [ ] Add "Bugs Fixed" or "Improvements" section documenting what was changed
- [ ] Add setup instructions for both TruthLens pipeline and Truth web app
- [ ] Add team member contributions

**Checkpoint:** README is accurate and matches the current code state.

---

## Phase 11: Write Project Report (If Required)
**Goal:** Produce a written report summarizing methodology and results.

- [ ] Problem statement and motivation
- [ ] Literature survey summary (reference the 8 papers from `data.md`)
- [ ] Methodology: preprocessing, feature engineering, model selection
- [ ] Results: in-domain, cross-dataset, bias probe, with tables and plots
- [ ] Discussion: why cross-dataset gap exists, what entity masking achieves
- [ ] Future work: transformer models, multimodal detection, Indian language support
- [ ] References in IEEE format

**Checkpoint:** Report is complete, formatted, and print-ready.

---

## Phase 12: Prepare for Presentation/Viva
**Goal:** Be ready to present and answer questions.

- [ ] Practice the 17-slide presentation (target: 15-20 minutes)
- [ ] Prepare answers for likely questions:
  - "Why is cross-dataset performance low?" -- Different text lengths/styles between ISOT and LIAR
  - "How does entity masking help?" -- Prevents model from using "Reuters" as shortcut for real news
  - "Why stacking over voting?" -- Meta-learner learns optimal weights; voting treats all models equally
  - "What do SHAP values tell you?" -- Which features drive predictions; audit for bias
  - "Why not use BERT/transformers?" -- Dataset size doesn't justify; traditional ML + good features is sufficient; better explainability
- [ ] Have a live demo ready (backend + frontend running)
- [ ] Have the Jupyter notebook open for code walkthrough

**Checkpoint:** Team is confident presenting and answering questions.

---

## Phase 13: Final Submission Checklist
**Goal:** Everything is submitted correctly.

- [ ] Dataset files (CSV/TSV) -- available in `data/raw/`
- [ ] Jupyter Notebook / Python pipeline -- `TruthLens_Pipeline.ipynb` or `main.py`
- [ ] GitHub repository link -- from Phase 9
- [ ] Block diagram image -- from Phase 7
- [ ] Presentation PDF -- from Phase 6
- [ ] Project report (if required) -- from Phase 11
- [ ] All team members' roll numbers on title slide

**Checkpoint:** All deliverables match the requirements from `ML_Project_Implementation_Phase-1.pdf`.

---

## Summary of Changes Made So Far

### Bugs Fixed (11 total)
| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `preprocessor.py` | `os.system()` for spaCy download (injection risk) | Replaced with `subprocess.run()` |
| 2 | `explainer.py` | Generic exception hides SHAP errors | Added traceback logging |
| 3 | `feature_engineer.py` | SVD explained variance not stored | Added `svd_explained_variance` attribute + save/load |
| 4 | `backend/main.py` | Unreachable return in `verify_wikipedia()` | Removed duplicate return |
| 5 | `backend/main.py` | Unreachable return in `verify_address()` | Removed duplicate return |
| 6 | `backend/main.py` | Stop_titles filtering "Delhi", "Mumbai", "India" | Removed from stop list |
| 7 | `backend/main.py` | Cache key uses first 100 chars (collision) | Changed to MD5 hash |
| 8 | `backend/main.py` | Unused GloVe preload wastes 128MB RAM | Removed |
| 9 | `backend/main.py` | Hard-coded confidence 0.99/0.96 ignores ML | Changed to `max(confidence, 0.90/0.85)` |
| 10 | `backend/main.py` | CORS wildcard `*` (security risk) | Restricted to localhost origins |
| 11 | `backend/main.py` | Silent exception in verify functions | Added error logging |

### Overfitting / Generalization Fixes
| # | File | Change | Expected Impact |
|---|------|--------|-----------------|
| 1 | `preprocessor.py` | Reuters/AP/AFP byline stripping | Removes source identity from text |
| 2 | `preprocessor.py` | Supplementary news source regex masking | Catches sources spaCy NER misses |
| 3 | `config.py` | SVD 300 -> 150 dimensions | Faster training, less overfitting to TF-IDF noise |
| 4 | `config.py` | CV_FOLDS 3 -> 5 | More robust cross-validation |
| 5 | `config.py` | RF `max_features="sqrt"` | Prevents RF from using all features (reduces overfitting) |
