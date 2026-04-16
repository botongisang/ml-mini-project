# TruthLens — Methodology Audit

## System Architecture (Current)

```
User Text
  |
  v
[Preprocessing] clean -> entity mask -> lemmatize
  |
  v
[Feature Engineering] TF-IDF+SVD(150d) + GloVe(100d) + Stylometric(15d) = 265d
  |
  v
[ML Ensemble] Stacking(SVM + LR + RF) -> LR meta-learner -> P(Real), P(Fake)
  |
  v
[RAG Fact-Check] Timeline | Wikipedia | OSM Maps | DuckDuckGo Web
  |
  v
[Override Logic] conflict -> Fake | 2+ verified -> Real | else -> ML prediction
  |
  v
[Explainability] Top 3 TF-IDF words highlighted
  |
  v
OUTPUT: prediction + confidence + factual_analysis + highlighted_words
```

---

## What Is Conceptually CORRECT

### 1. Entity Masking for Debiasing
**Method:** Replace PERSON/ORG/LOC/DATE with `[PERSON]`/`[ORG]`/`[LOC]`/`[DATE]` using spaCy NER before TF-IDF.  
**Why it's right:** Without masking, the model learns shortcuts like "Trump" -> Fake, "Reuters" -> Real. Entity masking forces the model to learn content patterns (writing style, claims, language) instead of entity associations. This is a recognized debiasing technique in NLP literature.  
**Evidence it works:** Entity bias probe dropped from near-random to 89.3% (still high due to spaCy NER limitations, but the technique is correct).

### 2. TF-IDF + SVD Pipeline
**Method:** TF-IDF(10k vocab, unigrams+bigrams, sublinear_tf) -> TruncatedSVD(150d).  
**Why it's right:** TF-IDF captures which words are uniquely important in a document vs the corpus. SVD reduces dimensionality while preserving the most informative variance (~85%). sublinear_tf = log(1 + tf) prevents long documents from dominating. This is a standard, well-understood text classification pipeline.

### 3. Stylometric Features
**Features:** word_count, avg_word_length, sentence_count, avg_sentence_length, vocabulary_richness, capital_ratio, exclamation_count, question_marks, digit_ratio, VADER sentiment (compound/pos/neg), Flesch readability, ARI readability.  
**Why it's right:** Fake news has measurable stylistic signatures: more exclamation marks, simpler vocabulary, more emotional language, lower readability scores. These features capture writing STYLE independent of content, which is exactly what you want alongside content-based TF-IDF features.

### 4. Stacking Ensemble with Meta-Learner
**Method:** 3 base models (SVM, LR, RF) each predict on held-out CV folds. Their predictions become features for a Logistic Regression meta-learner that makes the final decision.  
**Why it's right:** Stacking is superior to simple voting because the meta-learner learns WHEN to trust each base model. For example, RF might overfit on long articles but SVM generalizes better — the meta-learner captures this. Using `predict_proba` (not hard labels) preserves uncertainty information.

### 5. Cross-Dataset Evaluation
**Method:** Train on ISOT, test on LIAR (and vice versa).  
**Why it's right:** In-domain accuracy (99.2%) is meaningless if the model only memorized one dataset. Cross-dataset F1 (~50%) reveals the TRUE generalization ability. This is honest, rigorous evaluation.

---

## What Is Conceptually WRONG

### ISSUE 1: Explainability Is Fundamentally Broken

**The claim:** "We use LIME/SHAP for explainability"  
**What actually happens:** Backend picks the top 3 words by raw TF-IDF weight and highlights them.

**Why this is wrong:**

The model predicts using a **265-dimensional SVD-reduced + GloVe + stylometric** feature vector. The TF-IDF words are an INTERMEDIATE representation that gets compressed by SVD. The top TF-IDF words don't necessarily correspond to the SVD dimensions that actually drove the prediction.

Analogy: It's like explaining "why did you choose this house?" by listing the raw materials (bricks, wood, glass) instead of the actual reasons (location, price, size). The raw materials are an input to the house, but they don't explain the decision.

**What should happen:**
- **Option A (correct but slow):** Use LIME at inference time — perturb words, observe prediction changes, report which words actually shift the prediction. The code for this exists in `explainer.py` but is NOT used in the API.
- **Option B (correct and fast):** Since SVD is a linear projection, you CAN trace back which original TF-IDF features contribute most to each SVD component, and then report which words matter. This requires `svd.components_` matrix multiplication.
- **Option C (honest):** Rename from "LIME/SHAP explainability" to "Top Keywords" and acknowledge it's an approximation.

**Impact:** The highlighted words may be completely unrelated to why the model classified something as fake/real. A user sees "cure" highlighted red and thinks "the model detected cure is suspicious" — but the model may have classified it as fake because of stylometric features (too many exclamation marks), not because of the word "cure".

**File:** `Truth/backend/main.py` lines 699-729

---

### ISSUE 2: RAG Override Can Destroy Correct ML Predictions

**The flaw:** The "Master Architect Rule" — if ANY of the 4 RAG checks finds a conflict, the prediction is forced to "Fake News" with 90% confidence, regardless of what the ML model said.

**Why this is wrong:**

The RAG checks have significant false positive rates:
- **Wikipedia:** Searches for capitalized words → finds unrelated article → "myth" in summary → CONFLICT. Example: text mentions "Phoenix" (the city) but Wikipedia returns "Phoenix (mythology)" → "CONFLICT: flagged as mythological"
- **Spatial:** Extracts any capitalized 4+ letter word as a location → queries OSM → gets result in wrong country → CONFLICT
- **Web numeric:** Compares first number in text vs first number in search results without context
- **Temporal month-day regex:** `april\s+(\d{1,2})` on "april 2026" captures "20" (first 2 digits of year) as a day number

A single false positive from ANY check overrides a 99.5% confident ML prediction. This is backwards — the ML model was trained on 44,000+ labeled examples and is highly accurate. The RAG checks are fragile heuristics. The heuristic should not override the trained model on a single unreliable signal.

**What should happen:**
- RAG conflict should LOWER confidence, not force-override the prediction
- Multiple RAG conflicts should be needed to override ML (not just one)
- Weight RAG checks by reliability: Wikipedia entity match > DuckDuckGo keyword overlap
- Keep ML prediction as primary, RAG as supplementary evidence

**File:** `Truth/backend/main.py` lines 677-694

---

### ISSUE 3: Wikipedia Entity Extraction Is Wrong Method

**The flaw:** Extracts Wikipedia search subjects using REGEX on capitalized words:
```python
found_subjects = re.findall(r'\b[A-Z][A-Z\s]*\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
```

**Why this is wrong:**

1. **Already have NER from preprocessing:** The pipeline already runs spaCy NER during entity masking. Those entities are correct (PERSON, ORG, LOC). But instead of reusing them, the RAG module uses a SEPARATE regex-based extraction that is far less accurate.

2. **Regex fails on:**
   - Lowercase text: `"barack obama"` → finds nothing (no capitals)
   - Sentence-initial words: `"The economy is growing"` → extracts "The" as entity
   - Acronyms mixed with text: `"NASA said"` → extracts "NASA" (correct) but also "Said" at sentence start

3. **stop_titles list is a fragile blocklist:** Instead of positively identifying entities (NER), it tries to block non-entities with a growing list of words to exclude. This will always have holes.

**What should happen:**
- Run spaCy NER on the RAG input text (or pass entities from preprocessing)
- Use NER-detected PERSON/ORG/LOC as Wikipedia search subjects
- Fall back to regex only if NER finds nothing

**File:** `Truth/backend/main.py` lines 200-207

---

### ISSUE 4: Web RAG Numeric Matching Is Methodologically Unsound

**The method:**
```python
numbers_in_text = re.findall(r'\b\d+\b', main_claim)
claimed_val = int(numbers_in_text[0])  # First number in claim
# Compare against first numbers in DuckDuckGo results
if actual_val == claimed_val: → VERIFIED
if claimed_val == 0 and actual_val > 0: → CONFLICT
```

**Why this is wrong:**

Numbers without context are meaningless. The method compares the first number found in the claim against the first number found in search results.

- Input: `"7th april 2026"` → claimed_val = 7
- Web result about "7 Wonders of the World" → actual_val = 7 → "VERIFIED: Web consensus confirms the exact count of 7"
- This is a false verification — the numbers are completely unrelated.

Similarly: Input `"0 people died in the earthquake"` → claimed_val = 0. Web result: `"2024 earthquake killed 5"` → actual_val = 5 → CONFLICT. This happens to be correct, but the logic is fragile — it breaks when numbers appear in dates, IDs, or other contexts.

**What should happen:**
- Only do numeric comparison when the number IS the factual claim (statistics, counts, records)
- Need surrounding context to determine if the number is a factual assertion or just a date/ordinal/ID
- Better: don't do number matching at all — it's more harmful than helpful at this level of sophistication

**File:** `Truth/backend/main.py` lines 455-481

---

### ISSUE 5: Web Verification Word-Overlap Threshold Is Flawed

**The method:**
```python
overlap_words = set(main_claim.lower().split()) & set(results[0]['body'].lower().split())
if len(overlap_words) >= 5:
    return FactualCheck(status="verified")
```

**Why this is wrong:**

This counts ALL overlapping words including stopwords ("the", "is", "a", "in", "was", "of"). Any two English sentences of moderate length will share 5+ common words by chance.

Example:
- Claim: `"The president signed the new law in the White House"`  
- Web result (random news): `"The committee voted on the new budget in the Senate"`  
- Overlap: {"the", "in", "the", "new"} → 4 unique but "the" appears multiple times in both...  
- Actually using sets: {"the", "in", "new"} = 3. But with slightly longer text, 5 is easily reached.

A "verification" should mean "the web confirms this specific claim is true." Word overlap doesn't establish this — two articles about completely different topics can share enough common words to trigger false verification.

**What should happen:**
- Remove stopwords before computing overlap
- Use overlap RATIO (overlap / total_unique_words) not absolute count
- Better: use semantic similarity (cosine similarity of TF-IDF or sentence embeddings) instead of word overlap

**File:** `Truth/backend/main.py` lines 483-491

---

### ISSUE 6: ML Failure on Short Text Produces Invalid State

**The problem:** Text under ~10 words often becomes empty after preprocessing (stopword removal + lemmatization strips everything). The backend catches the exception and sets `prediction = "Error"` — a string that is NOT a valid classification label.

**Why this is wrong methodologically:**

A classification system must handle ALL inputs, even degenerate ones. The correct approaches:
1. **Abstain:** Return "Cannot classify — insufficient text" with confidence = 0 (the model honestly admits it can't decide)
2. **Heuristic fallback:** Use the sensational-word heuristic (exists at line 634 but is unreachable when ML is loaded)
3. **RAG-only mode:** If ML can't classify, let the RAG pipeline make the call independently

"Error" is an implementation detail, not a classification. It should never reach the user.

**File:** `Truth/backend/main.py` lines 596-637

---

### ISSUE 7: Confidence Is Not What the UI Claims

**The display:** "50% NEURAL ACCURACY"  
**The actual value:** `predict_proba` output — the model's estimated probability that its prediction is correct.

**Why this is wrong:**

- **Accuracy** = proportion of all predictions that are correct (a fixed property of the model, ~99.2%)
- **Confidence** = probability assigned to the chosen class for THIS specific input (varies per prediction)

Displaying confidence as "accuracy" misleads users into thinking the model is only 50% accurate (when it's actually 99.2% on test data). This is a conceptual error in the UI labeling.

Additionally, `predict_proba` probabilities from sklearn are often **uncalibrated** — a 90% probability doesn't mean the model is right 90% of the time. Platt scaling or isotonic regression would be needed for calibrated probabilities.

**File:** `Truth/frontend/src/App.jsx` lines 348-349

---

### ISSUE 8: Temporal Regex Has Two Bugs

**Bug A — Month-day pattern matches year digits:**
```python
pattern = rf'{month_name}\s+(\d{{1,2}})'
```
On `"april 2026"`: captures "20" (first 2 digits) as day number. Should use word boundary: `\d{1,2}(?!\d)` or `\d{1,2}\b`.

**Bug B — "Today is" check ignores month:**
```python
day_m, year_m = int(match_full.group(1)), int(match_full.group(2))
if year_m != current_year or day_m != now.day:  # No month check!
```
`"is today 7th march 2026"` would pass validation even though today is April (day=7 matches, year=2026 matches, but March != April). The month is never extracted or compared.

**File:** `Truth/backend/main.py` lines 148-171

---

## Summary: What Needs Fixing

| # | Issue | Type | Severity |
|---|-------|------|----------|
| 1 | Explainability shows TF-IDF words, not actual model reasoning | **Methodology** | High |
| 2 | Single RAG false-positive overrides correct ML prediction | **Methodology** | High |
| 3 | Wikipedia uses regex entity extraction instead of NER | **Methodology** | Medium |
| 4 | Web RAG number comparison is context-free | **Methodology** | Medium |
| 5 | Web verification word-overlap includes stopwords | **Methodology** | Medium |
| 6 | ML failure produces invalid "Error" prediction | **Implementation** | High |
| 7 | UI says "Accuracy" but shows prediction confidence | **Conceptual** | High |
| 8 | Temporal regex has month-day and month-validation bugs | **Bug** | Medium |

---

## Recommended Fix Order

**Round 1 — Fix broken methodology:**
1. Fix explainability: either use real LIME at inference, or use SVD component backprojection, or honestly label it "Top Keywords"
2. Fix RAG override: require 2+ conflicts to override ML, weight checks by reliability, lower confidence instead of force-switching
3. Fix Wikipedia entity extraction: use spaCy NER instead of regex

**Round 2 — Fix bugs:**
4. Fix temporal regex (word boundary + month validation)
5. Fix ML error fallback (heuristic + RAG-only mode instead of "Error")
6. Fix UI label ("CONFIDENCE" not "ACCURACY")

**Round 3 — Improve RAG quality:**
7. Remove or guard numeric matching in web RAG
8. Filter stopwords from word-overlap verification
9. Skip spatial validation when no geographic content detected
