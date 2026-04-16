# TruthLens Phase 3 — Every Concept Explained In Detail

This document explains **every single term, concept, and word** used in the TruthLens project. If you encounter any word in the presentation or code, you will find its meaning here.

---

## Table of Contents

1. [Project Title Breakdown](#1-project-title-breakdown)
2. [The Problem: Fake News & Misinformation](#2-the-problem-fake-news--misinformation)
3. [The 7 Critical Mistakes](#3-the-7-critical-mistakes)
4. [SDG Mapping](#4-sdg-mapping)
5. [Datasets Explained](#5-datasets-explained)
6. [Data Preprocessing Pipeline — Every Step](#6-data-preprocessing-pipeline--every-step)
7. [Feature Engineering — Every Feature Explained](#7-feature-engineering--every-feature-explained)
8. [Machine Learning Models — Every Algorithm](#8-machine-learning-models--every-algorithm)
9. [Hyperparameter Tuning — GridSearchCV](#9-hyperparameter-tuning--gridsearchcv)
10. [Evaluation Metrics — Every Metric](#10-evaluation-metrics--every-metric)
11. [Bias Probing — Every Probe](#11-bias-probing--every-probe)
12. [Explainability — LIME, SHAP, SVD Backprojection](#12-explainability--lime-shap-svd-backprojection)
13. [Dual-Mode System — Article vs Claim](#13-dual-mode-system--article-vs-claim)
14. [Fact Verification Pipeline](#14-fact-verification-pipeline)
15. [Tools & Technologies — Every Library](#15-tools--technologies--every-library)
16. [Results — Every Number Explained](#16-results--every-number-explained)
17. [Cross-Dataset Generalization](#17-cross-dataset-generalization)
18. [Error Analysis](#18-error-analysis)
19. [Glossary of Every Technical Term](#19-glossary-of-every-technical-term)

---

## 1. Project Title Breakdown

**"TruthLens: Debiased, Explainable Fake News Detection Using Hybrid Feature Engineering and Stacking Ensemble"**

Let's break down every word:

### TruthLens
- The project name. "Truth" = verifying what is true. "Lens" = a tool to see clearly. Together: a tool that helps you see the truth in news articles.

### Debiased
- **Bias** in ML means the model learns shortcuts instead of real patterns. For example, if all real news articles come from Reuters, the model learns "Reuters = Real" instead of actually understanding what makes news real or fake.
- **Debiased** means we actively removed these shortcuts. We did this through:
  - **Entity masking**: Replacing names like "Trump", "Reuters", "CNN" with generic tokens like `[PERSON]`, `[SOURCE]`
  - **Byline stripping**: Removing "WASHINGTON (Reuters) -" from the start of articles
  - **Source masking**: Using regex patterns to catch 51 news source names that spaCy missed

### Explainable
- Most ML models are **black boxes** — they give a prediction but don't tell you WHY.
- **Explainable** means our model tells you exactly which words and features caused the prediction.
- We use **LIME** (shows which words pushed toward Fake/Real) and **SHAP** (shows which features are most important globally).

### Fake News Detection
- **Fake news**: Deliberately false or misleading information presented as legitimate news.
- **Detection**: Automatically classifying a given text as Fake or Real using machine learning.
- This is a **binary classification** problem: input = text, output = Fake (1) or Real (0).

### Hybrid Feature Engineering
- **Feature**: A measurable property of the data that the model uses to make decisions. Example: word count, sentiment score, TF-IDF value.
- **Feature Engineering**: The process of creating/extracting features from raw data.
- **Hybrid**: We combine THREE different types of features:
  1. **TF-IDF + SVD** (150 dimensions) — statistical word importance
  2. **GloVe embeddings** (100 dimensions) — semantic word meaning
  3. **Stylometric features** (17 dimensions) — writing style patterns
- Total: **267 dimensions** per article (or 167 without GloVe)

### Stacking Ensemble
- **Ensemble**: Combining multiple models to get better predictions than any single model.
- **Stacking**: A specific ensemble technique where:
  1. Multiple **base models** (SVM, Logistic Regression, Random Forest) each make predictions
  2. A **meta-learner** (another Logistic Regression) takes those predictions as input and learns the best way to combine them
- This is better than **majority voting** (where you just pick whatever most models agree on) because the meta-learner learns which model to trust more for which type of input.

---

## 2. The Problem: Fake News & Misinformation

### What is misinformation?
- **Misinformation**: False or inaccurate information, regardless of intent.
- **Disinformation**: Deliberately false information spread to deceive.
- **Fake news**: A subset — news articles that are intentionally fabricated to mislead readers.

### Why does it matter?
- Misinformation spreads **6x faster** than factual news online (Vosoughi et al., Science, 2018).
  - This means: if you post a true story and a false story at the same time, the false story will reach 6 times more people.
  - Why? Because fake news is often more sensational, emotional, and surprising — it triggers stronger reactions and gets shared more.

### What's wrong with existing solutions?
- Existing ML fake news detectors report 95%+ accuracy on their test sets.
- **BUT** they fail catastrophically when tested on different datasets.
- **Root cause**: They learn **dataset-specific shortcuts**, not actual deception patterns.
  - Example: In the ISOT dataset, ALL real news comes from Reuters. So the model learns "Reuters = Real" instead of understanding what makes text trustworthy.
  - This is called **dataset bias** or **spurious correlation**.

---

## 3. The 7 Critical Mistakes

TruthLens was designed to fix 7 common problems in existing fake news detectors:

### Mistake 1: Dataset Bias
- **What**: Models memorize which entities (names, organizations) appear in fake vs. real news.
- **Example**: "Trump" appears more in fake articles in ISOT → model learns "Trump = Fake".
- **Our fix**: Entity masking — replace all entity names with generic tokens using spaCy NER.

### Mistake 2: Generalization Failure
- **What**: Models work great on one dataset but fail on another.
- **Example**: Train on ISOT (long articles) → test on LIAR (short claims) → F1 drops from 99% to 23%.
- **Our fix**: Cross-dataset evaluation to honestly measure this gap. Combined training mode.

### Mistake 3: Poor Semantics
- **What**: Using only TF-IDF misses the MEANING of words. "Good" and "great" are treated as completely different words.
- **Our fix**: Add GloVe embeddings that capture word meaning (semantic similarity).

### Mistake 4: Missing Style Context
- **What**: Ignoring HOW the text is written (writing style, tone, readability).
- **Our fix**: 17 stylometric features that capture writing patterns (sentiment, readability, exclamation rate, etc.).

### Mistake 5: No AI-Text Detection
- **What**: Not checking if text was generated by AI (ChatGPT, etc.).
- **Our fix**: Added **burstiness** and **Zipf coefficient** features that detect AI-generated text.

### Mistake 6: Black Box Models
- **What**: Models give predictions but no explanation.
- **Our fix**: LIME (word-level explanations) + SHAP (feature importance) + bias auditing.

### Mistake 7: Naive Voting
- **What**: Combining models by simple majority vote.
- **Our fix**: Stacking ensemble where a meta-learner learns optimal combination weights.

---

## 4. SDG Mapping

### What is SDG?
- **SDG** = Sustainable Development Goals — 17 goals adopted by the United Nations in 2015 to achieve a better and more sustainable future for all by 2030.

### SDG 16: Peace, Justice and Strong Institutions
- Goal 16 specifically targets: "Promote peaceful and inclusive societies, provide access to justice for all, and build effective, accountable institutions."
- **How TruthLens maps**: Combating misinformation strengthens democratic institutions because:
  - Informed citizens make better decisions
  - Reducing fake news reduces manipulation of public opinion
  - Fact-checking tools support journalists and researchers
  - Transparent AI (explainable predictions) builds trust in technology

---

## 5. Datasets Explained

### ISOT Fake News Dataset

- **Source**: University of Victoria, Canada (hosted on Kaggle)
- **Creator**: Hossein Ahmed, Ibrahim Traore, Sherif Saad (2017)
- **What it contains**: 44,898 full-length news articles
  - **23,481 fake articles** (52.3%) — collected from unreliable news websites flagged by PolitiFact and Wikipedia
  - **21,417 real articles** (47.7%) — collected from Reuters.com (a trusted news agency)
- **Columns**: `title`, `text`, `subject`, `date`
- **Average article length**: 500-3000 words (long, detailed articles)
- **Time period**: 2016-2017 (mostly US political news)
- **Key issue**: ALL real articles are from Reuters → model can learn "Reuters style = Real" (source bias)

### LIAR Dataset

- **Source**: William Yang Wang, UC Santa Barbara
- **Published at**: ACL 2017 (Association for Computational Linguistics — a top NLP conference)
- **What it contains**: 12,836 short political statements
  - Collected from PolitiFact.com (a fact-checking website)
  - Originally has 6 labels: pants-fire, false, barely-true, half-true, mostly-true, true
  - **We converted to binary**: {pants-fire, false, barely-true} → Fake (1), {half-true, mostly-true, true} → Real (0)
  - After conversion: ~6,426 fake (50.1%), ~6,410 real (49.9%)
- **Columns**: `id`, `label`, `statement`, `subject`, `speaker`, `job_title`, `state_info`, `party`, credit history counts, `context`
- **Average length**: 15-30 words (very short claims, not full articles)
- **Key issue**: Very different from ISOT — short claims vs long articles creates a domain mismatch.

### Why Two Datasets?
- Using only ISOT would give inflated accuracy (the model memorizes Reuters patterns).
- Testing on LIAR reveals how well (or poorly) the model generalizes to different types of text.
- **Combined dataset**: 57,734 total samples (ISOT + LIAR merged for combined training).

### Kaggle
- A platform owned by Google where people share datasets, run ML competitions, and host notebooks.
- We download ISOT from Kaggle using the `kagglehub` library.

### HuggingFace
- A company/platform that hosts ML models, datasets, and tools.
- We download LIAR from HuggingFace using their `datasets` library.

### Train/Test Split
- **80/20 split**: 80% of data for training, 20% for testing
- **Stratified**: The split maintains the same Fake/Real ratio in both train and test sets
  - Example: If 52% of the full data is Fake, then both train and test will have ~52% Fake
  - This prevents accidentally putting all Fake articles in training and none in testing
- **random_state=42**: A seed number that ensures the split is the same every time you run the code
  - "42" is a common convention (from "Hitchhiker's Guide to the Galaxy")
  - Any fixed number works — the point is reproducibility

---

## 6. Data Preprocessing Pipeline — Every Step

**Preprocessing** = cleaning and transforming raw text data before the model can use it.

### Why preprocess?
- Raw text has noise: URLs, HTML tags, typos, inconsistent casing
- Names and sources leak identity (bias)
- Different forms of same word ("running", "ran", "runs") should be treated as one word
- Models need consistent, clean input to learn meaningful patterns

### Step 1: Byline Stripping

**What is a byline?**
- A byline is the line at the beginning of a news article that tells you who wrote it and where.
- Example: `"WASHINGTON (Reuters) - The president announced..."` → The byline is `"WASHINGTON (Reuters) -"`

**Why strip it?**
- If ALL real articles start with "(Reuters)" and NO fake articles do, the model just learns to check for "Reuters" — that's a shortcut, not real understanding.
- We use **regex** (regular expressions) to remove these patterns.

**What is regex?**
- **Regex** (Regular Expression) = a pattern-matching language for text.
- `r'^[A-Z][A-Z\s,/]+\s*\(Reuters\)\s*[-–—]?\s*'` means:
  - `^` = start of text
  - `[A-Z][A-Z\s,/]+` = one or more uppercase letters, spaces, commas, slashes (like "WASHINGTON" or "NEW YORK")
  - `\s*` = zero or more spaces
  - `\(Reuters\)` = literal "(Reuters)"
  - `\s*[-–—]?\s*` = optional dash surrounded by optional spaces
- Same patterns for `(AP)` and `(AFP)` (other news agencies)

### Step 2: Text Cleaning

Removes noise from text:
- **URLs**: Links like `https://...` — removed because they don't indicate fake/real
- **HTML tags**: `<p>`, `<div>`, `<br>` etc. — leftover from web scraping
- **Email addresses**: `user@email.com` — irrelevant to classification
- **Special characters**: `!@#$%^&*()` — most are noise (but `!` and `?` counts are captured separately in stylometric features)
- **Truncation**: Text cut to max 5,000 characters — prevents extremely long articles from dominating memory
- **Whitespace normalization**: Multiple spaces/newlines collapsed to single space

**IMPORTANT**: Cleaning is done WITHOUT lowercasing first. Why? Because spaCy's NER (named entity recognition) needs capitalization to identify names.
- "Washington" (capitalized) → spaCy recognizes as a location
- "washington" (lowercase) → spaCy might miss it

### Step 3: Entity Masking (spaCy NER)

**What is NER?**
- **NER** = Named Entity Recognition — the task of finding and classifying proper nouns in text.
- **Named Entity**: A real-world object with a name — a person, organization, location, date, etc.

**What is spaCy?**
- spaCy is a Python library for NLP (Natural Language Processing).
- We use the `en_core_web_sm` model — a small English model trained on web text.
- "sm" = small (fast but less accurate). Alternatives: "md" (medium), "lg" (large), "trf" (transformer-based, most accurate).

**How entity masking works:**
1. Run spaCy NER on the text: `doc = nlp(text)`
2. spaCy identifies entities with labels:
   - `PERSON` = a human name → replaced with `[PERSON]`
   - `ORG` = an organization → replaced with `[ORG]`
   - `GPE` = geopolitical entity (country, city, state) → replaced with `[LOC]`
   - `LOC` = non-GPE location (mountain, river) → replaced with `[LOC]`
   - `FAC` = facility (building, airport) → replaced with `[LOC]`
   - `DATE` = a date or time → replaced with `[DATE]`
   - `NORP` = nationality, religious, political group → replaced with `[GROUP]`
   - `EVENT` = a named event (war, election) → replaced with `[EVENT]`
3. Entities are replaced in **reverse order** (from end of text to beginning) to avoid character offset issues
   - If you replace "Trump" at position 5-10 with "[PERSON]" (8 chars), every entity after position 10 has shifted. Processing in reverse avoids this.

**Example:**
- Before: `"Donald Trump met with Emmanuel Macron in Paris on June 5."`
- After: `"[PERSON] met with [PERSON] in [LOC] on [DATE]."`

**Why mask entities?**
- Without masking: the model learns "Trump → Fake" or "Reuters → Real"
- With masking: the model must learn from the CONTENT and STYLE, not from names
- Our bias probe shows entity masking reduces bias from ~95% to 91.45%

### Step 4: Source Masking (Supplementary Regex)

**Why is Step 3 not enough?**
- spaCy's small model misses some entities, especially news source names.
- "Breitbart" might not be recognized as an ORG by spaCy.
- So we use regex to catch 51 known news sources:

**The 51 sources masked:**
Reuters, Associated Press, AP, AFP, Fox News, CNN, BBC, MSNBC, NBC, ABC News, CBS, The Guardian, Washington Post, New York Times, NYT, Huffington Post, HuffPost, Daily Mail, BuzzFeed, PolitiFact, Snopes, FactCheck, Bloomberg, Wall Street Journal, WSJ, RT, Russia Today, Al Jazeera, USA Today, Newsweek, Time Magazine, Vox, Vice, Slate, The Atlantic, The Hill, Politico, Axios, TechCrunch, Wired, Fortune, Forbes, Business Insider, The Intercept, ProPublica, NPR, PBS, C-SPAN, The Economist, Financial Times, and Breitbart.

All are replaced with `[SOURCE]`.

### Step 5: Stopword Removal (NLTK)

**What are stopwords?**
- **Stopwords** = extremely common words that carry little meaningful information.
- Examples: "the", "is", "at", "which", "on", "a", "an", "in", "for", "to", "of", "and", "or", "but"
- They appear in EVERY text, so they don't help distinguish fake from real.

**What is NLTK?**
- **NLTK** = Natural Language Toolkit — a Python library for NLP tasks.
- We use `nltk.corpus.stopwords.words("english")` — a list of ~179 English stopwords.

**Why remove them?**
- Reduces noise in TF-IDF (stopwords would dominate term frequency counts)
- Reduces feature vector size
- Helps the model focus on meaningful content words

### Step 6: Lemmatization (spaCy)

**What is lemmatization?**
- **Lemmatization** = reducing words to their base/dictionary form (called a **lemma**).
- Examples:
  - "running", "ran", "runs" → "run"
  - "better" → "good"
  - "mice" → "mouse"
  - "is", "was", "are" → "be"

**How is it different from stemming?**
- **Stemming** just chops off word endings: "running" → "run", "studies" → "studi" (broken word!)
- **Lemmatization** uses vocabulary and grammar rules: "studies" → "study" (correct word)
- spaCy's lemmatizer is context-aware — it uses the word's part of speech.

**Why lemmatize?**
- "Investigation", "investigating", "investigated" are all about the same concept
- Without lemmatization: 3 separate TF-IDF features
- With lemmatization: 1 feature ("investigate") that captures all variants

### Output: 4 Text Versions Per Article

After preprocessing, each article has 4 versions stored:

| Column | Content | Used For |
|--------|---------|----------|
| `text` | Original raw text | Stylometric features (need original capitalization, punctuation) |
| `cleaned_text` | Cleaned + lowercased (NO masking) | GloVe embeddings (need real words, not `[PERSON]`) |
| `masked_text` | Cleaned + entity masked + source masked | Reference/debugging |
| `processed_text` | Masked + lemmatized + stopwords removed + lowercased | TF-IDF input |

---

## 7. Feature Engineering — Every Feature Explained

### What is a feature?
A **feature** is a numerical value that represents some property of the input data. ML models can only work with numbers, not raw text — so we must convert text into numbers (features).

### What is a feature vector?
A **feature vector** is a list of all features for one data point (one article). Our feature vector has **267 numbers** per article (150 + 100 + 17).

### What is a dimension?
In this context, **dimension** = the number of features. A 267-dimensional vector has 267 numbers. You can think of each article as a point in 267-dimensional space.

---

### Pipeline A: TF-IDF + SVD (150 dimensions)

#### TF-IDF (Term Frequency - Inverse Document Frequency)

**TF-IDF** is a way to measure how important a word is to a document relative to a collection of documents.

**Term Frequency (TF):**
- How often a word appears in THIS document.
- `TF(word, document) = count of word in document / total words in document`
- Example: If "vaccine" appears 5 times in a 100-word article → TF = 5/100 = 0.05

**Sublinear TF:**
- We use `sublinear_tf=True`, which changes the formula to: `TF = 1 + log(raw_count)`
- Why? Without log, a word appearing 100 times gets 100x the weight of a word appearing once. With log, it gets only about 5.6x.
- This prevents very frequent words from dominating.

**Inverse Document Frequency (IDF):**
- How rare a word is across ALL documents.
- `IDF(word) = log(total documents / documents containing this word) + 1`
- Words appearing in every document get low IDF (they're common, not distinctive)
- Words appearing in few documents get high IDF (they're rare, potentially important)
- Example: "the" appears in all 44,000 documents → IDF ≈ 0. "hydroxychloroquine" appears in 50 documents → IDF ≈ 6.8

**TF-IDF = TF x IDF**
- A word that's frequent in THIS document but rare overall gets a HIGH score (important for this specific document)
- A word that's common everywhere gets a LOW score (not distinctive)

**Our TF-IDF Configuration:**
- `max_features=10,000` → Keep only the top 10,000 most important terms (vocabulary size)
- `ngram_range=(1, 2)` → Use both unigrams and bigrams:
  - **Unigram**: single words ("vaccine", "government")
  - **Bigram**: pairs of consecutive words ("fake news", "breaking news")
  - Bigrams capture phrases that single words miss
- `min_df=2` → Ignore words appearing in fewer than 2 documents (too rare, probably noise)
- `max_df=0.95` → Ignore words appearing in more than 95% of documents (too common, like stopwords)
- `strip_accents="unicode"` → Normalize accented characters ("café" → "cafe")

**Output**: A sparse matrix with 10,000 columns (one per term) — most values are zero because any given article only contains a small fraction of the vocabulary.

#### TruncatedSVD (Singular Value Decomposition)

**What is SVD?**
- **SVD** (Singular Value Decomposition) is a mathematical technique that decomposes a matrix into three simpler matrices: `A = U × S × V^T`
  - `U` = document-topic matrix (how much each document relates to each "topic")
  - `S` = diagonal matrix of singular values (importance of each "topic")
  - `V^T` = topic-word matrix (which words define each "topic")
- "Topics" here are discovered automatically — they're not predefined categories. They're abstract patterns in word co-occurrence.

**What is TruncatedSVD?**
- **Truncated** means we only keep the top-k components (most important patterns).
- We keep **150 components** out of 10,000 — this compresses the TF-IDF matrix dramatically.
- We measured that 150 components retain ~85% of the variance (information) in the original data.

**Why use SVD?**
1. **Dimensionality reduction**: 10,000 columns → 150 columns. Makes training much faster.
2. **Noise reduction**: Small components are often noise. Removing them improves generalization.
3. **Dense representation**: TF-IDF produces sparse vectors (mostly zeros). SVD produces dense vectors (all non-zero). Many algorithms work better with dense vectors.
4. **Latent semantics**: SVD discovers hidden patterns — words that often appear together get grouped into the same component. This is related to **LSA** (Latent Semantic Analysis).

**Variance explained**:
- If 150 components explain 85% of variance, it means 85% of the information in the original 10,000-column matrix is captured by just 150 numbers.
- The remaining 15% is mostly noise and rare word patterns.

---

### Pipeline B: GloVe Embeddings (100 dimensions)

#### What are word embeddings?
- **Word embeddings** = representing words as dense vectors of real numbers.
- Unlike TF-IDF (which treats every word as independent), embeddings capture **meaning**.
- Words with similar meanings have similar vectors:
  - `vector("king") - vector("man") + vector("woman") ≈ vector("queen")`
  - `vector("good") ≈ vector("great")` (close in vector space)
  - `vector("dog") ≈ vector("puppy")` (close in vector space)

#### What is GloVe?
- **GloVe** = **Gl**obal **Ve**ctors for Word Representation
- Created by Stanford NLP group (Pennington, Socher, Manning, 2014)
- Trained on massive text corpora by analyzing word co-occurrence statistics
- We use: `glove-wiki-gigaword-100` = trained on Wikipedia + Gigaword corpus, 100-dimensional vectors

**How we use GloVe:**
1. Take the `cleaned_text` (no entity masking, because GloVe needs real words)
2. Split into words
3. Look up each word in the GloVe dictionary → get a 100-dimensional vector
4. **Average** all word vectors → one 100-dimensional vector per document
   ```
   doc_vector = mean([glove["scientists"], glove["discovered"], glove["water"], glove["mars"]])
   ```
5. If a word isn't in GloVe's vocabulary, skip it

**Why average?**
- GloVe gives vectors per WORD, but we need one vector per DOCUMENT.
- Averaging is simple and works well for document-level classification.
- More sophisticated methods (weighted averaging, pooling) exist but add complexity.

**GloVe incompatibility with Python 3.14:**
- GloVe is loaded via the `gensim` library.
- `gensim` is not compatible with Python 3.14+ (the latest Python version).
- Workaround: `--no-glove` flag disables Pipeline B, reducing features from 267 to 167 dimensions.
- Performance impact is minor because TF-IDF SVD already captures much of the same information.

---

### Pipeline C: Stylometric Features (17 dimensions)

**What is stylometry?**
- **Stylometry** = the statistical analysis of writing style.
- Different authors (and fake vs real news) have distinct writing patterns.
- These features capture HOW something is written, not WHAT it says.

#### Feature 1: `word_count`
- **What**: Total number of words in the article.
- **How**: `len(text.split())`
- **Why**: Fake news articles tend to be shorter and more sensational. Real news articles are typically longer with more detail.

#### Feature 2: `char_count`
- **What**: Total number of characters in the article.
- **How**: `len(text)`
- **Why**: Related to word_count but also captures whether the article uses long or short words.

#### Feature 3: `avg_word_length`
- **What**: Average number of characters per word.
- **How**: `char_count / word_count`
- **Why**: Higher average word length indicates more sophisticated vocabulary. Real news tends to use more precise, longer words. Fake news often uses simpler, shorter words.

#### Feature 4: `sentence_count`
- **What**: Number of sentences in the article.
- **How**: Split text on `.`, `!`, `?` — but handle abbreviations (Mr., Dr., U.S., etc.) using a regex pattern so they don't create false sentence breaks.
- **Why**: More sentences generally means more structured, detailed writing.

#### Feature 5: `avg_sentence_length`
- **What**: Average number of words per sentence.
- **How**: `word_count / sentence_count`
- **Why**: Real news tends to have longer, more complex sentences. Fake news often uses short, punchy sentences for emotional impact.

#### Feature 6: `vocabulary_richness`
- **What**: Type-token ratio (TTR) — the ratio of unique words to total words.
- **How**: `len(set(words)) / len(words)` (number of unique words / total words)
- **Why**: Higher vocabulary richness = more diverse word choice. Real news typically has higher richness because journalists use varied vocabulary. Fake news often repeats the same emotionally charged words.
- **Range**: 0 to 1 (1 = every word is unique, 0.1 = very repetitive)

#### Feature 7: `capital_ratio`
- **What**: Fraction of characters that are uppercase.
- **How**: `sum(1 for c in text if c.isupper()) / len(text)`
- **Why**: Fake news often uses ALL CAPS for emphasis: "BREAKING!!!", "EXPOSED!!!". Real news uses standard capitalization. High capital_ratio → likely fake.

#### Feature 8: `exclamation_rate`
- **What**: Number of exclamation marks per sentence.
- **How**: `text.count("!") / sentence_count`
- **Why**: Fake news uses excessive exclamation marks for sensationalism. "The government is HIDING the truth!!!" vs "The government released a statement."
- **Normalized per sentence**: So a long article with a few exclamation marks doesn't get the same score as a short article with the same number.

#### Feature 9: `question_rate`
- **What**: Number of question marks per sentence.
- **How**: `text.count("?") / sentence_count`
- **Why**: Fake news uses questions to create doubt and engagement: "Did the government LIE to you?" "Why aren't they telling us the truth?"

#### Feature 10: `digit_ratio`
- **What**: Fraction of characters that are digits.
- **How**: `sum(1 for c in text if c.isdigit()) / len(text)`
- **Why**: Real news often contains specific numbers (dates, statistics, figures). Fake news tends to make vague claims without specific numbers.

#### Feature 11: `sentiment_compound`
- **What**: Overall sentiment score from VADER.
- **Range**: -1 (extremely negative) to +1 (extremely positive), 0 = neutral.

**What is VADER?**
- **VADER** = Valence Aware Dictionary and sEntiment Reasoner
- A sentiment analysis tool specifically designed for social media and news text
- Uses a dictionary of words with pre-assigned sentiment scores
- Handles: negation ("not good" = negative), intensifiers ("very good" = more positive), punctuation ("good!!!" = even more positive), capitalization ("GOOD" = more intense)
- Returns 4 scores: positive, negative, neutral, compound

**What is sentiment analysis?**
- **Sentiment** = the emotional tone of text (positive, negative, neutral)
- **Sentiment analysis** = automatically determining the sentiment of text
- **Why for fake news**: Fake news tends to be more emotionally extreme (very positive or very negative) to trigger reactions. Real news is more neutral and balanced.

#### Feature 12: `sentiment_pos`
- **What**: Proportion of text that is positive sentiment (from VADER).
- **Range**: 0 to 1

#### Feature 13: `sentiment_neg`
- **What**: Proportion of text that is negative sentiment (from VADER).
- **Range**: 0 to 1
- **Why both pos and neg?** An article can be 30% positive, 40% negative, 30% neutral. The compound score alone doesn't capture this distribution.

#### Feature 14: `flesch_reading_ease`
- **What**: A readability score measuring how easy the text is to read.
- **Formula**: `206.835 - 1.015 × (total words / total sentences) - 84.6 × (total syllables / total words)`
- **Scale**:
  - 90-100: Very easy (5th grade level)
  - 60-70: Standard (8th-9th grade)
  - 30-50: Difficult (college level)
  - 0-30: Very difficult (college graduate)
  - Below 0: Extremely complex
- **Clamped to [-50, 120]**: To avoid extreme outliers from very short/long texts.
- **Why**: Real news from professional journalists typically has moderate readability (50-70). Fake news is often either very simple (to reach more people) or very complex (to seem authoritative).

**What is textstat?**
- A Python library that computes readability metrics. We use it for Flesch and ARI scores.

#### Feature 15: `automated_readability_index` (ARI)
- **What**: Another readability metric, expressed as a US school grade level.
- **Formula**: `4.71 × (characters/words) + 0.5 × (words/sentences) - 21.43`
- **Scale**: 1 = 1st grade, 8 = 8th grade, 14 = college level
- **Clamped to [0, 30]**: To avoid outliers.
- **Why two readability metrics?** Flesch uses syllable count; ARI uses character count. They capture slightly different aspects of readability.

#### Feature 16: `burstiness`
- **What**: How much sentence lengths vary within the article.
- **Formula**: `std(sentence_lengths) / mean(sentence_lengths)`
  - `std` = standard deviation = how spread out the values are
  - `mean` = average
- **What it measures**: If all sentences are the same length (low burstiness), the text is likely AI-generated. Humans naturally write with variable sentence lengths — some short, some long.
- **AI detection**: AI-generated text (ChatGPT, etc.) tends to produce unnaturally uniform sentence structures. Real human writing has "bursty" patterns — short sentences for emphasis, long sentences for explanation.
- **Example**:
  - AI text: "The sky is blue. The grass is green. The sun is warm." → all ~5 words → low burstiness
  - Human text: "Look. The sky — impossibly, achingly blue — stretched forever. Not a cloud." → lengths: 1, 8, 3 → high burstiness

#### Feature 17: `zipf_coefficient`
- **What**: The slope of the word frequency distribution on a log-log scale.

**What is Zipf's Law?**
- **Zipf's Law** states that in natural language, the frequency of a word is inversely proportional to its rank.
- The most common word ("the") appears about twice as often as the 2nd most common ("of"), three times as often as the 3rd, etc.
- When you plot `log(rank)` vs `log(frequency)`, natural text produces a straight line with slope ≈ -1.

**How we compute it:**
1. Count how often each word appears
2. Sort by frequency (descending)
3. Assign ranks (1st, 2nd, 3rd...)
4. Compute `log(rank)` and `log(frequency)` for each word
5. Fit a straight line through these points
6. The slope of that line is the **Zipf coefficient**

**Why for fake news:**
- Human-written text closely follows Zipf's law (slope ≈ -1.0)
- AI-generated text deviates from Zipf's law (slope might be -0.7 or -1.3)
- This is because AI models have different word distribution patterns than humans
- Combined with burstiness, this helps detect AI-generated fake news

---

### StandardScaler

**What is StandardScaler?**
- A preprocessing step that normalizes features to have **mean = 0** and **standard deviation = 1**.
- **Formula**: `z = (x - mean) / std` for each feature
- **Example**: If word_count has mean=500 and std=200, then an article with word_count=700 becomes: `(700-500)/200 = 1.0`

**Why scale?**
- Without scaling, features with large ranges (word_count: 10-5000) dominate features with small ranges (sentiment: -1 to 1)
- SVM and Logistic Regression are especially sensitive to feature scales
- After scaling, all features contribute equally

**CRITICAL: Fitted on training data only**
- The scaler learns mean and std from TRAINING data
- Then applies the same transformation to TEST data
- If you fit on test data too, that's **data leakage** — the model indirectly "sees" test data statistics during training

---

## 8. Machine Learning Models — Every Algorithm

### What is classification?
- **Classification** = predicting a categorical label for an input.
- **Binary classification** = two possible labels (Fake or Real).
- Each model learns a decision boundary that separates Fake from Real in 267-dimensional feature space.

---

### Model 1: SVM (Support Vector Machine)

#### What is SVM?
- **SVM** finds the **hyperplane** (decision boundary) that best separates two classes with the **maximum margin**.

**What is a hyperplane?**
- In 2D: a line that separates two groups of points
- In 3D: a plane
- In 267D: a 266-dimensional surface (hard to visualize, but the math works the same way)

**What is margin?**
- The distance between the hyperplane and the nearest data points from each class.
- SVM maximizes this margin — the bigger the gap, the more confident the classification.
- **Support vectors** = the data points closest to the hyperplane (they "support" the boundary). Only these points matter; all other points could move without affecting the hyperplane.

#### LinearSVC
- **LinearSVC** = Linear Support Vector Classification
- Uses a **linear** kernel — the decision boundary is a flat hyperplane (no curves)
- **Why linear?** With 267 features, linear boundaries work well. Non-linear kernels (like RBF) are O(n^2) which would be very slow on 40,000+ samples. LinearSVC is O(n) — much faster.

**What is O(n) vs O(n^2)?**
- **Big O notation** describes how an algorithm's time grows with data size.
- **O(n)**: Time grows linearly. 10x more data → 10x more time. (Fast)
- **O(n^2)**: Time grows quadratically. 10x more data → 100x more time. (Slow for large datasets)

#### CalibratedClassifierCV
- **Problem**: LinearSVC doesn't have a `predict_proba()` method — it only outputs class labels (0 or 1), not probabilities.
- **Why we need probabilities**: The stacking ensemble needs probability outputs from each base model (not just 0/1 labels) so the meta-learner can learn how confident each model is.
- **Solution**: `CalibratedClassifierCV` wraps LinearSVC and adds probability calibration.
- **How it works**: Uses **isotonic regression** (or Platt scaling) on cross-validated predictions to convert raw SVM scores into well-calibrated probabilities.
- **3-fold internal CV**: Trains the SVM on 2/3 of data, calibrates on remaining 1/3, repeats 3 times.

#### Hyperparameter C
- **C** = regularization strength (controls the tradeoff between margin width and misclassification)
- **Low C** (e.g., 0.01): Wider margin, allows more misclassifications. **More regularized** — prevents overfitting.
- **High C** (e.g., 10): Narrower margin, allows fewer misclassifications. **Less regularized** — might overfit.
- **Best C = 0.01**: The model prefers strong regularization, suggesting the 267-dimensional feature space is rich enough that overfitting is a risk.

---

### Model 2: Logistic Regression

#### What is Logistic Regression?
- Despite the name "regression", it's a **classification** algorithm.
- It models the **probability** that an input belongs to class 1 (Fake):
  - `P(Fake) = sigmoid(w1*x1 + w2*x2 + ... + w267*x267 + b)`
  - where `w1...w267` are learned weights, `x1...x267` are features, `b` is bias

**What is the sigmoid function?**
- `sigmoid(z) = 1 / (1 + e^(-z))`
- Takes any real number and squashes it to the range (0, 1) — perfect for probabilities
- If z is very large positive → sigmoid ≈ 1 (confident Fake)
- If z is very large negative → sigmoid ≈ 0 (confident Real)
- If z = 0 → sigmoid = 0.5 (uncertain)

**Why Logistic Regression?**
- **Interpretable**: Each weight tells you how much that feature pushes toward Fake or Real
- **Fast**: Training takes <1 second on our data
- **Calibrated probabilities**: Naturally outputs well-calibrated probabilities (no CalibratedClassifierCV needed)

#### Solver: L-BFGS
- **Solver** = the optimization algorithm used to find the best weights.
- **L-BFGS** = Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm
- It's an efficient quasi-Newton method for optimization. You don't need to understand the math — just know it's the recommended solver for medium-sized datasets.

#### Penalty: L2 (Ridge)
- **Regularization penalty** = a way to prevent the model from making weights too large (which causes overfitting).
- **L2 (Ridge)**: Adds `lambda * sum(w_i^2)` to the loss function. Penalizes large weights but doesn't force them to exactly zero.
- **L1 (Lasso)**: Would add `lambda * sum(|w_i|)`. Can force some weights to exactly zero (feature selection). We don't use this.
- **Why L2?**: We want all features to contribute, not to select a subset.

#### Hyperparameter C
- Same concept as SVM's C — controls regularization strength.
- **Best C = 0.1**: Moderate regularization.
- Note: In sklearn, higher C = LESS regularization (opposite of lambda in textbooks).

---

### Model 3: Random Forest

#### What is a Decision Tree?
- A tree-like model that makes decisions by asking yes/no questions about features.
- Example:
  ```
  Is exclamation_rate > 0.5?
    ├── Yes → Is sentiment_compound > 0.3?
    │         ├── Yes → FAKE
    │         └── No → Is word_count < 100?
    │                   ├── Yes → FAKE
    │                   └── No → REAL
    └── No → Is vocabulary_richness > 0.6?
              ├── Yes → REAL
              └── No → REAL
  ```
- Each internal node is a condition; each leaf is a prediction.

#### What is a Random Forest?
- **Random Forest** = an ensemble of many decision trees.
- **"Random" comes from two sources of randomness:**
  1. **Bagging (Bootstrap Aggregating)**: Each tree is trained on a random subset of the training data (sampled with replacement). If you have 35,000 training samples, each tree might see a different 35,000 samples (some duplicated, some missing).
  2. **Random feature subset**: At each split in each tree, only a random subset of features is considered. If you have 267 features and `max_features="sqrt"`, each split only considers √267 ≈ 16 random features.

**Why randomness helps:**
- Each tree sees slightly different data and features → each tree makes different mistakes.
- When you average (for regression) or vote (for classification) across all trees, individual mistakes cancel out.
- This is the key insight: **diverse weak learners combine to make a strong learner**.

#### Our Configuration:
- `n_estimators=100`: 100 decision trees (selected by GridSearchCV from [100, 300, 500])
- `max_depth=20`: Each tree can be at most 20 levels deep (prevents overfitting to noise)
- `min_samples_split=2`: A node can be split if it has at least 2 samples
- `max_features="sqrt"`: At each split, consider √267 ≈ 16 random features (prevents any single feature from dominating)
- `class_weight="balanced"`: Adjust weights inversely proportional to class frequency

---

### Model 4: Stacking Ensemble (Meta-LR)

#### What is a Stacking Ensemble?

Stacking is a two-level architecture:

**Level 1: Base Models**
- SVM, Logistic Regression, Random Forest — each trained independently on the same training data.
- Each produces probability predictions: `[P(Real), P(Fake)]` for each sample.

**Level 2: Meta-Learner**
- Takes the Level 1 predictions as INPUT features.
- For each training sample: 3 models × 2 probabilities = 6 input features to the meta-learner.
- The meta-learner (Logistic Regression) learns: "When SVM says 80% Fake and RF says 60% Real, what's the best final answer?"

**How stacking is trained (to avoid data leakage):**
1. Split training data into 5 folds (stratified)
2. For fold 1: Train base models on folds 2-5, predict on fold 1
3. For fold 2: Train base models on folds 1,3,4,5, predict on fold 2
4. ... repeat for all 5 folds
5. Now you have out-of-fold predictions for ALL training samples
6. Train meta-learner on these out-of-fold predictions
7. Finally, retrain all base models on the FULL training data

**Why is this better than majority voting?**
- Majority voting treats all models equally. But SVM might be much better than RF on this data.
- The meta-learner LEARNS which model to trust more. It might learn: "SVM's probability is the most reliable signal; RF only helps when SVM is uncertain."
- It can also learn non-obvious combinations: "When SVM says Fake but RF says Real, trust RF" (because RF captures non-linear patterns SVM misses).

#### `stack_method="predict_proba"`
- Use probability predictions (soft voting), not class labels (hard voting)
- Probabilities carry MORE information: "80% Fake" vs "52% Fake" tells the meta-learner about confidence
- Hard voting throws away this information: both become just "Fake"

#### `passthrough=False`
- Don't include the original 267 features alongside the base model predictions
- Only the 6 meta-features (3 models × 2 classes) go to the meta-learner
- This prevents the meta-learner from overfitting to the raw features

---

## 9. Hyperparameter Tuning — GridSearchCV

### What is a hyperparameter?
- **Hyperparameter**: A setting you choose BEFORE training (not learned from data).
  - Examples: C in SVM, number of trees in RF, learning rate
- **Parameter**: A value the model learns FROM data during training.
  - Examples: weights in Logistic Regression, split thresholds in decision trees

### What is GridSearchCV?
- **Grid Search** = try every combination of hyperparameter values in a predefined grid.
- **CV** = Cross-Validation = evaluate each combination using k-fold CV instead of a single train/test split.
- **Stratified K-Fold CV** = split data into k parts (folds) while maintaining class ratios.

**Example for SVM:**
- Grid: C ∈ [0.01, 0.1, 1, 10] → 4 values
- 5-fold CV → each value tested 5 times
- Total: 4 × 5 = 20 training runs
- Best C (highest average F1 across 5 folds) is selected

**Why use CV instead of a single validation set?**
- A single split might be lucky or unlucky (some easy/hard samples in test)
- 5-fold CV averages over 5 different splits → more reliable estimate
- Every sample is used for both training and validation exactly once

### `scoring="f1"`
- GridSearchCV optimizes for F1 score, not accuracy.
- **Why not accuracy?** In an imbalanced dataset (e.g., 90% Real, 10% Fake), a model that always predicts "Real" gets 90% accuracy but catches zero fake news. F1 penalizes this.

### `class_weight="balanced"`
- Automatically adjusts class weights inversely proportional to class frequency.
- Formula: `weight_class = n_samples / (n_classes × n_class_samples)`
- If Fake has 23,000 samples and Real has 21,000:
  - Fake weight = 44,000 / (2 × 23,000) = 0.957
  - Real weight = 44,000 / (2 × 21,000) = 1.048
- The minority class (Real) gets slightly higher weight → model pays more attention to getting Real predictions correct.

### `n_jobs=-1`
- Use ALL available CPU cores for parallel computation.
- -1 = all cores. 1 = single core. 4 = four cores.

---

## 10. Evaluation Metrics — Every Metric

### Confusion Matrix
A 2×2 table showing all possible prediction outcomes:

```
                    Predicted Real    Predicted Fake
Actual Real             TN                FP
Actual Fake             FN                TP
```

- **TP (True Positive)**: Model correctly identified Fake news as Fake
- **TN (True Negative)**: Model correctly identified Real news as Real
- **FP (False Positive)**: Model incorrectly said Real news was Fake (false alarm)
- **FN (False Negative)**: Model incorrectly said Fake news was Real (missed it)

### Accuracy
- `Accuracy = (TP + TN) / (TP + TN + FP + FN)` = fraction of all predictions that are correct
- **Our result**: 0.9931 = 99.31% of predictions are correct
- **Limitation**: Can be misleading with imbalanced classes

### Precision
- `Precision = TP / (TP + FP)` = "Of all articles the model called Fake, how many were actually Fake?"
- **Our result**: 0.9944 = when the model says "Fake", it's right 99.44% of the time
- **High precision** means few false alarms
- **Matters when**: The cost of false accusations is high (e.g., flagging a legitimate article as fake could harm a journalist's reputation)

### Recall (Sensitivity)
- `Recall = TP / (TP + FN)` = "Of all articles that ARE actually Fake, how many did the model catch?"
- **Our result**: 0.9905 = the model catches 99.05% of all fake news
- **High recall** means few missed fakes
- **Matters when**: The cost of missing fake news is high (misinformation spreads unchecked)

### F1 Score
- `F1 = 2 × (Precision × Recall) / (Precision + Recall)` = harmonic mean of precision and recall
- **Harmonic mean**: Penalizes models that sacrifice one metric for the other. If precision=100% but recall=1%, F1=1.98% (very low).
- **Our result**: 0.9924 = excellent balance between precision and recall
- **Why F1 is our primary metric**: Both false positives (censoring real news) and false negatives (missing fake news) are harmful. F1 balances both.

### AUC-ROC (Area Under the Receiver Operating Characteristic Curve)

**What is ROC?**
- **ROC curve** = a plot of True Positive Rate (Recall) vs False Positive Rate at different classification thresholds.
- **Threshold**: The probability cutoff for classifying as Fake. Default is 0.5, but you can change it.
  - Threshold = 0.3 → more articles classified as Fake (higher recall, lower precision)
  - Threshold = 0.7 → fewer articles classified as Fake (lower recall, higher precision)
- The ROC curve shows how precision-recall tradeoff changes as you move the threshold.

**What is AUC?**
- **AUC** = Area Under the Curve = the total area under the ROC curve.
- **Range**: 0 to 1
  - 1.0 = perfect classifier (100% recall at 0% false positive rate)
  - 0.5 = random classifier (diagonal line)
  - 0.0 = perfectly wrong classifier
- **Our result**: 0.9997 = almost perfect discrimination between Fake and Real
- **Why AUC matters**: It measures performance across ALL thresholds, not just 0.5. It tells you: "No matter what threshold you choose, this model will perform well."

---

## 11. Bias Probing — Every Probe

### What is bias probing?
- **Bias probing** = testing whether a simple model can predict the label using ONLY a specific type of feature (like entity names or text length).
- If a simple model achieves high accuracy using only entity names → the dataset has entity bias → a complex model trained on this data might be learning the same shortcut.

### Probe 1: Entity Bias (Accuracy: 91.45% — BIAS DETECTED)

**How it works:**
1. Sample 2,000 texts (stratified by label)
2. Extract ALL named entities using spaCy: "Donald Trump", "Google", "Paris"
3. Create a **bag of entities** (like bag of words, but only entity names)
4. Train a **Naive Bayes** classifier on entity bag alone
5. If accuracy > 60% → bias detected

**What is Naive Bayes?**
- A simple probabilistic classifier based on Bayes' theorem.
- "Naive" because it assumes all features are independent (which is rarely true, but works surprisingly well).
- It's deliberately simple — if even this simple model can predict well, the task is too easy (shortcut exists).

**Result: 91.45%** — A Naive Bayes using ONLY entity names can predict Fake/Real with 91% accuracy. This is terrible! It means the model can largely ignore the actual content and just look at WHO is mentioned.

**Why it's not 50%:**
- ISOT's real articles are all from Reuters → mention certain entities consistently
- ISOT's fake articles come from various sources → mention different entities
- Even with entity masking (replacing names with [PERSON]), some entities slip through (spaCy isn't perfect)
- The ISOT dataset has **structural bias** beyond just names (article formatting, topic distribution)

### Probe 2: Length Bias (Accuracy: 54.69% — OK)

**How it works:**
1. Compute text length (character count) for each article
2. Train Logistic Regression on length alone (1 feature)
3. If accuracy > 60% → bias detected

**Result: 54.69%** — Barely above random (50%). Text length alone can't predict Fake/Real. This means our entity masking and byline stripping successfully removed length-based shortcuts.

### Probe 3: Topic Bias (Accuracy: 54.21% — OK)

**How it works:**
1. Define keyword sets for 5 topics: politics, health, economy, science, military
2. For each text: count keyword matches per topic, assign the topic with most matches
3. Train Logistic Regression on topic labels alone
4. If accuracy > 60% → bias detected

**Result: 54.21%** — Near random. The model can't predict Fake/Real from topic alone. Good — it means the model isn't learning "health articles = fake".

### Threshold: 60%
- If a probe accuracy is above 60% (well above random 50%), we flag it as bias.
- Only entity bias exceeds this threshold.

---

## 12. Explainability — LIME, SHAP, SVD Backprojection

### Why explainability matters
- A model that says "FAKE" with no explanation is useless for a journalist.
- They need to know: "WHICH words triggered this? Is it because of sensational language, or because of a specific claim?"
- Explainability also helps detect bias — if the top reason is "Trump" instead of "sensational language", the model is biased.

### LIME (Local Interpretable Model-agnostic Explanations)

**Created by:** Ribeiro, Singh, Guestrin (2016, KDD conference)

**What each word in the name means:**
- **Local**: Explains ONE prediction at a time (not the entire model)
- **Interpretable**: The explanation is human-readable (list of words + importance scores)
- **Model-agnostic**: Works with ANY model — doesn't need to understand the model's internals
- **Explanations**: Outputs a list of features (words) that pushed the prediction toward Fake or Real

**How LIME works (step by step):**
1. Take one input text: "BREAKING!!! Government HIDING truth!!!"
2. Create 1,000 **perturbations**: randomly remove words
   - "BREAKING Government HIDING truth"
   - "BREAKING!!! HIDING truth!!!"
   - "Government truth!!!"
   - ... (1,000 variants)
3. Run EACH perturbation through the full model → get predictions
4. Train a simple **local linear model** (like Logistic Regression) on: which words were present/absent → how did the prediction change?
5. The linear model's weights = word importance scores
6. Output: "BREAKING pushed +0.3 toward Fake, HIDING pushed +0.2 toward Fake, Government pushed -0.1 toward Real"

**Our configuration:**
- `num_features=15`: Show top 15 most important words
- `num_samples=1000`: Create 1,000 perturbations per explanation
- Output: Interactive HTML file showing words colored by contribution (green = Real, red = Fake)

### SHAP (SHapley Additive exPlanations)

**Created by:** Lundberg & Lee (2017, NeurIPS conference)

**What is it?**
- SHAP assigns each feature a **Shapley value** — a concept from cooperative game theory.

**What is a Shapley value?**
- Imagine each feature is a "player" in a game where the "payout" is the model's prediction.
- The Shapley value answers: "How much does each player (feature) contribute to the final prediction?"
- It considers ALL possible combinations of features — what happens if you add/remove each feature in every possible order?
- This is the ONLY method that satisfies all fairness axioms (mathematical guarantees of fair attribution).

**Types of SHAP explainers we use:**
1. **TreeExplainer**: For Random Forest — exploits tree structure for fast exact computation. O(n) per instance.
2. **LinearExplainer**: For Logistic Regression — uses linear model structure.
3. **KernelExplainer**: For SVM and Ensemble — model-agnostic (like LIME but with Shapley values). Slower but works with any model.

**SHAP outputs:**
- **Summary plot (beeswarm)**: Shows top 20 features, with each dot = one data point. Position = SHAP value (positive = pushes toward Fake). Color = feature value (red = high, blue = low).
- **Bar plot**: Shows mean |SHAP value| for each feature — which features are most important overall.

**SHAP for bias auditing:**
- After computing SHAP values, we check if entity-related features (containing "person", "org", "reuters", "trump", etc.) appear in the top 20 most important features.
- If they do → the model is still relying on entity shortcuts despite masking.

### SVD Backprojection

**What is it?**
- SVD features are 150 abstract dimensions — not human-readable. Dimension 42 might represent a mix of "political vocabulary" and "formal tone" — you can't tell just by looking.
- **Backprojection** reverses the SVD transformation to map abstract dimensions back to original TF-IDF terms.
- This tells you: "The model's prediction was most influenced by the TF-IDF components corresponding to words like 'conspiracy', 'sources said', 'breaking'"

**In practice:** LIME already operates on original words (via the predict_fn that runs the full pipeline), so SVD backprojection is mainly useful for understanding which abstract SVD dimensions are important.

---

## 13. Dual-Mode System — Article vs Claim

### Why two modes?
- Long articles (500+ words) have enough text for statistical analysis (TF-IDF, stylometric features).
- Short claims ("India landed on the moon in 2023") don't — they need fact-checking against external sources.

### How input type is detected (claim_detector.py):

The system classifies input into 5 types using **rule-based** detection (no ML needed):

1. **Article** (>50 words, not a question): Routed to ML pipeline
2. **Temporal** (contains date/time: "Is today April 11th?"): Routed to temporal verification
3. **Question** (starts with "is", "are", "did", "who", "what"): Routed to fact-check
4. **Claim** (short factual assertion with verbs like "discovered", "announced", "landed"): Routed to fact-check
5. **Opinion** (contains "I think", "I believe", "probably"): Flagged as unverifiable

### Article Mode (ML Pipeline):
```
Text → Preprocess → Feature Engineering (267d) → Stacking Ensemble → Verdict + Confidence + Explanation
```

### Claim Mode (Fact Verification):
```
Claim → Temporal Check → Wikipedia Lookup → Web Search → Evidence Aggregation → Verdict + Sources
```

---

## 14. Fact Verification Pipeline

### Temporal Check
- **What**: Verifies date/time claims against the system clock.
- **Example**: "Is today April 13th 2026?" → Check `datetime.now()` → True/False
- **Uses**: regex for date parsing + `dateparser` library as fallback
- **Confidence**: Very high (0.95) when applicable — dates are objectively verifiable

### Wikipedia Lookup
- **What**: Searches Wikipedia for factual claims.
- **How**:
  1. Extract entity names from the claim (capitalized phrases)
  2. Query MediaWiki API (Wikipedia's search engine)
  3. Fetch article summaries
  4. Compute **word overlap similarity** between claim and Wikipedia article
- **Example**: "India landed on the moon in 2023" → Searches "India moon landing" → Finds Chandrayaan-3 article → Confirms claim
- **Similarity scoring**: Recall-based (what fraction of claim words appear in the Wikipedia article)
- **Free**: No API key needed

### Web Search (DuckDuckGo)
- **What**: Searches the web for supporting/contradicting evidence.
- **How**:
  1. Use DuckDuckGo's search API (free, no API key)
  2. Fetch top 5 results
  3. For each result, check:
     - **Similarity**: How much text overlap with the claim
     - **Stance**: Does it support or contradict?
       - Support keywords: "confirmed", "verified", "true", "announced"
       - Contradiction keywords: "false", "fake", "hoax", "debunked", "misleading"
     - **Reliability**: How trustworthy is the source?
       - `.gov`, `.edu`, WHO, UN → 0.9 (very reliable)
       - Wikipedia, NYT, BBC → 0.8
       - `.org`, CNN, CBS → 0.6
       - Unknown → 0.4
  4. Count supporting vs contradicting sources

### Evidence Aggregation
- Combines all evidence sources using **weighted voting**:
  - Temporal: weight 0.95 (near-certain)
  - Wikipedia: weight 0.80
  - Web search: weight 0.65
  - ML style analysis: weight 0.50 (supplementary)
- Computes a **truth score**: weighted average of all evidence
- **Final verdict**:
  - truth_score >= 0.75 → "Verified"
  - >= 0.60 → "Likely True"
  - >= 0.40 → "Uncertain"
  - >= 0.25 → "Likely False"
  - < 0.25 → "False"
- **Confidence capped at 95%**: Never claims 100% certainty

---

## 15. Tools & Technologies — Every Library

### Core ML

| Library | What it does |
|---------|-------------|
| **scikit-learn** | The main ML library. Provides all models (SVM, LR, RF), GridSearchCV, StackingClassifier, metrics (accuracy, F1, AUC), StandardScaler, TruncatedSVD, TfidfVectorizer, StratifiedKFold, train_test_split. Created by INRIA (French research institute). |
| **pandas** | Data manipulation library. DataFrames (tables) for loading CSVs, filtering rows, adding columns, merging datasets. Think of it as Excel for Python. |
| **numpy** | Numerical computing library. Arrays, matrix operations, mean, std, log. The foundation that pandas and scikit-learn are built on. |
| **scipy** | Scientific computing. Used internally by scikit-learn for sparse matrices and optimization. |
| **joblib** | Model serialization — saving trained models to disk (`.joblib` files) and loading them back. More efficient than Python's `pickle` for large numpy arrays. |

### NLP (Natural Language Processing)

| Library | What it does |
|---------|-------------|
| **spaCy** | Industrial-strength NLP library. We use it for: NER (Named Entity Recognition), lemmatization, tokenization. Model: `en_core_web_sm` (small English model, ~12MB, trained on web text). |
| **NLTK** | Natural Language Toolkit. We use it for: English stopword list (179 words). Older than spaCy, more academic. |
| **gensim** | Topic modeling and word embeddings library. We use it to load GloVe pre-trained word vectors (`glove-wiki-gigaword-100`). Incompatible with Python 3.14+. |
| **vaderSentiment** | Sentiment analysis tool. VADER = Valence Aware Dictionary and sEntiment Reasoner. Designed for social media text. Returns compound, positive, negative, neutral scores. |
| **textstat** | Readability metrics library. Computes Flesch Reading Ease, Automated Readability Index, and other readability scores. |

### Explainability

| Library | What it does |
|---------|-------------|
| **LIME** | Local Interpretable Model-agnostic Explanations. Creates perturbation-based explanations for individual predictions. Outputs interactive HTML visualizations. |
| **SHAP** | SHapley Additive exPlanations. Computes feature importance using game-theoretic Shapley values. Provides TreeExplainer, LinearExplainer, KernelExplainer. Generates summary and bar plots. |

### Visualization

| Library | What it does |
|---------|-------------|
| **matplotlib** | The foundational Python plotting library. Creates static plots: confusion matrices, ROC curves, calibration curves. |
| **seaborn** | Built on matplotlib. Provides prettier statistical visualizations: heatmaps, distribution plots. We use it for confusion matrix heatmaps. |

### Web & Data

| Library | What it does |
|---------|-------------|
| **Gradio** | Python library for creating web UIs for ML models. Creates interactive web apps with textboxes, labels, markdown output. Runs on `localhost:7860`. No frontend coding needed. |
| **kagglehub** | Downloads datasets from Kaggle. We use it to download the ISOT dataset. |
| **datasets** (HuggingFace) | Downloads datasets from HuggingFace Hub. We use it to download the LIAR dataset. |
| **wikipedia-api** | Python wrapper for MediaWiki API. Fetches Wikipedia article summaries for fact-checking. |
| **ddgs** (DuckDuckGo Search) | Python wrapper for DuckDuckGo search. Returns search results without needing an API key. Free and private. |
| **dateparser** | Robust date parsing library. Handles formats like "April 11th", "11/4/2026", "next Thursday". Used as fallback in temporal verification. |
| **requests** | HTTP library for making web requests. Used by fact-checker to query APIs. |
| **tqdm** | Progress bar library. Shows progress during long operations like preprocessing 44,000 articles. |

### Testing

| Library | What it does |
|---------|-------------|
| **pytest** | Python testing framework. We have 47 tests covering preprocessor, feature engineer, and fact verification. Fixtures, parametrize, and assert-based testing. |

---

## 16. Results — Every Number Explained

### In-Domain Test Results (ISOT, 20% held-out)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| SVM | 0.9935 (99.35%) | 0.9955 | 0.9902 | 0.9929 | 0.9997 |
| LR | 0.9932 (99.32%) | 0.9947 | 0.9905 | 0.9926 | 0.9997 |
| RF | 0.9748 (97.48%) | 0.9785 | 0.9662 | 0.9723 | 0.9963 |
| Ensemble | 0.9931 (99.31%) | 0.9944 | 0.9905 | 0.9924 | 0.9997 |

**What these numbers mean:**
- **SVM is the best individual model** (F1=0.9929) — linear SVMs excel on high-dimensional text features
- **Random Forest is the weakest** (F1=0.9723) — tree-based models can overfit to specific word patterns
- **Ensemble matches SVM** closely (F1=0.9924) — the meta-learner learned to weight SVM and LR highest
- **AUC-ROC of 0.9997** — near-perfect class separation (the model almost never confuses Fake and Real)

### Cross-Validation Results

| Metric | Mean | Std Dev |
|--------|------|---------|
| F1 | 0.9920 | ±0.0012 |
| AUC-ROC | 0.9995 | ±0.0001 |

- **Mean 0.9920**: Average F1 across 5 folds — very consistent
- **Std Dev 0.0012**: Extremely low variation — the model performs equally well on different subsets of data
- **Train-Test gap**: ~0.003 (0.33%) — no significant overfitting

### Training Times

| Model | Time |
|-------|------|
| SVM | 8.1 seconds |
| LR | 0.9 seconds |
| RF | 650.2 seconds (~11 minutes) |
| Ensemble | 47.0 seconds |

- LR is fastest (<1 second) because it's a simple linear model
- RF is slowest because GridSearchCV tries 18 parameter combinations × 5 folds × 100 trees each

---

## 17. Cross-Dataset Generalization

### What is cross-dataset generalization?
- Train the model on Dataset A, test it on completely different Dataset B.
- This tests: "Did the model learn real patterns, or just Dataset A's quirks?"

### Results

| Train → Test | Ensemble F1 | AUC-ROC |
|-------------|-------------|---------|
| ISOT → LIAR | 0.2345 | 0.5144 |
| LIAR → ISOT | 0.1269 | 0.5038 |

### What these numbers mean:
- **0.2345 F1**: Almost random. The model trained on ISOT articles is nearly useless on LIAR claims.
- **0.5144 AUC**: Barely above 0.5 (random). The model has no discriminative ability on unseen data.
- **76% F1 drop**: From 0.9924 (in-domain) to 0.2345 (cross-dataset) = 76% absolute drop.

### Why this happens:
1. **Text length mismatch**: ISOT articles are 500-3000 words. LIAR claims are 15-30 words. TF-IDF vocabulary learned on long articles doesn't transfer to short claims.
2. **Domain mismatch**: ISOT = news articles about politics, world events. LIAR = fact-checked political statements. Different writing styles entirely.
3. **Source bias**: ISOT real articles are all from Reuters. LIAR has no Reuters text. The model learned "Reuters-style writing = Real" which doesn't exist in LIAR.

### Why we report this honestly:
- Most fake news papers ONLY report in-domain accuracy (99%+) and claim success.
- We explicitly measure and report the cross-dataset gap — this is **scientifically honest**.
- It shows that the problem of fake news detection is far from solved.

---

## 18. Error Analysis

### Overview
- **54 errors** out of ~8,980 test samples (0.60% error rate)
- 34 False Negatives (Fake predicted as Real) — 63%
- 20 False Positives (Real predicted as Fake) — 37%

### What the confidence distribution tells us:
- **31 errors (57%) at low confidence (<0.4)**: The model was uncertain and guessed wrong. This is acceptable — the model "knows it doesn't know."
- **14 errors at high confidence (>0.8)**: The model was confidently wrong. These are the most dangerous cases.
- **9 errors at borderline (0.4-0.6)**: The model was on the fence.

### Key insight: "The model knows when it doesn't know"
- 57% of errors happen when the model is uncertain — it could flag these predictions as "needs human review" instead of giving a definitive answer.
- A confidence threshold system could dramatically reduce harmful errors.

---

## 19. Glossary of Every Technical Term

| Term | Meaning |
|------|---------|
| **Accuracy** | Fraction of correct predictions out of all predictions |
| **ARI** | Automated Readability Index — grade level readability score |
| **AUC** | Area Under the Curve — total area under the ROC curve |
| **Bagging** | Bootstrap Aggregating — training each model on a random subset of data |
| **Bias (ML)** | Model learning shortcuts instead of real patterns |
| **Bias (statistical)** | Systematic error in predictions |
| **Bigram** | A pair of consecutive words (e.g., "fake news") |
| **Binary classification** | Predicting one of two classes (Fake/Real) |
| **Black box** | A model whose internal logic is not understandable |
| **Bootstrap** | Sampling with replacement from a dataset |
| **Burstiness** | Variation in sentence lengths (std/mean) |
| **Calibration** | How well predicted probabilities match actual frequencies |
| **CalibratedClassifierCV** | Wrapper that adds probability calibration to models without predict_proba |
| **Class weight** | Adjusting importance of each class to handle imbalance |
| **Compound score** | VADER's overall sentiment score (-1 to +1) |
| **Confusion matrix** | Table showing TP, TN, FP, FN counts |
| **Cross-dataset** | Training on one dataset, testing on a completely different one |
| **Cross-validation (CV)** | Evaluating by splitting data into k folds and rotating train/test |
| **Data leakage** | Accidentally using test data information during training |
| **Decision boundary** | The surface that separates classes in feature space |
| **Decision tree** | Tree-like model making sequential yes/no decisions |
| **Dense vector** | A vector where most values are non-zero |
| **Dimensionality reduction** | Reducing the number of features while preserving information |
| **DuckDuckGo** | Privacy-focused search engine with a free API |
| **Embedding** | A dense vector representation of a word/document that captures meaning |
| **Ensemble** | Combining multiple models to improve predictions |
| **Entity** | A real-world object with a name (person, place, organization) |
| **Entity masking** | Replacing entity names with generic tokens to prevent bias |
| **F1 score** | Harmonic mean of precision and recall |
| **False negative (FN)** | Fake news incorrectly predicted as Real (missed) |
| **False positive (FP)** | Real news incorrectly predicted as Fake (false alarm) |
| **Feature** | A measurable property of data used by the model |
| **Feature engineering** | Creating/extracting features from raw data |
| **Feature vector** | The list of all features for one data point |
| **Flesch Reading Ease** | Readability score (0-100, higher = easier to read) |
| **GloVe** | Global Vectors — pre-trained word embeddings from Stanford |
| **Gradio** | Python library for creating ML web demos |
| **GridSearchCV** | Exhaustive search over hyperparameter combinations with CV |
| **Harmonic mean** | A type of average that penalizes extreme imbalance between values |
| **Hyperparameter** | A setting chosen before training (not learned from data) |
| **Hyperplane** | Decision boundary in high-dimensional space |
| **IDF** | Inverse Document Frequency — how rare a word is across documents |
| **In-domain** | Testing on data from the same distribution as training |
| **Isotonic regression** | Non-parametric method for probability calibration |
| **K-fold CV** | Cross-validation with k splits |
| **Kaggle** | Platform for ML datasets and competitions |
| **L-BFGS** | An optimization algorithm for finding model parameters |
| **L2 regularization** | Penalty on sum of squared weights to prevent overfitting |
| **Label encoding** | Converting categorical labels to numbers (Real=0, Fake=1) |
| **Latent** | Hidden/abstract (not directly observable) |
| **Lemma** | The base/dictionary form of a word |
| **Lemmatization** | Reducing words to their lemma form |
| **LIME** | Local Interpretable Model-agnostic Explanations |
| **Linear kernel** | SVM using a flat hyperplane (no curves) |
| **LinearSVC** | Linear Support Vector Classification (fast O(n) training) |
| **LSA** | Latent Semantic Analysis — using SVD on TF-IDF for topic discovery |
| **Majority voting** | Ensemble method where the prediction is whatever most models agree on |
| **Margin** | Distance between decision boundary and nearest data points (SVM) |
| **MediaWiki API** | Wikipedia's search and content retrieval API |
| **Meta-learner** | The second-level model in stacking that combines base model outputs |
| **Model-agnostic** | Works with any type of model |
| **N-gram** | A sequence of n consecutive words |
| **Naive Bayes** | Probabilistic classifier assuming feature independence |
| **Named Entity Recognition (NER)** | Identifying and classifying proper nouns in text |
| **NLP** | Natural Language Processing — teaching computers to understand text |
| **Normalization** | Scaling features to a standard range |
| **Overfitting** | Model memorizes training data noise instead of learning general patterns |
| **Parameter** | A value learned from data during training (e.g., weights) |
| **Passthrough** | Including original features alongside meta-features in stacking |
| **Perturbation** | Slightly modifying input to observe prediction changes |
| **Precision** | Of predicted positives, how many are actually positive |
| **predict_proba** | Method that returns probability predictions instead of class labels |
| **Random state** | Seed for random number generator (ensures reproducibility) |
| **Recall** | Of actual positives, how many were correctly predicted |
| **Regularization** | Techniques to prevent overfitting (L1, L2, dropout, etc.) |
| **Reproducibility** | Ability to get the same results when re-running the experiment |
| **ROC curve** | Plot of True Positive Rate vs False Positive Rate |
| **SDG** | Sustainable Development Goals (United Nations) |
| **Semantic** | Related to meaning (not just statistics) |
| **Sentiment** | The emotional tone of text (positive/negative/neutral) |
| **SHAP** | SHapley Additive exPlanations — game-theoretic feature attribution |
| **Shapley value** | Fair attribution of contribution from game theory |
| **Sigmoid function** | S-shaped function that maps any number to (0, 1) |
| **Sparse matrix/vector** | A matrix/vector where most values are zero |
| **spaCy** | Industrial NLP library for Python |
| **Stacking** | Ensemble method with base models + meta-learner |
| **StandardScaler** | Normalizes features to mean=0, std=1 |
| **Stemming** | Crudely chopping word endings (less accurate than lemmatization) |
| **Stopwords** | Very common words with little meaning ("the", "is", "at") |
| **Stratified** | Maintaining class proportions in splits |
| **Stylometric** | Related to statistical analysis of writing style |
| **Sublinear TF** | Using log(1 + count) instead of raw count for term frequency |
| **Support vector** | Data points closest to the decision boundary in SVM |
| **SVD** | Singular Value Decomposition — matrix factorization for dimensionality reduction |
| **TF-IDF** | Term Frequency-Inverse Document Frequency — word importance measure |
| **Threshold** | Cutoff value for converting probability to class label |
| **Token** | A word or word-piece (unit of text) |
| **Tokenization** | Splitting text into tokens |
| **True positive (TP)** | Correctly predicted positive (Fake correctly identified as Fake) |
| **True negative (TN)** | Correctly predicted negative (Real correctly identified as Real) |
| **TruncatedSVD** | SVD keeping only the top-k components |
| **Type-token ratio** | Unique words / total words (vocabulary diversity) |
| **Underfitting** | Model is too simple to capture patterns in data |
| **Unigram** | A single word |
| **VADER** | Valence Aware Dictionary and sEntiment Reasoner |
| **Variance** | How spread out values are; in SVD, how much information is captured |
| **Vocabulary** | The set of all unique terms in the corpus |
| **Zipf's law** | In natural language, word frequency is inversely proportional to rank |
| **Zipf coefficient** | The slope of the log-log word frequency plot |

---

*This document covers every concept used in the TruthLens Phase 3 presentation. Use it as a study guide for your viva/presentation — if an examiner asks about any term, the explanation is here.*
