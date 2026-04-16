# Gamma AI Prompts — TruthLens Phase 3 Final Presentation

Use each prompt separately in Gamma AI. Together they produce a 19-slide presentation.
Everything is written in simple language — no jargon walls.

---

## PROMPT 1 OF 3 (Slides 1–7: Title, Problem, Literature, Gaps, Architecture, Dataset)

```
Create a colorful, modern, easy-to-understand presentation (7 slides) for a college project. Use a clean dark/blue theme with big visuals, icons, and simple bullet points. NO walls of text. Write everything in plain, simple English — imagine explaining to a 12-year-old. Avoid technical jargon wherever possible, and when you must use a technical word, add a short "(= simple explanation)" next to it.

IMPORTANT: Wherever I ask for a block diagram or flowchart, you MUST generate an actual visual diagram with colored boxes and arrows — NOT text or ASCII art.

Slide 1 — Title Slide
Title: "TruthLens: A Smart Fake News Detector"
Subtitle: ML Mini Project — Phase 3 Final Submission
Team: Riddhi Patil (52), Prashant Jha (54), Aditya Soni (57), Riwan Pereira (65)
Guide: Dr. Joanne Gomes
Institute: St. Francis Institute of Technology, Mumbai
Course: C403 — Machine Learning Mini Project

Slide 2 — Problem Statement
Heading: "What Problem Are We Solving?"
Use simple language and icons:
- Fake news spreads 6 times faster than real news on the internet (proven by MIT researchers in 2018)
- Current computer programs that detect fake news "cheat" — they memorize which newspaper name appears, instead of actually reading and understanding the article. Example: They learn "if the article mentions Reuters, it's real" instead of checking if the content is truthful.
- Our goal: Build a system called TruthLens that actually reads the article's writing style, checks facts, and explains WHY it thinks something is fake or real
- Who uses this: Journalists, students, fact-checkers, anyone who reads news online
- What it gives you: A verdict (Fake or Real), a confidence percentage, and a clear explanation
- SDG Connection: UN Goal 16 — Peace & Justice. Fighting fake news helps people make better decisions and strengthens trust in society.
Add a simple before/after visual: "Old detectors: just check the newspaper name" vs "TruthLens: reads the content + checks facts + explains why"

Slide 3 — Literature Survey (What Others Did)
Heading: "What Did Researchers Before Us Do?"
Make a simple, colorful table:
| # | Who? | What They Did | Dataset Used | Method | What They Looked At |
| 1 | Ahmed et al., 2017 | Detected fake news articles | ISOT — 44,000 news articles | Counted word patterns (N-grams) + basic ML | Word frequencies |
| 2 | Wang, 2017 | Checked if political statements are true | LIAR — 12,800 short claims | Deep learning (CNN + LSTM) | Statement text + who said it |
| 3 | Shu et al., 2017 | Surveyed all fake news methods | Multiple sources | Literature review | Content + social media context |
| 4 | Vosoughi et al., 2018 | Studied how fake news spreads | Twitter data | Statistical analysis | How fast news travels |
| 5 | Ribeiro et al., 2016 | Made AI explain its decisions | Any ML model | LIME (asks "which words mattered?") | Individual word importance |
| 6 | Lundberg & Lee, 2017 | Made AI explain its decisions (different way) | Any ML model | SHAP (fair math-based scoring) | Feature importance scores |
| 7 | Pennington et al., 2014 | Taught computers word meanings | Wikipedia + news text | GloVe word vectors | Word relationships |

Slide 4 — Literature Survey (What Was Missing?)
Heading: "What Was Missing in Their Work? (And What We Added)"
Make a colorful table:
| # | Their Accuracy | What Was Wrong | The Gap We Found | What We Did About It |
| 1 | 92% | Didn't remove name-cheating. Model just memorized "Reuters = Real" | No debiasing (= removing cheating shortcuts) | We hide all names: people, companies, news sources — replaced with [PERSON], [ORG], [SOURCE] |
| 2 | 27% (6 classes) | Only tested on one type of text, no explanations given | Never tested on different types of data | We test across datasets: train on long articles, test on short claims (and honestly report the drop) |
| 3 | Survey only | No actual working system built | No way to measure if a dataset itself is biased | We built bias-checking tools that measure 3 types of cheating (name bias, length bias, topic bias) |
| 4 | N/A | Only studied spread, didn't build a detector | Didn't look at writing style | We extract 17 writing style features — including checks for AI-generated text |
| 5 | N/A | Explanation method is slow (needs 1000 tries per article) | Nobody combined explanations WITH debiasing | We combined LIME + SHAP + bias checking into one pipeline |
| 6 | N/A | Needs extra data to work | Only explains, doesn't debias | We integrated it directly into our system |
| 7 | N/A | Old-style word meanings, not tuned for fake news | No fake-news-specific features | We combined 3 types of features: word stats (150 numbers) + word meanings (100 numbers) + writing style (17 numbers) = 267 numbers per article |

Slide 5 — Research Gaps (Visual)
Heading: "5 Big Problems Nobody Solved (Until Now)"
Show as 5 large, colorful icon cards:
1. "Name Cheating" — Old models memorize names like "Trump" or "Reuters" instead of reading content. We hide all names before training.
2. "Never Tested on New Data" — Models claim 95%+ accuracy but only test on the SAME data they trained on. We tested on completely different data and found accuracy drops from 99% to 23%. Honestly reported.
3. "No Explanations" — Models say "FAKE" but never say WHY. Ours shows exactly which words triggered the decision.
4. "Can't Spot AI-Written Fakes" — Old models ignore writing patterns. We check if sentences are too uniform (a sign of AI writing) and if word usage follows natural patterns.
5. "Dumb Voting" — Old models combine multiple models by simple majority vote. Ours uses a smart "referee" model that learns which model to trust more.

Slide 6 — System Architecture (BLOCK DIAGRAM)
Heading: "How TruthLens Works — The Big Picture"
IMPORTANT: Generate an actual VISUAL BLOCK DIAGRAM with colored boxes and arrows. NOT text. This is the most important slide.

Show this flow:
Top: Big box "You paste text here"
Arrow down to: "What type of input is this?" decision box

LEFT PATH (labeled "Long Article — more than 50 words"):
  Arrow to "Step 1: Clean the text" box (light blue) — remove links, HTML, fix spacing
  Arrow to "Step 2: Hide all names" box (light blue) — replace Trump→[PERSON], Reuters→[SOURCE], Paris→[LOC]
  Arrow to "Step 3: Break into words" box (light blue) — remove common words like "the", "is", reduce to base form
  Arrow to 3 parallel boxes:
    Box A (blue): "Count important words → compress to 150 numbers" (TF-IDF + SVD)
    Box B (green): "Look up word meanings → 100 numbers" (GloVe)
    Box C (orange): "Measure writing style → 17 numbers" (sentiment, readability, caps, punctuation)
  All 3 merge into: "Combine into 267 numbers per article"
  Arrow to: "Normalize all numbers to same scale"
  Arrow to: "Ask 3 models + 1 referee model" box containing:
    SVM (finds the best dividing line)
    Logistic Regression (calculates probability)
    Random Forest (100 mini decision trees vote)
    → Referee (Logistic Regression) combines their answers
  Arrow to: "VERDICT: Fake or Real + Confidence %" (green box)
  Arrow to: "Explain WHY" box — which words mattered (LIME) + which features mattered (SHAP) + bias check

RIGHT PATH (labeled "Short Claim — less than 50 words"):
  Arrow to "Fact-Check Pipeline" box:
    Check 1: Is it a date/time claim? → verify against today's date
    Check 2: Search Wikipedia for facts
    Check 3: Search the web (DuckDuckGo)
  Arrow to: "Combine all evidence → VERDICT + Sources"

Also show a small box for "Opinion detected" → "Cannot verify — it's just an opinion"

Slide 7 — Dataset & Preprocessing
Heading: "Our Data & How We Cleaned It"
Simple dataset table with icons:
| Dataset | What Is It? | How Many? | Fake | Real | How Long? |
| ISOT | Full news articles (from Kaggle) | 44,898 | 23,481 (52%) | 21,417 (48%) | 500–3000 words each |
| LIAR | Short political claims (from PolitiFact) | 12,836 | 6,426 (50%) | 6,410 (50%) | 15–30 words each |
| Combined | Both merged | 57,734 | ~30,000 | ~28,000 | Mixed |

Split: 80% for training, 20% for testing (keeping the same fake/real ratio in both)

Preprocessing — show as a visual pipeline with 6 numbered steps and arrows:
1. "Remove bylines" — Delete things like "WASHINGTON (Reuters) -" that give away the source
2. "Clean the text" — Remove web links, HTML code, email addresses, weird symbols
3. "Hide people & org names" — Use AI (spaCy) to find names → replace with [PERSON], [ORG], [LOC], [DATE]
4. "Hide news source names" — Manually catch 51 news names (Reuters, CNN, BBC, Fox...) → replace with [SOURCE]
5. "Remove filler words" — Delete "the", "is", "and", "of" — words that appear everywhere and mean nothing for classification
6. "Reduce words to base form" — "running"→"run", "better"→"good", "studies"→"study"

Final output: Each article becomes 267 numbers that capture its content, meaning, and writing style.
```

---

## PROMPT 2 OF 3 (Slides 8–13: How It Works, Results, Demo, What's New)

```
Continue the same colorful, modern, easy-to-understand presentation. Same dark/blue theme. 6 more slides. Keep language very simple — explain like you're talking to a 12-year-old. Use icons, tables, and visuals. IMPORTANT: Where I ask for a block diagram, generate an actual visual diagram with boxes and arrows.

Slide 8 — How Our System Works (Methodology)
Heading: "The Brains Behind TruthLens"

Section 1 — "Our 4 Models (Think of them as 4 judges)":
Show 4 colorful cards:
1. SVM (Support Vector Machine) — "Draws the best possible line between Fake and Real articles. Like drawing a border on a map between two countries, making the gap as wide as possible." Best setting: C=0.01. Trained in 8 seconds.
2. Logistic Regression — "Calculates the probability of an article being fake. Like a weather forecast: '87% chance of being Real.' Very fast — trained in under 1 second."
3. Random Forest — "100 mini decision trees each vote. Each tree asks simple yes/no questions like 'Does it have lots of exclamation marks?' 'Is the sentiment angry?' Majority vote wins." Trained in 11 minutes.
4. Stacking Ensemble (The Referee) — "A smart referee that watches all 3 judges and learns who to trust more. If SVM says Fake and Random Forest says Real, the referee knows from experience which one is usually right."

Section 2 — "How We Found the Best Settings":
- We tried many different settings for each model (called GridSearchCV)
- For each setting, we tested it 5 different ways (5-fold cross-validation) to make sure it wasn't a fluke
- We picked the settings that got the highest F1 score (= the best balance between catching fakes and not false-alarming on real news)

Section 3 — "Tools We Used" (show as icon grid):
ML: scikit-learn (models), pandas (data tables), numpy (math)
Text Processing: spaCy (finds names, simplifies words), NLTK (common word list), GloVe via gensim (word meanings)
Style Analysis: VADER (mood detector), textstat (reading difficulty)
Explanations: LIME (which words mattered), SHAP (which features mattered)
Charts: matplotlib, seaborn
Web App: Gradio (instant web demo)
Fact-Checking: Wikipedia API, DuckDuckGo search, dateparser
Testing: pytest (47 automated tests)

Slide 9 — Model Comparison (Results Table)
Heading: "How Well Did Each Model Do?"

Main table (make it colorful — highlight the best numbers in green):
| Model | Accuracy | Precision (When it says Fake, is it right?) | Recall (Of all Fakes, how many did it catch?) | F1 Score (Overall balance) | AUC-ROC (How well it separates Fake from Real) |
| SVM | 99.35% | 99.55% | 99.02% | 99.29% | 99.97% |
| Logistic Regression | 99.32% | 99.47% | 99.05% | 99.26% | 99.97% |
| Random Forest | 97.48% | 97.85% | 96.62% | 97.23% | 99.63% |
| Stacking Ensemble (Referee) | 99.31% | 99.44% | 99.05% | 99.24% | 99.97% |

Below table:
"Consistency check: When we tested the Ensemble 5 different ways, it scored 99.20% ± 0.12% every time — very stable!"

Honest results (show with warning/red colors):
"BUT — when we trained on long articles (ISOT) and tested on short claims (LIAR):"
| Trained On | Tested On | F1 Score |
| Long articles (ISOT) | Short claims (LIAR) | 23.45% (almost random guessing!) |
| Short claims (LIAR) | Long articles (ISOT) | 12.69% (worse than guessing!) |
"This 76% drop proves the model learned article-specific patterns, not universal fake news patterns. We report this honestly — most researchers hide this."

Slide 10 — Best Model & Why We Chose It
Heading: "Our Winner: The Stacking Ensemble (Referee Model)"
Show the winning model like a trophy/podium:
- Accuracy: 99.31%
- F1 Score: 99.24%
- AUC-ROC: 99.97%

Why we picked the Referee over SVM (which scored slightly higher at 99.29%):
- "It combines the strengths of ALL 3 models — SVM's sharp boundary + Logistic Regression's probability estimates + Random Forest's pattern detection"
- "The referee learns WHO to trust — it doesn't just take a majority vote"
- "It gives well-calibrated confidence scores (when it says 90%, it's right ~90% of the time)"
- "It works with our explanation tools (LIME and SHAP)"
- "It's more stable — SVM might be slightly better today but the Ensemble is consistently good"

Simple analogy: "Imagine 3 teachers grade your essay. Instead of just averaging their scores, a principal who knows each teacher's strengths combines their feedback intelligently. That's our Ensemble."

Slide 11 — Results & Analysis
Heading: "What We Found"

Section 1 — "Errors (where the model got it wrong)":
- Only 54 mistakes out of ~8,980 test articles (0.60% error rate)
- 34 times it missed fake news (said Real when it was Fake) — 63% of errors
- 20 times it false-alarmed (said Fake when it was Real) — 37% of errors
- KEY INSIGHT: 57% of errors happened when the model was unsure (confidence below 40%). "The model knows when it doesn't know!" — these uncertain cases could be flagged for human review.

Section 2 — "Bias Check (did the model cheat?)":
Simple table:
| What We Checked | Score | Good or Bad? |
| Can you predict Fake/Real just from names? | 91.45% | BAD — names still leak info (should be ~50%) |
| Can you predict just from article length? | 54.69% | GOOD — almost random (50%), no length cheating |
| Can you predict just from topic keywords? | 54.21% | GOOD — almost random, no topic cheating |
"Our name-hiding reduced the cheating from ~95% to 91%, but didn't fully eliminate it. The ISOT dataset has deep structural bias beyond just names."

Section 3 — "Charts Generated":
- Confusion matrices for all 4 models (shows exactly where each model gets confused)
- ROC curves (shows how well models separate Fake from Real at different thresholds)
- Calibration curve (shows if predicted probabilities are accurate)
- SHAP plots (shows which features matter most across all articles)
- 3 LIME explanation reports (shows which words triggered specific predictions)

Slide 12 — Live Demo Examples
Heading: "See It In Action!"
Show 4 example cards with input→output:

Example 1 — Fake News:
Input: "BREAKING!!! Government HIDING the truth about vaccines!!! Share before they delete!!!"
Output: FAKE (94% confidence)
Why: Excessive exclamation marks, ALL CAPS words, sensationalist language, calls to action ("share before they delete")

Example 2 — Real News:
Input: "Scientists at NASA discovered evidence of water on Mars using spectroscopic analysis published in Nature."
Output: REAL (87% confidence)
Why: Neutral tone, specific source attribution (NASA, Nature), academic language, no emotional manipulation

Example 3 — Fact-Check:
Input: "India landed on the moon in 2023"
Output: VERIFIED (85% confidence)
How: Wikipedia confirms Chandrayaan-3 mission landed on August 23, 2023. Web search corroborates.

Example 4 — Opinion:
Input: "I think the government is hiding something"
Output: UNVERIFIABLE — This is a personal opinion, not a factual claim. Cannot be fact-checked.

"Try it yourself: Run 'python app.py' → opens a web page at localhost:7860"

Slide 13 — Our Contribution (What's New)
Heading: "What Makes TruthLens Different?"
Show 7 numbered contribution cards with simple icons:
1. "We hide names to stop cheating" — Replace all person/company/source names with tokens. 51 news sources caught by our custom list. Reduces name-bias from ~95% to 91%.
2. "267 numbers capture everything" — We combine word statistics (150) + word meanings (100) + writing style (17) into one powerful feature set. Includes AI-detection features.
3. "Smart referee, not dumb voting" — Our Stacking Ensemble learns which model to trust. Better than just counting votes.
4. "We explain every decision" — LIME shows which words mattered. SHAP shows which features mattered. Bias audit checks if names are still influencing predictions.
5. "We test honestly" — We report the 76% accuracy drop on different data. Most researchers hide this.
6. "Two modes: Articles + Claims" — Long articles go through ML analysis. Short claims get fact-checked against Wikipedia and the web.
7. "We measure 3 types of bias" — Name bias, length bias, topic bias — all quantified with numbers, not just guesses.
```

---

## PROMPT 3 OF 3 (Slides 14–19: Challenges, Future, Paper, Deliverables, Conclusion, References)

```
Final 6 slides of the same colorful, modern presentation. Same dark/blue theme. Keep language dead simple — a 12-year-old should understand every slide. Use icons, cards, and bullet points. No jargon without explanation.

Slide 14 — Challenges & How We Solved Them
Heading: "Problems We Faced (And How We Fixed Them)"
Show 6 challenge→solution pairs as colorful cards:

1. "The model broke on different data"
   Problem: Trained on long articles (ISOT), tested on short claims (LIAR) → accuracy crashed from 99% to 23%
   Why: Long articles and short claims are completely different types of text. It's like training a dog to fetch balls and then asking it to fetch frisbees.
   Fix: We offer a "combined training" mode that mixes both types. We also honestly report the gap.

2. "Name-hiding wasn't perfect"
   Problem: Even after hiding names, a simple model could still predict Fake/Real 91% of the time using leftover name patterns
   Why: Our name-finder (spaCy) missed some names. Also, the ISOT dataset has deeper patterns beyond names (all real articles follow Reuters writing style).
   Fix: Added manual list of 51 news source names. Added bias-checking tools to measure the problem.

3. "Random Forest was memorizing too much"
   Problem: Random Forest scored 99.98% on training data but only 97.23% on test data (overfitting = memorizing instead of learning)
   Fix: Limited tree depth to 20 levels. Made each tree only look at a random subset of features. Compressed word features from 10,000 to 150.

4. "GloVe word meanings library broke on new Python"
   Problem: The gensim library (which loads word meanings) doesn't work with Python 3.14+
   Fix: Added a --no-glove flag that skips word meanings. Feature vector shrinks from 267 to 167 numbers. Small performance impact.

5. "Explanations were too slow"
   Problem: Full LIME explanation needs 1,000 tries per article (~5 seconds each). Too slow for a web app.
   Fix: Use a faster explanation method (SVD word mapping) for the web app. Save full LIME/SHAP for detailed reports.

6. "Data was leaking clues"
   Problem: Real articles in ISOT all start with "WASHINGTON (Reuters) -" which is a dead giveaway
   Fix: Strip all such bylines before any processing happens.

Slide 15 — Future Work
Heading: "What Could Be Done Next?"
Show as a roadmap with icons:

Improvements:
- "Use modern AI (BERT/DistilBERT)" — These understand context much better than word counting. Could close the gap between long articles and short claims.
- "Better name-finder" — Upgrade from the small spaCy model to the large one. Catches more names = less cheating.
- "Check images too" — Fake news often comes with manipulated images. Add image analysis alongside text analysis.
- "Support Indian languages" — Hindi and Marathi fake news is a huge problem. Extend TruthLens to work with Indian languages.
- "Browser extension" — A Chrome/Firefox plugin that automatically checks news articles as you browse the web.
- "Learn from user feedback" — When users correct wrong predictions, feed that back to improve the model over time.
- "Test against intentional fakes" — Create fake articles that deliberately mimic real news writing style and test if the model can still catch them.

Slide 16 — Research Paper Overview
Heading: "Our Research Paper — Structure"
Show paper sections as a visual flow/timeline:
1. Abstract — Short summary: what we built, how it works, key results (99.31% accuracy, bias probing)
2. Introduction — Why fake news is a problem, what motivates our work, UN SDG 16
3. Literature Review — 7 papers we studied, the gaps we found (from Slides 3-4)
4. Methodology — Our full pipeline: how we clean text, extract features, train models, explain predictions
5. Dataset Description — ISOT + LIAR details, how we split and preprocessed them
6. Results — In-domain accuracy, cross-dataset drop, bias probe numbers, error analysis
7. Discussion — Why entity bias persists, why cross-dataset fails, what masking vs no-masking comparison shows
8. Conclusion & Future Work — What we achieved, what's left to do
Format: College research paper style, IEEE references

Slide 17 — Phase 3 Deliverables (What We're Submitting)
Heading: "What We're Submitting"
Show as a checklist with green checkmarks:
1. This Presentation (Final PPT) — 19 slides covering the entire project
2. Working Code (Jupyter Notebook) — TruthLens_Pipeline.ipynb — runs the full pipeline from data loading to predictions and explanations
3. Dataset — ISOT: 44,898 articles (332MB) + LIAR: 12,836 claims (5MB) — cleaned, processed, ready to use
4. GitHub Repository — Organized code: 8 Python modules, configuration file, 47 automated tests, README with setup instructions
5. Research Paper (Mandatory) — Full write-up: methodology, results, ablation study (masking vs no masking), limitations, references

Slide 18 — Conclusion
Heading: "What We Built & What We Learned"

What we built:
- TruthLens: a fake news detector that tries to be fair, transparent, and honest
- 99.31% accuracy on news articles using a team of 3 models + 1 smart referee
- Each article is described by 267 numbers: word stats (150) + word meanings (100) + writing style (17)
- Names are hidden to stop the model from cheating — reduces name-bias from 95% to 91%
- Two modes: ML analysis for articles + fact-checking for short claims (Wikipedia + DuckDuckGo)
- Every prediction comes with an explanation: which words mattered and which features mattered

What we learned:
- "99% accuracy on one dataset does NOT mean it works everywhere" — accuracy dropped 76% on different data. This is the biggest lesson.
- "Removing cheating shortcuts makes numbers slightly worse but the model more honest" — our accuracy dipped from 99.7% (Phase 1) to 99.3% (Phase 3) because we removed shortcuts. That small drop is a GOOD thing.
- "If the AI can't explain itself, don't trust it" — a prediction without reasoning is worthless
- "Old-school ML with smart features still works great" — we didn't need expensive deep learning (BERT, GPT) to hit 99%+ accuracy on structured text

Slide 19 — References
Heading: "References"
Two columns:

Left column — "Research Papers":
[1] Ahmed, Traore, Saad (2017) — "Detection of Online Fake News Using N-Gram Analysis" — ISOT dataset source
[2] Wang (2017) — "Liar, Liar Pants on Fire" — LIAR dataset source, ACL conference
[3] Shu et al. (2017) — "Fake News Detection on Social Media" — survey paper, ACM SIGKDD
[4] Vosoughi, Roy, Aral (2018) — "The spread of true and false news online" — Science journal, the "6x faster" statistic
[5] Ribeiro et al. (2016) — "Why Should I Trust You?" — LIME paper, KDD conference
[6] Lundberg & Lee (2017) — "A Unified Approach to Interpreting Model Predictions" — SHAP paper, NeurIPS
[7] Pennington et al. (2014) — "GloVe: Global Vectors for Word Representation" — word meanings, EMNLP

Right column — "Tools Used":
Python 3.14, scikit-learn (ML models), spaCy (name finding + word simplification), NLTK (filler word list), gensim (word meanings), VADER (mood detection), textstat (reading difficulty), LIME & SHAP (explanations), matplotlib + seaborn (charts), Gradio (web demo), pandas + numpy (data handling), Wikipedia API + DuckDuckGo (fact-checking), pytest (testing)

Add "Thank You!" at the bottom with a subtle background.
```

---

## How to Use These Prompts

1. Go to [gamma.app](https://gamma.app)
2. Click "Create new" → "Paste in text"
3. Copy **Prompt 1** (everything inside the first code block) → paste → generate slides 1–7
4. Start new or continue → Copy **Prompt 2** → paste → generate slides 8–13
5. Copy **Prompt 3** → paste → generate slides 14–19
6. Merge into one 19-slide deck

**Theme:** Dark mode, blue/purple accents, "Professional" or "Tech" template.
**Key difference from Phase 2 prompts:** Everything is written in plain English with analogies. No unexplained jargon.
