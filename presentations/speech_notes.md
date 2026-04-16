# TruthLens Phase 2 — Speech Notes (Slide-by-Slide)

> Estimated total: ~12-15 minutes. Each slide is ~45-60 seconds unless marked otherwise.

---

## Slide 1 — Title Slide (~20 seconds)

Good morning everyone. Our project is **TruthLens** — a debiased, explainable fake news detection system using hybrid feature engineering and a stacking ensemble. This is our Phase 2 implementation review. Our team consists of Riddhi, Prashant, Aditya, and Riwan, under the guidance of Dr. Joanne Gomes.

---

## Slide 2 — Problem Recap (~50 seconds)

So why does this project matter? Research shows that misinformation spreads **6 times faster** than factual news online. That's a massive problem.

But here's the deeper issue — most existing ML-based fake news detectors don't actually learn what makes news fake. They learn **shortcuts**. For example, if every real article in the training set starts with "Reuters," the model just learns "Reuters = Real" instead of understanding writing patterns. This means they **fail completely** on new, unseen data.

Our objectives for TruthLens are: first, to address 7 critical mistakes we identified in existing models. Second, achieve over 99% accuracy while being **transparent** about generalization. Third, provide explainable predictions using LIME and SHAP. Fourth, actively detect and measure dataset bias. And finally, deploy an interactive Gradio web demo for real-time verification.

---

## Slide 3 — Dataset Overview (~50 seconds)

We used two complementary datasets. **ISOT** contains about 44,900 full-length news articles — these are 500 to 3000 words each, with a roughly balanced split of 52% fake and 48% real. **LIAR** contains about 12,800 short claims — just 15 to 30 words each — also balanced at 50-50.

Combined, we have nearly 57,700 samples. The reason we use two datasets is to test generalization — ISOT is long-form articles while LIAR is short claims, so they represent very different types of text.

Our feature vector is **167 dimensions** — 150 from TF-IDF with SVD compression, plus 17 handcrafted stylometric features. We use an 80/20 stratified split with a fixed random state of 42 for reproducibility. The primary training happens on ISOT, giving us about 35,918 training samples.

---

## Slide 4 — Data Preprocessing Pipeline (~60 seconds)

Our preprocessing pipeline has 6 key steps, and this is where a lot of the debiasing happens.

**Step 1 — Byline Stripping.** We remove Reuters, AP, and AFP bylines from the text. This is critical because if we leave them in, the model just learns "article starts with Reuters = Real news" — that's a shortcut, not a real pattern.

**Step 2 — Text Cleaning.** Standard cleanup — remove URLs, HTML tags, emails, special characters, and truncate to 5000 characters.

**Step 3 — Entity Masking** using spaCy's Named Entity Recognition. We replace person names, organizations, locations, and dates with generic tokens like [PERSON] and [ORG]. This prevents the model from memorizing that "Trump" or "Obama" correlates with a specific label.

**Step 4 — Source Masking** using regex. We mask 51 known news source names with a [SOURCE] token — CNN, Fox News, BBC, and so on.

**Step 5 and 6** are standard NLP — stopword removal with NLTK and lemmatization with spaCy.

Additionally, we handle missing values, encode labels as binary, fit StandardScaler on training data only to prevent data leakage, and use batch processing with spaCy pipes for a 3 to 5x speedup.

---

## Slide 5 — Hybrid Feature Engineering (~50 seconds)

Our feature engineering uses two parallel pipelines that merge into a single 167-dimensional vector.

**Pipeline 1 — the Lexical Pipeline.** We take the masked, cleaned text, apply TF-IDF with a 10,000 word vocabulary including bigrams, then compress it down to 150 dimensions using Truncated SVD. The SVD step is important — it reduces dimensionality, removes noise, and helps prevent overfitting.

**Pipeline 2 — the Stylometric Pipeline.** From the raw text, we extract 17 handcrafted features that capture **writing style** rather than content. These include word count, average word length, vocabulary richness, capital letter ratio, exclamation rate, sentiment scores, reading ease, burstiness, and the Zipf coefficient.

The key pattern we discovered: fake news tends to have more exclamation marks, higher emotional sentiment, and lower readability scores. Real news maintains a neutral tone with higher vocabulary richness. These stylometric features capture **how** something is written, not just **what** is written.

---

## Slide 6 — Models Implemented (~50 seconds)

We implemented 4 models, each chosen for a specific reason.

**SVM** — using LinearSVC with CalibratedClassifierCV for probability estimates. SVM is great for high-dimensional data like ours, with O(n) training complexity and strong margin-based classification.

**Logistic Regression** — our interpretable baseline. It trains in under a second, gives fully interpretable coefficients, and produces well-calibrated probabilities.

**Random Forest** — captures non-linear feature interactions that linear models miss, and provides built-in feature importance rankings.

**Stacking Ensemble** — this is our final model. It combines all three base models using a Logistic Regression meta-learner. The meta-learner takes the probability outputs from all three models via 5-fold cross-validation and learns the **optimal weights** for combining them. This is significantly better than naive majority voting because it learns which model to trust in which situations.

---

## Slide 7 — Hyperparameter Tuning (~50 seconds)

We used **GridSearchCV** with stratified 5-fold cross-validation, F1 as the scoring metric, and class_weight set to balanced.

For **SVM**, we searched C values of 0.01, 0.1, 1, and 10. The best was C=0.01, achieving a CV F1 of 0.9920 in just 8 seconds.

For **Logistic Regression**, same C candidates with L2 penalty and LBFGS solver. Best C was 0.1, CV F1 of 0.9917, training in under 1 second.

For **Random Forest**, we searched over n_estimators, max_depth, and min_samples_split — a much larger grid. Best configuration was 100 trees, depth 20, min split 2, with CV F1 of 0.9702. This took about 650 seconds.

The **key insight** here is that both SVM and LR selected stronger regularization — low C values and bounded depth. This is evidence that the models are actively preventing overfitting on our 167-dimensional feature space, which is exactly what we want.

---

## Slide 8 — Model Comparison (~50 seconds)

Here are our test set results across all four models.

**SVM** leads with the highest individual F1 of **0.9929** and accuracy of 99.35%.

**Logistic Regression** is very close — F1 of 0.9926, accuracy 99.32%.

**Random Forest** is the weakest performer at F1 of 0.9723, about 2 points lower.

The **Ensemble** achieves F1 of 0.9924 and the highest AUC-ROC of **0.9997**.

Our 5-fold cross-validation shows F1 of 0.9920 plus or minus 0.0012 — that very tight standard deviation tells us the model is **stable** and not dependent on a lucky split.

Now, you might ask — if SVM has a slightly higher F1, why not just use SVM? That brings us to the next slide.

---

## Slide 9 — Best Model: Stacking Ensemble (~50 seconds)

We selected the **Stacking Ensemble** as our best model for six reasons.

**Model Diversity** — it combines SVM's margin optimization, LR's calibration, and RF's non-linearity. Each model captures different patterns.

**Optimal Weights** — the meta-classifier learns the optimal combination weights from probability outputs, not just a simple vote.

**Robustness** — it's resistant to individual model failures and more stable across different confidence thresholds.

**Discrimination** — AUC-ROC of 0.9997, the highest among all models.

**Calibration** — verified via calibration curve analysis, meaning when the model says 90% confidence, it's right about 90% of the time.

**Explainability** — fully compatible with LIME and SHAP for transparent predictions.

Yes, SVM has a marginally higher F1 — 0.9929 versus 0.9924 — but that 0.0005 difference is negligible. The ensemble's **robustness** and **stability** make it the better choice for production use.

---

## Slide 10 — Error Analysis (~60 seconds)

Out of approximately 8,980 test samples, our model made only **54 errors** — a 0.60% error rate. Let's understand where and why it fails.

By **error type**: 34 were false negatives — fake news predicted as real — making up 63%. 20 were false positives — real news predicted as fake — at 37%. So the model is more likely to miss fake news than to falsely flag real news.

By **confidence**: 31 of the 54 errors, that's 57%, occurred on uncertain predictions where confidence was below 0.4. Only 14 errors were made with high confidence above 0.8. And 9 were borderline.

Let me show two examples. **Example 1** — a false negative: "BOOM! Trump Drains Obama Swamp..." — this was predicted as Real with only 0.25 confidence. Why? Because the article body contained legitimate policy content that mimicked real news structure, even though the headline was sensational.

**Example 2** — a false positive: "Factbox: Trump's tweets vs CEO comments" from Reuters — predicted as Fake with 0.99 confidence. Why? The informal comparison style resembled sensationalist patterns.

The **key insight**: 57% of errors happen when the model is uncertain — meaning the model effectively **knows when it doesn't know**. This is a desirable property.

---

## Slide 11 — Phase 1 vs Phase 2 (~60 seconds)

Comparing Phase 1 to Phase 2, you'll notice the numbers actually went **down** slightly. Accuracy dropped from 0.9970 to 0.9931, F1 from 0.9971 to 0.9924, and AUC-ROC from 0.9999 to 0.9997.

This might seem like a step backward, but it's **intentional**. Phase 2 introduced aggressive debiasing — entity masking, byline stripping, and SVD reduction from 300 to 150 dimensions. These changes **remove the shortcut features** that were inflating Phase 1's metrics. The Phase 1 model was partially learning "Reuters = Real" rather than genuine deception patterns. Phase 2's slightly lower numbers reflect a model that's learning **real patterns** instead of dataset artifacts.

What **actually improved** in Phase 2: first, we added **bias probing** — measuring entity, length, and topic bias systematically. Second, **systematic error analysis** with categorization by type and confidence. Third, a **SHAP bias audit** to check if entity-related features dominate predictions. Fourth, an **ablation study** comparing masked versus unmasked performance to validate that debiasing works. And fifth, overall code quality with bug fixes and refactoring.

---

## Slide 12 — System Architecture (~50 seconds)

This slide shows our complete system architecture. The flow is straightforward.

Raw text comes in — either an article or a headline. It goes through **feature extraction** via our parallel pipelines: Pipeline A cleans the text, applies entity and source masking, then TF-IDF with SVD to get 150 dimensions. Pipeline B extracts 17 stylometric features. These concatenate and pass through StandardScaler to produce our 167-dimensional vector.

This vector feeds into the **Stacking Ensemble** — SVM, LR, and RF as base models, with a meta-LR combining their outputs. The output is a prediction with a confidence score.

On the explainability side, we generate **LIME** explanations for local, word-level interpretability, **SHAP** for global feature importance, and a **bias audit** for transparency.

For deployment, the entire system is wrapped in a **Gradio web app** that auto-detects input type — long articles go through the ML pipeline, short claims trigger fact verification, and opinions are flagged as unverifiable.

---

## Slide 13 — Demo & Results (~50 seconds)

Here are three prediction examples showing the system in action.

**Example 1 — Fake News Detection**: Input is "BREAKING!!! Government HIDING the truth about vaccines!!! Share before they delete!!!" — the model correctly identifies this as **FAKE with 94% confidence**. The indicators are excessive punctuation, heavy capitalization, and sensationalist emotional manipulation.

**Example 2 — Real News Classification**: "Scientists at NASA discovered evidence of water on Mars using spectroscopic analysis published in Nature." — correctly classified as **REAL with 87% confidence**. The indicators are neutral tone, academic language, and specific attribution to a credible source.

**Example 3 — Fact Verification**: "India landed on the moon in 2023" — this short claim triggers the fact verification pipeline and is **VERIFIED at 85% confidence** by cross-referencing Wikipedia, confirming the Chandrayaan-3 mission on August 23, 2023.

Our pipeline also generates comprehensive outputs — confusion matrices, ROC curves, calibration plots, SHAP summary and bar plots, LIME HTML explanations, error analysis CSV, and cross-dataset evaluation results.

---

## Slide 14 — Challenges Faced (~60 seconds)

We faced five major technical challenges.

**First — Cross-Dataset Generalization.** When we trained on ISOT and tested on LIAR, F1 dropped to 0.23. The reverse was even worse at 0.13. The root cause is a 10 to 100x difference in text length between the datasets. Traditional ML features can't bridge this gap — you need transformer architectures like BERT for that.

**Second — Persistent Entity Bias.** Our bias probe accuracy was 0.91, but ideally it should be around 0.50, meaning the model can't distinguish entities from labels. Even after masking, spaCy misses some entities, and ISOT has structural source bias that goes beyond just named entities.

**Third — Random Forest Overfitting.** Training F1 was 99.98% versus test F1 of 97.23% — a clear overfitting gap. We addressed this with max_depth of 20, max_features set to square root, and SVD compression.

**Fourth — Data Imbalance.** Handled with class_weight='balanced' in all models, using F1 as our primary metric instead of accuracy, and stratified splits.

**Fifth — GloVe Incompatibility.** The gensim library breaks on Python 3.14+. Our workaround is a --no-glove flag that drops from 267 to 167 dimensions with minor performance impact.

---

## Slide 15 — Future Work (~40 seconds)

Five priority directions for future development.

**Transformer Models** — implementing BERT or DistilBERT to close the cross-dataset generalization gap and handle variable-length inputs.

**Improved NER** — upgrading to spaCy's transformer-based model, en_core_web_trf, for better entity masking.

**Multimodal Detection** — combining image and text analysis since modern misinformation often uses manipulated images alongside text.

**Indian Language Support** — extending to Hindi and Marathi fake news detection, which is highly relevant for our context.

**Browser Extension** — a real-time news verification tool with active learning from user feedback.

For the final paper, we plan to include comprehensive methodology, the ablation study, limitations analysis, and literature comparison.

---

## Slide 16 — Phase-2 Deliverables (~40 seconds)

All six deliverables for Phase 2 are complete.

**One** — Cleaned Dataset: ISOT at 332MB plus LIAR at 5MB, no missing values, 80/20 stratified split with consistent labeling.

**Two** — Jupyter Notebook: a 21-cell pipeline covering preprocessing, feature engineering, all 4 ML models, GridSearchCV tuning, and model comparison.

**Three** — Results & Evaluation: confusion matrices, ROC curves, calibration plot, comparison table, with Stacking Ensemble identified as the best model.

**Four** — GitHub Repository: organized with src/, data/, models/, and results/ directories, with a README including the architecture diagram.

**Five** — System Integration: Gradio web app with auto input-type detection and the block diagram showing data flow.

**Six** — Progress Evidence: debiasing implementation, bias probing, error analysis, ablation study, and SHAP audit — all demonstrating rigorous validation.

---

## Closing (~10 seconds)

That concludes our Phase 2 presentation. Thank you. We're happy to take any questions.

---

> **Tips for delivery:**
> - Speak slowly on numbers — give the audience time to read the table while you explain
> - On Slide 11 (Phase 1 vs 2), emphasize the word "intentional" — the examiners may question the drop
> - On Slide 10, pause briefly after each example to let it sink in
> - On Slide 14 (Challenges), be honest about limitations — examiners respect transparency over perfection
> - Keep a steady pace — don't rush the technical slides (7, 8, 9)
