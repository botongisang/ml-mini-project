# Fake News Detection Using Machine Learning — Complete Research Data

---

# PART 1: KNOWLEDGE & UNDERSTANDING (Learning about the topic)

---

## 1.1 What is Fake News?

Fake news refers to deliberately fabricated information presented as legitimate news, designed to mislead readers. It includes:
- **Misinformation**: False information shared without harmful intent
- **Disinformation**: Intentionally false information spread to deceive
- **Satire/Parody**: Misinterpreted as real news
- **Clickbait**: Sensationalized headlines to attract clicks

Fake news spreads 6x faster than real news on social media (MIT study, 2018). Platforms like Facebook, Twitter (X), and WhatsApp are the primary vectors.

---

## 1.2 Why is this a Problem?

- **Political manipulation**: Fake news influenced elections globally (e.g., 2016 US elections, Indian elections)
- **Health misinformation**: COVID-19 fake cures led to deaths
- **Financial fraud**: Fake stock news manipulates markets
- **Social unrest**: Communal violence triggered by WhatsApp rumors in India
- **Trust erosion**: Public loses trust in legitimate journalism

**Scale**: Over 3.5 billion social media users worldwide are exposed to fake news daily.

---

## 1.3 How Machine Learning Helps

Traditional fact-checking is manual and slow. ML automates this by:
1. **Text preprocessing** — cleaning, tokenizing, removing stopwords
2. **Feature extraction** — converting text to numbers (TF-IDF, BOW, N-grams)
3. **Classification** — training models to predict Fake vs Real

### ML Pipeline for Fake News Detection:
```
Raw Text News → Preprocessing → Feature Extraction → Model Training → Prediction (Fake/Real)
```

---

## 1.4 Key Concepts You Must Know

### A. Text Preprocessing
| Step | What it does | Example |
|------|-------------|---------|
| Lowercasing | Converts all text to lowercase | "BREAKING News" → "breaking news" |
| Tokenization | Splits text into words | "fake news" → ["fake", "news"] |
| Stop word removal | Removes common words (the, is, a) | "this is fake" → "fake" |
| Stemming | Reduces words to root form | "running" → "run" |
| Lemmatization | Reduces to dictionary form | "better" → "good" |
| Punctuation removal | Strips special characters | "fake!!!" → "fake" |

### B. Feature Extraction Techniques

**Bag of Words (BOW)**:
- Creates a vocabulary of all unique words
- Represents each document as a vector of word counts
- Simple but ignores word importance

**TF-IDF (Term Frequency - Inverse Document Frequency)**:
- TF = (Number of times term appears in document) / (Total terms in document)
- IDF = log(Total documents / Documents containing the term)
- TF-IDF = TF x IDF
- **Gives higher weight to rare, important words** and lower weight to common words
- Most commonly used in fake news detection papers

**N-grams**:
- Unigrams: single words → ["fake", "news"]
- Bigrams: word pairs → ["fake news", "news detected"]
- Trigrams: word triplets → ["fake news detected"]
- Captures word context and phrases

### C. ML Algorithms Used in Fake News Detection

| Algorithm | Type | How it works (simplified) | Typical Accuracy |
|-----------|------|--------------------------|-----------------|
| **Naive Bayes** | Probabilistic | Uses Bayes theorem; assumes features are independent | 72-93% |
| **Logistic Regression** | Linear | Fits a sigmoid curve to classify binary outcomes | 91-98% |
| **SVM** | Margin-based | Finds the best hyperplane separating fake from real | 93-99.6% |
| **Decision Tree** | Tree-based | Creates if-else rules from features | 85-95% |
| **Random Forest** | Ensemble | Combines multiple decision trees, takes majority vote | 90-97% |
| **LSTM** | Deep Learning | Recurrent neural network; captures word sequence/order | 92-95% |
| **Passive Aggressive** | Online learning | Aggressively updates on misclassified samples | 92-96% |
| **XGBoost** | Boosting | Gradient boosted trees; handles complex patterns | 89-91% |

### D. Evaluation Metrics
- **Accuracy** = (TP + TN) / Total — overall correctness
- **Precision** = TP / (TP + FP) — of all predicted fake, how many were actually fake?
- **Recall** = TP / (TP + FN) — of all actual fake news, how many did we catch?
- **F1-Score** = 2 x (Precision x Recall) / (Precision + Recall) — harmonic mean
- **Confusion Matrix** — table showing TP, TN, FP, FN

### E. Common Datasets Used
| Dataset | Size | Source | Labels |
|---------|------|--------|--------|
| ISOT Fake News | 44,898 articles | Reuters + Kaggle | Fake/Real |
| LIAR | 12,836 statements | PolitiFact | 6 fine-grained labels |
| Kaggle Fake News | 26,000 articles | Kaggle | Fake/Real |
| Facebook News Posts | ~5,000 posts | BuzzFeed + Facebook | Fake/Real |
| FakeNewsNet | 23,196 articles | PolitiFact + GossipCop | Fake/Real |

---

## 1.5 SDG 16 — Peace, Justice & Strong Institutions

- **Target 16.10**: Ensure public access to information and protect fundamental freedoms
- Fake news undermines democratic institutions, public discourse, and informed decision-making
- Automated detection supports media literacy and information integrity
- The UN recognizes misinformation as a threat to sustainable development

---

# PART 2: PRESENTATION CONTENT (Slide-by-slide data)

---

## SELECTED 8 PAPERS

### Paper 1
- **Title**: Fake News Detection Using Naive Bayes Classifier
- **Authors**: Mykhailo Granik, Volodymyr Mesyura
- **Year**: 2017
- **Conference**: IEEE First Ukraine Conference on Electrical and Computer Engineering (UKRCON)
- **Link**: [https://ieeexplore.ieee.org/document/8100379/](https://ieeexplore.ieee.org/document/8100379/)

### Paper 2
- **Title**: Comparison of Various Machine Learning Models for Accurate Detection of Fake News
- **Authors**: K. Poddar, G. B. Amali D., K. S. Umadevi
- **Year**: 2019
- **Conference**: 2019 Innovations in Power and Advanced Computing Technologies (i-PACT), Vellore, India
- **Link**: [https://ieeexplore.ieee.org/document/8960044/](https://ieeexplore.ieee.org/document/8960044/)

### Paper 3
- **Title**: Fake News Detection Using Machine Learning Approaches: A Systematic Review
- **Authors**: Syed Ishfaq Manzoor, Jimmy Singla, Nikita
- **Year**: 2019
- **Conference**: 3rd International Conference on Trends in Electronics and Informatics (ICOEI)
- **Link**: [https://ieeexplore.ieee.org/document/8862770/](https://ieeexplore.ieee.org/document/8862770/)

### Paper 4
- **Title**: Classifying Fake News Detection Using SVM, Naive Bayes and LSTM
- **Authors**: P. Jain, S. Sharma, Monica, P. K. Aggarwal
- **Year**: 2022
- **Conference**: 12th International Conference on Cloud Computing, Data Science & Engineering (Confluence)
- **Link**: [https://ieeexplore.ieee.org/document/9734129/](https://ieeexplore.ieee.org/document/9734129/)

### Paper 5
- **Title**: Exploring the Effect of N-grams with BOW and TF-IDF Representations on Detecting Fake News
- **Authors**: Amal Esmail Qasem, Mohammad Sajid
- **Year**: 2022
- **Conference**: International Conference on Data Analytics for Business and Industry (ICDABI)
- **Link**: [https://ieeexplore.ieee.org/document/10041537/](https://ieeexplore.ieee.org/document/10041537/)

### Paper 6
- **Title**: An Empirical Analysis of Naive Bayes, SVM, Logistic Regression and Random Forest to Spot False Information in Real-World Networks
- **Authors**: (IEEE Conference Publication, 2022)
- **Year**: 2022
- **Conference**: IEEE Conference Publication
- **Link**: [https://ieeexplore.ieee.org/document/9862430/](https://ieeexplore.ieee.org/document/9862430/)

### Paper 7
- **Title**: Unveiling the Truth: Detecting Fake News Using SVM and TF-IDF
- **Authors**: Randi Rizal, Ade Faturahman, Ali Impron, Alam Rahmatulloh
- **Year**: 2025
- **Conference**: IEEE Conference Publication
- **Link**: [https://ieeexplore.ieee.org/abstract/document/10933324/](https://ieeexplore.ieee.org/abstract/document/10933324/)

### Paper 8
- **Title**: A Scalable Approach to Fake News Detection Using TF-IDF and Logistic Regression
- **Authors**: (IEEE Conference Publication, 2025)
- **Year**: 2025
- **Conference**: IEEE Conference Publication
- **Link**: [https://ieeexplore.ieee.org/document/11135996/](https://ieeexplore.ieee.org/document/11135996/)

---

## SLIDE 1: Title Slide

- **Project Title**: Fake News Detection Using Machine Learning and Natural Language Processing
- **Course**: C403 — Machine Learning Mini Project
- **SDG**: SDG 16 — Peace, Justice & Strong Institutions
- **Institute**: St. Francis Institute of Technology, Mumbai

---

## SLIDE 2: Motivation & SDG Mapping

### Problem Motivation
- Social media platforms (Facebook, Twitter, WhatsApp) have become primary news sources for billions of people
- Fake news spreads 6 times faster and reaches more people than real news
- Manual fact-checking is too slow to keep up with the volume of online content

### Real-World Importance
- Political manipulation through fake news during elections
- Health misinformation (e.g., COVID-19 fake cures) causing real harm
- Financial market manipulation through fabricated stories
- Communal violence triggered by fake WhatsApp forwards in India

### Mapped SDG
- **SDG 16: Peace, Justice and Strong Institutions**
  - Target 16.10: Ensure public access to information
  - Fake news undermines democratic processes, public trust, and informed decision-making

### Why This Problem Matters
- Over 3.5 billion social media users exposed to fake news daily
- 86% of internet users have been deceived by fake news at least once
- Automated ML-based detection can provide scalable, real-time solutions that manual fact-checking cannot

---

## SLIDE 3: Problem Statement

### Clear Problem Definition
To develop a machine learning-based classification system that can automatically distinguish between fake and real news articles using Natural Language Processing (NLP) techniques for text feature extraction and supervised learning algorithms for binary classification.

### Scope of Study
- Focus on **text-based** English news articles (not images/videos)
- Use **TF-IDF and BOW** feature extraction techniques
- Compare **multiple ML algorithms**: Naive Bayes, SVM, Logistic Regression, Random Forest, Decision Tree
- Evaluate using standard metrics: Accuracy, Precision, Recall, F1-Score
- Use publicly available labeled datasets (Kaggle / ISOT)

### Research Objective
1. To study and compare existing ML approaches for fake news detection through a literature survey of 8 research papers
2. To identify the most effective combination of feature extraction technique and ML algorithm
3. To identify research gaps and propose improvements for a future implementation

---

## SLIDE 4: Paper Details (All 8 Papers)

| Paper # | Title | Authors | Year | Publisher |
|---------|-------|---------|------|-----------|
| 1 | Fake News Detection Using Naive Bayes Classifier | M. Granik, V. Mesyura | 2017 | IEEE UKRCON |
| 2 | Comparison of Various ML Models for Accurate Detection of Fake News | K. Poddar, G.B. Amali D., K.S. Umadevi | 2019 | IEEE i-PACT |
| 3 | Fake News Detection Using ML Approaches: A Systematic Review | S.I. Manzoor, J. Singla, Nikita | 2019 | IEEE ICOEI |
| 4 | Classifying Fake News Detection Using SVM, Naive Bayes and LSTM | P. Jain, S. Sharma, Monica, P.K. Aggarwal | 2022 | IEEE Confluence |
| 5 | Exploring the Effect of N-grams with BOW and TF-IDF on Detecting Fake News | A.E. Qasem, M. Sajid | 2022 | IEEE ICDABI |
| 6 | Empirical Analysis of NB, SVM, LR and RF to Spot False Information | IEEE Conference | 2022 | IEEE |
| 7 | Unveiling the Truth: Detecting Fake News Using SVM and TF-IDF | R. Rizal, A. Faturahman, A. Impron, A. Rahmatulloh | 2025 | IEEE |
| 8 | A Scalable Approach to Fake News Detection Using TF-IDF and LR | IEEE Conference | 2025 | IEEE |

---

## SLIDE 5: Introduction of Paper 1 — Granik & Mesyura (2017)

### Background Context
- One of the earliest IEEE papers applying Naive Bayes specifically to fake news detection
- The rise of social media made Facebook a major channel for fake news distribution
- At the time, no widely adopted automated detection systems existed

### Why the Problem is Important
- Facebook had 2 billion monthly users in 2017, making it the largest news distribution platform
- Users often could not distinguish between fake and real posts
- Manual fact-checking was insufficient for the scale of the problem

### Key Idea of the Paper
- Apply a **Naive Bayes classifier** (one of the simplest ML models) on Facebook news posts
- Demonstrate that even basic ML approaches can meaningfully detect fake news
- Test the system on a dataset of posts from 9 Facebook pages (3 left-leaning, 3 right-leaning, 3 mainstream: CNN, Politico, ABC News)

---

## SLIDE 6: Methodology of Paper 1 — Granik & Mesyura (2017)

### Dataset Used
- Facebook news posts collected from BuzzFeed's dataset
- 9 Facebook pages covering different political orientations
- Approximately 5,000 news posts labeled as Fake or Real

### Features
- Text content of Facebook posts
- **Bag of Words (BOW)** model for feature extraction
- Stopword removal and text cleaning as preprocessing

### ML Algorithm
- **Multinomial Naive Bayes Classifier**
  - Based on Bayes' Theorem: P(Fake|Text) = P(Text|Fake) x P(Fake) / P(Text)
  - Assumes word features are independent (naive assumption)
  - Fast to train, works well with small datasets

### Workflow
```
Facebook Posts → Text Cleaning → Tokenization → BOW Vectorization → Naive Bayes Training → Prediction
```

---

## SLIDE 7: Results & Evaluation of Paper 1 — Granik & Mesyura (2017)

### Performance Metrics
| Metric | Value |
|--------|-------|
| **Accuracy** | ~74% |
| Model | Multinomial Naive Bayes |
| Feature Extraction | Bag of Words |
| Dataset | Facebook posts (~5,000) |

### Key Findings
- Even a simple NB classifier can detect fake news with 74% accuracy
- Proved that ML-based fake news detection is feasible
- Accuracy was limited due to the small dataset and basic feature extraction (BOW only)
- The paper served as a foundational baseline for future research

### Limitations Observed
- Only 74% accuracy — not sufficient for production use
- BOW loses word order information
- Small dataset from limited sources
- No comparison with other ML models

---

## SLIDE 8: Introduction of Paper 2 — Poddar et al. (2019)

### Background Context
- Built upon earlier work like Paper 1 to provide a more comprehensive comparison
- Recognized that a single model cannot be declared best without comparing multiple approaches
- Used a much larger dataset from Kaggle (26,000 articles)

### Why the Problem is Important
- Fake news continued to escalate during the 2018-2019 period globally
- Needed systematic evaluation of which ML model performs best for this task
- Previous studies used different datasets, making direct comparison difficult

### Key Idea of the Paper
- Compare **5 different ML classifiers** (NB, SVM, LR, DT, ANN) on the **same dataset**
- Compare two vectorization techniques: **Count Vectorizer vs TF-IDF Vectorizer**
- Determine which model + vectorizer combination is most accurate

---

## SLIDE 9: Methodology of Paper 2 — Poddar et al. (2019)

### Dataset Used
- **Kaggle Fake News Dataset**: 26,000 articles (roughly equal fake and real)
- Split into 80% training and 20% testing

### Features
- Text content of news articles
- Two vectorizers compared:
  - **Count Vectorizer (BOW)**: Simple word frequency counts
  - **TF-IDF Vectorizer**: Weighted by word importance across documents

### ML Algorithms Compared
1. Naive Bayes (NB)
2. Support Vector Machine (SVM)
3. Logistic Regression (LR)
4. Decision Tree (DT)
5. Artificial Neural Network (ANN)

### Workflow
```
Kaggle Dataset → Preprocessing (cleaning, stopword removal) → Count Vectorizer / TF-IDF → 5 ML Models → Accuracy Comparison
```

---

## SLIDE 10: Results & Evaluation of Paper 2 — Poddar et al. (2019)

### Performance Comparison Table

| Algorithm | Count Vectorizer | TF-IDF Vectorizer |
|-----------|:---:|:---:|
| Naive Bayes | ~85% | ~88% |
| SVM | ~93% | **~95%** (Best) |
| Logistic Regression | ~91% | ~93% |
| Decision Tree | ~85% | ~87% |
| ANN | ~90% | ~92% |

### Key Findings
- **SVM with TF-IDF gave the highest accuracy** (~95%)
- TF-IDF consistently outperformed Count Vectorizer across all models
- SVM and Logistic Regression are the most suitable for text classification
- NB was the fastest to train but had the lowest accuracy
- ANN performed well but required significantly more computation

### Significance
- Established SVM + TF-IDF as the go-to combination for text-based fake news detection
- Provided a reproducible benchmark on a publicly available dataset

---

## PAPER 3: Introduction — Manzoor et al. (2019) [Survey Paper]

### Background Context
- By 2019, numerous individual studies had proposed different ML methods for fake news detection
- However, no comprehensive review existed that compared and categorized all these approaches
- Researchers needed a consolidated reference to understand the landscape of techniques, datasets, and results

### Why the Problem is Important
- The volume of fake news research was growing rapidly, making it hard for new researchers to identify the best approaches
- Different papers used different datasets and metrics, making direct comparison impossible
- The shift from traditional ML to deep learning needed documentation and analysis

### Key Idea of the Paper
- Conduct a **systematic literature review** of all ML-based fake news detection approaches published between 2015-2019
- Categorize methods by: feature extraction type, ML algorithm, dataset used, and evaluation metrics
- Identify limitations of traditional ML approaches and suggest deep learning as an improvement path
- Provide a reference framework for future researchers in this domain

---

## PAPER 3: Methodology — Manzoor et al. (2019)

### Review Methodology
- **Systematic review** following standard survey methodology
- Collected and analyzed research papers from IEEE, Springer, ACM, ScienceDirect, and other databases
- Focused on papers published between **2015-2019**

### Categories of Techniques Reviewed
1. **Knowledge-based approaches**: Fact-checking using external knowledge bases
2. **Style-based approaches**: Analyzing writing style, sensationalism, grammar quality
3. **Propagation-based approaches**: How news spreads through social networks
4. **ML-based approaches**: Classification using NB, SVM, LR, DT, RF, and neural networks

### Feature Extraction Methods Covered
- Bag of Words (BOW)
- TF-IDF
- Word2Vec
- N-grams
- Linguistic features (sentiment, readability scores)

### Datasets Identified
- LIAR dataset (PolitiFact), ISOT, Kaggle, BuzzFeed, FakeNewsNet

### Workflow
```
Literature Collection → Screening & Filtering → Categorization by Technique → Comparative Analysis → Gap Identification
```

---

## PAPER 3: Results & Evaluation — Manzoor et al. (2019)

### Key Findings from the Review

| Approach | Best Algorithm | Typical Accuracy | Limitation |
|----------|:---:|:---:|---|
| Traditional ML + BOW | Random Forest | ~98.8% (FARN Dataset) | Loses word order |
| Traditional ML + TF-IDF | SVM | ~93-95% | Ignores semantics |
| Deep Learning (CNN/LSTM) | LSTM | ~95-97% | Needs large data & compute |
| Knowledge-based | N/A | Varies | Needs curated knowledge base |

### Major Conclusions
- **TF-IDF outperforms BOW** as a feature extraction technique in most studies
- **SVM and Random Forest** are the most commonly used and best-performing traditional ML models
- **Deep learning approaches** (CNN, LSTM) show promise but require significantly more data and compute
- Most studies only focus on **English text** — multilingual detection is a major gap
- **No single approach works universally** — performance depends heavily on the dataset used

### Limitations Identified
- Most reviewed papers tested on a single dataset only
- Very few papers addressed real-time detection
- Limited work on multimodal fake news (text + image + video)
- No adversarial robustness testing in any reviewed paper

---

## PAPER 4: Introduction — Jain et al. (2022)

### Background Context
- By 2022, both traditional ML and deep learning had been applied to fake news detection independently
- However, few papers directly compared traditional ML (SVM, NB) with deep learning (LSTM) on the same task and dataset
- This paper bridges the gap between classical and modern approaches

### Why the Problem is Important
- Social media misinformation continued to grow during the COVID-19 pandemic (2020-2022)
- Choosing between traditional ML and deep learning for fake news detection was an open question
- Practitioners needed guidance on which approach to choose given their computational constraints

### Key Idea of the Paper
- Directly compare **SVM, Naive Bayes (traditional ML)** with **LSTM (deep learning)** for fake news classification
- Use **NLP preprocessing pipeline** including tokenization, stopword removal, and stemming
- Determine if the added complexity of deep learning provides meaningful accuracy improvement over simpler models

---

## PAPER 4: Methodology — Jain et al. (2022)

### Dataset Used
- Standard fake news datasets from publicly available sources
- Binary labeled: Fake (0) and Real (1)
- Split into training and testing sets

### Preprocessing Pipeline
1. Text cleaning (removing HTML tags, URLs, special characters)
2. Tokenization (splitting text into individual words)
3. Stopword removal (removing common words: "the", "is", "a")
4. Stemming (reducing words to root form: "running" → "run")

### Feature Extraction
- **TF-IDF Vectorizer** for SVM and Naive Bayes
- **Word Embeddings** (sequence representation) for LSTM

### ML Algorithms Compared
1. **Naive Bayes** — probabilistic classifier using Bayes' theorem; assumes feature independence
2. **SVM (Support Vector Machine)** — finds the optimal hyperplane separating fake from real in high-dimensional space
3. **LSTM (Long Short-Term Memory)** — recurrent neural network that captures sequential word dependencies and long-range context

### Workflow
```
Dataset → Text Cleaning → Tokenization → Stopword Removal → Stemming → TF-IDF / Word Embedding → NB / SVM / LSTM → Prediction
```

---

## PAPER 4: Results & Evaluation — Jain et al. (2022)

### Performance Comparison

| Algorithm | Type | Accuracy | Training Speed |
|-----------|------|:---:|:---:|
| Naive Bayes | Traditional ML | ~85% | Very Fast |
| SVM | Traditional ML | ~93% | Fast |
| **LSTM** | Deep Learning | **~95%** | Slow |

### Detailed Analysis
- **Naive Bayes**: Fastest to train but lowest accuracy. The "naive" independence assumption limits its ability to capture word relationships.
- **SVM**: Strong performance with TF-IDF features. Best balance of accuracy and training speed. Works well even with limited compute.
- **LSTM**: Highest accuracy because it captures sequential word patterns and long-range dependencies. However, requires significantly more training time and GPU resources.

### Key Findings
- LSTM outperformed both traditional ML models due to its ability to capture **sequential dependencies** in text
- However, the accuracy gap between SVM (~93%) and LSTM (~95%) is only ~2%, while LSTM requires **10-50x more training time**
- **SVM remains the best practical choice** when computational resources are limited
- LSTM is preferred only when maximum accuracy is critical and compute is available

### Limitations
- No transformer-based models (BERT) compared
- No explainability analysis — cannot determine why a news article was classified as fake
- Limited hyperparameter tuning for LSTM

---

## PAPER 5: Introduction — Qasem & Sajid (2022)

### Background Context
- Previous studies showed TF-IDF outperforms BOW, but did not explore **how N-gram variations** affect performance
- N-grams capture local word context: unigrams (single words), bigrams (word pairs), trigrams (word triplets)
- The question remained: does using bigrams/trigrams with TF-IDF improve fake news detection over simple unigrams?

### Why the Problem is Important
- Feature extraction is the most critical step in text classification — the quality of features directly determines model accuracy
- Social media fake news is **unstructured text** that needs to be converted to numerical representations
- Choosing the right combination of vectorizer + N-gram size can make or break a model's performance

### Key Idea of the Paper
- Systematically compare **6 feature extraction combinations**: BOW and TF-IDF each with unigrams, bigrams, and trigrams
- Test each combination with **3 classifiers**: SVM, Naive Bayes, and Decision Tree
- Determine the **optimal feature extraction strategy** for fake news detection

---

## PAPER 5: Methodology — Qasem & Sajid (2022)

### Dataset Used
- Fake news datasets collected from social media platforms (Twitter, Facebook)
- Unstructured text data requiring heavy preprocessing
- Binary labeled: Fake and Real

### Preprocessing
1. Lowercasing all text
2. Removing URLs, mentions, hashtags, special characters
3. Tokenization
4. Stopword removal
5. Stemming/Lemmatization

### Feature Extraction (CORE FOCUS)
Six combinations tested:

| Vectorizer | N-gram Type | Example |
|------------|-------------|---------|
| BOW | Unigrams | ["fake", "news", "spread"] |
| BOW | Bigrams | ["fake news", "news spread"] |
| BOW | Trigrams | ["fake news spread"] |
| TF-IDF | Unigrams | weighted ["fake", "news", "spread"] |
| TF-IDF | Bigrams | weighted ["fake news", "news spread"] |
| TF-IDF | Trigrams | weighted ["fake news spread"] |

### ML Classifiers
1. **SVM** — margin-based, effective for high-dimensional text data
2. **Naive Bayes** — probabilistic, fast, good baseline
3. **Decision Tree** — rule-based, interpretable but prone to overfitting

### Workflow
```
Social Media Data → Preprocessing → 6 Feature Extraction Combos → 3 Classifiers each → 18 Experiments → Compare Results
```

---

## PAPER 5: Results & Evaluation — Qasem & Sajid (2022)

### Performance Results (18 experiments)

| Feature Extraction | SVM | Naive Bayes | Decision Tree |
|---|:---:|:---:|:---:|
| BOW + Unigrams | Good | Moderate | Low |
| BOW + Bigrams | Good | Moderate | Low |
| BOW + Trigrams | Moderate | Low | Low |
| **TF-IDF + Unigrams** | **Highest** | Good | Moderate |
| TF-IDF + Bigrams | High | Good | Moderate |
| TF-IDF + Trigrams | High | Moderate | Low |

### Key Findings
1. **TF-IDF + Unigrams + SVM** is the most effective combination overall
2. **TF-IDF consistently outperformed BOW** across all N-gram sizes and all classifiers
3. **N-grams beyond unigrams did NOT significantly improve accuracy** — bigrams and trigrams added computational cost without meaningful accuracy gains
4. **SVM was the best classifier** across all 6 feature extraction combinations
5. **Decision Tree performed worst** — prone to overfitting on text data
6. Trigrams actually **decreased accuracy** in some cases due to sparse feature space

### Significance
- Settled the N-gram debate: **simple unigrams with TF-IDF are sufficient**
- No need for complex bigram/trigram features — saves computation without sacrificing accuracy
- Confirmed SVM's dominance for text-based fake news classification

### Limitations
- Only 3 classifiers tested — no ensemble methods, no deep learning
- No word embedding techniques (Word2Vec, GloVe) compared
- Dataset limited to social media text only

---

## PAPER 6: Introduction — Empirical Analysis (2022)

### Background Context
- Most fake news detection papers test on curated, clean datasets (Kaggle, ISOT)
- Real-world social networks contain noisy, informal, and messy text data
- This paper focuses on testing ML models on **real-world network data** rather than curated datasets

### Why the Problem is Important
- A model that works well on a clean Kaggle dataset may fail on real social media posts
- Real-world fake news contains slang, abbreviations, emojis, and informal grammar
- Practical deployment requires models tested on realistic data conditions

### Key Idea of the Paper
- Empirically evaluate **4 ML classifiers** (Naive Bayes, SVM, Logistic Regression, Random Forest) on real-world social network data
- Use **TF-IDF** for text vectorization
- Provide a practical comparison focused on **real-world applicability** rather than benchmark performance

---

## PAPER 6: Methodology — Empirical Analysis (2022)

### Dataset Used
- Real-world social network data
- Total of **8,980 test samples**
- Binary classification: Fake news vs True news
- Data sourced from actual social media platforms with real user posts

### Preprocessing
1. Text cleaning (removing noise, URLs, special characters)
2. Tokenization
3. Stopword removal
4. **TF-IDF Vectorization** for feature extraction

### ML Algorithms Compared
1. **Naive Bayes** — probabilistic, fast, assumes feature independence
2. **SVM (Support Vector Machine)** — finds optimal separating hyperplane
3. **Logistic Regression** — linear model with sigmoid activation
4. **Random Forest** — ensemble of decision trees with majority voting

### Workflow
```
Real-World Social Network Data → Text Cleaning → TF-IDF Vectorization → NB / SVM / LR / RF → Evaluate Accuracy, Precision, Recall, F1
```

---

## PAPER 6: Results & Evaluation — Empirical Analysis (2022)

### Performance Comparison

| Algorithm | Precision (Fake) | Precision (True) | Recall | F1-Score | Overall |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Naive Bayes | 0.95 | 0.93 | 0.94 | 0.94 | Good |
| **SVM** | **0.96** | **0.95** | **0.96** | **0.96** | **Best** |
| Logistic Regression | 0.95 | 0.94 | 0.95 | 0.95 | Very Good |
| Random Forest | 0.94 | 0.93 | 0.93 | 0.93 | Good |

### Key Findings
1. **SVM exhibited the greatest efficacy** in detecting fake news on real-world network data
2. **Logistic Regression performed nearly as well** as SVM — strong practical choice due to simplicity and interpretability
3. **Naive Bayes** performed surprisingly well despite its simplicity — precision of 0.95 for fake news detection
4. **Random Forest** showed slightly lower performance, possibly due to overfitting on noisy real-world text
5. All models achieved **F1-scores above 0.93** — indicating ML approaches are viable for real-world deployment

### Significance
- Validated that ML models trained with TF-IDF work on **real-world noisy data**, not just clean benchmarks
- SVM's dominance holds even on messy social media text
- High precision across all models means **low false positive rate** — real news rarely misclassified as fake

### Limitations
- No cross-domain evaluation (testing model trained on one platform on another)
- No multimodal analysis (images, videos, user metadata)
- Dataset details (exact source platform) limited in available information

---

## PAPER 7: Introduction — Rizal et al. (2025)

### Background Context
- By 2025, SVM and TF-IDF had been established as strong performers, but few studies focused on **optimizing SVM specifically** for fake news detection
- Previous studies compared many models but did not deeply tune any single one
- This paper focuses on extracting maximum performance from SVM by optimizing it with TF-IDF

### Why the Problem is Important
- Despite many published methods, fake news remains a growing global problem
- Social media platforms need highly accurate (>99%) automated systems to filter content at scale
- Even small accuracy improvements matter when processing millions of articles daily

### Key Idea of the Paper
- Optimize the **SVM algorithm specifically** using TF-IDF feature extraction
- Compare optimized SVM against Random Forest and Logistic Regression
- Demonstrate that proper optimization can push SVM + TF-IDF to near-perfect accuracy

---

## PAPER 7: Methodology — Rizal et al. (2025)

### Dataset Used
- **Two publicly available Kaggle datasets** containing labeled fake and real news articles
- Large-scale dataset with balanced classes
- Binary classification: Fake vs Real

### Preprocessing Pipeline
1. Text cleaning (removing punctuation, numbers, special characters)
2. Lowercasing
3. Tokenization
4. Stopword removal
5. Stemming

### Feature Extraction
- **TF-IDF (Term Frequency - Inverse Document Frequency)**
- Optimized TF-IDF parameters: max_features, ngram_range, min_df, max_df
- Focus on extracting the most discriminative features

### ML Algorithms
1. **SVM (optimized)** — hyperparameter tuning for kernel type, C parameter, gamma
2. **Random Forest** — ensemble baseline
3. **Logistic Regression** — linear baseline

### Workflow
```
2 Kaggle Datasets → Heavy Preprocessing → Optimized TF-IDF Extraction → SVM (tuned) / RF / LR → Accuracy Comparison
```

---

## PAPER 7: Results & Evaluation — Rizal et al. (2025)

### Performance Results

| Algorithm | Accuracy | Precision | Recall |
|-----------|:---:|:---:|:---:|
| Logistic Regression | ~97% | ~97% | ~97% |
| Random Forest | ~96% | ~96% | ~96% |
| **SVM + TF-IDF (Optimized)** | **99.6%** | **~99%** | **~99%** |

### Key Findings
1. **SVM with optimized TF-IDF achieved 99.6% accuracy** — the highest reported accuracy among all 8 papers
2. SVM outperformed both Random Forest and Logistic Regression by a significant margin
3. **TF-IDF optimization** (parameter tuning) was a key contributor to the high accuracy
4. Even without optimization, SVM outperformed RF and LR, confirming its inherent strength for text classification

### Why So High?
- Proper TF-IDF parameter tuning extracted highly discriminative features
- SVM hyperparameter optimization (kernel, regularization) improved generalization
- Kaggle datasets are relatively clean and well-separated compared to real-world data

### Limitations
- **99.6% accuracy may not generalize** to noisy real-world social media data
- No cross-dataset validation (trained and tested on same source)
- No adversarial testing (deliberately crafted fake news designed to evade detection)
- Possible dataset leakage or overfitting not addressed
- No explainability — cannot tell users WHY an article was classified as fake

---

## PAPER 8: Introduction — Scalable TF-IDF + LR (2025)

### Background Context
- Most research focuses on maximizing accuracy, but **scalability** for production deployment is equally important
- Complex models (SVM with RBF kernel, deep learning) are accurate but slow at inference time
- This paper prioritizes building a **scalable, production-ready** fake news detection pipeline

### Why the Problem is Important
- Real-world systems need to process **thousands of articles per second** in real-time
- Heavy models create bottlenecks in production — latency matters as much as accuracy
- A slightly less accurate but fast model is often more practical than a perfect but slow model

### Key Idea of the Paper
- Use **TF-IDF + Logistic Regression** as a lightweight, scalable pipeline
- Logistic Regression is inherently fast at both training and inference
- Feature coefficients from LR provide **built-in interpretability** — can explain which words made the model flag an article as fake
- Design the pipeline for **horizontal scaling** in production environments

---

## PAPER 8: Methodology — Scalable TF-IDF + LR (2025)

### Dataset Used
- Large-scale news article dataset
- Binary labeled: Fake vs Real
- Designed to simulate production-scale data volumes

### Preprocessing Pipeline
1. Text normalization (lowercasing, unicode handling)
2. Removal of HTML tags, URLs, email addresses
3. Tokenization
4. Stopword removal
5. Stemming / Lemmatization

### Feature Extraction
- **TF-IDF Vectorizer**
- Each article represented as a high-dimensional sparse vector
- Optimized parameters: max_features tuned for balance between accuracy and computation

### ML Algorithm
- **Logistic Regression (LR)**
  - Linear model with sigmoid activation for binary classification
  - Fast training: O(n) complexity
  - Fast inference: single matrix multiplication
  - **Interpretable**: Feature coefficients show which words contribute most to the fake/real decision

### Scalability Features
- Sparse matrix operations for memory efficiency
- Pipeline designed for horizontal scaling
- Lightweight model size — easily deployable on web servers

### Workflow
```
Large-Scale Data → Preprocessing Pipeline → TF-IDF (sparse) → Logistic Regression → Real-Time Prediction + Explanation (top contributing words)
```

---

## PAPER 8: Results & Evaluation — Scalable TF-IDF + LR (2025)

### Performance Results

| Metric | Value |
|--------|:---:|
| Accuracy | ~98.3% |
| Precision | ~98% |
| Recall | ~98% |
| F1-Score | ~98% |
| Training Time | Very Fast (seconds) |
| Inference Time | <1ms per article |

### Key Findings
1. **Logistic Regression achieved ~98.3% accuracy** — nearly matching complex models like SVM and Random Forest
2. **Training completed in seconds** vs minutes for SVM and hours for deep learning
3. **Inference takes <1ms per article** — suitable for real-time production use
4. **Built-in interpretability**: LR coefficients reveal which headline terms (e.g., "BREAKING", "shocking", "you won't believe") are most associated with fake news
5. The model is **lightweight** — entire trained model fits in a few MBs, easily deployable

### Interpretability Example
Top words with highest "fake" coefficients:
- "shocking", "unbelievable", "secret", "exposed", "breaking"

Top words with highest "real" coefficients:
- "reported", "according", "officials", "statement", "confirmed"

### Significance
- Proves that **simple models can achieve near-state-of-the-art accuracy** while being 100x faster
- LR's interpretability addresses the **explainability gap** identified in other papers
- Most practical model for real-world deployment among all 8 papers

### Limitations
- LR is a **linear model** — may miss complex non-linear patterns in adversarial fake news
- No comparison with ensemble methods (stacking, boosting)
- Accuracy slightly lower than optimized SVM (98.3% vs 99.6%)

---

## SLIDE 15: Summary — Strengths of Papers

| S.N. | Paper (Authors, Title, Year, Publisher) | What is Good | Innovations | Assumptions |
|:---:|---|---|---|---|
| 1 | Granik & Mesyura, "Fake News Detection Using NB Classifier", 2017, IEEE UKRCON | Foundational work; simple and reproducible approach | First to apply NB specifically to Facebook fake news posts | Features are independent (Naive assumption); Facebook posts are representative of all fake news |
| 2 | Poddar et al., "Comparison of Various ML Models", 2019, IEEE i-PACT | Comprehensive comparison of 5 models with 2 vectorizers on same dataset | Systematic comparison of Count Vectorizer vs TF-IDF across multiple classifiers | Kaggle dataset is representative; text features alone are sufficient |
| 3 | Manzoor et al., "Systematic Review", 2019, IEEE ICOEI | Broad coverage of existing literature; identifies trends and gaps | Categorizes approaches by technique type; highlights shift to deep learning | Published papers are representative of all approaches |
| 4 | Jain et al., "SVM, NB and LSTM", 2022, IEEE Confluence | Bridges traditional ML and deep learning comparison | Compares classical ML (SVM, NB) with DL (LSTM) on same task | LSTM captures more context; sequential information matters for detection |
| 5 | Qasem & Sajid, "N-grams with BOW and TF-IDF", 2022, IEEE ICDABI | Thorough analysis of feature extraction impact on performance | Systematically varies n-gram sizes with both BOW and TF-IDF | N-grams should capture contextual information better than unigrams |
| 6 | "Empirical Analysis of NB, SVM, LR, RF", 2022, IEEE | Tests on real-world social network data; practical applicability | Analysis on real-world network data rather than curated datasets | Real-world networks have consistent patterns of fake news |
| 7 | Rizal et al., "SVM and TF-IDF", 2025, IEEE | Highest accuracy (99.6%); demonstrates TF-IDF+SVM dominance | Optimization of SVM hyperparameters specifically for fake news | Dataset is well-curated and separable |
| 8 | "TF-IDF and Logistic Regression", 2025, IEEE | Focuses on scalability; practical for large-scale deployment | Scalable pipeline design using lightweight model | LR is sufficient for production-level accuracy |

---

## SLIDE 16: Summary — Limitations of Papers

| Paper (Authors, Title, Year, Publisher) | Weaknesses | Missing Aspects | Assumptions |
|---|---|---|---|
| Granik & Mesyura (2017) | Only 74% accuracy; single model tested; small dataset | No comparison with other models; no TF-IDF; no deep learning | BOW is sufficient for text representation |
| Poddar et al. (2019) | No deep learning models compared; no feature engineering beyond vectorizers | No hyperparameter tuning details; no cross-validation mentioned | Single train-test split is sufficient |
| Manzoor et al. (2019) | Survey only, no implementation; limited to papers up to 2019 | Does not cover post-2019 advances (BERT, transformers) | Literature is comprehensive enough to draw conclusions |
| Jain et al. (2022) | LSTM training is computationally expensive; limited hyperparameter exploration | No transformer/BERT comparison; no explainability analysis | Sequential models always capture better context |
| Qasem & Sajid (2022) | Only 3 classifiers tested; no deep learning models | No word embeddings (Word2Vec, GloVe) comparison; no ensemble methods | N-gram analysis alone reveals optimal feature extraction |
| Empirical Analysis (2022) | Limited details available on dataset specifics | No cross-domain evaluation; no multimodal analysis | Patterns in one network generalize to others |
| Rizal et al. (2025) | 99.6% accuracy may indicate overfitting or dataset leakage | No cross-dataset validation; no adversarial testing | High accuracy = robust model (not always true) |
| TF-IDF + LR (2025) | LR is a linear model; may miss non-linear patterns | No comparison with non-linear models on same data | Linear separability of features is sufficient |

---

## SLIDE 17: Research Gaps

### Gaps Identified from Literature Survey

1. **No Multimodal Analysis**: All 8 papers focus only on text. Real fake news also contains images, videos, and user metadata (shares, comments, profile age).

2. **No Cross-Dataset Validation**: Most papers train and test on the same dataset. Performance drops significantly when tested on a different dataset (domain shift problem).

3. **No Indian Language Support**: All papers work on English text. Fake news in Hindi, Marathi, and other Indian languages (especially on WhatsApp) remains unaddressed.

4. **Limited Explainability**: None of the papers explain *why* a particular news article was classified as fake. Users need to understand the reasoning, not just the label.

5. **No Real-Time Detection**: Papers test on static datasets. A real-world system needs to process news articles in real-time as they appear on social media.

6. **No Ensemble Methods**: Most papers test individual models. Combining multiple models (ensemble/stacking) could improve robustness and accuracy.

7. **No Adversarial Testing**: None of the papers test against adversarial fake news (deliberately crafted to evade ML detection).

8. **Temporal Drift**: Models trained on 2017 data may not work well on 2025 fake news. Writing styles and techniques of fake news creators evolve over time.

---

## SLIDE 18: Your Proposed Idea

### Improvement Idea
**Ensemble-based Fake News Detection System combining TF-IDF + Multiple ML Models with Explainability**

### New Approach
1. Use **TF-IDF (unigrams)** for feature extraction (proven best from literature)
2. Train 3 models: **SVM, Logistic Regression, Random Forest**
3. Use **Majority Voting Ensemble** — final prediction = what 2 out of 3 models agree on
4. Add **feature importance visualization** — show which words contributed most to the "Fake" prediction (using model coefficients from LR/SVM)
5. **Cross-dataset validation** — train on Kaggle dataset, test on ISOT dataset to check generalization

### Expected Outcome
- Higher robustness than any single model (ensemble reduces variance)
- ~95-97% accuracy with better generalization across datasets
- Explainable output — user can see *why* the system flagged an article as fake
- Addresses 3 research gaps: no ensemble methods, no explainability, no cross-dataset testing

---

## SLIDE 19: Conclusion

### Key Learnings
1. **TF-IDF is the most effective feature extraction** technique for text-based fake news detection, consistently outperforming BOW across all papers
2. **SVM is the best-performing traditional ML model** for this task, achieving up to 99.6% accuracy when optimized with TF-IDF
3. **Deep learning (LSTM) captures sequential patterns** but requires significantly more compute and does not always outperform well-tuned SVM
4. **N-grams beyond unigrams** do not provide significant accuracy improvement, making unigram TF-IDF the most efficient choice
5. **Key research gaps remain**: multimodal detection, cross-dataset validation, Indian language support, real-time deployment, and model explainability
6. **Ensemble approaches** combining multiple models with explainability could address current limitations and improve both accuracy and trustworthiness

---

## SLIDE 20: References

1. M. Granik and V. Mesyura, "Fake news detection using naive Bayes classifier," *2017 IEEE First Ukraine Conference on Electrical and Computer Engineering (UKRCON)*, pp. 900-903, 2017. [https://ieeexplore.ieee.org/document/8100379/](https://ieeexplore.ieee.org/document/8100379/)

2. K. Poddar, G. B. Amali D., and K. S. Umadevi, "Comparison of Various Machine Learning Models for Accurate Detection of Fake News," *2019 Innovations in Power and Advanced Computing Technologies (i-PACT)*, Vellore, India, 2019. [https://ieeexplore.ieee.org/document/8960044/](https://ieeexplore.ieee.org/document/8960044/)

3. S. I. Manzoor, J. Singla, and Nikita, "Fake News Detection Using Machine Learning Approaches: A Systematic Review," *2019 3rd International Conference on Trends in Electronics and Informatics (ICOEI)*, pp. 230-234, 2019. [https://ieeexplore.ieee.org/document/8862770/](https://ieeexplore.ieee.org/document/8862770/)

4. P. Jain, S. Sharma, Monica, and P. K. Aggarwal, "Classifying Fake News Detection Using SVM, Naive Bayes and LSTM," *2022 12th International Conference on Cloud Computing, Data Science & Engineering (Confluence)*, pp. 460-464, 2022. [https://ieeexplore.ieee.org/document/9734129/](https://ieeexplore.ieee.org/document/9734129/)

5. A. E. Qasem and M. Sajid, "Exploring the Effect of N-grams with BOW and TF-IDF Representations on Detecting Fake News," *2022 International Conference on Data Analytics for Business and Industry (ICDABI)*, 2022. [https://ieeexplore.ieee.org/document/10041537/](https://ieeexplore.ieee.org/document/10041537/)

6. "An Empirical Analysis of Naive Bayes, SVM, Logistic Regression and Random Forest to Spot False Information in Real-World Networks," *IEEE Conference Publication*, 2022. [https://ieeexplore.ieee.org/document/9862430/](https://ieeexplore.ieee.org/document/9862430/)

7. R. Rizal, A. Faturahman, A. Impron, and A. Rahmatulloh, "Unveiling the Truth: Detecting Fake News Using SVM and TF-IDF," *IEEE Conference Publication*, 2025. [https://ieeexplore.ieee.org/abstract/document/10933324/](https://ieeexplore.ieee.org/abstract/document/10933324/)

8. "A Scalable Approach to Fake News Detection Using TF-IDF and Logistic Regression," *IEEE Conference Publication*, 2025. [https://ieeexplore.ieee.org/document/11135996/](https://ieeexplore.ieee.org/document/11135996/)

---

## QUICK REFERENCE: Key Numbers to Remember

| Fact | Value |
|------|-------|
| Best algorithm overall | SVM + TF-IDF |
| Highest accuracy reported | 99.6% (Paper 7) |
| Lowest accuracy reported | 74% (Paper 1, NB with BOW) |
| Best feature extraction | TF-IDF (unigrams) |
| Most common dataset | Kaggle Fake News (26,000 articles) |
| SDG mapped | SDG 16 — Peace, Justice & Strong Institutions |
| Total papers surveyed | 8 (6 research + 1 survey + 1 scalability focused) |
| Year range | 2017 - 2025 |
