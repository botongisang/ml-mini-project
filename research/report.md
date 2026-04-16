# IEEE-Format ML Project Report — Template & Rules
**For Mumbai University BE Projects (IEEE-compliant)**

---

## TABLE OF CONTENTS

1. [Formatting Rules](#formatting-rules)
2. [Report Structure](#report-structure)
3. [Citation Rules](#citation-rules)
4. [Writing Rules](#writing-rules)
5. [Full Template](#full-template)

---

## FORMATTING RULES

### Page Setup
| Element | Specification |
|---|---|
| Paper size | A4 (210mm × 297mm) |
| Top margin | 1 inch (25mm) |
| Bottom margin | 1 inch (25mm) |
| Right margin | 1 inch (25mm) |
| Left margin | 1.25 inch (30mm) — binding space |
| Font | Times New Roman throughout |
| Body text size | 12 pt |
| Line spacing | 1.5 (body text) |
| Alignment | Justified |

### Font Hierarchy
| Element | Size | Style | Alignment |
|---|---|---|---|
| University / Institute name | 14 pt | Bold, CAPS | Center |
| Report main title | 16 pt | Bold, CAPS | Center |
| Chapter title | 16 pt | Bold, CAPS | Left |
| Section heading (1.1) | 14 pt | Bold, CAPS | Left |
| Subsection heading (1.1.1) | 13 pt | Bold | Left |
| Body text | 12 pt | Regular | Justified |
| Table/Figure caption | 11 pt | Bold label + regular text | Center |
| Table content | 10–11 pt | Regular | Center/Left |
| Footnotes | 10 pt | Regular | Left |

### Page Numbering
- **Front matter** (title page → list of tables): lowercase Roman numerals — i, ii, iii …
- **Main body** (Introduction onward): Arabic numerals — 1, 2, 3 …
- Number starts at Chapter 1 / Introduction, not at title page

### Figure and Table Numbering
- Numbered by chapter: Figure 1.1, Figure 1.2, Figure 2.1
- Tables same: Table 1.1, Table 2.1
- Caption for figures: **below** the figure
- Caption for tables: **above** the table
- Caption format: **Figure 1.1:** Description of figure.

---

## REPORT STRUCTURE

### Front Matter (Roman numerals, not counted in page total)

```
Title Page                  — i (not printed)
Certificate of Approval     — ii
Declaration                 — iii
Acknowledgments             — iv
Abstract                    — v
Keywords                    — v (same page as abstract)
Table of Contents           — vi onward
List of Figures             — after ToC
List of Tables              — after List of Figures
List of Abbreviations       — if applicable
```

### Main Body (Arabic numerals, 1 onward)

```
Chapter 1  — Introduction
Chapter 2  — Literature Review
Chapter 3  — System Design and Methodology
Chapter 4  — Implementation
Chapter 5  — Results and Discussion
Chapter 6  — Conclusion and Future Work
References
Appendices (A, B, C …)
```

### Heading Numbering Convention

```
1.         CHAPTER TITLE          ← Chapter heading
1.1        Section Title          ← Section
1.1.1      Subsection Title       ← Subsection
```

No heading deeper than 3 levels. If you need more nesting, restructure.

---

## CITATION RULES

### In-Text Citations (IEEE Numbered Style)
- Number citations in **order of first appearance**: [1], [2], [3]
- Place citation **before** closing punctuation: "...was demonstrated [1]."
- Multiple sources at once: [1], [2], [3] or compressed range [1]–[3]
- Once assigned, a citation number **never changes**
- Do **not** use author names in-text — only the number

### Reference List Rules
- Section title: **REFERENCES** (bold, all caps)
- Listed in **citation order** (not alphabetical)
- Numbers flush left: [1], [2], [3]
- Hanging indent on continuation lines

### Reference Formats

**Journal article:**
```
[N] A. Author and B. Author, "Article title in sentence case,"
    Abbrev. Journal, vol. X, no. Y, pp. XXX–XXX, Mon. YYYY,
    doi: XX.XXXX/XXXXX.
```

**Conference paper:**
```
[N] A. Author, "Paper title," in Proc. Conf. Name (ABBREV.),
    City, Country, Mon. YYYY, pp. XXX–XXX.
```

**Book:**
```
[N] A. Author, Book Title in Title Case. City: Publisher, YYYY.
```

**Book chapter:**
```
[N] A. Author, "Chapter title," in Book Title, A. Editor, Ed.
    City: Publisher, YYYY, ch. X, pp. XXX–XXX.
```

**Website / online resource:**
```
[N] A. Author, "Page title," Site Name. [Online].
    Available: https://url.com. [Accessed: DD Mon. YYYY].
```

**Dataset / Kaggle / UCI:**
```
[N] A. Author, "Dataset name," Platform Name, YYYY. [Online].
    Available: https://url.com. [Accessed: DD Mon. YYYY].
```

---

## WRITING RULES

### Abstract
- Single paragraph only — **no bullet points, no subsections**
- 150–250 words (target: 200)
- Must include: problem, approach, key method, main result, significance
- No abbreviations (unless defined), no citations, no equations, no figures
- Written in past tense (what was done) + present tense (what the paper does)

### Keywords
- 3–5 keywords, comma-separated
- Placed immediately after abstract
- Format: **Keywords—** Machine Learning, Fake News Detection, NLP, BERT

### Introduction
- Starts broad, narrows to your specific problem
- Must state: motivation, problem definition, objectives, paper structure
- Last paragraph should outline what each section covers

### Literature Review
- Group related works by theme or approach, not by chronology
- Every claim must have a citation
- End with a gap analysis: what existing work does not address that yours does

### Methodology
- Enough detail to replicate your work
- State dataset source, size, features
- State all preprocessing steps
- State all model architectures, hyperparameters, training setup
- Use past tense throughout

### Results and Discussion
- Present results objectively first, then interpret
- Every table/figure must be referenced in text before it appears
- Compare against baseline or prior work
- Acknowledge limitations honestly

### Conclusion
- No new information
- Summarise findings, connect back to objectives
- State each conclusion as a numbered point
- Include a Future Work paragraph: concrete, specific next steps

### Style Rules
- Use passive voice sparingly — prefer active where possible
- Avoid first person singular ("I") — use "we" or passive construction
- Avoid informal language: "basically," "really," "lots of," "kind of"
- Spell out acronyms on first use: Natural Language Processing (NLP)
- Numbers: spell out one through nine, use numerals for 10 and above
- Equations get numbers: (1), (2), (3) — right-aligned, referenced in text

---

## FULL TEMPLATE

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    TITLE PAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**[UNIVERSITY NAME]**
*[Institute / College Name]*
*[Department Name]*

---

# [PROJECT TITLE IN FULL CAPS]

*A Project Report submitted in partial fulfillment of the requirements for the degree of*
**Bachelor of Engineering in [Branch Name]**
*University of Mumbai*

---

Submitted by:

| Roll No. | Name |
|---|---|
| [Roll No.] | [Student Name 1] |
| [Roll No.] | [Student Name 2] |
| [Roll No.] | [Student Name 3] |

---

**Project Guide:** [Guide Name], [Designation]

**Academic Year:** 2025–26

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    CERTIFICATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

This is to certify that the project titled **"[Project Title]"** is a bonafide work carried out by **[Student Names]** in partial fulfillment of the requirements for the degree of Bachelor of Engineering in [Branch] under the University of Mumbai, during the academic year 2025–26.

This project has been carried out under the guidance of **[Guide Name]**, [Designation], Department of [Department Name], [Institute Name].

---

| | |
|---|---|
| **Project Guide** | **Head of Department** |
| [Name] | [Name] |
| [Designation] | [Designation] |

**Internal Examiner:** \_\_\_\_\_\_\_\_\_\_\_ &nbsp;&nbsp;&nbsp; **External Examiner:** \_\_\_\_\_\_\_\_\_\_\_

**Date:** \_\_\_\_\_\_\_\_\_\_\_ &nbsp;&nbsp;&nbsp; **Place:** \_\_\_\_\_\_\_\_\_\_\_

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    DECLARATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

We, the undersigned, declare that this project report titled **"[Project Title]"** submitted to the University of Mumbai in partial fulfillment of the requirements for the degree of Bachelor of Engineering in [Branch] is our own original work.

The work contained in this report has not been submitted elsewhere, in full or in part, for any other degree or examination. All sources consulted have been acknowledged through proper citations.

---

| Roll No. | Name | Signature |
|---|---|---|
| [Roll No.] | [Student Name 1] | \_\_\_\_\_\_\_\_ |
| [Roll No.] | [Student Name 2] | \_\_\_\_\_\_\_\_ |
| [Roll No.] | [Student Name 3] | \_\_\_\_\_\_\_\_ |

**Date:** \_\_\_\_\_\_\_\_\_\_\_ &nbsp;&nbsp;&nbsp; **Place:** \_\_\_\_\_\_\_\_\_\_\_

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ACKNOWLEDGMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

We express our sincere gratitude to **[Guide Name]**, [Designation], for their invaluable guidance, continuous support, and constructive feedback throughout this project.

We are grateful to **[HoD Name]**, Head of the Department of [Department Name], and the faculty members of [Institute Name] for providing the necessary resources and encouragement.

We also acknowledge the University of Mumbai for the academic framework that made this work possible.

Finally, we thank our families and peers for their constant motivation.

---

**[Student Names]**
[Department], [Institute]

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ABSTRACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

[Single paragraph. 150–250 words. State the problem, the approach taken, the key technique or model used, the primary result, and its significance. No citations. No abbreviations unless defined inline. No bullet points. Past tense for what was done; present tense for what the paper presents.]

**Keywords—** [Keyword 1], [Keyword 2], [Keyword 3], [Keyword 4], [Keyword 5]

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                TABLE OF CONTENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

| Section | Page |
|---|---|
| Certificate | ii |
| Declaration | iii |
| Acknowledgments | iv |
| Abstract | v |
| Table of Contents | vi |
| List of Figures | vii |
| List of Tables | viii |
| List of Abbreviations | ix |
| **Chapter 1 — Introduction** | **1** |
| 1.1 Background and Motivation | 1 |
| 1.2 Problem Statement | 2 |
| 1.3 Objectives | 3 |
| 1.4 Scope of Work | 3 |
| 1.5 Report Organization | 4 |
| **Chapter 2 — Literature Review** | **5** |
| 2.1 [Theme / Approach Group 1] | 5 |
| 2.2 [Theme / Approach Group 2] | 7 |
| 2.3 [Theme / Approach Group 3] | 10 |
| 2.4 Research Gap | 12 |
| **Chapter 3 — System Design and Methodology** | **14** |
| 3.1 System Architecture | 14 |
| 3.2 Dataset | 16 |
| 3.3 Data Preprocessing | 17 |
| 3.4 Feature Engineering | 19 |
| 3.5 Model Selection | 20 |
| 3.6 Training Setup | 22 |
| **Chapter 4 — Implementation** | **24** |
| 4.1 Development Environment | 24 |
| 4.2 [Module / Component 1] | 25 |
| 4.3 [Module / Component 2] | 27 |
| 4.4 [Module / Component 3] | 29 |
| **Chapter 5 — Results and Discussion** | **32** |
| 5.1 Evaluation Metrics | 32 |
| 5.2 Baseline Comparison | 33 |
| 5.3 Experimental Results | 34 |
| 5.4 Discussion | 37 |
| 5.5 Limitations | 39 |
| **Chapter 6 — Conclusion and Future Work** | **41** |
| 6.1 Conclusions | 41 |
| 6.2 Future Work | 42 |
| References | 43 |
| Appendix A — [Title] | 46 |
| Appendix B — [Title] | 48 |

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              CHAPTER 1 — INTRODUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 1.1 Background and Motivation

[Contextualise the domain broadly. What is the broader problem area? Why does it matter in the real world today? Use 2–3 paragraphs moving from the general domain to the specific problem your project addresses. Every factual claim must have a citation [N].]

[Example: The proliferation of misinformation on social media platforms has emerged as a significant societal challenge [1]. Studies indicate that false information spreads six times faster than verified news online [2]. Automated detection systems leveraging Machine Learning (ML) and Natural Language Processing (NLP) have shown promise in addressing this challenge [3], [4].]

## 1.2 Problem Statement

[State the specific problem in 1–2 precise paragraphs. Avoid vague language. Define any domain-specific terms. The reader must be able to understand exactly what is being solved.]

**Formal Problem Definition:** [One clear sentence. Example: "Given a news article as text input, classify it as real or fake with high accuracy using a machine learning pipeline trained on labelled data."]

## 1.3 Objectives

The primary objectives of this project are:

1. To [Objective 1 — specific and measurable]
2. To [Objective 2]
3. To [Objective 3]
4. To [Objective 4]

## 1.4 Scope of Work

[Define the boundaries of what this project covers and what it explicitly does not cover. This prevents examiner questions about excluded scenarios.]

**In scope:**
- [Item 1]
- [Item 2]

**Out of scope:**
- [Item 1]
- [Item 2]

## 1.5 Report Organization

The remainder of this report is organized as follows. Chapter 2 presents a review of related literature. Chapter 3 describes the system design and methodology. Chapter 4 details the implementation. Chapter 5 presents and discusses experimental results. Chapter 6 concludes the report and outlines directions for future work.

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            CHAPTER 2 — LITERATURE REVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 2.1 [Thematic Group 1 — e.g., Classical ML Approaches]

[Review 3–5 related works grouped by theme, not by date. Discuss each work's approach, dataset, result, and limitation. Do not just list papers — synthesise them.]

[Example paragraph: Shu et al. [5] proposed a framework combining linguistic features with social context signals for fake news detection. Their approach achieved 89.2% accuracy on the FakeNewsNet dataset but was limited by its dependence on social graph data, which is unavailable in real-time scenarios [5]. Ahmed et al. [6] extended this work by incorporating TF-IDF and n-gram features, reporting an F1-score of 0.91 on a balanced corpus [6]. However, both approaches relied on handcrafted features, making them brittle in the face of evolving linguistic patterns.]

## 2.2 [Thematic Group 2 — e.g., Deep Learning Approaches]

[Same approach — synthesize, compare, critique.]

## 2.3 [Thematic Group 3 — e.g., Transformer / BERT-based Approaches]

[Same approach.]

## 2.4 Research Gap

[This section is critical. Summarise what the reviewed literature does not address. Then state directly how your project addresses that gap.]

**Table 2.1:** Comparison of existing approaches

| Reference | Technique | Dataset | Accuracy | Limitation |
|---|---|---|---|---|
| [5] | SVM + Linguistic | FakeNewsNet | 89.2% | Requires social graph |
| [6] | TF-IDF + n-gram | Custom | F1: 0.91 | Brittle to style variation |
| [7] | LSTM | LIAR | 73.4% | Long-range dependency loss |
| [8] | BERT fine-tuned | ISOT | 95.1% | High compute cost |
| **This work** | **[Your approach]** | **[Your dataset]** | **[Your result]** | **[Your limitation]** |

Based on the reviewed literature, a gap exists in [specific gap]. This project addresses that gap by [specific contribution].

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       CHAPTER 3 — SYSTEM DESIGN AND METHODOLOGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 3.1 System Architecture

[Describe the overall system at a high level. This should be accompanied by a system architecture diagram.]

**Figure 3.1:** High-level system architecture

```
[Input Layer] → [Preprocessing] → [Feature Extraction] → [Model] → [Output]
```

*(Replace the above with an actual diagram image in the final report)*

[Describe each component in the diagram briefly. The rest of the chapter expands on each.]

## 3.2 Dataset

[Describe the dataset completely: source, collection method, size, class distribution, features, format, and any known limitations.]

**Table 3.1:** Dataset summary

| Property | Value |
|---|---|
| Dataset name | [Name] |
| Source | [URL / paper reference] |
| Total records | [N] |
| Features | [List of features] |
| Class distribution | Real: [N] / Fake: [N] |
| Train / Val / Test split | [%] / [%] / [%] |
| Collection period | [Date range] |

[Discuss class imbalance if present. Discuss any known biases in the dataset.]

## 3.3 Data Preprocessing

[Describe every transformation applied to the raw data, in order. Be precise enough that another person could reproduce your pipeline.]

The preprocessing pipeline consists of the following steps:

1. **Text normalisation:** [e.g., lowercasing, removal of HTML tags, special characters]
2. **Tokenisation:** [e.g., word-level, subword using BPE]
3. **Stop word removal:** [tool and word list used]
4. **Stemming / Lemmatisation:** [algorithm used]
5. **Handling missing values:** [strategy used]
6. **Class balancing:** [e.g., SMOTE, under-sampling, weighted loss]

## 3.4 Feature Engineering

[Describe the features used as model input. If using a pretrained model, describe how text is encoded.]

## 3.5 Model Selection

[Justify your choice of model(s). Compare candidate models and explain why the selected approach is appropriate for the problem.]

**Table 3.2:** Candidate model comparison

| Model | Strengths | Weaknesses | Selected? |
|---|---|---|---|
| Logistic Regression | Fast, interpretable | Linear boundary | Baseline |
| Random Forest | Robust, non-linear | Feature-dependent | No |
| LSTM | Sequential context | Gradient issues | No |
| [Your model] | [Strengths] | [Weaknesses] | Yes |

### 3.5.1 Model Architecture

[Detail the architecture: layers, dimensions, activation functions, loss function, optimiser. For neural models, provide a layer-by-layer description.]

## 3.6 Training Setup

| Hyperparameter | Value |
|---|---|
| Learning rate | [value] |
| Batch size | [value] |
| Epochs | [value] |
| Optimiser | [e.g., Adam] |
| Loss function | [e.g., Binary Cross-Entropy] |
| Random seed | [value — for reproducibility] |
| Hardware | [CPU/GPU details] |

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              CHAPTER 4 — IMPLEMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 4.1 Development Environment

| Component | Specification |
|---|---|
| Operating System | [e.g., Ubuntu 22.04 / Windows 11] |
| Programming Language | Python 3.x |
| Primary Libraries | [e.g., scikit-learn, PyTorch, TensorFlow, HuggingFace] |
| IDE | [e.g., VS Code, Jupyter Notebook] |
| Version Control | Git / GitHub |

## 4.2 [Module 1 — e.g., Data Pipeline]

[Describe the implementation of this module. Include relevant code snippets (not entire scripts — only key logic). Explain design decisions made during implementation.]

```python
# Example code snippet — show only non-obvious, key logic
def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text
```

*Figure 4.1: [Caption for any relevant diagram/screenshot]*

## 4.3 [Module 2 — e.g., Model Training Module]

[Same structure — describe, justify, show key code, show relevant figure.]

## 4.4 [Module 3 — e.g., Inference / Web Interface]

[Same structure.]

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         CHAPTER 5 — RESULTS AND DISCUSSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 5.1 Evaluation Metrics

[Define every metric used. Do not assume the examiner knows them.]

The following metrics are used to evaluate model performance:

- **Accuracy:** Proportion of correctly classified samples out of total samples.
- **Precision:** Of all samples predicted positive, the fraction that are truly positive.
- **Recall (Sensitivity):** Of all actual positives, the fraction correctly identified.
- **F1-Score:** Harmonic mean of Precision and Recall, balancing both measures.
- **AUC-ROC:** Area under the Receiver Operating Characteristic curve, measuring discrimination ability.

## 5.2 Baseline Comparison

[State your baseline clearly. A baseline is the simplest reasonable approach — e.g., majority class, TF-IDF + Logistic Regression.]

**Table 5.1:** Baseline vs. proposed model

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Majority Class (baseline) | [%] | [%] | [%] | [%] | [%] |
| TF-IDF + LR (baseline) | [%] | [%] | [%] | [%] | [%] |
| **[Your Model] (proposed)** | **[%]** | **[%]** | **[%]** | **[%]** | **[%]** |

## 5.3 Experimental Results

[Present all results systematically. Use tables for quantitative comparisons. Use figures for trends, confusion matrices, ROC curves.]

**Table 5.2:** Detailed classification report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Real | [value] | [value] | [value] | [N] |
| Fake | [value] | [value] | [value] | [N] |
| **Weighted Avg** | **[value]** | **[value]** | **[value]** | **[N]** |

**Figure 5.1:** Confusion matrix

**Figure 5.2:** ROC curve

**Figure 5.3:** Training and validation loss curves

## 5.4 Discussion

[Interpret the results. Do not just restate numbers — explain what they mean, why the model performs well or poorly on certain classes, and how the results relate to the objectives set in Chapter 1.]

[Paragraph 1: Overall performance analysis]

[Paragraph 2: Analysis of errors — what is the model getting wrong and why?]

[Paragraph 3: Comparison to literature — how do your results compare to the works in Table 2.1?]

## 5.5 Limitations

[State limitations honestly. This shows intellectual maturity and is expected at this level.]

1. [Limitation 1 — e.g., dataset restricted to English-language news]
2. [Limitation 2 — e.g., model not tested on out-of-domain data]
3. [Limitation 3 — e.g., inference latency may not meet real-time requirements]

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       CHAPTER 6 — CONCLUSION AND FUTURE WORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 6.1 Conclusions

This project presented [brief description of the work]. The following conclusions are drawn from the work:

1. [Conclusion 1 — specific finding, with evidence]
2. [Conclusion 2]
3. [Conclusion 3]
4. [Conclusion 4]

The proposed system achieved [key metric result], demonstrating [significance relative to baseline / prior work].

## 6.2 Future Work

The following directions are identified for extending this work:

1. **[Direction 1]:** [Specific, concrete description — e.g., "Extend the model to support multilingual inputs by incorporating multilingual BERT (mBERT) embeddings [N]."]
2. **[Direction 2]:** [e.g., "Deploy the system as a real-time browser extension to provide inline verification of news content."]
3. **[Direction 3]:** [e.g., "Investigate model explainability using SHAP or LIME to increase transparency for end-users."]

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                      REFERENCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

[1] A. Author and B. Author, "Article title in sentence case," *Abbrev. Journal*, vol. X, no. Y, pp. XXX–XXX, Mon. YYYY, doi: XX.XXXX/XXXXX.

[2] A. Author, "Paper title," in *Proc. Conf. Name (ABBREV.)*, City, Country, Mon. YYYY, pp. XXX–XXX.

[3] A. Author, *Book Title in Title Case*. City: Publisher, YYYY.

[4] A. Author, "Page title," *Site Name*. [Online]. Available: https://example.com. [Accessed: DD Mon. YYYY].

*(Continue numbering in order of first citation in the text)*

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                      APPENDICES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Appendix A — [Title, e.g., Full Dataset Statistics]

[Content]

## Appendix B — [Title, e.g., Full Model Configuration]

[Content]

## Appendix C — [Title, e.g., Sample Output Screenshots]

[Content]

---

## QUICK REFERENCE — COMMON MISTAKES TO AVOID

| Mistake | Correct Approach |
|---|---|
| Citing in alphabetical order | Cite in order of first appearance [1], [2], [3] |
| "I did this..." | "The system was implemented..." or "We implemented..." |
| Abbreviation without definition | "Natural Language Processing (NLP)" on first use |
| Figure with no in-text reference | Always write "as shown in Figure X.Y" before the figure |
| Conclusion that introduces new content | Conclusions only summarise; all content must appear earlier |
| Results without baseline | Always compare against at least one baseline |
| Claiming performance without test set | All reported numbers must come from the held-out test set |
| References listed alphabetically | References are in citation order, not alphabetical |
| Table caption below the table | Table captions go **above** the table |
| Figure caption above the figure | Figure captions go **below** the figure |

---

*Template based on: IEEE Editorial Style Manual, IEEE Author Guidelines, and University of Mumbai BE Project Report Guidelines.*
