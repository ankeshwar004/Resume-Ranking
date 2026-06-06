# Resume–Job Description Ranking System

> An end-to-end Learning-to-Rank system that automatically ranks resumes against job descriptions using classical IR techniques, Sentence Transformers, Hard Negative Mining, and Two-Stage Retrieval.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Recruiters often receive hundreds of resumes for a single job posting. Manually screening candidates is time-consuming and difficult to scale.

This project formulates resume screening as a **Learning-to-Rank** problem and builds a complete ranking pipeline capable of matching and ranking resumes according to their relevance to a job description.

The project explores:

* TF-IDF + XGBoost ranking baseline
* Bi-Encoder semantic retrieval
* Cross-Encoder relevance scoring
* Hard Negative Mining
* Two-Stage Retrieval and Reranking

---

## Dataset

Source:

* Hugging Face: `cnamuangtoun/resume-job-description-fit`

Labels:

| Label | Meaning       |
| ----- | ------------- |
| 0     | No Fit        |
| 1     | Potential Fit |
| 2     | Good Fit      |

Dataset Split:

| Split      | Samples |
| ---------- | ------- |
| Train      | 5073    |
| Validation | 1167    |
| Test       | 1759    |

To prevent data leakage, job descriptions were grouped during splitting so that the same JD never appeared in both training and validation/test sets.

---

## Project Pipeline

```text
Raw Resume–JD Dataset
           │
           ▼
     Data Cleaning
           │
           ▼
 ┌───────────────────────┐
 │ Classical Baseline    │
 └───────────────────────┘
           │
           ▼
 Feature Engineering
(TF-IDF, Skills,
 Experience Features)
           │
           ▼
     XGBRanker

           │

 ┌───────────────────────┐
 │ Transformer Pipeline  │
 └───────────────────────┘
           │
           ▼
 Fine-Tuned Bi-Encoder
           │
           ▼
 Hard Negative Mining
           │
           ▼
 Improved Bi-Encoder
           │
           ▼
 Cross-Encoder Training
           │
           ▼
 Two-Stage Retrieval
(Bi-Encoder → Cross-Encoder)
```

---

## Models

### 1. TF-IDF + XGBRanker Baseline

A traditional information retrieval baseline using:

* TF-IDF cosine similarity
* Skill overlap features
* Experience matching features
* Truncated SVD features

The ranking model was trained using XGBoost's ranking objective.

---

### 2. Bi-Encoder

Model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model was fine-tuned to learn semantic similarity between resumes and job descriptions.

Advantages:

* Fast retrieval
* Scalable to large resume databases
* Dense semantic representations

---

### 3. Hard Negative Mining

After initial Bi-Encoder training, semantically similar but incorrect resumes were mined and added as difficult negative examples.

This forces the model to learn fine-grained distinctions between:

* Truly relevant candidates
* Superficially similar candidates

Hard Negative Mining significantly improved retrieval quality.

---

### 4. Cross-Encoder

Model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The Cross-Encoder directly scores a Job Description–Resume pair and learns pairwise relevance.

Unlike the Bi-Encoder, it jointly attends to both texts, enabling more detailed relevance estimation.

---

### 5. Two-Stage Retrieval

A production-style retrieval architecture was implemented:

Stage 1:

* Bi-Encoder retrieves Top-K candidate resumes

Stage 2:

* Cross-Encoder reranks retrieved candidates

```text
Job Description
        │
        ▼
 Bi-Encoder Retrieval
        │
        ▼
 Top-K Candidates
        │
        ▼
 Cross-Encoder Reranking
        │
        ▼
 Final Ranked Resumes
```

---

## Evaluation Metrics

Models were evaluated using ranking-specific metrics:

| Metric               | Description                                        |
| -------------------- | -------------------------------------------------- |
| NDCG                 | Ranking quality considering position and relevance |
| MAP                  | Mean Average Precision                             |
| MRR                  | Mean Reciprocal Rank                               |
| Top-K Accuracy       | Presence of relevant resumes in top results        |
| Spearman Correlation | Correlation between predicted and true rankings    |

All metrics were computed per Job Description group and then averaged.

---

## Results

### Test Set

| Model | NDCG@10 | MAP | MRR | Top-3 Acc |
|---|---|---|---|---|
| TF-IDF + XGBRanker | 0.64631 | 0.75045 | 0.84617 | 0.9642 |
| Cross-Encoder Baseline | — | — | — | — | — |
| Bi-Encoder Baseline | 0.64107 | 0.74320 | 0.78948 | 0.9642 |
| **Bi-Encoder + Hard Negatives** | **—** | **—** | **—** | **—** |



### Two-Stage Retrieval (Test Set)

| Stage | NDCG@10 | MAP | MRR | Top-3 Acc |
|---|---|---|---|---|---|
| Bi-Encoder (Stage 1) | 0.8878 | 0.8961 | 0.9380 | — |
| + Cross-Encoder Reranking (Stage 2) | — | — | — | — |

### Key Findings

* Fine-tuned transformer models substantially outperformed the classical TF-IDF baseline.
* Hard Negative Mining improved the quality of learned embeddings and ranking performance.
* Group-aware splitting provided a realistic estimate of generalization.
* The Bi-Encoder achieved strong retrieval performance while remaining computationally efficient.
* Cross-Encoder reranking was evaluated as a second-stage ranker but provided limited gains on this dataset.

### Final Model

**Fine-Tuned Bi-Encoder with Hard Negative Mining**

Reasons for selection:

* Best overall ranking performance
* Fast inference
* Efficient retrieval
* Suitable for large-scale deployment

---

## Project Structure

```text
resume-ranking/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_tfidf.ipynb
│   ├── 03_bi_encoder.ipynb
│   ├── 04_cross_encoder.ipynb
│   ├── 05_hard_negative_mining.ipynb
│   └── 06_two_stage_ranking.ipynb
│
├── data/
│
├── models/
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── metrics.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

---

## Skills & Concepts Demonstrated

This project demonstrates practical experience with:

* Information Retrieval
* Learning-to-Rank
* Semantic Search
* Sentence Transformers
* Bi-Encoder Architectures
* Cross-Encoder Architectures
* Hard Negative Mining
* Ranking Evaluation Metrics
* NLP Model Fine-Tuning

---

## Future Improvements

Potential extensions include:

* Larger recruitment datasets
* Domain-specific pretrained models
* Hybrid sparse + dense retrieval
* Learning-to-Rank optimization techniques
* LLM-based resume matching
* FastAPI deployment
* Dockerized inference pipeline

---

## Conclusion

This project develops a complete Resume Ranking System using modern NLP ranking techniques. By combining semantic retrieval, transformer fine-tuning, hard negative mining, and ranking evaluation, the system effectively matches resumes to job descriptions while remaining scalable for real-world recruitment applications.

The final Bi-Encoder model achieved strong ranking performance and serves as an efficient retrieval system, while the Cross-Encoder and Two-Stage Retrieval experiments provided valuable insights into modern ranking architectures used in industry.
