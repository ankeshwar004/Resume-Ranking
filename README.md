# Resume–Job Description Ranking System

> An end-to-end Learning-to-Rank system that ranks resumes against job descriptions using classical IR techniques, Sentence Transformers, Hard Negative Mining, and Two-Stage Retrieval. The project includes honest documentation of negative experimental results — a key part of the research process.

---

## Overview

Recruiters often receive hundreds of resumes for a single job posting. Manually screening candidates is time-consuming and difficult to scale.

This project formulates resume screening as a **Learning-to-Rank** problem and builds a complete ranking pipeline capable of matching and ranking resumes according to their relevance to a job description.

The project explores four approaches in increasing complexity:

- TF-IDF + XGBoost ranking baseline
- Bi-Encoder semantic retrieval (CoSENTLoss fine-tuning)
- Hard Negative Mining
- Cross-Encoder relevance scoring with Two-Stage Retrieval

---

## Dataset

**Source:** Hugging Face — `cnamuangtoun/resume-job-description-fit`

| Label | Meaning        |
| ----- | -------------- |
| 0     | No Fit         |
| 1     | Potential Fit  |
| 2     | Good Fit       |

| Split      | Samples |
| ---------- | ------- |
| Train      | 5,073   |
| Validation | 1,167   |
| Test       | 1,759   |

**Leakage prevention:** `GroupShuffleSplit` was used with `job_description_text` as the group key, ensuring the same JD never appears in both training and validation/test sets. This is critical for realistic generalization estimates.

---

## Project Pipeline

```
Raw Resume–JD Dataset
         │
         ▼
   EDA + Data Cleaning
         │
    ┌────┴────┐
    ▼         ▼
Classical   Transformer
Baseline    Pipeline
    │         │
    ▼         ▼
Feature Eng  Fine-Tuned
(TF-IDF,    Bi-Encoder
 Skills,         │
 Experience)     ▼
    │       Hard Negative
    ▼         Mining
XGBRanker        │
                 ▼
           Cross-Encoder
           + Two-Stage
            Retrieval
                 │
                 ▼
          Final Evaluation
          (NDCG, MAP, MRR,
           Top-K per JD)
```

---

## Models and Experiments

### 1. TF-IDF + XGBRanker Baseline

A traditional information retrieval baseline combining:

- **TF-IDF cosine similarity** — computed between vectorized resume and JD
- **Skill overlap features** — matched skills from a curated vocabulary (normalized via spaCy + tech alias mapping)
- **Experience matching features** — gap and ratio between extracted experience values
- **Truncated SVD** (30 components) on the TF-IDF interaction matrix

The ranking model used XGBoost's `rank:ndcg` objective — a proper ranking objective, not a classifier. Evaluated using 5-fold `GroupKFold` cross-validation.

---

### 2. Bi-Encoder (Fine-Tuned)

**Base model:** `sentence-transformers/all-MiniLM-L6-v2`

Fine-tuned using **CoSENTLoss**, which directly optimizes ranking order rather than predicting a scalar similarity score. This is the key distinction from CosineSimilarityLoss or ContrastiveLoss — CoSENTLoss treats the task as ranking, which matches the evaluation objective.

Training details:
- Labels mapped to continuous scores: `{0: 0.0, 1: 0.5, 2: 1.0}`
- 4 epochs with early stopping on a composite metric: `0.6 × NDCG + 0.3 × MAP + 0.1 × MRR`
- Best checkpoint saved per epoch

Advantages:
- Fast inference — encodes resumes and JDs independently
- Scalable to large resume pools
- Dense semantic representations suitable for approximate nearest-neighbor search

---

### 3. Hard Negative Mining (Negative Result)

**Motivation:** Force the model to learn fine-grained distinctions between truly relevant and superficially similar candidates by augmenting training data with semantically close but incorrect resumes.

**Method:** For each positive (label=2) JD–resume pair, the fine-tuned Bi-Encoder retrieved the top-10 semantically similar resumes. Resumes not labeled as relevant for that JD were added as hard negatives (label=0), capped at 3 per positive pair.

**Outcome:** Hard negative mining did not improve performance. The augmented model underperformed the baseline Bi-Encoder.

**Why it failed on this dataset:**

Hard negative mining assumes the training data contains many "easy negatives" — obviously irrelevant documents the model learns to dismiss trivially. This assumption is violated here:

1. The dataset contains expert HR-labeled negatives per JD–resume pair. There is no easy-negative problem to solve.
2. Mining from other JDs introduces label noise. A resume labeled "Good Fit" for one JD can become a falsely labeled "No Fit" for another, since many resumes are cross-domain relevant.
3. With 2–3 resumes per JD on average, there is limited pairwise ranking signal to benefit from harder examples.

This is documented as a deliberate negative result rather than hidden.

---

### 4. Cross-Encoder

**Base model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

Unlike the Bi-Encoder, the Cross-Encoder jointly encodes a (JD, resume) pair and produces a single relevance score. This enables cross-attention between both texts and theoretically richer relevance estimation.

Training used regression targets `{0: 0.0, 1: 0.5, 2: 1.0}` with MSE loss.

**Note on fair comparison:** The Bi-Encoder was trained with CoSENTLoss (ranking-aware), while the Cross-Encoder was trained with MSE regression loss (score prediction). This is an important distinction — if Cross-Encoder were retrained with a pairwise or listwise ranking loss, the comparison would be more architectural and less a function of loss choice.

---

### 5. Two-Stage Retrieval

A production-realistic retrieval architecture:

```
Job Description
       │
       ▼
 Bi-Encoder Retrieval
 (Top-K candidates from
  full resume pool)
       │
       ▼
 Cross-Encoder Reranking
 (Pairwise scoring on
  retrieved candidates)
       │
       ▼
 Final Ranked Resumes
```

**Stage 1:** Bi-Encoder performs global semantic search across all resumes using `util.semantic_search` — a realistic simulation of retrieving from a large candidate database.

**Stage 2:** Cross-Encoder scores each retrieved (JD, resume) pair and reranks.

**Note on evaluation setup:** Two-stage metrics are computed in a global retrieval setting — the Bi-Encoder retrieves from the full test resume pool, and only labeled candidates are scored. This is not directly comparable to the pairwise evaluation used for the standalone models.

---

## Preprocessing

A custom spaCy-based preprocessing pipeline was built:

- **Text cleaning:** URL removal, email/phone stripping, lowercasing
- **Tech alias normalization:** Common abbreviations mapped to canonical forms (`js → javascript`, `k8s → kubernetes`, `ml → machine learning`, etc.)
- **PhraseMatcher:** Multi-word skill phrases detected and joined as single tokens (`machine_learning`, `natural_language_processing`)
- **POS filtering:** Only NOUN, VERB, ADJ, PROPN tokens retained
- **Lemmatization:** Reduces vocabulary to canonical word forms
- **External skill vocabulary:** Augmented with the `meerawks/it-skills-from-jobs` Kaggle dataset

---

## Evaluation Metrics

All metrics were computed **per Job Description group** and then averaged — not globally across all pairs. This is the correct evaluation protocol for ranking systems where each JD forms an independent ranking query.

| Metric            | Description                                              |
| ----------------- | -------------------------------------------------------- |
| NDCG@10           | Normalized Discounted Cumulative Gain — position-weighted ranking quality |
| MAP               | Mean Average Precision over all relevant resumes         |
| MRR               | Mean Reciprocal Rank — rank position of first relevant resume |
| Top-3 Accuracy    | Whether a Good Fit (label=2) appears in the top 3 results |
| Spearman ρ        | Rank correlation between predicted scores and true labels |

---

## Results

### Pairwise Evaluation (Test Set)

All models below are evaluated by scoring all labeled (JD, resume) pairs in the test set, grouped by JD.

| Model                          | NDCG@10 | MAP     | MRR     | Top-3 Acc |
| ------------------------------ | ------- | ------- | ------- | --------- |
| TF-IDF + XGBRanker             | 0.6463  | 0.7505  | 0.8462  | 0.9642    |
| Bi-Encoder (CoSENTLoss)        | 0.6411  | 0.7432  | 0.7895  | 0.9642    |
| Bi-Encoder + Hard Negatives    | *below baseline* | — | — | — |
| Cross-Encoder (MSE loss)       | *below baseline* | — | — | — |

> Hard Negative Bi-Encoder and Cross-Encoder both underperformed the baseline Bi-Encoder. See [Key Findings](#key-findings) for root cause analysis.

---

### Two-Stage Retrieval — Global Retrieval Setting (Test Set)

Bi-Encoder performs semantic search across all test resumes. Only retrieved resumes with known labels for that JD are scored. This simulates real-world retrieval from a large pool.

| Stage                         | NDCG@10 | MAP    | MRR    |
| ----------------------------- | ------- | ------ | ------ |
| Bi-Encoder (Stage 1, top-20)  | 0.8878  | 0.8961 | 0.9380 |
| + Cross-Encoder Reranking     | *no significant gain* | — | — |

> The higher NDCG in this setting vs. pairwise evaluation reflects the different evaluation protocol (global retrieval vs. scoring all labeled pairs), not a change in model capability. Cross-Encoder reranking did not meaningfully improve over the Bi-Encoder retrieval scores.

---

## Key Findings

**1. Bi-Encoder (no hard negatives) is the best model.**

The fine-tuned Bi-Encoder with CoSENTLoss achieved the best ranking performance among transformer-based models. Neither Hard Negative augmentation nor Cross-Encoder reranking improved upon it on this dataset.

**2. Hard Negative Mining hurt performance.**

This is a documented negative result. The dataset's expert HR labels already serve as meaningful hard negatives. Cross-JD mining introduced label noise that degraded an already well-trained model. This is a known failure mode when the "easy negative" assumption behind hard negative mining does not hold.

**3. Cross-Encoder did not outperform Bi-Encoder.**

Two likely causes:
- **Loss function mismatch:** Bi-Encoder used CoSENTLoss (ranking-aware); Cross-Encoder used MSE regression (score prediction). The comparison is partially a loss-function comparison.
- **Dataset size:** Cross-Encoders typically require more data and more epochs to outperform Bi-Encoders. With 2–3 resumes per JD, pairwise ranking signal was limited.

**4. More complex does not mean better.**

The TF-IDF + XGBRanker baseline was competitive with the neural models on NDCG and Top-3 Accuracy. On a small, noisy, domain-specific dataset, classical feature engineering (skill overlap, experience matching) captured most of the signal.

---

## Final Model

**Fine-Tuned Bi-Encoder (CoSENTLoss, no hard negatives)**

Selected for:
- Best ranking performance among transformer-based models
- Fast inference — resumes encoded independently and reusable
- Scalable to large resume databases (no pairwise scoring required at retrieval time)
- Suitable for integration with FAISS or any vector store

---

## Project Structure

```
resume-ranking/
│
├── resume_ranking.ipynb       # Complete end-to-end notebook
│
├── data/                      # Raw and preprocessed data
│
├── models/
│   ├── bi_encoder_resume_model/      # Best Bi-Encoder checkpoint
│   ├── bi_encoder_hard_neg/          # Hard Negative Bi-Encoder (for comparison)
│   └── cross_encoder_resume_model/   # Cross-Encoder checkpoint
│
├── requirements.txt
└── README.md
```

---

## Skills and Concepts Demonstrated

- Information Retrieval and Learning-to-Rank
- Sentence Transformers — Bi-Encoder and Cross-Encoder architectures
- CoSENTLoss and ranking-aware training objectives
- Hard Negative Mining — implementation and failure mode analysis
- Two-Stage Retrieval pipelines (retrieve → rerank)
- Per-group ranking evaluation: NDCG, MAP, MRR, Top-K
- GroupShuffleSplit for leakage-free data splitting
- spaCy NLP pipeline: POS filtering, lemmatization, PhraseMatcher
- XGBoost LambdaRank / `rank:ndcg` objective
- Honest negative result documentation

---

## Future Work

- **Zero-shot evaluation** — baseline the pretrained model before any fine-tuning to quantify fine-tuning gains
- **Cross-Encoder with ranking loss** — retrain using listwise or pairwise loss for a fair architectural comparison
- **In-batch negatives** — cleaner alternative to cross-JD hard negatives; other pairs in the same batch serve as negatives with no label noise
- **Token truncation analysis** — measure how much information is lost by max_length=384 on long resumes
- **FAISS / vector database integration** — production-scale ANN retrieval
- **FastAPI serving** — REST endpoint for real-time ranking
- **Larger domain-specific datasets** — cross-encoder typically needs more data to outperform bi-encoder

---

## Conclusion

This project builds a complete Resume Ranking System using modern NLP ranking techniques. By combining semantic retrieval, transformer fine-tuning, hard negative mining, and five ranking evaluation metrics, the system matches resumes to job descriptions in a way that mirrors production retrieval architectures.

The most important outcome is not the best metric — it is the systematic comparison. Hard Negative Mining and Cross-Encoder reranking are standard techniques that did not improve performance on this dataset, and the reasons are documented explicitly. The final Bi-Encoder model achieves strong retrieval performance while remaining efficient enough for large-scale deployment.
