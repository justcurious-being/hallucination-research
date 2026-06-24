# 🧠 Training Data Quality & Generative AI Hallucinations

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)](https://www.kaggle.com/)
[![Paper](https://img.shields.io/badge/Research-AIxSET%202025-orange)](docs/paper_reference.md)


> *"Role of Training Data Quality in Generative AI Hallucinations"*  
> Sahil Garg (Strayer University) 

This repository provides a **fully reproducible pipeline** to test the paper's central hypothesis:

> *Hallucination in generative AI is an emergent property of imperfect data ecosystems —  
> measurably influenced by training data accuracy, consistency, representativeness, noise, and entropy.*

---

## 📋 Table of Contents
- [Research Hypotheses](#research-hypotheses)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Expected Results](#expected-results)
- [Citation](#citation)

---

## Research Hypotheses

| # | Hypothesis | Metric |
|---|---|---|
| H1 | Higher contradiction density → higher HI | Pearson r |
| H2 | Higher noise ratio → higher HI | Pearson r |
| H3 | Lower representational balance → higher HI | Pearson r |
| H4 | Higher Shannon entropy → higher HI | Pearson r |
| H5 | RAG reduces HI but cannot override training-time deficits | Δ HI pre/post RAG |

---

## Datasets

### Primary — [LLM Hallucination Evaluation](https://www.kaggle.com/datasets/thedevastator/chatgpt-and-llm-hallucination-dataset)
Human-annotated hallucination labels across factual QA prompts. Used as the evaluation benchmark.

### Secondary — [Fake & Real News Corpus](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
Used to simulate noise-augmented and contradiction-enriched corpora by injecting low-credibility content.

```bash
# Install Kaggle CLI and place kaggle.json in ~/.kaggle/
pip install kaggle
kaggle datasets download -d thedevastator/chatgpt-and-llm-hallucination-dataset -p data/raw/
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw/
unzip "data/raw/*.zip" -d data/raw/
```

---

## Project Structure

```
hallucination-research/
├── README.md
├── requirements.txt
├── setup.py
├── run_pipeline.py                  # End-to-end runner
├── .env.example
│
├── src/
│   ├── data_quality/
│   │   ├── corpus_builder.py        # Builds 4 experimental corpora
│   │   ├── quality_metrics.py       # 5 data quality dimension scores
│   │   └── noise_injector.py        # Noise / contradiction injection
│   ├── evaluation/
│   │   ├── hallucination_index.py   # Composite HI computation
│   │   ├── factual_checker.py       # Factual accuracy scoring
│   │   └── semantic_coherence.py    # Semantic consistency
│   ├── models/
│   │   ├── llm_wrapper.py           # OpenAI / HuggingFace interface
│   │   └── rag_pipeline.py          # RAG sensitivity analysis
│   └── analysis/
│       ├── correlation.py           # Pearson/Spearman + regression
│       └── visualization.py         # Reproduces paper figures
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_corpus_construction.ipynb
│   ├── 03_quality_metrics.ipynb
│   ├── 04_hallucination_evaluation.ipynb
│   └── 05_results_visualization.ipynb
│
├── tests/
│   ├── test_corpus_builder.py
│   ├── test_quality_metrics.py
│   └── test_hallucination_index.py
│
└── docs/
    ├── paper_reference.md
    ├── methodology.md
    └── results_interpretation.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/hallucination-research.git
cd hallucination-research
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # Add your OpenAI key (optional)
```

---

## Quick Start

```bash
# Full pipeline (no LLM API key required — uses HuggingFace by default)
python run_pipeline.py --full

# Individual stages
python -m src.data_quality.corpus_builder   # Build corpora
python -m src.data_quality.quality_metrics  # Score quality dimensions
python -m src.evaluation.hallucination_index # Compute HI per corpus
python -m src.analysis.correlation          # Run stats + generate plots
```

---

## Pipeline Stages

| Stage | Script | Output |
|---|---|---|
| 1. Corpus Construction | `corpus_builder.py` | 4 CSV corpora in `data/processed/` |
| 2. Quality Scoring | `quality_metrics.py` | `results/quality_scores.csv` |
| 3. LLM Evaluation | `hallucination_index.py` | `results/hallucination_index.csv` |
| 4. Correlation Analysis | `correlation.py` | `results/correlation_table.csv` |
| 5. Visualization | `visualization.py` | `results/figures/` |

---

## Expected Results

```
Corpus Condition         HI (%)   Contradiction r   Noise r   p-value
─────────────────────────────────────────────────────────────────────
Baseline (high-quality)   7.2          —               —         —
Noise-Augmented          17.8        +0.77           +0.85    <0.01
Contradiction-Enriched   24.1        +0.85           +0.62    <0.001
Distributionally Imbal.  20.6        +0.58           +0.71    <0.05
```

---
```

---

## License
MIT — see [LICENSE](LICENSE)
