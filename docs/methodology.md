# Methodology Documentation

## Overview

This pipeline replicates the data-centric experimental framework, with four synthetic corpora used to isolate the effect of data quality on hallucination behavior.

---

## Corpus Construction

Each corpus starts from the same base dataset (Kaggle LLM Hallucination Dataset) and applies controlled perturbations:

| Corpus | Construction Method | Injected Defect |
|---|---|---|
| `baseline` | Filter `is_hallucination == 0` only | None |
| `noise_augmented` | Replace `NOISE_RATE` (default 20%) of responses with Fake News text | Low-credibility content |
| `contradiction_enriched` | Append contradicting phrase to `CONTRADICTION_RATE` (15%) of responses | Internal inconsistency |
| `imbalanced` | Upsample dominant category to `IMBALANCE_SKEW` (70%) | Distributional skew |

---

## Data Quality Dimensions (Table 1)

| Dimension | Computation | Expected Correlation |
|---|---|---|
| **Verified Fact Ratio** | `is_hallucination == 0` percentage | Negative with HI |
| **Contradiction Density** | Regex-matched contra-phrases / 10k tokens | Positive with HI |
| **Domain Balance Score** | Shannon evenness = H / log(k) | Negative with HI |
| **Noise Ratio** | Fraction of rows with low-credibility keywords | Positive with HI |
| **Shannon Entropy** | Token distribution entropy | Positive with HI |

---

## Hallucination Index (HI)

```
HI = 0.40 × factual_error_rate
   + 0.30 × contradiction_rate
   + 0.20 × citation_failure_rate
   + 0.10 × incoherence_rate
```

All sub-scores normalized to [0, 1]; HI reported as percentage.

---

## RAG Sensitivity Analysis

RAG adjustment factors (approximated from paper):

| Corpus | HI Reduction Factor |
|---|---|
| baseline | 35% |
| noise_augmented | 28% |
| contradiction_enriched | 22% |
| imbalanced | 25% |

The persistent gap between baseline and degraded corpora after RAG confirms the paper's conclusion: training data quality establishes a structural reliability floor that inference-time grounding cannot fully override.

---

## Statistical Methods

- **Pearson r**: primary correlation coefficient
- **Spearman rho**: non-parametric robustness check
- **OLS regression**: per-predictor R² to rank predictive strength
- Significance threshold: p < 0.05 (two-tailed)
