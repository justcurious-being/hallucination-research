"""
quality_metrics.py
==================
Computes the five data quality dimensions from Table 1 of the paper:

  1. Data Accuracy          – Verified Fact Ratio (%)
  2. Data Consistency       – Contradiction Density (per 10k tokens)
  3. Representativeness     – Domain Balance Score (0–1, Shannon evenness)
  4. Noise Level            – Low-Credibility Content Ratio (%)
  5. Data Entropy           – Shannon Entropy Index over token distribution

Usage:
    python -m src.data_quality.quality_metrics
    # Reads all corpora in data/processed/, writes results/quality_scores.csv
"""

import logging
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results")

# Simple keyword list for noise/low-credibility detection
_LOW_CRED_PATTERNS = re.compile(
    r"\b(unverified|disputed|allegedly|rumor|hoax|fake|fabricated|"
    r"no evidence|sources say|anonymous claim|the opposite is|entirely false)\b",
    re.IGNORECASE,
)

# Contradiction signal keywords
_CONTRA_PATTERNS = re.compile(
    r"\b(however|on the contrary|yet other|reverse|disputed|the opposite)\b",
    re.IGNORECASE,
)


# ── 1. Data Accuracy ─────────────────────────────────────────────────────────

def verified_fact_ratio(df: pd.DataFrame) -> float:
    """
    Percentage of rows where is_hallucination == 0 (verified/factual).
    Range: 0–100 (higher = better quality).
    """
    if "is_hallucination" not in df.columns:
        return np.nan
    ratio = (df["is_hallucination"] == 0).mean() * 100
    return round(ratio, 2)


# ── 2. Data Consistency ───────────────────────────────────────────────────────

def contradiction_density(df: pd.DataFrame, text_col: str = "response") -> float:
    """
    Number of contradiction-signal phrases per 10,000 tokens.
    Higher values → more internal inconsistency.
    """
    texts  = df[text_col].dropna().astype(str)
    tokens = sum(len(t.split()) for t in texts)
    hits   = sum(_CONTRA_PATTERNS.subn("", t)[1] for t in texts)

    if tokens == 0:
        return 0.0
    density = (hits / tokens) * 10_000
    return round(density, 4)


# ── 3. Representativeness ────────────────────────────────────────────────────

def domain_balance_score(df: pd.DataFrame, category_col: str = "category") -> float:
    """
    Shannon evenness index over domain/category distribution.
    Range: 0 (maximally imbalanced) – 1 (perfectly uniform).
    """
    if category_col not in df.columns:
        return np.nan

    counts = df[category_col].value_counts()
    total  = counts.sum()
    k      = len(counts)

    if k <= 1 or total == 0:
        return 0.0

    probs    = counts / total
    entropy  = -sum(p * math.log(p) for p in probs if p > 0)
    max_ent  = math.log(k)
    evenness = entropy / max_ent if max_ent > 0 else 0.0
    return round(evenness, 4)


# ── 4. Noise Level ────────────────────────────────────────────────────────────

def noise_ratio(df: pd.DataFrame, text_col: str = "response") -> float:
    """
    Percentage of rows containing low-credibility signal phrases.
    Higher → noisier corpus.
    """
    texts = df[text_col].dropna().astype(str)
    noisy = texts.apply(lambda t: bool(_LOW_CRED_PATTERNS.search(t)))
    return round(noisy.mean() * 100, 2)


# ── 5. Data Entropy ───────────────────────────────────────────────────────────

def shannon_entropy_index(df: pd.DataFrame, text_col: str = "response") -> float:
    """
    Shannon entropy computed over the unigram token distribution of the corpus.
    Higher entropy → more dispersed / less predictable token distribution.
    """
    texts  = df[text_col].dropna().astype(str)
    tokens = []
    for t in texts:
        tokens.extend(t.lower().split())

    if not tokens:
        return 0.0

    counts  = Counter(tokens)
    total   = sum(counts.values())
    probs   = [v / total for v in counts.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return round(entropy, 4)


# ── Composite scorer ─────────────────────────────────────────────────────────

def compute_all_metrics(df: pd.DataFrame, corpus_name: str) -> dict:
    """
    Compute all five quality dimensions for a single corpus DataFrame.
    Returns a flat dict ready for a results table row.
    """
    logger.info(f"  Computing metrics for corpus: {corpus_name}")
    return {
        "corpus":                corpus_name,
        "n_rows":                len(df),
        "verified_fact_ratio":   verified_fact_ratio(df),
        "contradiction_density": contradiction_density(df),
        "domain_balance_score":  domain_balance_score(df),
        "noise_ratio":           noise_ratio(df),
        "shannon_entropy":       shannon_entropy_index(df),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus_files = sorted(PROCESSED_DIR.glob("*.csv"))
    if not corpus_files:
        logger.error(f"No corpora found in {PROCESSED_DIR}. Run corpus_builder.py first.")
        return

    rows = []
    for fp in tqdm(corpus_files, desc="Scoring corpora"):
        df   = pd.read_csv(fp)
        name = fp.stem
        rows.append(compute_all_metrics(df, name))

    results = pd.DataFrame(rows)
    out     = RESULTS_DIR / "quality_scores.csv"
    results.to_csv(out, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("DATA QUALITY SCORES")
    logger.info("=" * 70)
    logger.info(results.to_string(index=False))
    logger.info(f"\nSaved to: {out}")
    return results


if __name__ == "__main__":
    run()
