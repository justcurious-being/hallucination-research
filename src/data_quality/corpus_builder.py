"""
corpus_builder.py
=================
Constructs the four experimental training corpora described in the paper:

  1. baseline            – high-quality curated data (no injection)
  2. noise_augmented     – low-credibility content injected at NOISE_RATE
  3. contradiction_enriched – contradictory claims inserted at CONTRADICTION_RATE
  4. imbalanced          – domain distribution skewed to IMBALANCE_SKEW

Input datasets (Kaggle):
  - data/raw/hallucination_dataset.csv   (LLM hallucination benchmark)
  - data/raw/Fake.csv + data/raw/True.csv (Fake & Real News corpus)

Outputs:
  data/processed/{baseline,noise_augmented,contradiction_enriched,imbalanced}.csv
"""

import os
import random
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR        = Path("data/raw")
PROCESSED_DIR  = Path("data/processed")
NOISE_RATE     = float(os.getenv("NOISE_INJECTION_RATE", 0.20))
CONTRA_RATE    = float(os.getenv("CONTRADICTION_RATE",   0.15))
IMBAL_SKEW     = float(os.getenv("IMBALANCE_SKEW",       0.70))
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_hallucination_dataset() -> pd.DataFrame:
    """
    Load the Kaggle LLM Hallucination dataset.
    Expected columns: prompt, response, is_hallucination (0/1), category
    Falls back to a synthetic stub if the file is not yet downloaded.
    """
    path = RAW_DIR / "hallucination_dataset.csv"
    if path.exists():
        df = pd.read_csv(path)
        logger.info(f"Loaded hallucination dataset: {len(df):,} rows")
    else:
        logger.warning("Hallucination dataset not found — generating synthetic stub.")
        df = _synthetic_hallucination_stub(n=500)
    return df


def load_fake_news_dataset() -> pd.DataFrame:
    """
    Load Fake & Real News from Kaggle (Fake.csv + True.csv).
    Returns a combined frame with columns: text, label ('fake'|'real'), subject
    """
    fake_path = RAW_DIR / "Fake.csv"
    true_path = RAW_DIR / "True.csv"

    if fake_path.exists() and true_path.exists():
        fake = pd.read_csv(fake_path).assign(label="fake")
        real = pd.read_csv(true_path).assign(label="real")
        df   = pd.concat([fake, real], ignore_index=True)
        logger.info(f"Loaded news dataset: {len(df):,} rows (fake={len(fake):,}, real={len(real):,})")
    else:
        logger.warning("News dataset not found — generating synthetic stub.")
        df = _synthetic_news_stub(n=1000)
    return df


# ── Corpus builders ───────────────────────────────────────────────────────────

def build_baseline(halluc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Corpus 1 — Baseline: only verified/real content, no injection.
    """
    baseline = halluc_df[halluc_df["is_hallucination"] == 0].copy()
    baseline["corpus"] = "baseline"
    baseline["injected"] = False
    logger.info(f"Baseline corpus: {len(baseline):,} rows")
    return baseline


def build_noise_augmented(
    halluc_df: pd.DataFrame,
    news_df: pd.DataFrame,
    noise_rate: float = NOISE_RATE,
) -> pd.DataFrame:
    """
    Corpus 2 — Noise-augmented: inject fake/low-credibility content at `noise_rate`.

    Strategy: replace `noise_rate` fraction of real responses with text drawn
    from the Fake news corpus (simulates web-scraped misinformation).
    """
    df = halluc_df.copy()
    n_inject = int(len(df) * noise_rate)
    inject_idx = df.sample(n=n_inject, random_state=RANDOM_SEED).index

    fake_texts = news_df[news_df["label"] == "fake"]["text"].dropna().tolist()
    inject_texts = random.choices(fake_texts, k=n_inject)

    df.loc[inject_idx, "response"] = inject_texts
    df.loc[inject_idx, "is_hallucination"] = 1
    df["corpus"]   = "noise_augmented"
    df["injected"] = df.index.isin(inject_idx)
    logger.info(f"Noise-augmented corpus: {len(df):,} rows, {n_inject} injected ({noise_rate:.0%})")
    return df


def build_contradiction_enriched(
    halluc_df: pd.DataFrame,
    contra_rate: float = CONTRA_RATE,
) -> pd.DataFrame:
    """
    Corpus 3 — Contradiction-enriched: insert semantically opposite claims.

    Strategy: For `contra_rate` fraction of rows, append a negated or contrary
    statement to the response, simulating internally inconsistent training data.
    """
    df = halluc_df.copy()
    n_contra = int(len(df) * contra_rate)
    contra_idx = df.sample(n=n_contra, random_state=RANDOM_SEED).index

    def _add_contradiction(text: str) -> str:
        """Append a generic negation to simulate contradictory training signal."""
        negations = [
            " However, the opposite is also widely reported to be true.",
            " On the contrary, multiple sources confirm this is entirely false.",
            " Yet other evidence suggests the complete reverse is accurate.",
            " This claim has been disputed and the reverse has been documented.",
        ]
        return str(text) + random.choice(negations)

    df.loc[contra_idx, "response"] = df.loc[contra_idx, "response"].apply(_add_contradiction)
    df.loc[contra_idx, "is_hallucination"] = 1
    df["corpus"]   = "contradiction_enriched"
    df["injected"] = df.index.isin(contra_idx)
    logger.info(f"Contradiction-enriched corpus: {len(df):,} rows, {n_contra} contradictions ({contra_rate:.0%})")
    return df


def build_imbalanced(
    halluc_df: pd.DataFrame,
    skew: float = IMBAL_SKEW,
) -> pd.DataFrame:
    """
    Corpus 4 — Distributionally imbalanced: over-represent one domain.

    Strategy: upsample one category to `skew` fraction of the corpus,
    leaving remaining categories to share (1 - skew).
    """
    df = halluc_df.copy()

    if "category" not in df.columns:
        df["category"] = np.random.choice(
            ["science", "history", "medicine", "finance", "geography"],
            size=len(df)
        )

    dominant = df["category"].value_counts().idxmax()
    dominant_df  = df[df["category"] == dominant]
    remaining_df = df[df["category"] != dominant]

    n_total    = len(df)
    n_dominant = int(n_total * skew)
    n_remain   = n_total - n_dominant

    dominant_sample  = dominant_df.sample(n=n_dominant, replace=True,  random_state=RANDOM_SEED)
    remaining_sample = remaining_df.sample(n=n_remain,  replace=True,  random_state=RANDOM_SEED)

    imbalanced = pd.concat([dominant_sample, remaining_sample]).sample(frac=1, random_state=RANDOM_SEED)
    imbalanced["corpus"]   = "imbalanced"
    imbalanced["injected"] = False
    logger.info(
        f"Imbalanced corpus: {len(imbalanced):,} rows | "
        f"dominant='{dominant}' at {skew:.0%}"
    )
    return imbalanced


# ── Synthetic stubs (fallback when Kaggle files absent) ──────────────────────

def _synthetic_hallucination_stub(n: int = 500) -> pd.DataFrame:
    """Generate a minimal synthetic hallucination dataset for offline testing."""
    categories = ["science", "history", "medicine", "finance", "geography"]
    rows = []
    for i in range(n):
        is_hall = int(i % 5 == 0)
        rows.append({
            "prompt":           f"Question about {categories[i % 5]} topic {i}",
            "response":         f"Answer {i}: This is {'incorrect fabricated' if is_hall else 'verified factual'} information.",
            "is_hallucination": is_hall,
            "category":         categories[i % 5],
        })
    return pd.DataFrame(rows)


def _synthetic_news_stub(n: int = 1000) -> pd.DataFrame:
    """Generate a minimal synthetic news dataset for offline testing."""
    rows = []
    for i in range(n):
        label = "fake" if i % 2 == 0 else "real"
        rows.append({
            "text":    f"{'Unverified claim' if label == 'fake' else 'Verified report'} number {i}. "
                       f"This story {'lacks credible sources' if label == 'fake' else 'cites authoritative sources'}.",
            "label":   label,
            "subject": random.choice(["politics", "world", "sports", "tech"]),
        })
    return pd.DataFrame(rows)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_corpus(df: pd.DataFrame, name: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved: {out}  ({len(df):,} rows)")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def build_all_corpora():
    logger.info("=" * 60)
    logger.info("Building all four experimental corpora")
    logger.info("=" * 60)

    halluc_df = load_hallucination_dataset()
    news_df   = load_fake_news_dataset()

    corpora = {
        "baseline":               build_baseline(halluc_df),
        "noise_augmented":        build_noise_augmented(halluc_df, news_df),
        "contradiction_enriched": build_contradiction_enriched(halluc_df),
        "imbalanced":             build_imbalanced(halluc_df),
    }

    for name, df in corpora.items():
        save_corpus(df, name)

    logger.info("All corpora built successfully.")
    return corpora


if __name__ == "__main__":
    build_all_corpora()
