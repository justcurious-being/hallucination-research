"""
tests/test_quality_metrics.py
Tests for all five data quality dimension scorers.
Run: pytest tests/ -v
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_quality.quality_metrics import (
    contradiction_density,
    domain_balance_score,
    noise_ratio,
    shannon_entropy_index,
    verified_fact_ratio,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_df():
    """Perfectly clean corpus — all verified, no noise, balanced."""
    return pd.DataFrame({
        "response":         ["This is a verified factual statement."] * 20,
        "is_hallucination": [0] * 20,
        "category":         ["science", "history", "medicine", "finance", "geography"] * 4,
    })


@pytest.fixture
def noisy_df():
    """Corpus with injected low-credibility signals."""
    return pd.DataFrame({
        "response": [
            "This is unverified information with no evidence.",
            "A verified fact.",
            "Rumor suggests this may be a hoax.",
            "Another accurate statement.",
            "This claim is disputed and allegedly fake.",
        ],
        "is_hallucination": [1, 0, 1, 0, 1],
        "category": ["science", "history", "medicine", "finance", "geography"],
    })


@pytest.fixture
def contradiction_df():
    """Corpus with contradiction signals."""
    return pd.DataFrame({
        "response": [
            "The result is positive. However, on the contrary, it is also negative.",
            "A simple clean statement.",
            "Evidence shows X. Yet other sources confirm the reverse is true.",
            "Plain factual response.",
            "The answer is yes. On the contrary, sources say otherwise.",
        ],
        "is_hallucination": [1, 0, 1, 0, 1],
        "category": ["science"] * 5,
    })


@pytest.fixture
def imbalanced_df():
    """Corpus heavily skewed toward one domain."""
    return pd.DataFrame({
        "response":         ["Text"] * 100,
        "is_hallucination": [0] * 100,
        "category":         ["science"] * 80 + ["history"] * 10 + ["medicine"] * 10,
    })


# ── Tests: verified_fact_ratio ────────────────────────────────────────────────

class TestVerifiedFactRatio:
    def test_all_clean(self, clean_df):
        assert verified_fact_ratio(clean_df) == 100.0

    def test_all_hallucinations(self):
        df = pd.DataFrame({"is_hallucination": [1, 1, 1, 1]})
        assert verified_fact_ratio(df) == 0.0

    def test_mixed(self, noisy_df):
        ratio = verified_fact_ratio(noisy_df)
        assert 0 < ratio < 100

    def test_missing_column(self, clean_df):
        df = clean_df.drop(columns=["is_hallucination"])
        result = verified_fact_ratio(df)
        assert math.isnan(result)

    def test_exact_value(self):
        df = pd.DataFrame({"is_hallucination": [0, 0, 0, 1]})  # 75% clean
        assert verified_fact_ratio(df) == 75.0


# ── Tests: contradiction_density ─────────────────────────────────────────────

class TestContradictionDensity:
    def test_clean_corpus_low_density(self, clean_df):
        density = contradiction_density(clean_df)
        assert density == 0.0

    def test_contradiction_corpus_higher(self, contradiction_df, clean_df):
        cd_contra = contradiction_density(contradiction_df)
        cd_clean  = contradiction_density(clean_df)
        assert cd_contra > cd_clean

    def test_nonnegative(self, noisy_df):
        assert contradiction_density(noisy_df) >= 0.0

    def test_empty_df(self):
        df = pd.DataFrame({"response": []})
        assert contradiction_density(df) == 0.0


# ── Tests: domain_balance_score ───────────────────────────────────────────────

class TestDomainBalanceScore:
    def test_perfectly_balanced(self, clean_df):
        score = domain_balance_score(clean_df)
        assert score == pytest.approx(1.0, abs=0.05)

    def test_imbalanced_lower_score(self, imbalanced_df, clean_df):
        imbal_score = domain_balance_score(imbalanced_df)
        clean_score = domain_balance_score(clean_df)
        assert imbal_score < clean_score

    def test_single_category(self):
        df = pd.DataFrame({"category": ["science"] * 50})
        assert domain_balance_score(df) == 0.0

    def test_range(self, noisy_df):
        score = domain_balance_score(noisy_df)
        assert 0.0 <= score <= 1.0

    def test_missing_column(self, clean_df):
        df = clean_df.drop(columns=["category"])
        result = domain_balance_score(df)
        assert math.isnan(result)


# ── Tests: noise_ratio ────────────────────────────────────────────────────────

class TestNoiseRatio:
    def test_clean_corpus_zero_noise(self, clean_df):
        assert noise_ratio(clean_df) == 0.0

    def test_noisy_corpus_higher(self, noisy_df, clean_df):
        assert noise_ratio(noisy_df) > noise_ratio(clean_df)

    def test_range(self, noisy_df):
        nr = noise_ratio(noisy_df)
        assert 0.0 <= nr <= 100.0

    def test_all_noise(self):
        df = pd.DataFrame({
            "response": ["This is unverified and allegedly a hoax."] * 10
        })
        assert noise_ratio(df) == 100.0


# ── Tests: shannon_entropy_index ──────────────────────────────────────────────

class TestShannonEntropyIndex:
    def test_uniform_text_higher_entropy(self):
        """Diverse vocabulary → higher entropy than repetitive text."""
        repetitive = pd.DataFrame({"response": ["cat cat cat cat"] * 50})
        diverse    = pd.DataFrame({"response": [
            f"The quick brown fox jumps over the lazy {i} unique words here" for i in range(50)
        ]})
        assert shannon_entropy_index(diverse) > shannon_entropy_index(repetitive)

    def test_nonnegative(self, clean_df):
        assert shannon_entropy_index(clean_df) >= 0.0

    def test_empty(self):
        df = pd.DataFrame({"response": []})
        assert shannon_entropy_index(df) == 0.0

    def test_contradiction_corpus_higher_than_clean(self, contradiction_df, clean_df):
        # Contradiction corpus has more diverse vocabulary
        assert shannon_entropy_index(contradiction_df) > 0


# ── Integration: all metrics on same corpus ───────────────────────────────────

class TestIntegration:
    def test_all_metrics_run_on_all_corpora(self, clean_df, noisy_df, contradiction_df, imbalanced_df):
        for df in [clean_df, noisy_df, contradiction_df, imbalanced_df]:
            assert verified_fact_ratio(df)   is not None
            assert contradiction_density(df) is not None
            assert domain_balance_score(df)  is not None
            assert noise_ratio(df)           is not None
            assert shannon_entropy_index(df) is not None

    def test_baseline_beats_noisy_on_accuracy(self, clean_df, noisy_df):
        assert verified_fact_ratio(clean_df) > verified_fact_ratio(noisy_df)

    def test_imbalanced_lowest_balance_score(self, clean_df, imbalanced_df):
        assert domain_balance_score(imbalanced_df) < domain_balance_score(clean_df)
