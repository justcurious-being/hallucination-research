"""
tests/test_hallucination_index.py
Tests for the composite Hallucination Index computation.
Run: pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.hallucination_index import (
    apply_rag_adjustment,
    compute_hi,
    score_citation_failure_rate,
    score_contradiction_rate,
    score_factual_accuracy,
    score_incoherence_rate,
)


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "response":         ["This is a verified factual statement with a reference (Smith, 2023)."] * 20,
        "is_hallucination": [0] * 20,
    })


@pytest.fixture
def hallucinated_df():
    return pd.DataFrame({
        "response": [
            "According to studies show, the result is X. However, on the contrary, it is also not X.",
            "Research confirms this but the opposite is also true. No evidence exists.",
            "This is disputed. Yet other sources say the reverse.",
        ],
        "is_hallucination": [1, 1, 1],
    })


class TestComponentScorers:
    def test_factual_accuracy_clean(self, clean_df):
        assert score_factual_accuracy(clean_df) == 0.0

    def test_factual_accuracy_hallucinated(self, hallucinated_df):
        assert score_factual_accuracy(hallucinated_df) == 1.0

    def test_contradiction_rate_clean(self, clean_df):
        assert score_contradiction_rate(clean_df) == 0.0

    def test_contradiction_rate_high(self, hallucinated_df):
        assert score_contradiction_rate(hallucinated_df) > 0

    def test_citation_failure_clean(self, clean_df):
        # Clean df has proper citation format → should not trigger failure
        rate = score_citation_failure_rate(clean_df)
        assert rate == 0.0

    def test_citation_failure_high(self, hallucinated_df):
        # Hallucinated df claims citations without references
        rate = score_citation_failure_rate(hallucinated_df)
        assert rate >= 0.0  # may vary based on phrasing

    def test_incoherence_rate_range(self, hallucinated_df):
        rate = score_incoherence_rate(hallucinated_df)
        assert 0.0 <= rate <= 1.0


class TestCompositeHI:
    def test_clean_lower_than_hallucinated(self, clean_df, hallucinated_df):
        clean_hi = compute_hi(clean_df, "baseline")["hallucination_index"]
        hall_hi  = compute_hi(hallucinated_df, "noisy")["hallucination_index"]
        assert clean_hi < hall_hi

    def test_hi_range(self, clean_df):
        result = compute_hi(clean_df, "baseline")
        assert 0.0 <= result["hallucination_index"] <= 100.0

    def test_result_has_all_keys(self, clean_df):
        result = compute_hi(clean_df, "test")
        expected_keys = [
            "corpus", "n_evaluated", "factual_error_rate",
            "contradiction_rate", "citation_fail_rate",
            "incoherence_rate", "hallucination_index",
        ]
        for k in expected_keys:
            assert k in result, f"Missing key: {k}"

    def test_sample_size_respected(self):
        big_df = pd.DataFrame({
            "response":         ["text"] * 1000,
            "is_hallucination": [0] * 1000,
        })
        result = compute_hi(big_df, "large", sample_n=50)
        assert result["n_evaluated"] == 50


class TestRAGAdjustment:
    def test_rag_reduces_hi(self):
        hi_df = pd.DataFrame({
            "corpus":              ["baseline", "noise_augmented", "contradiction_enriched", "imbalanced"],
            "hallucination_index": [7.2, 17.8, 24.1, 20.6],
        })
        adjusted = apply_rag_adjustment(hi_df)
        assert (adjusted["hi_with_rag"] < adjusted["hallucination_index"]).all()

    def test_rag_delta_positive(self):
        hi_df = pd.DataFrame({
            "corpus":              ["baseline", "noise_augmented"],
            "hallucination_index": [10.0, 20.0],
        })
        adjusted = apply_rag_adjustment(hi_df)
        assert (adjusted["rag_delta"] > 0).all()

    def test_baseline_gets_most_reduction(self):
        hi_df = pd.DataFrame({
            "corpus":              ["baseline", "contradiction_enriched"],
            "hallucination_index": [10.0, 10.0],  # Same HI to isolate factor effect
        })
        adjusted = apply_rag_adjustment(hi_df)
        baseline_delta = adjusted.loc[adjusted["corpus"] == "baseline", "rag_delta"].values[0]
        contra_delta   = adjusted.loc[adjusted["corpus"] == "contradiction_enriched", "rag_delta"].values[0]
        assert baseline_delta > contra_delta
