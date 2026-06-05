"""
tests/test_corpus_builder.py
Tests for the four corpus construction functions.
Run: pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_quality.corpus_builder import (
    _synthetic_hallucination_stub,
    _synthetic_news_stub,
    build_baseline,
    build_contradiction_enriched,
    build_imbalanced,
    build_noise_augmented,
)


@pytest.fixture
def halluc_df():
    return _synthetic_hallucination_stub(n=200)


@pytest.fixture
def news_df():
    return _synthetic_news_stub(n=400)


class TestSyntheticStubs:
    def test_hallucination_stub_shape(self):
        df = _synthetic_hallucination_stub(n=100)
        assert len(df) == 100
        assert "prompt" in df.columns
        assert "response" in df.columns
        assert "is_hallucination" in df.columns

    def test_news_stub_labels(self):
        df = _synthetic_news_stub(n=100)
        assert set(df["label"].unique()) == {"fake", "real"}


class TestBaselineCorpus:
    def test_only_clean_rows(self, halluc_df):
        baseline = build_baseline(halluc_df)
        assert (baseline["is_hallucination"] == 0).all()

    def test_corpus_label(self, halluc_df):
        baseline = build_baseline(halluc_df)
        assert (baseline["corpus"] == "baseline").all()

    def test_injected_false(self, halluc_df):
        baseline = build_baseline(halluc_df)
        assert not baseline["injected"].any()


class TestNoiseAugmented:
    def test_injection_rate(self, halluc_df, news_df):
        rate   = 0.20
        corpus = build_noise_augmented(halluc_df, news_df, noise_rate=rate)
        injected_fraction = corpus["injected"].mean()
        assert abs(injected_fraction - rate) < 0.05

    def test_corpus_label(self, halluc_df, news_df):
        corpus = build_noise_augmented(halluc_df, news_df)
        assert (corpus["corpus"] == "noise_augmented").all()

    def test_injected_rows_are_hallucinations(self, halluc_df, news_df):
        corpus = build_noise_augmented(halluc_df, news_df, noise_rate=0.20)
        injected = corpus[corpus["injected"] == True]
        assert (injected["is_hallucination"] == 1).all()

    def test_row_count_preserved(self, halluc_df, news_df):
        corpus = build_noise_augmented(halluc_df, news_df)
        assert len(corpus) == len(halluc_df)


class TestContradictionEnriched:
    def test_contradiction_rate(self, halluc_df):
        rate   = 0.15
        corpus = build_contradiction_enriched(halluc_df, contra_rate=rate)
        injected_fraction = corpus["injected"].mean()
        assert abs(injected_fraction - rate) < 0.05

    def test_contradictions_marked_hallucination(self, halluc_df):
        corpus = build_contradiction_enriched(halluc_df, contra_rate=0.15)
        contra_rows = corpus[corpus["injected"] == True]
        assert (contra_rows["is_hallucination"] == 1).all()

    def test_contradiction_text_appended(self, halluc_df):
        """Contradicted responses should be longer than originals."""
        original_len = halluc_df["response"].str.len().mean()
        corpus = build_contradiction_enriched(halluc_df, contra_rate=0.50)
        new_len = corpus["response"].str.len().mean()
        assert new_len > original_len

    def test_corpus_label(self, halluc_df):
        corpus = build_contradiction_enriched(halluc_df)
        assert (corpus["corpus"] == "contradiction_enriched").all()


class TestImbalanced:
    def test_dominant_category_at_skew(self, halluc_df):
        skew   = 0.70
        corpus = build_imbalanced(halluc_df, skew=skew)
        dominant = corpus["category"].value_counts(normalize=True).iloc[0]
        assert abs(dominant - skew) < 0.10

    def test_row_count_reasonable(self, halluc_df):
        corpus = build_imbalanced(halluc_df)
        assert len(corpus) == len(halluc_df)

    def test_corpus_label(self, halluc_df):
        corpus = build_imbalanced(halluc_df)
        assert (corpus["corpus"] == "imbalanced").all()


class TestIntegration:
    def test_all_four_corpora_distinct(self, halluc_df, news_df):
        baseline = build_baseline(halluc_df)
        noisy    = build_noise_augmented(halluc_df, news_df)
        contra   = build_contradiction_enriched(halluc_df)
        imbal    = build_imbalanced(halluc_df)

        corpus_names = {baseline["corpus"].iloc[0], noisy["corpus"].iloc[0],
                        contra["corpus"].iloc[0],   imbal["corpus"].iloc[0]}
        assert len(corpus_names) == 4

    def test_noise_has_more_hallucinations_than_baseline(self, halluc_df, news_df):
        baseline = build_baseline(halluc_df)
        noisy    = build_noise_augmented(halluc_df, news_df)
        assert noisy["is_hallucination"].mean() > baseline["is_hallucination"].mean()
