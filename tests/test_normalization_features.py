import pytest

from fuzzy_matching.normalization_features import (
    fraction_tokens_removed,
    normalization_score_delta,
    token_count_removed,
)


def test_token_removal_features():
    conservative = "cincinnati insurance company"
    aggressive = "cincinnati"
    assert token_count_removed(conservative, aggressive) == 2
    assert fraction_tokens_removed(conservative, aggressive) == pytest.approx(2 / 3)


def test_normalization_score_delta_can_improve_match():
    delta = normalization_score_delta(
        "cincinnati insurance company",
        "cincinnati financial corporation",
        "cincinnati",
        "cincinnati financial",
    )
    assert isinstance(delta, float)
