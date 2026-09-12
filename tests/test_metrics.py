import pytest

from fuzzy_matching.string_metrics import (
    damerau_levenshtein,
    jaro_winkler,
    rapidfuzz_ratio,
    token_set_ratio,
)


def test_identical_strings_are_perfect_matches():
    text = "cincinnati insurance company"
    assert rapidfuzz_ratio(text, text) == pytest.approx(1.0)
    assert jaro_winkler(text, text) == pytest.approx(1.0)
    assert damerau_levenshtein(text, text) == pytest.approx(1.0)


def test_typo_remains_high_similarity():
    assert jaro_winkler("cincinnatti", "cincinnati") > 0.9
    assert damerau_levenshtein("travleers", "travelers") > 0.7


def test_token_set_can_be_perfect_for_subset():
    # Documents why token_set_ratio must not be interpreted as probability.
    assert token_set_ratio("cincinnati", "cincinnati insurance company") == pytest.approx(1.0)
