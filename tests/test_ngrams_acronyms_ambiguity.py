import pytest

from fuzzy_matching.acronyms import acronym_exact, acronym_similarity
from fuzzy_matching.ambiguity import margin_1_2, n_within_5, score_softmax_entropy
from fuzzy_matching.ngrams import CharNgramTFIDF


def test_acronym_features():
    candidate = "cincinnati insurance company"
    assert acronym_exact("cic", candidate)
    assert acronym_similarity("cic", candidate) == pytest.approx(1.0)


def test_char_ngram_similarity_prefers_typo_to_unrelated_name():
    refs = [
        "cincinnati insurance company",
        "liberty mutual insurance group",
    ]
    model = CharNgramTFIDF.fit(refs)
    sims = model.similarities("cincinnatti insurance")
    assert sims[0] > sims[1]
    assert sims[0] > 0.5


def test_candidate_landscape_margin_and_counts():
    clear = [96, 70, 65, 60]
    crowded = [96, 94, 93, 70]
    assert margin_1_2(clear) > margin_1_2(crowded)
    assert n_within_5(clear) == 1
    assert n_within_5(crowded) == 3
    assert score_softmax_entropy(clear) < score_softmax_entropy(crowded)
