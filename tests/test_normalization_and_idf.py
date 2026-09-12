import pytest

from fuzzy_matching.idf import TokenIDF, query_idf_coverage, weighted_jaccard
from fuzzy_matching.normalization import normalize_aggressive, normalize_conservative


def test_reference_and_user_normalization_behave_consistently():
    assert normalize_conservative("Cincinnati Ins. Co.") == "cincinnati insurance company"
    assert normalize_conservative("CINCINNATI INS CO") == "cincinnati insurance company"


def test_aggressive_normalization_is_more_lossy():
    conservative = normalize_conservative("Cincinnati Insurance Company, Inc.")
    aggressive = normalize_aggressive("Cincinnati Insurance Company, Inc.")
    assert conservative == "cincinnati insurance company"
    assert aggressive == "cincinnati"


def test_query_idf_coverage_is_directional():
    refs = [
        "cincinnati financial corporation",
        "american family insurance group",
        "liberty mutual insurance group",
    ]
    idf = TokenIDF.fit(refs)

    assert query_idf_coverage("cincinnati", "cincinnati financial corporation", idf) == pytest.approx(1.0)
    assert weighted_jaccard("cincinnati", "cincinnati financial corporation", idf) < 1.0


def test_rare_token_has_more_weight_than_common_token():
    refs = [
        "alpha insurance group",
        "beta insurance group",
        "gamma insurance group",
        "cincinnati insurance group",
    ]
    idf = TokenIDF.fit(refs)
    assert idf.idf("cincinnati") > idf.idf("insurance")
