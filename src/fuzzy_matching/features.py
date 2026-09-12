from __future__ import annotations

from .acronyms import acronym_exact, acronym_similarity, query_looks_like_acronym
from .ambiguity import margin_1_2, margin_1_3, n_within_5, n_within_10, score_softmax_entropy
from .idf import (
    TokenIDF,
    candidate_idf_coverage,
    max_shared_idf,
    query_idf_coverage,
    query_total_idf,
    shared_idf,
    unmatched_candidate_idf,
    unmatched_query_idf,
    weighted_jaccard,
)
from .string_metrics import (
    damerau_levenshtein,
    jaro_winkler,
    partial_ratio,
    rapidfuzz_ratio,
    token_set_ratio,
    token_sort_ratio,
    wratio,
)


def pairwise_features(
    query: str,
    candidate: str,
    *,
    idf: TokenIDF,
    generic_tokens: set[str] | None = None,
) -> dict[str, float]:
    generic_tokens = generic_tokens or set()
    q_tokens = query.split()

    return {
        "rapidfuzz_ratio": rapidfuzz_ratio(query, candidate),
        "partial_ratio": partial_ratio(query, candidate),
        "token_sort_ratio": token_sort_ratio(query, candidate),
        "token_set_ratio": token_set_ratio(query, candidate),
        "wratio": wratio(query, candidate),
        "jaro_winkler": jaro_winkler(query, candidate),
        "damerau_levenshtein": damerau_levenshtein(query, candidate),
        "shared_idf": shared_idf(query, candidate, idf),
        "max_shared_idf": max_shared_idf(query, candidate, idf),
        "weighted_jaccard": weighted_jaccard(query, candidate, idf),
        "query_idf_coverage": query_idf_coverage(query, candidate, idf),
        "candidate_idf_coverage": candidate_idf_coverage(query, candidate, idf),
        "unmatched_query_idf": unmatched_query_idf(query, candidate, idf),
        "unmatched_candidate_idf": unmatched_candidate_idf(query, candidate, idf),
        "query_total_idf": query_total_idf(query, idf),
        "query_n_chars": float(len(query)),
        "query_n_tokens": float(len(q_tokens)),
        "query_generic_fraction": (
            sum(token in generic_tokens for token in q_tokens) / len(q_tokens)
            if q_tokens else 0.0
        ),
        "query_looks_like_acronym": float(query_looks_like_acronym(query)),
        "full_acronym_exact": float(acronym_exact(query, candidate)),
        "full_acronym_similarity": acronym_similarity(query, candidate),
        "core_acronym_exact": float(acronym_exact(query, candidate, generic_tokens)),
        "core_acronym_similarity": acronym_similarity(query, candidate, generic_tokens),
    }


def candidate_landscape_features(scores: list[float]) -> dict[str, float]:
    return {
        "margin_1_2": margin_1_2(scores),
        "margin_1_3": margin_1_3(scores),
        "n_within_5": float(n_within_5(scores)),
        "n_within_10": float(n_within_10(scores)),
        "score_entropy": score_softmax_entropy(scores),
    }
