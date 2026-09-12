from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log
from typing import Iterable, Sequence


def tokenize(text: str) -> list[str]:
    return text.split()


def unique_tokens(text: str) -> set[str]:
    return set(tokenize(text))


@dataclass(frozen=True)
class TokenIDF:
    """IDF weights fit on the normalized AM Best candidate universe."""

    weights: dict[str, float]
    unseen_weight: float

    @classmethod
    def fit(cls, reference_names: Sequence[str]) -> "TokenIDF":
        if not reference_names:
            raise ValueError("reference_names must contain at least one name")

        n = len(reference_names)
        df: Counter[str] = Counter()
        for name in reference_names:
            df.update(unique_tokens(name))

        weights = {token: log((1 + n) / (1 + freq)) + 1 for token, freq in df.items()}
        return cls(weights=weights, unseen_weight=log(1 + n) + 1)

    def idf(self, token: str) -> float:
        return self.weights.get(token, self.unseen_weight)

    def total_idf(self, tokens: Iterable[str]) -> float:
        return sum(self.idf(token) for token in set(tokens))


def shared_idf(query: str, candidate: str, idf: TokenIDF) -> float:
    return idf.total_idf(unique_tokens(query) & unique_tokens(candidate))


def max_shared_idf(query: str, candidate: str, idf: TokenIDF) -> float:
    shared = unique_tokens(query) & unique_tokens(candidate)
    return max((idf.idf(token) for token in shared), default=0.0)


def query_total_idf(query: str, idf: TokenIDF) -> float:
    return idf.total_idf(unique_tokens(query))


def candidate_total_idf(candidate: str, idf: TokenIDF) -> float:
    return idf.total_idf(unique_tokens(candidate))


def query_idf_coverage(query: str, candidate: str, idf: TokenIDF) -> float:
    denominator = query_total_idf(query, idf)
    return shared_idf(query, candidate, idf) / denominator if denominator else 0.0


def candidate_idf_coverage(query: str, candidate: str, idf: TokenIDF) -> float:
    denominator = candidate_total_idf(candidate, idf)
    return shared_idf(query, candidate, idf) / denominator if denominator else 0.0


def weighted_jaccard(query: str, candidate: str, idf: TokenIDF) -> float:
    q, c = unique_tokens(query), unique_tokens(candidate)
    union = q | c
    return idf.total_idf(q & c) / idf.total_idf(union) if union else 0.0


def unmatched_query_idf(query: str, candidate: str, idf: TokenIDF) -> float:
    return idf.total_idf(unique_tokens(query) - unique_tokens(candidate))


def unmatched_candidate_idf(query: str, candidate: str, idf: TokenIDF) -> float:
    return idf.total_idf(unique_tokens(candidate) - unique_tokens(query))
