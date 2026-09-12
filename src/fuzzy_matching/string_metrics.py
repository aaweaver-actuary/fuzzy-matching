from rapidfuzz import fuzz
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler


def rapidfuzz_ratio(query: str, candidate: str) -> float:
    return fuzz.ratio(query, candidate) / 100.0


def partial_ratio(query: str, candidate: str) -> float:
    return fuzz.partial_ratio(query, candidate) / 100.0


def token_sort_ratio(query: str, candidate: str) -> float:
    return fuzz.token_sort_ratio(query, candidate) / 100.0


def token_set_ratio(query: str, candidate: str) -> float:
    return fuzz.token_set_ratio(query, candidate) / 100.0


def wratio(query: str, candidate: str) -> float:
    return fuzz.WRatio(query, candidate) / 100.0


def jaro_winkler(query: str, candidate: str) -> float:
    return JaroWinkler.normalized_similarity(query, candidate)


def damerau_levenshtein(query: str, candidate: str) -> float:
    return DamerauLevenshtein.normalized_similarity(query, candidate)
