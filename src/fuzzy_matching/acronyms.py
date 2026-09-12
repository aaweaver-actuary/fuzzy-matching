from __future__ import annotations

from rapidfuzz import fuzz


def acronym(text: str, ignored_tokens: set[str] | None = None) -> str:
    ignored_tokens = ignored_tokens or set()
    return "".join(
        token[0]
        for token in text.split()
        if token and token not in ignored_tokens
    )


def compact(text: str) -> str:
    return "".join(text.split())


def query_looks_like_acronym(query: str, max_length: int = 8) -> bool:
    tokens = query.split()
    return len(tokens) == 1 and 1 < len(tokens[0]) <= max_length


def acronym_exact(
    query: str,
    candidate: str,
    ignored_tokens: set[str] | None = None,
) -> bool:
    return compact(query) == acronym(candidate, ignored_tokens)


def acronym_similarity(
    query: str,
    candidate: str,
    ignored_tokens: set[str] | None = None,
) -> float:
    candidate_acronym = acronym(candidate, ignored_tokens)
    if not candidate_acronym:
        return 0.0
    return fuzz.ratio(compact(query), candidate_acronym) / 100.0
