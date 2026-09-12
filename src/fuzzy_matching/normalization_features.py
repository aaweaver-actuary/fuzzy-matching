from __future__ import annotations

from collections.abc import Callable

from .string_metrics import wratio


def token_count_removed(conservative_query: str, aggressive_query: str) -> int:
    return max(0, len(conservative_query.split()) - len(aggressive_query.split()))


def fraction_tokens_removed(conservative_query: str, aggressive_query: str) -> float:
    n = len(conservative_query.split())
    return token_count_removed(conservative_query, aggressive_query) / n if n else 0.0


def normalization_score_delta(
    query_conservative: str,
    candidate_conservative: str,
    query_aggressive: str,
    candidate_aggressive: str,
    scorer: Callable[[str, str], float] = wratio,
) -> float:
    """Aggressive similarity minus conservative similarity."""
    return scorer(query_aggressive, candidate_aggressive) - scorer(
        query_conservative, candidate_conservative
    )
