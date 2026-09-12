from __future__ import annotations

from typing import Sequence

import numpy as np


def _scores(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("scores must not be empty")
    return np.sort(values)[::-1]


def top_score(scores: Sequence[float]) -> float:
    return float(_scores(scores)[0])


def second_score(scores: Sequence[float]) -> float:
    values = _scores(scores)
    return float(values[1]) if len(values) > 1 else 0.0


def margin_1_2(scores: Sequence[float]) -> float:
    values = _scores(scores)
    return float(values[0] - values[1]) if len(values) > 1 else float(values[0])


def margin_1_3(scores: Sequence[float]) -> float:
    values = _scores(scores)
    return float(values[0] - values[2]) if len(values) > 2 else float(values[0])


def n_within_delta(scores: Sequence[float], delta: float) -> int:
    values = _scores(scores)
    return int(np.sum(values >= values[0] - delta))


def n_within_5(scores: Sequence[float]) -> int:
    return n_within_delta(scores, 5.0)


def n_within_10(scores: Sequence[float]) -> int:
    return n_within_delta(scores, 10.0)


def top_n_mean(scores: Sequence[float], n: int = 5) -> float:
    return float(np.mean(_scores(scores)[:n]))


def top_n_std(scores: Sequence[float], n: int = 5) -> float:
    return float(np.std(_scores(scores)[:n]))


def score_softmax_entropy(scores: Sequence[float], temperature: float = 5.0) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    values = _scores(scores)
    z = (values - values.max()) / temperature
    probs = np.exp(z) / np.exp(z).sum()
    return float(-np.sum(probs * np.log(probs + 1e-15)))
