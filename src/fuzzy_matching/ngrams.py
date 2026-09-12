from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CharNgramTFIDF:
    """Character n-gram TF-IDF model fit once on normalized reference names."""

    vectorizer: TfidfVectorizer
    reference_names: tuple[str, ...]
    reference_matrix: object

    @classmethod
    def fit(
        cls,
        reference_names: Sequence[str],
        ngram_range: tuple[int, int] = (3, 5),
    ) -> "CharNgramTFIDF":
        names = tuple(reference_names)
        if not names:
            raise ValueError("reference_names must contain at least one name")

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            lowercase=False,
            norm="l2",
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(names)
        return cls(vectorizer, names, matrix)

    def similarity(self, query: str, candidate_index: int) -> float:
        q = self.vectorizer.transform([query])
        return float(cosine_similarity(q, self.reference_matrix[candidate_index])[0, 0])

    def similarities(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return cosine_similarity(q, self.reference_matrix)[0]
