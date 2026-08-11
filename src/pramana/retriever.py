"""Threshold-gated retrieval.

A retriever that always returns something regardless of relevance is not a
grounding layer — it's a noise pipe. `Retriever.retrieve` returns evidence
only when it clears a calibrated similarity threshold; otherwise it returns
`None`, so the caller has an honest signal to fall back on rather than
evidence that merely happens to be the closest thing available.
"""

from __future__ import annotations

from dataclasses import dataclass

from pramana.embedding import Embedder
from pramana.store import VectorStore
from pramana.types import RetrievedChunk


class CalibrationError(ValueError):
    """Raised when labeled calibration queries don't separate cleanly —
    i.e. the highest score among queries that should NOT retrieve is not
    below the lowest score among queries that SHOULD retrieve. This usually
    means the chunking or embedding needs work before a threshold can be
    chosen reliably, not that the number is simply hard to find.
    """


@dataclass
class Retriever:
    embedder: Embedder
    store: VectorStore
    threshold: float = 0.75
    k: int = 3

    def retrieve(
        self, query: str, where: dict | None = None
    ) -> list[RetrievedChunk] | None:
        query_vector = self.embedder.embed_query(query)
        results = self.store.query(query_vector, k=self.k, where=where)
        passed = [r for r in results if r.similarity >= self.threshold]
        return passed or None

    def top_similarity(self, query: str, where: dict | None = None) -> float:
        """The top-1 similarity score for a query, ignoring the threshold.
        Used by `calibrate_threshold` to observe the raw score distribution.
        """
        query_vector = self.embedder.embed_query(query)
        results = self.store.query(query_vector, k=1, where=where)
        return results[0].similarity if results else 0.0


def calibrate_threshold(
    retriever: Retriever, labeled_queries: list[tuple[str, bool]]
) -> float:
    """Automates the book's manual threshold-calibration process (ch.6):
    run each labeled query, record its top-1 similarity score, and find the
    value that separates queries that should retrieve from queries that
    should not.

    `labeled_queries` is a list of `(query, should_retrieve)` pairs — write
    them before calling this function, the same discipline the evaluation
    test set requires: labels decided in advance, not fitted to a result
    you've already seen.

    Returns the midpoint between the highest negative score and the lowest
    positive score. Raises `CalibrationError` if the two distributions
    overlap — there is no single threshold that gets every labeled query
    right, and the fix is upstream (chunking, embeddings), not the number.
    """
    positive_scores = []
    negative_scores = []
    for query, should_retrieve in labeled_queries:
        score = retriever.top_similarity(query)
        (positive_scores if should_retrieve else negative_scores).append(score)

    if not positive_scores or not negative_scores:
        raise CalibrationError(
            "calibrate_threshold needs at least one True and one False "
            "labeled query to find a separating threshold."
        )

    max_negative = max(negative_scores)
    min_positive = min(positive_scores)

    if max_negative >= min_positive:
        raise CalibrationError(
            "No threshold cleanly separates the labeled queries: the "
            f"highest score among should-not-retrieve queries ({max_negative:.4f}) "
            f"is not below the lowest score among should-retrieve queries "
            f"({min_positive:.4f}). This usually means chunking or embedding "
            "quality needs work before a threshold can be chosen reliably."
        )

    return (max_negative + min_positive) / 2
