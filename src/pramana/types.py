"""Shared data types used across every pramana component."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """One retrievable unit of evidence.

    `concern` is caller-defined — it names the *kind* of question this chunk
    answers (e.g. "care", "clause", "dosage"), not a fixed enum. Pramana's
    chunking philosophy is one chunk answers one concern; it never prescribes
    which concerns your domain has.
    """

    id: str
    text: str
    concern: str
    source_id: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity: float


@dataclass
class TestCase:
    """A labeled evaluation case. `relevant_chunk_ids` and `expected_answer`
    are ground truth, written before the system runs, not after seeing scores.
    """

    query: str
    expected_answer: str
    relevant_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    faithfulness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    answer_relevancy: float | None = None
    per_query: list[dict] = field(default_factory=list)
