"""Concern-based chunking.

Generalizes the "one chunk, one question, one clear answer" rule: a document
diluted into a single blob of text loses retrieval precision because a query
about one attribute competes with every other attribute for the embedding's
meaning. Splitting into single sentences overcorrects and loses the context
that anchors a chunk to what it's about.

The fix is neither: split each document into one chunk per *concern* — a
caller-defined category of question your domain actually gets asked
("care", "clause", "dosage", "eligibility", ...) — with each chunk carrying
enough of its own context to be retrievable in isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from pramana.types import Chunk


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: dict) -> list[Chunk]:
        """Split one source document into concern-focused chunks."""


@dataclass
class ConcernSpec:
    """Describes how to build one concern's chunk from a document.

    `build_text` returns the chunk text for this concern, or `None`/`""` to
    skip it when the document has no data for that concern (e.g. a product
    with no return policy produces no "policy" chunk). Callers are
    responsible for anchoring the text with enough identifying context
    (a name, a title) that the chunk means something read on its own —
    Pramana does not silently prepend anything for you, so nothing is added
    to a chunk you didn't put there.
    """

    name: str
    build_text: Callable[[dict], str | None]
    metadata: Callable[[dict], dict] = field(default=lambda d: {})


class ConcernChunker(Chunker):
    """A `Chunker` driven by a list of `ConcernSpec`s.

    One `Chunk` is produced per concern whose `build_text` returns non-empty
    text. The concerns themselves are entirely up to the caller — this class
    carries no domain vocabulary.
    """

    def __init__(self, concerns: list[ConcernSpec], id_field: str = "id"):
        if not concerns:
            raise ValueError("ConcernChunker requires at least one ConcernSpec")
        self.concerns = concerns
        self.id_field = id_field

    def chunk(self, document: dict) -> list[Chunk]:
        if self.id_field not in document:
            raise KeyError(
                f"document is missing id_field {self.id_field!r}: {document!r}"
            )
        source_id = str(document[self.id_field])
        chunks: list[Chunk] = []
        for concern in self.concerns:
            text = concern.build_text(document)
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=f"{source_id}_{concern.name}",
                    text=text,
                    concern=concern.name,
                    source_id=source_id,
                    metadata=concern.metadata(document),
                )
            )
        return chunks
