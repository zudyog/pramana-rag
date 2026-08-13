"""Vector storage. `Retriever` only ever calls this interface, never a
specific database — swap `InMemoryStore` for `ChromaStore` (or your own
`VectorStore`) without touching any calling code, the same abstraction
boundary the book argues for when ChromaDB is later replaced by a
production vector database.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pramana.types import Chunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Store chunks alongside their embedding vectors."""

    @abstractmethod
    def query(
        self, vector: list[float], k: int, where: dict | None = None
    ) -> list[RetrievedChunk]:
        """Return the k nearest chunks to `vector`, ranked by similarity
        (highest first). `where` optionally restricts the search to chunks
        whose metadata matches every key/value pair given.
        """


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _matches(metadata: dict, where: dict) -> bool:
    return all(metadata.get(key) == value for key, value in where.items())


class InMemoryStore(VectorStore):
    """A pure-numpy cosine-similarity store. No external service, no extra
    dependency beyond numpy. Intended for tests, small catalogs, and
    examples that shouldn't require installing/running a vector database
    just to try the library.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[np.ndarray] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        self._chunks.extend(chunks)
        self._vectors.extend(np.asarray(v, dtype=float) for v in vectors)

    def query(
        self, vector: list[float], k: int, where: dict | None = None
    ) -> list[RetrievedChunk]:
        query_vec = np.asarray(vector, dtype=float)
        scored: list[RetrievedChunk] = []
        for chunk, stored_vec in zip(self._chunks, self._vectors):
            if where and not _matches(chunk.metadata, where):
                continue
            similarity = _cosine_similarity(query_vec, stored_vec)
            scored.append(RetrievedChunk(chunk=chunk, similarity=similarity))
        scored.sort(key=lambda r: r.similarity, reverse=True)
        return scored[:k]


class ChromaStore(VectorStore):
    """Wraps a local ChromaDB persistent collection, cosine-indexed —
    matching the setup used throughout the book. Requires the `chromadb`
    package (`pip install pramana-rag[chroma]`).
    """

    def __init__(
        self,
        path: str = ".pramana_db",
        collection_name: str = "pramana",
        client=None,
    ):
        if client is None:
            try:
                import chromadb
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "ChromaStore requires the 'chromadb' package. "
                    "Install with: pip install pramana-rag[chroma]"
                ) from exc
            client = chromadb.PersistentClient(path=path)
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"concern": c.concern, "source_id": c.source_id, **c.metadata} for c in chunks],
        )

    def query(
        self, vector: list[float], k: int, where: dict | None = None
    ) -> list[RetrievedChunk]:
        results = self._collection.query(
            query_embeddings=[vector],
            n_results=k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        retrieved: list[RetrievedChunk] = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances):
            meta = dict(meta)
            concern = meta.pop("concern", "")
            source_id = meta.pop("source_id", "")
            chunk = Chunk(
                id=chunk_id, text=doc, concern=concern, source_id=source_id, metadata=meta
            )
            retrieved.append(RetrievedChunk(chunk=chunk, similarity=1 - distance))
        return retrieved
