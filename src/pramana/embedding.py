"""Text-to-vector embedding. Pluggable — Pramana ships one default provider."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts, in the same order they were given."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""


class OpenAIEmbedder(Embedder):
    """Wraps OpenAI's embeddings API. Requires the `openai` package
    (`pip install pramana-rag[openai]`) and an `OPENAI_API_KEY`.
    """

    def __init__(self, model: str = "text-embedding-3-small", client=None):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "OpenAIEmbedder requires the 'openai' package. "
                    "Install with: pip install pramana-rag[openai]"
                ) from exc
            client = OpenAI()
        self._client = client
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
