"""Test fixtures with zero network calls.

FakeEmbedder produces deterministic bag-of-words vectors (word -> hashed
dimension, counted) so texts that share vocabulary score a predictably
higher cosine similarity than texts that don't — enough to exercise
threshold gating and ranking without calling any real embedding API.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from pramana.embedding import Embedder
from pramana.llm import LLM

DIMS = 64


def _text_to_vector(text: str) -> list[float]:
    vec = np.zeros(DIMS)
    for word in text.lower().split():
        index = int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIMS
        vec[index] += 1.0
    if vec.sum() == 0:
        vec[0] = 1.0
    return vec.tolist()


class FakeEmbedder(Embedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _text_to_vector(text)


class FakeLLM(LLM):
    """Echoes the context back so tests can assert on what the grounding
    prompt actually assembled, without any real generation call.
    """

    def complete(self, system: str, user: str) -> str:
        return f"[fake-llm response to: {user!r}]\nsystem-was:\n{system}"


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
