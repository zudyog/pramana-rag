"""Pramana — a grounding-first RAG library.

Pramana (Sanskrit: a valid source of knowledge, from the Nyaya school of
classical Indian epistemology) treats retrieve-or-refuse, concern-based
chunking, a locked grounding prompt, and faithfulness evaluation as the
default path for a retrieval-augmented system, not optional pieces you
assemble yourself. See the README for the full architecture and an honest
comparison to existing grounding/guardrail tooling.
"""

from pramana.chunking import Chunker, ConcernChunker, ConcernSpec
from pramana.embedding import Embedder, OpenAIEmbedder
from pramana.evaluate import evaluate
from pramana.grounding import DEFAULT_REFUSAL, GroundedAnswerer, GroundingPrompt, format_chunks
from pramana.llm import LLM, OpenAIChat
from pramana.retriever import CalibrationError, Retriever, calibrate_threshold
from pramana.store import ChromaStore, InMemoryStore, VectorStore
from pramana.types import Chunk, EvalResult, RetrievedChunk, TestCase

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "RetrievedChunk",
    "TestCase",
    "EvalResult",
    "Chunker",
    "ConcernChunker",
    "ConcernSpec",
    "Embedder",
    "OpenAIEmbedder",
    "VectorStore",
    "InMemoryStore",
    "ChromaStore",
    "Retriever",
    "calibrate_threshold",
    "CalibrationError",
    "LLM",
    "OpenAIChat",
    "GroundingPrompt",
    "GroundedAnswerer",
    "format_chunks",
    "DEFAULT_REFUSAL",
    "evaluate",
]
