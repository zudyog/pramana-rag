import pytest

from pramana.grounding import DEFAULT_REFUSAL, GroundedAnswerer, GroundingPrompt, format_chunks
from pramana.retriever import Retriever
from pramana.store import InMemoryStore
from pramana.types import Chunk, RetrievedChunk


def _build_answerer(embedder, llm, threshold):
    store = InMemoryStore()
    chunk = Chunk(id="a", text="dry clean only silk saree", concern="care", source_id="a")
    store.add([chunk], embedder.embed_documents([chunk.text]))
    retriever = Retriever(embedder=embedder, store=store, threshold=threshold, k=1)
    prompt = GroundingPrompt(persona="You are a test assistant.")
    return GroundedAnswerer(retriever=retriever, llm=llm, prompt=prompt)


def test_answer_returns_refusal_when_nothing_retrieved(fake_embedder, fake_llm):
    answerer = _build_answerer(fake_embedder, fake_llm, threshold=0.99)

    result = answerer.answer("completely unrelated weather report today")

    assert result == DEFAULT_REFUSAL


def test_answer_calls_llm_with_grounded_context_when_evidence_found(fake_embedder, fake_llm):
    answerer = _build_answerer(fake_embedder, fake_llm, threshold=0.5)

    result = answerer.answer("dry clean only silk saree")

    assert "dry clean only silk saree" in result
    assert "care" in result  # concern label made it into the assembled system prompt


def test_grounding_prompt_includes_persona_context_and_refusal():
    prompt = GroundingPrompt(persona="You are ShopBot.", refusal_message="I cannot help with that.")

    system = prompt.system_prompt(context="[1 — care]\ndry clean only")

    assert "You are ShopBot." in system
    assert "I cannot help with that." in system
    assert "dry clean only" in system


def test_format_chunks_labels_each_chunk_by_concern():
    evidence = [
        RetrievedChunk(chunk=Chunk(id="a", text="text A", concern="care", source_id="a"), similarity=0.9),
        RetrievedChunk(chunk=Chunk(id="b", text="text B", concern="policy", source_id="b"), similarity=0.8),
    ]

    formatted = format_chunks(evidence)

    assert "[1 — care]" in formatted
    assert "[2 — policy]" in formatted
    assert "text A" in formatted
    assert "text B" in formatted
