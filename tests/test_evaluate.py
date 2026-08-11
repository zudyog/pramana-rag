import pytest

pytest.importorskip("ragas")

from pramana.evaluate import evaluate
from pramana.grounding import GroundedAnswerer, GroundingPrompt
from pramana.retriever import Retriever
from pramana.store import InMemoryStore
from pramana.types import Chunk, TestCase


def test_evaluate_returns_faithfulness_and_context_precision(fake_embedder, fake_llm):
    store = InMemoryStore()
    chunk = Chunk(id="a", text="dry clean only silk saree", concern="care", source_id="a")
    store.add([chunk], fake_embedder.embed_documents([chunk.text]))
    retriever = Retriever(embedder=fake_embedder, store=store, threshold=0.5, k=1)
    answerer = GroundedAnswerer(
        retriever=retriever,
        llm=fake_llm,
        prompt=GroundingPrompt(persona="You are a test assistant."),
    )

    test_cases = [
        TestCase(query="dry clean only silk saree", expected_answer="Dry clean only."),
    ]

    result = evaluate(answerer, test_cases)

    assert result.faithfulness is not None
    assert result.context_precision is not None
