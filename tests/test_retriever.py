import pytest

from pramana.retriever import CalibrationError, Retriever, calibrate_threshold
from pramana.store import InMemoryStore
from pramana.types import Chunk


def _store_with(chunks_and_texts, embedder):
    store = InMemoryStore()
    chunks = [
        Chunk(id=cid, text=text, concern="test", source_id=cid)
        for cid, text in chunks_and_texts
    ]
    vectors = embedder.embed_documents([c.text for c in chunks])
    store.add(chunks, vectors)
    return store


def test_retrieve_returns_none_below_threshold(fake_embedder):
    store = _store_with([("a", "alpha beta gamma delta")], fake_embedder)
    retriever = Retriever(embedder=fake_embedder, store=store, threshold=0.99, k=3)

    result = retriever.retrieve("completely unrelated weather report today")

    assert result is None


def test_retrieve_returns_exact_match_above_threshold(fake_embedder):
    store = _store_with([("a", "alpha beta gamma delta")], fake_embedder)
    retriever = Retriever(embedder=fake_embedder, store=store, threshold=0.5, k=3)

    result = retriever.retrieve("alpha beta gamma delta")

    assert result is not None
    assert result[0].chunk.id == "a"
    assert result[0].similarity == pytest.approx(1.0)


def test_retrieve_ranks_more_similar_chunk_first(fake_embedder):
    store = _store_with(
        [
            ("care", "dry clean only silk saree care instructions"),
            ("policy", "return policy seven days exchange window"),
        ],
        fake_embedder,
    )
    retriever = Retriever(embedder=fake_embedder, store=store, threshold=0.0, k=2)

    result = retriever.retrieve("dry clean silk saree care")

    assert result[0].chunk.id == "care"


def test_calibrate_threshold_finds_separating_midpoint(fake_embedder):
    store = _store_with([("care", "dry clean care instructions silk saree")], fake_embedder)
    retriever = Retriever(embedder=fake_embedder, store=store, threshold=0.0, k=1)

    labeled_queries = [
        ("dry clean care instructions silk saree", True),   # identical -> similarity 1.0
        ("completely unrelated weather report", False),      # disjoint vocab -> similarity ~0
    ]

    threshold = calibrate_threshold(retriever, labeled_queries)

    assert 0.0 < threshold < 1.0


def test_calibrate_threshold_raises_when_scores_overlap(fake_embedder):
    store = _store_with([("care", "dry clean care instructions silk saree")], fake_embedder)
    retriever = Retriever(embedder=fake_embedder, store=store, threshold=0.0, k=1)

    # Deliberately mislabeled so the negative example scores higher than the
    # positive example — no threshold can separate them.
    labeled_queries = [
        ("dry clean care instructions silk saree", False),  # identical -> similarity 1.0
        ("completely unrelated weather report", True),       # disjoint vocab -> similarity ~0
    ]

    with pytest.raises(CalibrationError):
        calibrate_threshold(retriever, labeled_queries)


def test_calibrate_threshold_requires_both_labels(fake_embedder):
    store = _store_with([("a", "alpha beta")], fake_embedder)
    retriever = Retriever(embedder=fake_embedder, store=store)

    with pytest.raises(CalibrationError):
        calibrate_threshold(retriever, [("alpha beta", True)])
