import pytest

from pramana.chunking import ConcernChunker, ConcernSpec


def test_concern_chunker_builds_one_chunk_per_concern_with_data():
    concerns = [
        ConcernSpec(
            name="identity",
            build_text=lambda d: f"{d['name']}. {d.get('description', '')}".strip(),
        ),
        ConcernSpec(
            name="care",
            build_text=lambda d: d.get("care_instructions"),
        ),
    ]
    chunker = ConcernChunker(concerns)

    document = {
        "id": "doc1",
        "name": "Cotton Kurta",
        "description": "Breathable, ideal for summer.",
        "care_instructions": "Machine wash cold.",
    }

    chunks = chunker.chunk(document)

    assert [c.concern for c in chunks] == ["identity", "care"]
    assert chunks[0].id == "doc1_identity"
    assert chunks[0].source_id == "doc1"
    assert "Cotton Kurta" in chunks[0].text
    assert chunks[1].text == "Machine wash cold."


def test_concern_chunker_skips_concerns_with_no_data():
    concerns = [
        ConcernSpec(name="identity", build_text=lambda d: d["name"]),
        ConcernSpec(name="policy", build_text=lambda d: d.get("return_policy")),
    ]
    chunker = ConcernChunker(concerns)

    document = {"id": "doc1", "name": "Woolen Shawl"}  # no return_policy key

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].concern == "identity"


def test_concern_chunker_attaches_metadata():
    concerns = [
        ConcernSpec(
            name="identity",
            build_text=lambda d: d["name"],
            metadata=lambda d: {"category": d["category"]},
        ),
    ]
    chunker = ConcernChunker(concerns)

    chunks = chunker.chunk({"id": "doc1", "name": "Saree", "category": "festive"})

    assert chunks[0].metadata == {"category": "festive"}


def test_concern_chunker_raises_on_missing_id_field():
    chunker = ConcernChunker([ConcernSpec(name="identity", build_text=lambda d: d["name"])])
    with pytest.raises(KeyError):
        chunker.chunk({"name": "no id here"})


def test_concern_chunker_requires_at_least_one_concern():
    with pytest.raises(ValueError):
        ConcernChunker([])
