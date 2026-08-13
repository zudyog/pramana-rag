# pramana-rag

[![CI](https://github.com/zudyog/pramana-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/zudyog/pramana-rag/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pramana-rag.svg)](https://pypi.org/project/pramana-rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/pramana-rag.svg)](https://pypi.org/project/pramana-rag/)

A grounding-first RAG library: threshold-gated retrieval, concern-based
chunking, a locked grounding prompt, and built-in faithfulness evaluation as
the *default path* — not pieces you have to assemble yourself.

```bash
pip install pramana-rag[openai,chroma,eval]
```

**Full documentation**: [www.zudyog.com/docs/pramana-rag/overview](https://www.zudyog.com/docs/pramana-rag/overview) — overview & quickstart, [API reference](https://www.zudyog.com/docs/pramana-rag/api-reference), and [verified examples](https://www.zudyog.com/docs/pramana-rag/examples).

## What "Pramana" means

*Pramana* (प्रमाण) is a Sanskrit term from the Nyaya school of classical
Indian epistemology — a *valid source of knowledge*. Not belief. Not
assumption. Not the most plausible guess. A source that has been retrieved,
verified, and bounded.

Applied to a RAG system: the retrieval step is the system's Pramana — its
valid source, the boundary it is not permitted to answer beyond. This
library's architecture (embed → chunk by concern → store → retrieve above a
calibrated threshold → generate only from what was retrieved → measure
faithfulness) is the *Pramana Framework* taught across
[**RAG Essentials Book 1**](https://zudyog.com) (zUdyog / ShopBot), credited
here as its origin.

## Examples

- **`examples/shopbot/`** — the book's own fashion e-commerce assistant,
  ported onto this library.
- **`examples/hr_policy_bot/`** — a deliberately different domain (an
  internal HR policy assistant for a fictional company), using entirely
  different concern names, to demonstrate the abstractions aren't
  secretly e-commerce-shaped.

## Honest positioning

Threshold-gated "retrieve or refuse" is not a new idea — it has real prior
art (e.g. ConfRAG, Microsoft's Confidence-Aware RAG). NVIDIA's NeMo
Guardrails already ships a production fact-checking rail for RAG
groundedness. RAGAS is already integrated into LangChain, LlamaIndex,
Langfuse, and Braintrust. This library does not claim to be first at any of
that.

What it does differently is **packaging**. Today, getting this full
discipline — refuse when evidence is weak, chunk by concern instead of
character count, lock the prompt to retrieved context only, measure
faithfulness — means assembling 2-3 separate, heavier systems: an
orchestration framework, a guardrails middleware with its own DSL, and a
separate eval tool. `pramana-rag` is a single lightweight library where
that discipline is the default call path, not an optional rail you wire in
later.

## Core abstractions

| Concern | Type | Default implementation |
|---|---|---|
| Chunking | `Chunker` | `ConcernChunker` — one chunk per caller-defined *concern* |
| Embedding | `Embedder` | `OpenAIEmbedder` (`text-embedding-3-small`) |
| Vector storage | `VectorStore` | `InMemoryStore` (zero deps) / `ChromaStore` |
| Retrieval | `Retriever` | threshold-gated: returns evidence or `None`, never "closest anyway" |
| Threshold selection | `calibrate_threshold()` | finds the score that separates labeled should-retrieve / should-not-retrieve queries |
| Generation | `LLM` | `OpenAIChat` (temperature 0 by default) |
| Grounding | `GroundingPrompt` + `GroundedAnswerer` | locks the model to retrieved context, refuses explicitly otherwise |
| Evaluation | `evaluate()` | RAGAS Faithfulness / Context Precision (+ Recall / Relevancy with ground truth) |

Every abstraction is domain-agnostic: `concern` is a string you define
(`"care"`, `"clause"`, `"dosage"`, `"eligibility"`, ...), not a fixed enum.
`src/pramana/` contains no e-commerce vocabulary anywhere — that only
appears in `examples/`.

## Quickstart

```python
from pramana import (
    ConcernChunker, ConcernSpec, InMemoryStore, OpenAIEmbedder,
    Retriever, calibrate_threshold, GroundingPrompt, GroundedAnswerer, OpenAIChat,
)

# 1. Define the concerns your domain actually gets asked about.
concerns = [
    ConcernSpec(name="definition", build_text=lambda d: d.get("definition")),
    ConcernSpec(name="eligibility", build_text=lambda d: d.get("eligibility")),
]
chunker = ConcernChunker(concerns)

# 2. Chunk your documents, embed them, store them.
documents = [{"id": "policy-1", "definition": "...", "eligibility": "..."}]
embedder = OpenAIEmbedder()
store = InMemoryStore()  # or ChromaStore() for persistence
for doc in documents:
    chunks = chunker.chunk(doc)
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))

# 3. Build a retriever and calibrate its threshold against labeled examples.
retriever = Retriever(embedder=embedder, store=store)
retriever.threshold = calibrate_threshold(retriever, [
    ("who is eligible for this policy?", True),
    ("what's the weather today?", False),
])

# 4. Wrap it in a locked grounding prompt.
answerer = GroundedAnswerer(
    retriever=retriever,
    llm=OpenAIChat(),
    prompt=GroundingPrompt(persona="You are a policy assistant."),
)

print(answerer.answer("who is eligible for this policy?"))
```

## Architecture

```
document ──▶ ConcernChunker ──▶ Chunk(s) ──▶ Embedder ──▶ VectorStore
                                                                │
query ──▶ Embedder ──▶ Retriever (threshold-gated) ────────────┘
                             │
                   evidence ▼ or None
                    ┌────────┴────────┐
                    ▼                 ▼
          GroundingPrompt      refusal_message
          + LLM.complete
                    │
              grounded answer
```

## Evaluation

```python
from pramana import evaluate

result = evaluate(answerer, test_cases)  # list[TestCase]
print(result.faithfulness, result.context_precision)
```

Requires `pip install pramana-rag[eval]`.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The core test suite runs fully offline (fake embedder/LLM, in-memory
store) — no API key required.

## License

MIT — see `LICENSE`.
