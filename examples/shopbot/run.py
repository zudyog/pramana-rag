"""ShopBot, rebuilt on the generic pramana library.

Same catalog, same "one chunk, one question" chunking philosophy, same
threshold-gated retrieval and locked grounding prompt as RAG Essentials
Book 1 — expressed through pramana's domain-agnostic API instead of
hand-written, book-specific code.

Requires: pip install pramana-rag[openai] python-dotenv
Requires: OPENAI_API_KEY set in the environment, or in a .env file in
either this directory or the repo root.

Run:
    python examples/shopbot/run.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from products import PRODUCTS

from pramana import (
    ConcernChunker,
    ConcernSpec,
    GroundedAnswerer,
    GroundingPrompt,
    InMemoryStore,
    OpenAIChat,
    OpenAIEmbedder,
    Retriever,
    calibrate_threshold,
)

# ── Chunking: one concern, one chunk, anchored with the product name ────────

CONCERNS = [
    ConcernSpec(
        name="identity",
        build_text=lambda p: (
            f"{p['name']}. {p['description']} "
            f"Category: {p['category']}. Occasion: {p.get('occasion', 'general')}."
        ),
        metadata=lambda p: {"category": p["category"], "price": p["price"]},
    ),
    ConcernSpec(
        name="variants",
        build_text=lambda p: (
            f"{p['name']} is available in colours: {', '.join(p.get('colors', []))}. "
            f"Available sizes: {', '.join(p.get('sizes', []))}. Price: Rs.{p['price']}."
        ),
        metadata=lambda p: {"category": p["category"], "price": p["price"]},
    ),
    ConcernSpec(
        name="policy",
        build_text=lambda p: (
            f"Return and exchange policy for {p['name']}: {p['return_policy']} "
            f"Exchange policy: {p.get('exchange_policy', 'contact support for exchanges')}."
            if p.get("return_policy")
            else None
        ),
        metadata=lambda p: {"category": p["category"], "price": p["price"]},
    ),
    ConcernSpec(
        name="care",
        build_text=lambda p: (
            f"Care instructions for {p['name']}: {p['care_instructions']}"
            if p.get("care_instructions")
            else None
        ),
        metadata=lambda p: {"category": p["category"], "price": p["price"]},
    ),
]


def faq_concerns(product: dict) -> list[ConcernSpec]:
    """FAQs are one-per-entry rather than one-per-product, so they're built
    per product and merged in below instead of living in the static list.
    """
    specs = []
    for i, faq in enumerate(product.get("faqs", [])):
        specs.append(
            ConcernSpec(
                name=f"faq_{i}",
                build_text=lambda p, faq=faq: (
                    f"Question about {p['name']}: {faq['question']} Answer: {faq['answer']}"
                ),
                metadata=lambda p: {"category": p["category"], "price": p["price"]},
            )
        )
    return specs


def build_store(embedder: OpenAIEmbedder) -> InMemoryStore:
    store = InMemoryStore()
    for product in PRODUCTS:
        chunker = ConcernChunker(CONCERNS + faq_concerns(product))
        chunks = chunker.chunk(product)
        vectors = embedder.embed_documents([c.text for c in chunks])
        store.add(chunks, vectors)
        print(f"  {product['name']}: {len(chunks)} chunks")
    return store


def main() -> None:
    embedder = OpenAIEmbedder()
    print("Ingesting catalog...")
    store = build_store(embedder)

    # A default threshold of 0.75 is the book's number for the book's exact
    # chunk text — a different catalog, even one this similar, produces a
    # different real score distribution. Calibrating against labeled
    # queries (some that should retrieve, some that shouldn't) is the
    # library's own recommended fix, not a magic number.
    retriever = Retriever(embedder=embedder, store=store, k=3)
    labeled_queries = [
        ("Does the silk saree need dry cleaning?", True),
        ("What colours does the anarkali suit come in?", True),
        ("Is the cotton kurta good for summer?", True),
        ("Do you sell men's sherwanis?", False),
        ("What's the weather like today?", False),
    ]
    retriever.threshold = calibrate_threshold(retriever, labeled_queries)
    print(f"Calibrated similarity threshold: {retriever.threshold:.4f}")

    prompt = GroundingPrompt(
        persona=(
            "You are ShopBot, a warm and knowledgeable product assistant for "
            "zUdyog Fashion, an Indian fashion e-commerce store."
        ),
        refusal_message=(
            "I don't have that specific information. Please reach our support "
            "team at support@zudyog.com for help."
        ),
    )
    answerer = GroundedAnswerer(retriever=retriever, llm=OpenAIChat(), prompt=prompt)

    test_queries = [
        "Does the silk saree need dry cleaning?",
        "What colours does the anarkali suit come in?",
        "Do you sell men's sherwanis?",
    ]
    for query in test_queries:
        print(f"\nCustomer: {query}")
        print(f"ShopBot:  {answerer.answer(query)}")


if __name__ == "__main__":
    main()
