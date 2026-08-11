"""ShopBot, rebuilt on the generic pramana library.

Same catalog, same "one chunk, one question" chunking philosophy, same
threshold-gated retrieval and locked grounding prompt as RAG Essentials
Book 1 — expressed through pramana's domain-agnostic API instead of
hand-written, book-specific code.

Requires: pip install pramana-rag[openai]
Requires: OPENAI_API_KEY set in the environment.

Run:
    python examples/shopbot/run.py
"""

from __future__ import annotations

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

    retriever = Retriever(embedder=embedder, store=store, threshold=0.75, k=3)
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
