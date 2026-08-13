"""PolicyBot — an internal HR assistant, built on the generic pramana library.

A second, deliberately non-e-commerce example (see examples/shopbot/ for
the first) to demonstrate that pramana's abstractions are not fashion-
retail-shaped: the concerns below (summary/eligibility/procedure/
exceptions/faq) share no vocabulary with ShopBot's
(identity/variants/policy/care/faq), and the wiring is identical.

Requires: pip install pramana-rag[openai] python-dotenv
Requires: OPENAI_API_KEY set in the environment, or in a .env file in
either this directory or the repo root.

Run:
    python examples/hr_policy_bot/run.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from policies import POLICIES

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

# ── Chunking: one concern, one chunk, anchored with the policy name ─────────

CONCERNS = [
    ConcernSpec(
        name="summary",
        build_text=lambda p: f"{p['name']}. {p['summary']}",
    ),
    ConcernSpec(
        name="eligibility",
        build_text=lambda p: (
            f"Eligibility for the {p['name']}: {p['eligibility']}"
            if p.get("eligibility")
            else None
        ),
    ),
    ConcernSpec(
        name="procedure",
        build_text=lambda p: (
            f"Procedure for the {p['name']}: {p['procedure']}"
            if p.get("procedure")
            else None
        ),
    ),
    ConcernSpec(
        name="exceptions",
        build_text=lambda p: (
            f"Exceptions to the {p['name']}: {p['exceptions']}"
            if p.get("exceptions")
            else None
        ),
    ),
]


def faq_concerns(policy: dict) -> list[ConcernSpec]:
    """FAQs are one-per-entry rather than one-per-policy, built per policy
    and merged in below — the same pattern examples/shopbot/run.py uses.
    """
    specs = []
    for i, faq in enumerate(policy.get("faqs", [])):
        specs.append(
            ConcernSpec(
                name=f"faq_{i}",
                build_text=lambda p, faq=faq: (
                    f"Question about the {p['name']}: {faq['question']} "
                    f"Answer: {faq['answer']}"
                ),
            )
        )
    return specs


def build_store(embedder: OpenAIEmbedder) -> InMemoryStore:
    store = InMemoryStore()
    for policy in POLICIES:
        chunker = ConcernChunker(CONCERNS + faq_concerns(policy))
        chunks = chunker.chunk(policy)
        vectors = embedder.embed_documents([c.text for c in chunks])
        store.add(chunks, vectors)
        print(f"  {policy['name']}: {len(chunks)} chunks")
    return store


def main() -> None:
    embedder = OpenAIEmbedder()
    print("Ingesting policy handbook...")
    store = build_store(embedder)

    # See examples/shopbot/run.py's comment on why 0.75 is not reused as a
    # default here: each catalog's real embedding-score distribution
    # differs, so the threshold is calibrated against labeled queries.
    # k=5 rather than the book's k=3: each policy here has up to 5 chunks
    # (summary/eligibility/procedure/exceptions/faq), all plausibly close
    # in score for a query about that policy, so k=3 was cutting off the
    # summary chunk in favour of narrower ones — verified against real
    # embedding scores, not assumed.
    retriever = Retriever(embedder=embedder, store=store, k=5)
    labeled_queries = [
        ("How many weeks of parental leave do I get?", True),
        ("Can I expense a business dinner with alcohol?", True),
        ("How much PTO do I accrue per year?", True),
        ("What's the company policy on pet insurance?", False),
        ("What's the weather like today?", False),
    ]
    retriever.threshold = calibrate_threshold(retriever, labeled_queries)
    print(f"Calibrated similarity threshold: {retriever.threshold:.4f}")

    prompt = GroundingPrompt(
        persona=(
            "You are PolicyBot, an internal HR assistant for Nimbus Labs. "
            "You help employees understand company policy."
        ),
        refusal_message=(
            "I don't have policy information that specifically answers "
            "that. Please reach out to hr@nimbuslabs.example for help."
        ),
    )
    answerer = GroundedAnswerer(retriever=retriever, llm=OpenAIChat(), prompt=prompt)

    test_queries = [
        "How many weeks of parental leave do I get?",
        "Can I expense a business dinner with alcohol?",
        "What's the company policy on pet insurance?",  # not covered — should refuse
    ]
    for query in test_queries:
        print(f"\nEmployee: {query}")
        print(f"PolicyBot: {answerer.answer(query)}")


if __name__ == "__main__":
    main()
