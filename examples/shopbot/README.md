# ShopBot, on pramana

This is *RAG Essentials Book 1*'s ShopBot — a product Q&A assistant for a
fashion e-commerce catalog — rebuilt on the generic `pramana` library
instead of the book's hand-written, chapter-by-chapter code.

It exists to prove the library's abstractions actually hold for the domain
they were generalized from, and to give you a concrete starting point for
your own domain: swap `products.py` and the `CONCERNS` list in `run.py` for
your own documents and question categories, and the rest — threshold-gated
retrieval, the locked grounding prompt, refusal-on-no-evidence — carries
over unchanged.

## Run it

```bash
pip install pramana-rag[openai]
export OPENAI_API_KEY=sk-...
python run.py
```

## What's real vs. reconstructed

`products.py` uses the exact structured data the book gives for the cotton
kurta and silk saree (Chapter 5), the exact flat-catalog fields the book
gives for the woolen shawl and linen co-ord set (Chapter 4), and the one
verified anarkali FAQ that appears in the book's Chapter 9 test cases. It
does not invent care instructions, policies, or FAQs the book never
specified for those three products — some products in this example
therefore have fewer chunk types than others, which is also a correct
demonstration of `ConcernChunker`: a concern with no data produces no
chunk, rather than a fabricated one.
