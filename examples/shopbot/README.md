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
pip install pramana-rag[openai] python-dotenv
export OPENAI_API_KEY=sk-...   # or put it in a .env file in this directory or the repo root
python run.py
```

This has been run end-to-end against real OpenAI embeddings and chat
completions — not just the offline fake-embedder tests. It calibrates its
own similarity threshold with `calibrate_threshold()` rather than reusing
the book's `0.75`: the book's number was tuned for the book's exact chunk
text, and a different (even similar) catalog produces a different real
score distribution — reusing 0.75 here actually caused two of the three
demo queries to wrongly refuse until it was calibrated against this
catalog's own scores.

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
