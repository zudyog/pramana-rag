# PolicyBot, on pramana

A second example, deliberately **not** e-commerce (see `examples/shopbot/`
for the first) — an internal HR policy assistant for a fictional company,
Nimbus Labs. It exists to demonstrate that pramana's abstractions aren't
secretly fashion-retail-shaped: this example's concerns (`summary`,
`eligibility`, `procedure`, `exceptions`, `faq`) share no vocabulary with
ShopBot's (`identity`, `variants`, `policy`, `care`, `faq`), yet the wiring
in `run.py` is structurally identical — define concerns, chunk, embed,
store, retrieve above threshold, answer only from what was retrieved.

All company/policy data here is invented for this example. It is not a
template for real HR policy and should not be treated as legal or HR
guidance.

## What it shows

- **Concerns are yours to define** — nothing in `src/pramana/` knows what
  "eligibility" or "procedure" means; you supply that shape.
- **Missing concerns are skipped, not fabricated** — the PTO policy has no
  `exceptions` or `faqs` field, so `ConcernChunker` simply produces fewer
  chunks for it. Nothing is invented to fill the gap.
- **Refusal on out-of-scope questions** — the third test query
  ("pet insurance") isn't covered by any policy, so `GroundedAnswerer`
  should return the refusal message rather than guessing.

## Run it

```bash
pip install pramana-rag[openai] python-dotenv
export OPENAI_API_KEY=sk-...   # or put it in a .env file in this directory or the repo root
python run.py
```

Run end-to-end against real OpenAI embeddings and chat completions. Uses
`k=5` rather than the shopbot example's `k=3` — with only 3-5 chunks per
policy, the chunk that actually answered "how many weeks of parental
leave" ranked 4th in real similarity scores, behind three related-but-
non-answering chunks from the same policy, so `k=3` cut it off. Verified
against the real scores, not assumed; see the comment in `run.py`.
