"""The grounding prompt — the final lock.

The retriever decides what evidence reaches the model. This module decides
what the model does with that evidence: answer only from what it was given,
in its own voice, and say so explicitly — using a caller-provided refusal
message — when nothing qualified. Between a threshold-gated retriever and a
locked grounding prompt, there is no entry point left for the model to
extend past verified evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from pramana.llm import LLM
from pramana.retriever import Retriever
from pramana.types import RetrievedChunk

DEFAULT_REFUSAL = (
    "I don't have information that specifically answers that question."
)


def format_chunks(evidence: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a single context block, each one
    labeled by its concern so the model can tell multiple pieces of
    evidence apart rather than reading an undifferentiated wall of text.
    """
    parts = []
    for i, item in enumerate(evidence, 1):
        parts.append(f"[{i} — {item.chunk.concern}]\n{item.chunk.text}")
    return "\n\n---\n\n".join(parts)


@dataclass
class GroundingPrompt:
    """Assembles a system prompt that constrains the model to speak only
    from retrieved context, generalizing the book's tested third-draft
    pattern: a positive instruction for what to do when evidence is
    present, and a principled instruction — with the refusal message
    spelled out verbatim — for when it is not.
    """

    persona: str
    refusal_message: str = DEFAULT_REFUSAL
    extra_instructions: str = ""

    def system_prompt(self, context: str) -> str:
        extra = f"\n{self.extra_instructions}\n" if self.extra_instructions else ""
        return f"""{self.persona}

YOUR ONLY SOURCE OF TRUTH is the information provided below. You have no
other knowledge about this subject.

When you have the information:
- Answer directly and specifically, using only what is provided.
- Answer in natural language — synthesize an answer, don't just transcribe.
{extra}
When you do NOT have the information:
- Say exactly: "{self.refusal_message}"
- Do not guess. Do not approximate. Do not use outside knowledge to fill
  gaps in what was provided.

Information:
{context}
"""


@dataclass
class GroundedAnswerer:
    """Ties a `Retriever` and an `LLM` together behind the grounding
    prompt. `answer()` is the single entry point: retrieve, and either
    generate from the evidence or return the refusal — the model is never
    invoked without evidence to ground it.
    """

    retriever: Retriever
    llm: LLM
    prompt: GroundingPrompt

    def answer(self, query: str, where: dict | None = None) -> str:
        evidence = self.retriever.retrieve(query, where=where)
        if evidence is None:
            return self.prompt.refusal_message
        context = format_chunks(evidence)
        system = self.prompt.system_prompt(context)
        return self.llm.complete(system=system, user=query)
