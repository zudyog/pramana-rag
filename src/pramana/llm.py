"""Chat completion. Pluggable — Pramana ships one default provider."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Generate a response given a system prompt and a user message."""


class OpenAIChat(LLM):
    """Wraps OpenAI's chat completions API. Requires the `openai` package
    (`pip install pramana-rag[openai]`) and an `OPENAI_API_KEY`.

    Defaults to temperature 0. A grounding contract relies on the model
    consistently choosing its highest-probability path — the faithful
    rendering of retrieved evidence — rather than occasionally sampling a
    plausible-sounding detail that isn't in the context. Non-zero
    temperature doesn't break anything outright, but it reintroduces the
    randomness the grounding prompt is trying to remove, so it only warns.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        client=None,
    ):
        if temperature != 0.0:
            warnings.warn(
                "OpenAIChat temperature != 0: the grounding contract is "
                "validated at temperature=0, where the same input always "
                "produces the same output. Non-zero temperature "
                "reintroduces a small chance of fabricated detail.",
                stacklevel=2,
            )
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "OpenAIChat requires the 'openai' package. "
                    "Install with: pip install pramana-rag[openai]"
                ) from exc
            client = OpenAI()
        self._client = client
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content
