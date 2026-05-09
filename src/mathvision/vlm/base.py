"""Base interface for all visual-language model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class VLMBackend(ABC):
    """Abstract VLM backend."""

    name: str

    @abstractmethod
    def generate(
        self,
        image_path: str,
        question: str,
        context: str | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate an answer from a local image path and question."""
