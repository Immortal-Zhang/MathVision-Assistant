"""A lightweight multimodal-index facade.

The default implementation intentionally uses text metadata only, because it
must run cleanly on Mac without FAISS or CUDA-only dependencies. The interface
is kept separate so image embeddings can be added later.
"""

from __future__ import annotations

from typing import Any

from mathvision.retrieval.text_index import TextRetriever


class MultiModalRetriever(TextRetriever):
    """Text-backed retriever with a multimodal-ready name."""

    def build(self, documents: list[dict[str, Any]]) -> None:
        super().build(documents)
