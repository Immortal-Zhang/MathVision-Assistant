"""RAG pipeline for MathVision-Assistant."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

from mathvision.retrieval.text_index import TextRetriever
from mathvision.vlm import build_vlm_backend
from mathvision.vlm.base import VLMBackend


class MathVisionRAGPipeline:
    """Image + question -> retrieval -> VLM answer."""

    def __init__(self, backend: str | VLMBackend, retriever: TextRetriever) -> None:
        self.backend = build_vlm_backend(backend) if isinstance(backend, str) else backend
        self.retriever = retriever

    def answer(self, image_path: str, question: str, top_k: int = 3) -> dict[str, Any]:
        """Answer a question about a local image path."""

        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"图片不存在: {image_file}")
        if not question.strip():
            raise ValueError("question 不能为空")

        start = time.perf_counter()
        try:
            with Image.open(image_file) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"无法读取图片，请确认文件有效: {image_file}") from exc

        evidence = self.retriever.search(question, top_k=top_k)
        context = self._format_context(evidence)
        answer = self.backend.generate(
            image_path=str(image_file),
            question=question,
            context=context,
        )
        latency = time.perf_counter() - start
        return {
            "backend": self.backend.name,
            "backend_name": self.backend.name,
            "question": question,
            "image_path": str(image_file),
            "answer": answer,
            "evidence": evidence,
            "latency": latency,
            "latency_seconds": latency,
        }

    @staticmethod
    def _format_context(evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return ""
        lines: list[str] = []
        for idx, item in enumerate(evidence, start=1):
            title = item.get("title", "")
            content = item.get("content", "")
            lines.append(f"[证据{idx}] {title}: {content}")
        return "\n".join(lines)
