"""TF-IDF text retriever for local-friendly RAG."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextRetriever:
    """A small TF-IDF retriever with pickle save/load support."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.matrix: Any | None = None

    def build(self, documents: list[dict[str, Any]]) -> None:
        """Build an index from document dictionaries."""

        if not documents:
            raise ValueError("documents 不能为空，无法构建检索索引")
        normalized_docs: list[dict[str, Any]] = []
        texts: list[str] = []
        for idx, doc in enumerate(documents):
            doc_id = str(doc.get("id", f"doc_{idx}"))
            title = str(doc.get("title", ""))
            content = str(doc.get("content", ""))
            source_image = doc.get("source_image")
            normalized = {
                "id": doc_id,
                "title": title,
                "content": content,
                "source_image": source_image,
            }
            normalized_docs.append(normalized)
            texts.append(f"{title}\n{content}")

        self.documents = normalized_docs
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search documents and return evidence records."""

        if self.matrix is None or not self.documents:
            raise RuntimeError("索引尚未构建，请先调用 build() 或 load()")
        if top_k <= 0:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        order = np.argsort(scores)[::-1][: min(top_k, len(self.documents))]

        results: list[dict[str, Any]] = []
        for doc_idx in order:
            doc = self.documents[int(doc_idx)]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "score": float(scores[int(doc_idx)]),
                    "source_image": doc.get("source_image"),
                }
            )
        return results

    def save(self, path: str | Path) -> None:
        """Save the retriever to a pickle file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(
                {
                    "documents": self.documents,
                    "vectorizer": self.vectorizer,
                    "matrix": self.matrix,
                },
                file,
            )

    @classmethod
    def load(cls, path: str | Path) -> "TextRetriever":
        """Load a retriever from a pickle file."""

        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {input_path}")
        with input_path.open("rb") as file:
            payload = pickle.load(file)

        retriever = cls()
        retriever.documents = payload["documents"]
        retriever.vectorizer = payload["vectorizer"]
        retriever.matrix = payload["matrix"]
        return retriever
