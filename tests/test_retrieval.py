from __future__ import annotations

from mathvision.retrieval.text_index import TextRetriever


def test_text_retriever_top_k() -> None:
    documents = [
        {
            "id": "kb_line",
            "title": "Line slope",
            "content": "The slope of y=2x+1 is 2.",
            "source_image": None,
        },
        {
            "id": "kb_circle",
            "title": "Circle area",
            "content": "Area equals pi times radius squared.",
            "source_image": None,
        },
        {
            "id": "kb_triangle",
            "title": "Triangle angles",
            "content": "Triangle angles sum to 180 degrees.",
            "source_image": None,
        },
    ]
    retriever = TextRetriever()
    retriever.build(documents)
    results = retriever.search("What is the slope of the line?", top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "kb_line"
    assert "score" in results[0]
