from __future__ import annotations

from pathlib import Path

from PIL import Image

from mathvision.rag.pipeline import MathVisionRAGPipeline
from mathvision.retrieval.text_index import TextRetriever


def test_pipeline_with_mock_backend(tmp_path: Path) -> None:
    image_path = tmp_path / "line_slope_01.png"
    Image.new("RGB", (64, 64), color="white").save(image_path)

    retriever = TextRetriever()
    retriever.build(
        [
            {
                "id": "kb_line_slope_01",
                "title": "Line slope y=2x+1",
                "content": "The slope of y=2x+1 is 2.",
                "source_image": str(image_path),
            }
        ]
    )
    pipeline = MathVisionRAGPipeline(backend="mock", retriever=retriever)
    result = pipeline.answer(
        image_path=str(image_path),
        question="What is the slope of the line?",
        top_k=1,
    )
    assert result["backend"] == "mock"
    assert "答案" in result["answer"]
    assert "2" in result["answer"]
    assert len(result["evidence"]) == 1
