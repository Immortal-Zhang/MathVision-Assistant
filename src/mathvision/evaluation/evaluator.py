"""End-to-end evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from mathvision.evaluation.metrics import (
    exact_match,
    keyword_coverage,
    numeric_match,
    retrieval_recall_at_k,
)
from mathvision.io_utils import read_jsonl, write_text
from mathvision.rag.pipeline import MathVisionRAGPipeline


class Evaluator:
    """Run QA evaluation with a RAG pipeline."""

    def __init__(self, pipeline: MathVisionRAGPipeline, top_k: int = 3) -> None:
        self.pipeline = pipeline
        self.top_k = top_k

    def evaluate(self, qa_file: str | Path, out_dir: str | Path = "reports") -> dict[str, Any]:
        """Evaluate all records and save CSV plus Markdown summary."""

        records = read_jsonl(qa_file)
        rows: list[dict[str, Any]] = []
        for item in tqdm(records, desc="Evaluating", unit="sample"):
            result = self.pipeline.answer(
                image_path=str(item["image"]),
                question=str(item["question"]),
                top_k=self.top_k,
            )
            retrieved_ids = [str(evidence["id"]) for evidence in result["evidence"]]
            row = {
                "id": item.get("id"),
                "image": item.get("image"),
                "question": item.get("question"),
                "reference_answer": item.get("answer"),
                "prediction": result["answer"],
                "exact_match": exact_match(result["answer"], str(item.get("answer", ""))),
                "numeric_match": numeric_match(result["answer"], str(item.get("answer", ""))),
                "keyword_coverage": keyword_coverage(
                    result["answer"], list(item.get("keywords", []))
                ),
                "retrieval_recall_at_k": retrieval_recall_at_k(
                    retrieved_ids,
                    [str(x) for x in item.get("related_knowledge_ids", [])],
                    self.top_k,
                ),
                "latency_seconds": result["latency_seconds"],
                "retrieved_ids": ",".join(retrieved_ids),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / "eval_results.csv"
        df.to_csv(csv_path, index=False)

        summary = {
            "num_samples": int(len(df)),
            "exact_match": float(df["exact_match"].mean()) if len(df) else 0.0,
            "numeric_match": float(df["numeric_match"].mean()) if len(df) else 0.0,
            "keyword_coverage": float(df["keyword_coverage"].mean()) if len(df) else 0.0,
            "retrieval_recall_at_k": float(df["retrieval_recall_at_k"].mean())
            if len(df)
            else 0.0,
            "average_latency": float(df["latency_seconds"].mean()) if len(df) else 0.0,
        }
        md_path = out_path / "eval_summary.md"
        write_text(md_path, self._build_markdown_summary(summary, csv_path))
        return {"summary": summary, "csv_path": str(csv_path), "summary_path": str(md_path)}

    def _build_markdown_summary(self, summary: dict[str, Any], csv_path: Path) -> str:
        lines = [
            "# MathVision-Assistant 评测摘要",
            "",
            "本报告由 `scripts/run_eval.py` 自动生成，默认评测本地合成 demo 数据。",
            "",
            "| metric | value |",
            "|---|---:|",
        ]
        for key, value in summary.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.4f} |")
            else:
                lines.append(f"| {key} | {value} |")
        lines.extend(
            [
                "",
                f"逐样本结果已保存到 `{csv_path}`。",
                "",
                "说明：mock backend 是规则型本地后端，用于验证工程链路；真实模型结果请使用 SmolVLM 或 Qwen2.5-VL 重新评测。",
            ]
        )
        return "\n".join(lines) + "\n"
