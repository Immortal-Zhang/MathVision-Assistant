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

        qa_path = Path(qa_file)
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
                "task_type": item.get("task_type", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
                "split": item.get("split", "unknown"),
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
        split_name = self._infer_split(records, qa_path)
        by_task_type = self._summarize_by_task_type(df)
        md_path = out_path / "eval_summary.md"
        write_text(
            md_path,
            self._build_markdown_summary(
                summary=summary,
                csv_path=csv_path,
                qa_file=qa_path,
                split_name=split_name,
                by_task_type=by_task_type,
            ),
        )
        return {"summary": summary, "csv_path": str(csv_path), "summary_path": str(md_path)}

    def _build_markdown_summary(
        self,
        summary: dict[str, Any],
        csv_path: Path,
        qa_file: Path,
        split_name: str,
        by_task_type: list[dict[str, Any]],
    ) -> str:
        lines = [
            "# MathVision-Assistant 评测摘要",
            "",
            "本报告由 `scripts/run_eval.py` 自动生成，用于本地合成 demo 数据的原型阶段评测。",
            "",
            f"- qa_file: `{qa_file}`",
            f"- split: `{split_name}`",
            f"- num_samples: {summary['num_samples']}",
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
                "## By task_type",
                "",
                "| task_type | num_samples | exact_match | numeric_match | keyword_coverage | retrieval_recall_at_k | average_latency |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if by_task_type:
            for item in by_task_type:
                lines.append(
                    "| {task_type} | {num_samples} | {exact_match:.4f} | {numeric_match:.4f} | "
                    "{keyword_coverage:.4f} | {retrieval_recall_at_k:.4f} | {average_latency:.4f} |".format(
                        **item
                    )
                )
        else:
            lines.append("| unknown | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |")
        lines.extend(
            [
                "",
                f"逐样本结果已保存到 `{csv_path}`。",
                "",
                "说明：`retrieval_recall_at_k` 是 evidence id 级召回，不等同于最终回答正确率；不同 backend 的结果会有差异，建议结合逐样本 CSV 查看错误样本。",
            ]
        )
        return "\n".join(lines) + "\n"

    def _summarize_by_task_type(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        task_df = df.copy()
        task_df["task_type"] = task_df["task_type"].fillna("unknown")
        grouped = task_df.groupby("task_type", dropna=False)
        rows: list[dict[str, Any]] = []
        for task_type, group in grouped:
            rows.append(
                {
                    "task_type": str(task_type),
                    "num_samples": int(len(group)),
                    "exact_match": float(group["exact_match"].mean()),
                    "numeric_match": float(group["numeric_match"].mean()),
                    "keyword_coverage": float(group["keyword_coverage"].mean()),
                    "retrieval_recall_at_k": float(group["retrieval_recall_at_k"].mean()),
                    "average_latency": float(group["latency_seconds"].mean()),
                }
            )
        return sorted(rows, key=lambda item: str(item["task_type"]))

    def _infer_split(self, records: list[dict[str, Any]], qa_file: Path) -> str:
        splits = {str(item.get("split", "unknown")) for item in records if item.get("split")}
        if len(splits) == 1:
            return next(iter(splits))
        if qa_file.name == "qa.jsonl":
            return "all"
        if not splits:
            return "unknown"
        return "mixed"
