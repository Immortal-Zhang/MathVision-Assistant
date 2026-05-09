#!/usr/bin/env python3
"""Offline smoke test for the whole local pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathvision.data.synthetic import generate_demo_dataset
from mathvision.evaluation.evaluator import Evaluator
from mathvision.io_utils import read_jsonl
from mathvision.rag.pipeline import MathVisionRAGPipeline
from mathvision.retrieval.text_index import TextRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 smoke test：数据、索引、pipeline、评测")
    parser.add_argument("--backend", default="mock", choices=["mock", "smolvlm", "qwen-vl"])
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa_file = Path("data/demo/qa.jsonl")
    kb_file = Path("data/demo/knowledge_base.jsonl")
    if not qa_file.exists() or not kb_file.exists():
        print("未发现 demo 数据，正在生成...")
        generate_demo_dataset("data/demo")

    print("正在构建检索索引...")
    retriever = TextRetriever()
    retriever.build(read_jsonl(kb_file))
    retriever.save("data/outputs/index.pkl")

    print("正在运行单条问答...")
    qa_records = read_jsonl(qa_file)
    first = qa_records[0]
    pipeline = MathVisionRAGPipeline(backend=args.backend, retriever=retriever)
    result = pipeline.answer(first["image"], first["question"], top_k=args.top_k)
    if not result["answer"] or not result["evidence"]:
        raise RuntimeError("pipeline 输出不完整")
    print(result["answer"])

    print("正在运行完整评测...")
    evaluator = Evaluator(pipeline=pipeline, top_k=args.top_k)
    eval_result = evaluator.evaluate(qa_file, "reports")
    summary = eval_result["summary"]

    print("Smoke test 成功完成。")
    print(f"- backend: {args.backend}")
    print(f"- 样本数: {summary['num_samples']}")
    print(f"- exact_match: {summary['exact_match']:.4f}")
    print(f"- retrieval_recall_at_k: {summary['retrieval_recall_at_k']:.4f}")
    print("- 输出文件: reports/eval_results.csv, reports/eval_summary.md")


if __name__ == "__main__":
    main()
