#!/usr/bin/env python3
"""Run end-to-end evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathvision.evaluation.evaluator import Evaluator
from mathvision.io_utils import read_jsonl
from mathvision.rag.pipeline import MathVisionRAGPipeline
from mathvision.retrieval.text_index import TextRetriever
from mathvision.vlm.smolvlm_backend import SmolVLMBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 MathVision-Assistant 自动评测")
    parser.add_argument("--backend", default="mock", choices=["mock", "smolvlm", "qwen-vl"])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--qa_file", default="data/demo/qa.jsonl")
    parser.add_argument("--kb_file", default="data/demo/knowledge_base.jsonl")
    parser.add_argument("--out_dir", default="reports")
    parser.add_argument(
        "--model_name",
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="真实模型名称，仅 smolvlm backend 使用",
    )
    parser.add_argument(
        "--lora_adapter",
        default=None,
        help="LoRA adapter 目录，仅 smolvlm backend 使用",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = TextRetriever()
    retriever.build(read_jsonl(args.kb_file))
    backend = args.backend
    if args.backend == "smolvlm" and args.lora_adapter:
        backend = SmolVLMBackend(
            model_name=args.model_name,
            lora_adapter=args.lora_adapter,
        )
    pipeline = MathVisionRAGPipeline(backend=backend, retriever=retriever)
    evaluator = Evaluator(pipeline=pipeline, top_k=args.top_k)
    result = evaluator.evaluate(args.qa_file, args.out_dir)
    print("评测完成：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
