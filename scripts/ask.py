#!/usr/bin/env python3
"""Single-image question answering CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathvision.io_utils import read_jsonl
from mathvision.rag.pipeline import MathVisionRAGPipeline
from mathvision.retrieval.text_index import TextRetriever
from mathvision.vlm.smolvlm_backend import SmolVLMBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MathVision-Assistant 单条问答")
    parser.add_argument("--image", required=True, help="本地图片路径")
    parser.add_argument("--question", required=True, help="问题")
    parser.add_argument("--backend", default="mock", choices=["mock", "smolvlm", "qwen-vl"])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--kb_file", default="data/demo/knowledge_base.jsonl")
    parser.add_argument("--index_file", default="data/outputs/index.pkl")
    parser.add_argument(
        "--model_name",
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="真实模型名称，仅 smolvlm backend 使用",
    )
    parser.add_argument(
        "--lora_adapter",
        default=None,
        help="LoRA adapter 目录，仅 smolvlm backend 使用，例如 checkpoints/smolvlm500m-lora-mathvision-debug",
    )
    return parser.parse_args()


def load_or_build_retriever(kb_file: str, index_file: str) -> TextRetriever:
    index_path = Path(index_file)
    if index_path.exists():
        return TextRetriever.load(index_path)
    documents = read_jsonl(kb_file)
    retriever = TextRetriever()
    retriever.build(documents)
    retriever.save(index_path)
    return retriever


def main() -> None:
    args = parse_args()
    retriever = load_or_build_retriever(args.kb_file, args.index_file)
    backend = args.backend
    if args.backend == "smolvlm" and args.lora_adapter:
        backend = SmolVLMBackend(
            model_name=args.model_name,
            lora_adapter=args.lora_adapter,
        )
    pipeline = MathVisionRAGPipeline(backend=backend, retriever=retriever)
    result = pipeline.answer(args.image, args.question, top_k=args.top_k)
    payload = {
        "backend": result["backend"],
        "model_name": args.model_name if args.backend == "smolvlm" else None,
        "lora_adapter": args.lora_adapter,
        "question": result["question"],
        "image_path": result["image_path"],
        "answer": result["answer"],
        "evidence": result["evidence"],
        "latency_seconds": result["latency_seconds"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
