#!/usr/bin/env python3
"""Build a TF-IDF retrieval index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathvision.io_utils import read_jsonl
from mathvision.retrieval.text_index import TextRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建本地 TF-IDF 检索索引")
    parser.add_argument("--kb_file", default="data/demo/knowledge_base.jsonl")
    parser.add_argument("--index_file", default="data/outputs/index.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = read_jsonl(args.kb_file)
    retriever = TextRetriever()
    retriever.build(documents)
    retriever.save(args.index_file)
    print(f"索引构建完成：{args.index_file}，文档数：{len(documents)}")


if __name__ == "__main__":
    main()
