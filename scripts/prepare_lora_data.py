#!/usr/bin/env python3
"""Convert QA JSONL into a Qwen-VL style supervised fine-tuning JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathvision.io_utils import ensure_parent, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 Qwen2.5-VL LoRA 微调数据")
    parser.add_argument("--qa_file", default="data/demo/qa.jsonl")
    parser.add_argument("--output_file", default="data/outputs/lora_qwen_vl.jsonl")
    return parser.parse_args()


def convert_record(record: dict[str, object]) -> dict[str, object]:
    image_path = str(record["image"])
    question = str(record["question"])
    answer = str(record["answer"])
    return {
        "id": record.get("id", ""),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
    }


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.qa_file)
    output_path = ensure_parent(args.output_file)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(convert_record(record), ensure_ascii=False) + "\n")
    print(f"LoRA 数据已生成：{output_path}，样本数：{len(records)}")


if __name__ == "__main__":
    main()
