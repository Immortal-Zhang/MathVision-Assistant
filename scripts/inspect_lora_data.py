#!/usr/bin/env python3
"""Inspect Qwen-VL LoRA training JSONL before launching GPU training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 LoRA 训练数据的问答格式")
    parser.add_argument("--train_file", default="data/outputs/lora_qwen_vl_train.jsonl")
    parser.add_argument("--num_examples", type=int, default=3)
    return parser.parse_args()


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return ""


def _extract_roles(record: dict[str, Any]) -> tuple[str, str]:
    question = ""
    answer = ""
    for message in record.get("messages", []):
        role = message.get("role")
        text = _extract_text(message.get("content", []))
        if role == "user" and not question:
            question = text
        if role == "assistant" and not answer:
            answer = text
    return question, answer


def main() -> None:
    args = parse_args()
    path = Path(args.train_file)
    if not path.exists():
        raise FileNotFoundError(f"训练文件不存在：{path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    answers: list[str] = []
    print(f"训练文件：{path}")
    print(f"样本数：{len(records)}")
    for index, record in enumerate(records[: args.num_examples], start=1):
        question, answer = _extract_roles(record)
        answers.append(answer)
        print(f"\n[样本 {index}]")
        print(f"User question: {question}")
        print(f"Assistant answer: {answer}")

    if len(records) > args.num_examples:
        for record in records[args.num_examples :]:
            _, answer = _extract_roles(record)
            answers.append(answer)

    lengths = [len(answer) for answer in answers]
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
    else:
        avg_len = 0.0
        min_len = 0
        max_len = 0
    too_short_count = sum(1 for length in lengths if length < 20)
    print("\n长度统计：")
    print(f"- average_length: {avg_len:.2f}")
    print(f"- min_length: {min_len}")
    print(f"- max_length: {max_len}")
    print(f"- too_short_count(<20 chars): {too_short_count}")
    if avg_len < 20:
        print("警告：assistant 平均答案长度小于 20 个字符，可能仍然过短。")


if __name__ == "__main__":
    main()
