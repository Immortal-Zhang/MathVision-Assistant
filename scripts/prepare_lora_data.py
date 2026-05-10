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
    parser.add_argument(
        "--qa_file",
        default="data/demo/qa_train.jsonl",
        help="输入 QA JSONL，默认只使用 train split，避免评测数据泄漏。",
    )
    parser.add_argument(
        "--out_file",
        "--output_file",
        dest="out_file",
        default="data/outputs/lora_qwen_vl_train.jsonl",
        help="输出 LoRA 训练 JSONL。--output_file 作为旧参数别名保留。",
    )
    parser.add_argument(
        "--answer_style",
        choices=["short", "explain"],
        default="explain",
        help="assistant 答案格式。explain 会生成“答案 + 一句话依据”。",
    )
    parser.add_argument(
        "--knowledge_file",
        default="data/demo/knowledge_base.jsonl",
        help="用于构造依据的知识库 JSONL。",
    )
    return parser.parse_args()


def load_knowledge(knowledge_file: str | Path) -> dict[str, str]:
    path = Path(knowledge_file)
    if not path.exists():
        return {}
    records = read_jsonl(path)
    return {str(record.get("id", "")): str(record.get("content", "")) for record in records}


def build_basis(record: dict[str, object], knowledge_by_id: dict[str, str]) -> str:
    related_ids = [str(item) for item in record.get("related_knowledge_ids", [])]  # type: ignore[arg-type]
    evidence = [knowledge_by_id[item] for item in related_ids if knowledge_by_id.get(item)]
    if evidence:
        return " ".join(evidence[:2])

    keywords = [str(item) for item in record.get("keywords", [])]  # type: ignore[arg-type]
    if keywords:
        return f"该题需要识别关键词：{', '.join(keywords)}。"
    return "根据图像和题目信息可得出该答案。"


def build_assistant_text(
    record: dict[str, object], answer_style: str, knowledge_by_id: dict[str, str]
) -> str:
    answer = str(record["answer"])
    if answer_style == "short":
        return answer
    basis = build_basis(record, knowledge_by_id)
    return f"答案：{answer}\n依据：{basis}"


def convert_record(
    record: dict[str, object], answer_style: str, knowledge_by_id: dict[str, str]
) -> dict[str, object]:
    image_path = str(record["image"])
    question = str(record["question"])
    assistant_text = build_assistant_text(record, answer_style, knowledge_by_id)
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
                "content": [{"type": "text", "text": assistant_text}],
            },
        ],
    }


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.qa_file)
    test_records = [record for record in records if str(record.get("split", "")) == "test"]
    if test_records:
        raise ValueError(
            f"输入文件包含 {len(test_records)} 条 test split 样本。LoRA 训练数据不能包含测试集。"
        )
    knowledge_by_id = load_knowledge(args.knowledge_file)
    output_path = ensure_parent(args.out_file)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            converted = convert_record(record, args.answer_style, knowledge_by_id)
            file.write(json.dumps(converted, ensure_ascii=False) + "\n")
    print(
        f"LoRA 数据已生成：{output_path}，样本数：{len(records)}，"
        f"answer_style={args.answer_style}"
    )


if __name__ == "__main__":
    main()
