#!/usr/bin/env python3
"""Compare Qwen base and LoRA evaluation CSV files and export bad cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 Qwen LoRA 退化样本")
    parser.add_argument("--base_csv", required=True)
    parser.add_argument("--lora_csv", required=True)
    parser.add_argument("--out_dir", default="reports/qwen_bad_cases")
    return parser.parse_args()


def _read_csv(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return {str(row.get("id", "")): row for row in reader if row.get("id")}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _possible_reasons(base: dict[str, str], lora: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    base_keyword = _as_float(base.get("keyword_hit"))
    lora_keyword = _as_float(lora.get("keyword_hit"))
    base_numeric = _as_float(base.get("numeric_match"))
    lora_numeric = _as_float(lora.get("numeric_match"))
    base_len = _as_float(base.get("answer_length"))
    lora_len = _as_float(lora.get("answer_length"))

    if _as_bool(lora.get("too_short")):
        reasons.append("LoRA 输出过短")
    if base_keyword > lora_keyword:
        reasons.append("LoRA 漏掉关键词")
    if base_numeric >= 1.0 and lora_numeric < 1.0:
        reasons.append("LoRA 数值匹配失败")
    if base_len > 0 and lora_len < max(10.0, base_len * 0.5):
        reasons.append("LoRA 回答长度显著短于 base")
    return reasons


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# Qwen LoRA Bad Case Analysis",
        "",
        "本报告基于本地合成 demo 数据的功能性评测输出生成，不等同于正式 benchmark。",
        "",
        "| id | possible_reason | base_keyword_hit | lora_keyword_hit |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['possible_reason']} | "
            f"{float(row['base_keyword_hit']):.4f} | {float(row['lora_keyword_hit']):.4f} |"
        )
    if not rows:
        lines.append("| - | 未发现符合规则的 bad case | 0.0000 | 0.0000 |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = _read_csv(args.base_csv)
    lora_rows = _read_csv(args.lora_csv)
    bad_cases: list[dict[str, str]] = []
    for sample_id, lora in lora_rows.items():
        base = base_rows.get(sample_id)
        if not base:
            continue
        reasons = _possible_reasons(base, lora)
        if not reasons:
            continue
        bad_cases.append(
            {
                "id": sample_id,
                "question": lora.get("question", ""),
                "reference_answer": lora.get("reference_answer", ""),
                "base_answer": base.get("model_answer", ""),
                "lora_answer": lora.get("model_answer", ""),
                "base_keyword_hit": str(_as_float(base.get("keyword_hit"))),
                "lora_keyword_hit": str(_as_float(lora.get("keyword_hit"))),
                "possible_reason": "；".join(reasons),
            }
        )

    csv_path = out_dir / "bad_cases.csv"
    fieldnames = [
        "id",
        "question",
        "reference_answer",
        "base_answer",
        "lora_answer",
        "base_keyword_hit",
        "lora_keyword_hit",
        "possible_reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bad_cases)

    md_path = out_dir / "bad_cases.md"
    _write_markdown(md_path, bad_cases)
    print(f"bad case CSV 已保存：{csv_path}")
    print(f"bad case Markdown 已保存：{md_path}")
    print(f"bad case 数量：{len(bad_cases)}")


if __name__ == "__main__":
    main()
