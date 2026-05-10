#!/usr/bin/env python3
"""Evaluate Qwen2.5-VL base model or a LoRA adapter on demo QA samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathvision.evaluation.metrics import extract_numbers, keyword_coverage, normalize_text
from mathvision.io_utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 Qwen2.5-VL / Qwen2.5-VL LoRA")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter_dir", default=None, help="LoRA adapter 目录；为空则评测基座模型")
    parser.add_argument("--qa_file", default="data/demo/qa_test.jsonl")
    parser.add_argument("--out_dir", default="reports/qwen_lora_eval")
    parser.add_argument("--limit_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument(
        "--prompt_style",
        choices=["plain", "answer_then_reason"],
        default="answer_then_reason",
    )
    parser.add_argument("--min_answer_length_warning", type=int, default=10)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    return parser.parse_args()


def _import_deps() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "评测依赖缺失。请安装 transformers、accelerate、peft、qwen-vl-utils；"
            "不要安装 flash-attn，也不要重新安装服务器已有的 torch。"
        ) from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "process_vision_info": process_vision_info,
        "AutoProcessor": AutoProcessor,
        "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLForConditionalGeneration,
    }


def _load_model(
    model_cls: Any,
    model_name: str,
    torch_module: Any,
    bf16: bool,
    attn_implementation: str,
) -> Any:
    dtype = torch_module.bfloat16 if bf16 else torch_module.float16
    base_kwargs = {"device_map": "auto", "trust_remote_code": True}

    def attempt(attention: str) -> Any:
        kwargs = {**base_kwargs, "attn_implementation": attention}
        try:
            return model_cls.from_pretrained(model_name, dtype=dtype, **kwargs)
        except TypeError:
            return model_cls.from_pretrained(model_name, torch_dtype=dtype, **kwargs)

    try:
        return attempt(attn_implementation)
    except Exception:
        if attn_implementation == "eager":
            raise
        print(
            f"[warn] 使用 attention={attn_implementation} 加载失败，自动退回 eager。",
            flush=True,
        )
        return attempt("eager")


def _abs_image_path(value: str) -> str:
    if value.startswith(("http://", "https://", "file://")):
        return value
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def _generate(
    record: dict[str, Any],
    model: Any,
    processor: Any,
    process_vision_info: Any,
    torch_module: Any,
    max_new_tokens: int,
    prompt_style: str,
) -> str:
    image_path = _abs_image_path(str(record["image"]))
    question = _build_eval_question(str(record["question"]), prompt_style)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = "cuda" if torch_module.cuda.is_available() else "cpu"
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch_module.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    decoded = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip() if decoded else ""


def _build_eval_question(question: str, prompt_style: str) -> str:
    if prompt_style == "plain":
        return question
    return (
        f"{question}\n"
        "请先给出答案，再用一句话说明依据。回答尽量简洁，但不要只输出一个数字或一个词。"
    )


def _strict_exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def _first_numeric_match(prediction: str, reference: str, tol: float = 1e-3) -> float:
    pred_numbers = extract_numbers(prediction)
    ref_numbers = extract_numbers(reference)
    if not pred_numbers or not ref_numbers:
        return 0.0
    return 1.0 if abs(pred_numbers[0] - ref_numbers[0]) <= tol else 0.0


def _write_markdown(out_file: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Qwen2.5-VL LoRA Demo Evaluation",
        "",
        "本报告用于小规模功能性评测，不等同于正式 benchmark。",
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
            "## Samples",
            "",
            "| id | keyword_hit | exact_match | numeric_match | too_short | answer_length |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['id']} | {float(row['keyword_hit']):.4f} | "
            f"{float(row['exact_match']):.4f} | {float(row['numeric_match']):.4f} | "
            f"{row['too_short']} | {int(row['answer_length'])} |"
        )
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.attn_implementation == "flash_attention_2":
        raise ValueError("本项目服务器流程不安装 flash-attn，请使用 sdpa 或 eager。")

    deps = _import_deps()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU。Qwen2.5-VL 评测应在服务器上运行。")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.qa_file)
    if args.limit_samples > 0:
        records = records[: args.limit_samples]

    processor = deps["AutoProcessor"].from_pretrained(args.model_name, trust_remote_code=True)
    model = _load_model(
        deps["Qwen2_5_VLForConditionalGeneration"],
        args.model_name,
        torch,
        bf16=args.bf16,
        attn_implementation=args.attn_implementation,
    )
    if args.adapter_dir:
        model = deps["PeftModel"].from_pretrained(model, args.adapter_dir)
    model.eval()

    rows: list[dict[str, Any]] = []
    for record in records:
        started = time.perf_counter()
        answer = _generate(
            record=record,
            model=model,
            processor=processor,
            process_vision_info=deps["process_vision_info"],
            torch_module=torch,
            max_new_tokens=args.max_new_tokens,
            prompt_style=args.prompt_style,
        )
        latency = time.perf_counter() - started
        keywords = [str(item) for item in record.get("keywords", [])]
        hit = keyword_coverage(answer, keywords)
        reference_answer = str(record.get("answer", ""))
        answer_length = len(answer)
        exact = _strict_exact_match(answer, reference_answer)
        numeric = _first_numeric_match(answer, reference_answer)
        too_short = answer_length < args.min_answer_length_warning
        rows.append(
            {
                "id": record.get("id", ""),
                "image_path": record.get("image", ""),
                "question": record.get("question", ""),
                "reference_answer": reference_answer,
                "model_answer": answer,
                "keyword_hit": hit,
                "too_short": too_short,
                "exact_match": exact,
                "numeric_match": numeric,
                "answer_length": answer_length,
                "latency_seconds": latency,
            }
        )

    summary = {
        "num_samples": len(rows),
        "keyword_coverage": sum(float(row["keyword_hit"]) for row in rows) / len(rows)
        if rows
        else 0.0,
        "non_empty_rate": sum(1 for row in rows if str(row["model_answer"]).strip()) / len(rows)
        if rows
        else 0.0,
        "too_short_rate": sum(1 for row in rows if bool(row["too_short"])) / len(rows)
        if rows
        else 0.0,
        "exact_match": sum(float(row["exact_match"]) for row in rows) / len(rows)
        if rows
        else 0.0,
        "numeric_match": sum(float(row["numeric_match"]) for row in rows) / len(rows)
        if rows
        else 0.0,
        "average_answer_length": sum(int(row["answer_length"]) for row in rows) / len(rows)
        if rows
        else 0.0,
        "average_latency_seconds": sum(float(row["latency_seconds"]) for row in rows) / len(rows)
        if rows
        else 0.0,
        "adapter_dir": args.adapter_dir or "",
        "qa_file": args.qa_file,
        "prompt_style": args.prompt_style,
        "max_new_tokens": args.max_new_tokens,
    }

    csv_path = out_dir / "eval_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = out_dir / "summary.md"
    _write_markdown(markdown_path, summary, rows)
    print(
        json.dumps(
            {
                "csv_path": str(csv_path),
                "summary_path": str(summary_path),
                "markdown_path": str(markdown_path),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
