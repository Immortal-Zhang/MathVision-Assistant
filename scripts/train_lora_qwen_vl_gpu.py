#!/usr/bin/env python3
"""Train a Qwen2.5-VL LoRA adapter on a CUDA server.

This script is designed for the RTX 5090 server workflow. It is not used by
local Mac smoke tests and should not be run without a CUDA GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL LoRA 微调")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--train_file", default="data/outputs/lora_qwen_vl_train.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/qwen25vl-lora-mathvision")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--limit_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="默认使用 sdpa；本项目不安装 flash-attn。",
    )
    return parser.parse_args()


def _import_training_deps() -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from qwen_vl_utils import process_vision_info
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "训练依赖缺失。请在服务器环境安装 transformers、accelerate、datasets、"
            "peft、trl、qwen-vl-utils 等依赖；不要重新安装 torch，也不要安装 flash-attn。"
        ) from exc

    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "process_vision_info": process_vision_info,
        "AutoProcessor": AutoProcessor,
        "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLForConditionalGeneration,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _json_dump(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _abs_image_path(value: str) -> str:
    if value.startswith(("http://", "https://", "file://")):
        return value
    image_path = Path(value)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    return str(image_path.resolve())


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", [])
        new_content: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                updated = dict(item)
                updated["image"] = _abs_image_path(str(updated["image"]))
                new_content.append(updated)
            elif isinstance(item, dict):
                new_content.append(dict(item))
        normalized.append({"role": message.get("role", "user"), "content": new_content})
    return normalized


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


def _choose_lora_targets(model: Any, torch_module: Any) -> list[str]:
    linear_suffixes = sorted(
        {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if isinstance(module, torch_module.nn.Linear)
        }
    )
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj"]
    selected = [name for name in preferred if name in linear_suffixes]
    if not selected:
        fallback = [
            name
            for name in linear_suffixes
            if name not in {"lm_head"} and ("proj" in name or name in {"w1", "w2", "w3"})
        ]
        selected = fallback[:8]
    if not selected:
        raise RuntimeError("未找到可用于 LoRA 的 Linear 模块，请检查模型结构。")
    print(f"可用 Linear 模块后缀：{linear_suffixes}", flush=True)
    print(f"LoRA target_modules：{selected}", flush=True)
    return selected


def main() -> None:
    args = parse_args()
    deps = _import_training_deps()
    torch = deps["torch"]

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU。该脚本只应在 RTX 5090 / CUDA 服务器上运行。")
    if args.attn_implementation == "flash_attention_2":
        raise ValueError("本项目服务器流程不安装 flash-attn，请使用 sdpa 或 eager。")

    train_file = Path(args.train_file)
    if not train_file.exists():
        raise FileNotFoundError(f"训练文件不存在：{train_file}。请先运行 scripts/prepare_lora_data.py。")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = {
        "model_name": args.model_name,
        "train_file": str(train_file),
        "output_dir": str(output_dir),
        "max_steps": args.max_steps,
        "limit_samples": args.limit_samples,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": args.bf16,
        "attn_implementation": args.attn_implementation,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    _json_dump(output_dir / "train_config.json", env_config)

    processor = deps["AutoProcessor"].from_pretrained(args.model_name, trust_remote_code=True)
    model = _load_model(
        deps["Qwen2_5_VLForConditionalGeneration"],
        args.model_name,
        torch,
        bf16=args.bf16,
        attn_implementation=args.attn_implementation,
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    target_modules = _choose_lora_targets(model, torch)
    lora_config = deps["LoraConfig"](
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = deps["get_peft_model"](model, lora_config)
    model.print_trainable_parameters()

    dataset = deps["load_dataset"]("json", data_files=str(train_file), split="train")
    if args.limit_samples > 0:
        dataset = dataset.select(range(min(args.limit_samples, len(dataset))))
    process_vision_info = deps["process_vision_info"]

    def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        all_images: list[Any] = []
        all_videos: list[Any] = []
        for feature in features:
            messages = _normalize_messages(feature["messages"])
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            texts.append(text)
            if image_inputs:
                all_images.extend(image_inputs)
            if video_inputs:
                all_videos.extend(video_inputs)

        batch = processor(
            text=texts,
            images=all_images or None,
            videos=all_videos or None,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        batch["labels"] = labels
        return batch

    training_args = deps["TrainingArguments"](
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=max(1, min(25, args.max_steps)),
        save_total_limit=2,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = deps["Trainer"](
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    _json_dump(output_dir / "train_metrics.json", train_result.metrics)
    _json_dump(output_dir / "train_log.json", trainer.state.log_history)
    print(f"Qwen2.5-VL LoRA adapter 已保存到：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
