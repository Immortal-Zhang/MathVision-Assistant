#!/usr/bin/env python3
"""Qwen2.5-VL LoRA training script for cloud GPU environments.

This script is intentionally not part of the local smoke test. It expects a
CUDA GPU, recent transformers, peft, trl, datasets, and qwen-vl-utils.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="云 GPU 上运行 Qwen2.5-VL LoRA 微调")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--train_file", default="data/outputs/lora_qwen_vl.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/qwen25vl-lora-mathvision")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=2048)
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

        try:
            import trl  # noqa: F401
        except Exception as exc:
            raise RuntimeError("缺少 trl，请执行 pip install -r requirements-gpu.txt") from exc
    except Exception as exc:
        raise RuntimeError(
            "训练依赖缺失。请在云 GPU 环境执行：\n"
            "pip install -r requirements.txt\n"
            "pip install -r requirements-gpu.txt\n"
            "并确认 transformers 版本支持 Qwen2_5_VLForConditionalGeneration。"
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


def main() -> None:
    args = parse_args()
    deps = _import_training_deps()
    torch = deps["torch"]

    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到 CUDA GPU。该脚本用于云 GPU 进阶训练，不建议在 Mac 本地运行。"
        )
    if not Path(args.train_file).exists():
        raise FileNotFoundError(
            f"训练文件不存在：{args.train_file}。请先运行 python scripts/prepare_lora_data.py"
        )

    processor = deps["AutoProcessor"].from_pretrained(
        args.model_name, trust_remote_code=True
    )
    model = deps["Qwen2_5_VLForConditionalGeneration"].from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = deps["LoraConfig"](
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = deps["get_peft_model"](model, lora_config)
    model.print_trainable_parameters()

    dataset = deps["load_dataset"]("json", data_files=args.train_file, split="train")
    process_vision_info = deps["process_vision_info"]

    def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        all_images: list[Any] = []
        all_videos: list[Any] = []
        for feature in features:
            messages = feature["messages"]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
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
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    training_args = deps["TrainingArguments"](
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = deps["Trainer"](
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"LoRA 训练完成，权重已保存到：{args.output_dir}")


if __name__ == "__main__":
    main()
