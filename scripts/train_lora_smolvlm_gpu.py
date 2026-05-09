#!/usr/bin/env python3
"""SmolVLM LoRA training script for cloud GPU environments.

This script fine-tunes HuggingFaceTB/SmolVLM-500M-Instruct with PEFT LoRA on
the MathVision QA JSONL converted by ``scripts/prepare_lora_data.py``.

It is intentionally not part of the local smoke test. A Mac can prepare data,
but actual training is recommended on a CUDA GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="云 GPU 上运行 SmolVLM LoRA 微调")
    parser.add_argument(
        "--model_name",
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="Hugging Face 模型名",
    )
    parser.add_argument(
        "--train_file",
        default="data/outputs/lora_qwen_vl.jsonl",
        help="由 scripts/prepare_lora_data.py 生成的训练 JSONL",
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints/smolvlm500m-lora-mathvision",
        help="LoRA adapter 输出目录",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument(
        "--max_length",
        type=int,
        default=0,
        help=(
            "保留给纯文本长度控制使用。SmolVLM 图像 token 较多，默认 0 表示不截断，"
            "避免 image token count mismatch。"
        ),
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="训练最大 step 数。Mac 本地 dry-run 可设置为 1。",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=0,
        help="只取前 N 条样本训练。0 表示使用全部样本。",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="启用 gradient checkpointing，降低显存/内存占用但会变慢。",
    )
    parser.add_argument(
        "--allow_non_cuda",
        action="store_true",
        help="允许在 CPU/MPS 上调试脚本。训练会很慢，不推荐正式使用。",
    )
    return parser.parse_args()


def _import_training_deps() -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Trainer, TrainingArguments

        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass
    except Exception as exc:
        raise RuntimeError(
            "SmolVLM LoRA 训练依赖缺失。请在云 GPU 环境执行：\n"
            "pip install -r requirements.txt\n"
            "pip install peft datasets\n"
            "如 transformers 版本过旧，请执行 pip install -U transformers accelerate"
        ) from exc

    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "AutoProcessor": AutoProcessor,
        "ModelClass": ModelClass,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _extract_image_path(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") == "image":
                image_path = part.get("image")
                if image_path:
                    return str(image_path)
    raise ValueError("训练样本缺少 image 字段")


def _format_messages(processor: Any, messages: list[dict[str, Any]]) -> str:
    if hasattr(processor, "apply_chat_template"):
        return str(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

    user_text = ""
    assistant_text = ""
    for message in messages:
        content = message.get("content", [])
        text_parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        if message.get("role") == "user":
            user_text = "\n".join(text_parts)
        elif message.get("role") == "assistant":
            assistant_text = "\n".join(text_parts)
    return f"User: <image>\n{user_text}\nAssistant: {assistant_text}"


def _infer_lora_target_modules(model: Any, torch: Any) -> list[str]:
    candidate_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "proj",
        "linear",
    }
    found: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in candidate_suffixes:
                found.add(suffix)

    if found:
        return sorted(found)
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    args = parse_args()
    deps = _import_training_deps()
    torch = deps["torch"]

    if not torch.cuda.is_available() and not args.allow_non_cuda:
        raise RuntimeError(
            "未检测到 CUDA GPU。SmolVLM 虽然比 Qwen2.5-VL 小，但 LoRA 训练仍建议在云 GPU 上运行。\n"
            "如果只是想本地检查脚本，可加 --allow_non_cuda，但会非常慢。"
        )
    if not Path(args.train_file).exists():
        raise FileNotFoundError(
            f"训练文件不存在：{args.train_file}。请先运行 python scripts/prepare_lora_data.py"
        )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    if device != "cuda":
        dtype = torch.float32

    processor = deps["AutoProcessor"].from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["attn_implementation"] = "eager"

    model = deps["ModelClass"].from_pretrained(args.model_name, **model_kwargs)
    if device != "cuda":
        model.to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    target_modules = _infer_lora_target_modules(model, torch)
    print(f"LoRA target_modules: {target_modules}")
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

    dataset = deps["load_dataset"]("json", data_files=args.train_file, split="train")
    if args.limit_samples > 0:
        dataset = dataset.select(range(min(args.limit_samples, len(dataset))))

    def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        images: list[Image.Image] = []
        for feature in features:
            messages = feature["messages"]
            image_path = Path(_extract_image_path(messages))
            if not image_path.exists():
                raise FileNotFoundError(f"图片不存在：{image_path}")
            images.append(Image.open(image_path).convert("RGB"))
            texts.append(_format_messages(processor, messages))

        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "images": images,
            "padding": True,
            "return_tensors": "pt",
        }
        # Do not truncate by default. For VLM processors, image placeholders can
        # expand to hundreds/thousands of tokens; truncating them causes a
        # mismatch between image token count in text and input_ids.
        if args.max_length > 0:
            processor_kwargs["truncation"] = False
        batch = processor(**processor_kwargs)
        labels = batch["input_ids"].clone()
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
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
        max_steps=args.max_steps,
        bf16=device == "cuda" and torch.cuda.is_bf16_supported(),
        fp16=device == "cuda" and not torch.cuda.is_bf16_supported(),
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
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"SmolVLM LoRA 训练完成，adapter 已保存到：{args.output_dir}")


if __name__ == "__main__":
    main()
