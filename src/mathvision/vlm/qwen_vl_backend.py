"""Optional Qwen2.5-VL backend for cloud GPU inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mathvision.device import get_best_device
from mathvision.prompts import build_vqa_prompt
from mathvision.vlm.base import VLMBackend


class QwenVLBackend(VLMBackend):
    """Qwen2.5-VL backend.

    This backend is intended for a cloud GPU environment. It is implemented as
    optional code and is not imported during local smoke tests unless selected.
    """

    name = "qwen-vl"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> None:
        self.model_name = model_name
        self.device = get_best_device()
        self.processor: Any | None = None
        self.model: Any | None = None

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "加载 Qwen2.5-VL 失败：缺少依赖。请在云 GPU 环境安装："
                "pip install -r requirements.txt && pip install -r requirements-gpu.txt。"
                "Qwen2.5-VL 通常需要较新的 transformers 和 qwen-vl-utils。"
            ) from exc

        try:
            dtype = torch.float32
            kwargs: dict[str, Any] = {"trust_remote_code": True}
            if self.device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                kwargs["torch_dtype"] = dtype
                kwargs["device_map"] = "auto"
            else:
                kwargs["torch_dtype"] = dtype
                kwargs["attn_implementation"] = "eager"

            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, **kwargs
            )
            if self.device in {"mps", "cpu"}:
                self.model.to(self.device)
            self.model.eval()
            self._process_vision_info = process_vision_info
        except Exception as exc:
            raise RuntimeError(
                "加载 Qwen2.5-VL 模型失败。建议在 CUDA 云 GPU 上运行，并确认模型名称、"
                "网络和显存满足要求。Mac 本地建议使用 mock 或 smolvlm。"
            ) from exc

    def generate(
        self,
        image_path: str,
        question: str,
        context: str | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        self._load()
        assert self.model is not None
        assert self.processor is not None

        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"图片不存在: {image_file}")

        try:
            import torch

            prompt = build_vqa_prompt(question, context)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_file.resolve())},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = self._process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            generated_trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
            ]
            decoded = self.processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return decoded[0].strip() if decoded else "模型未生成有效回答。"
        except Exception as exc:
            raise RuntimeError(
                "Qwen2.5-VL 推理失败。请检查本地图片路径、依赖版本和 GPU 显存。"
            ) from exc
