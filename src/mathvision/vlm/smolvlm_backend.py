"""SmolVLM backend based on Hugging Face Transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from mathvision.device import get_best_device
from mathvision.prompts import build_vqa_prompt
from mathvision.vlm.base import VLMBackend


class SmolVLMBackend(VLMBackend):
    """Real lightweight multimodal backend.

    The model is downloaded on first use. This backend is optional and is not
    used by smoke tests.
    """

    name = "smolvlm"

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        lora_adapter: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.lora_adapter = lora_adapter
        self.device = get_best_device()
        self.processor: Any | None = None
        self.model: Any | None = None

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor

            try:
                from transformers import AutoModelForImageTextToText as ModelClass
            except ImportError:
                from transformers import AutoModelForVision2Seq as ModelClass
        except Exception as exc:
            raise RuntimeError(
                "加载 SmolVLM 失败：缺少 torch/transformers 等依赖。"
                "请先执行 pip install -r requirements.txt，或先使用 --backend mock 跑通本地流程。"
            ) from exc

        try:
            dtype = torch.float32
            model_kwargs: dict[str, Any] = {"trust_remote_code": True}
            if self.device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                model_kwargs["torch_dtype"] = dtype
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = dtype
                model_kwargs["attn_implementation"] = "eager"

            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = ModelClass.from_pretrained(self.model_name, **model_kwargs)
            if self.lora_adapter:
                adapter_path = Path(self.lora_adapter)
                if not adapter_path.exists():
                    raise FileNotFoundError(f"LoRA adapter 路径不存在: {adapter_path}")
                try:
                    from peft import PeftModel
                except Exception as exc:
                    raise RuntimeError(
                        "加载 LoRA adapter 失败：缺少 peft。请执行 pip install peft，"
                        "或不传 --lora_adapter 使用原始 SmolVLM。"
                    ) from exc
                self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            if self.device in {"mps", "cpu"}:
                self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                "加载 SmolVLM 模型失败。请确认网络可访问 Hugging Face、磁盘空间充足，"
                "并且 transformers 版本较新。Mac 本地首次下载会比较慢；"
                "如果使用 LoRA，请确认 adapter 目录存在且已安装 peft；"
                "如果只是验证项目，请先运行：python scripts/run_smoke_test.py --backend mock"
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

            image = Image.open(image_file).convert("RGB")
            prompt = build_vqa_prompt(question, context)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            else:
                text = prompt

            inputs = self.processor(text=text, images=[image], return_tensors="pt")
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

            input_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
            new_tokens = generated_ids[:, input_length:]
            decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
            answer = decoded[0].strip() if decoded else ""
            if not answer:
                full = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                answer = full[0].replace(prompt, "").strip() if full else ""
            return answer or "模型未生成有效回答。"
        except Exception as exc:
            raise RuntimeError(
                "SmolVLM 推理失败。请检查图片路径、transformers 版本和设备内存。"
                "若在 Mac 上资源不足，请使用 --backend mock 或租用云 GPU。"
            ) from exc
