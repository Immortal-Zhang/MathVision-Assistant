"""Backend factory."""

from __future__ import annotations

from mathvision.vlm.base import VLMBackend
from mathvision.vlm.mock_backend import MockVLMBackend


def build_vlm_backend(name: str) -> VLMBackend:
    """Create a VLM backend by name."""

    normalized = name.strip().lower()
    if normalized == "mock":
        return MockVLMBackend()
    if normalized == "smolvlm":
        from mathvision.vlm.smolvlm_backend import SmolVLMBackend

        return SmolVLMBackend()
    if normalized in {"qwen-vl", "qwen_vl", "qwen2.5-vl", "qwen"}:
        from mathvision.vlm.qwen_vl_backend import QwenVLBackend

        return QwenVLBackend()
    raise ValueError(f"未知 backend: {name}，可选值为 mock / smolvlm / qwen-vl")
