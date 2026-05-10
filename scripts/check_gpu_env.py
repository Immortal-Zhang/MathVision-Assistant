#!/usr/bin/env python3
"""Collect GPU-server environment information for reproducible LoRA runs."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "peft": "peft",
    "accelerate": "accelerate",
    "qwen_vl_utils": "qwen-vl-utils",
    "numpy": "numpy",
    "pandas": "pandas",
    "pillow": "Pillow",
    "gradio": "gradio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查并记录 GPU 服务器环境")
    parser.add_argument("--out_file", default="runs/env.json")
    return parser.parse_args()


def _run_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def _package_info(module_name: str, dist_name: str) -> dict[str, Any]:
    info: dict[str, Any] = {"installed": False, "version": None, "import_ok": False}
    try:
        info["version"] = importlib.metadata.version(dist_name)
        info["installed"] = True
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        importlib.import_module(module_name)
        info["import_ok"] = True
    except Exception as exc:
        info["import_error"] = repr(exc)
    return info


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"import_ok": False, "import_error": repr(exc)}

    gpu_names: list[str] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(index))
    return {
        "import_ok": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    }


def collect_env() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "platform": platform.platform(),
        },
        "project": {
            "root": str(ROOT),
            "git_commit": _run_command(["git", "rev-parse", "HEAD"]),
            "git_status_short": _run_command(["git", "status", "--short"]),
        },
        "packages": {
            module: _package_info(module, dist_name)
            for module, dist_name in PACKAGES.items()
            if module != "torch"
        },
        "torch": _torch_info(),
        "nvidia_smi": _run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    return payload


def main() -> None:
    args = parse_args()
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = collect_env()
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"环境信息已保存：{out_file}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
