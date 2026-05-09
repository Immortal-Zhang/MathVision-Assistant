"""YAML configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    """Typed view over the most commonly used configuration paths."""

    qa_file: str
    knowledge_base_file: str
    index_file: str
    default_backend: str
    top_k: int


def load_yaml_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load a YAML configuration file."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 YAML object: {config_path}")
    return data


def load_app_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    """Load default project configuration as ``AppConfig``."""

    data = load_yaml_config(path)
    return AppConfig(
        qa_file=str(data.get("data", {}).get("qa_file", "data/demo/qa.jsonl")),
        knowledge_base_file=str(
            data.get("data", {}).get(
                "knowledge_base_file", "data/demo/knowledge_base.jsonl"
            )
        ),
        index_file=str(data.get("data", {}).get("index_file", "data/outputs/index.pkl")),
        default_backend=str(data.get("backend", {}).get("default", "mock")),
        top_k=int(data.get("retrieval", {}).get("top_k", 3)),
    )
