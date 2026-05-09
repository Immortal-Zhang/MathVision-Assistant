"""Small IO helpers used across scripts and modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def project_root() -> Path:
    """Return the repository root based on the package location."""

    return Path(__file__).resolve().parents[2]


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a path and return it as ``Path``."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""

    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")

    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path} 第 {line_no} 行不是合法 JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{jsonl_path} 第 {line_no} 行必须是 JSON object")
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to a JSONL file."""

    jsonl_path = ensure_parent(path)
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_text(path: str | Path) -> str:
    """Read UTF-8 text."""

    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    """Write UTF-8 text, creating parent directories."""

    text_path = ensure_parent(path)
    text_path.write_text(content, encoding="utf-8")
