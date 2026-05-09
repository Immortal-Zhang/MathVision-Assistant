#!/usr/bin/env python3
"""Generate offline demo data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathvision.data.synthetic import generate_demo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 MathVision-Assistant 合成 demo 数据")
    parser.add_argument("--output_dir", default="data/demo", help="输出目录，默认 data/demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_demo_dataset(args.output_dir)
    print("Demo 数据生成完成：")
    for key, value in result.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
