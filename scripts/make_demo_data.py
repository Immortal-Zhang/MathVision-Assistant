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
    parser.add_argument("--num_samples", type=int, default=1000, help="生成样本数，默认 1000")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例，默认 0.8")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例，默认 0.1")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_demo_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print("Demo 数据生成完成：")
    for key, value in result.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
