"""Score a local MVTec test split with a saved memory bank."""

from __future__ import annotations

import argparse
import sys

from train import selected_categories
from utils.config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", action="append", dest="categories")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    categories = selected_categories(config, args.categories)
    from evaluation.run import evaluate_category

    for category in categories:
        evaluate_category(config, category)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
