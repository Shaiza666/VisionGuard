"""Build a nominal memory bank for one or more MVTec categories."""

from __future__ import annotations

import argparse
import sys

from data.layout import summarize_category
from utils.config import load_config
from utils.seed import set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Repeat to select categories. Defaults to every category in the config.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Count local images and exit. Does not load weights or build a memory bank.",
    )
    return parser.parse_args(argv)


def selected_categories(config, requested: list[str] | None) -> tuple[str, ...]:
    if not requested:
        return config.data.categories
    unknown = [category for category in requested if category not in config.data.categories]
    if unknown:
        allowed = ", ".join(config.data.categories)
        raise ValueError(f"Categories {unknown} are not listed in the config. Config allows: {allowed}.")
    return tuple(requested)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    categories = selected_categories(config, args.categories)
    if args.check:
        for category in categories:
            summary = summarize_category(config.data.root, category)
            print(summary.render(), flush=True)
        return 0

    set_seed(config.seed)
    from training.memory_bank import build_category_memory

    for category in categories:
        build_category_memory(config, category)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
