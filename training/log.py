"""Append-only text logs for memory builds and evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class RunLog:
    def __init__(self, directory: Path, category: str, kind: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = directory / f"{category}-{kind}-{stamp}.log"
        self.latest = directory / f"{category}-{kind}.log"
        self.write(f"start {kind} category={category} utc={stamp}")

    def write(self, message: str) -> None:
        line = message.rstrip() + "\n"
        for path in (self.path, self.latest):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
