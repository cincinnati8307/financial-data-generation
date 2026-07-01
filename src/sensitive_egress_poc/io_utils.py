from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    if path.parent:
        ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    if path.parent:
        ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
