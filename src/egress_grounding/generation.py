from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def grounding_split_key(row: dict[str, Any], fallback: str) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    grounding = meta.get("grounding") if isinstance(meta.get("grounding"), dict) else None
    if grounding and grounding.get("dataset") and grounding.get("source_group_id"):
        return f"grounding:{grounding['dataset']}:{grounding['source_group_id']}"
    return fallback


def grounding_manifest_section(
    *,
    rows: Iterable[dict[str, Any]],
    mode: str,
    requested_ratio: float,
    grounding_paths: Iterable[str | Path],
    dataset_filters: Iterable[str],
    generator: Any,
) -> dict[str, Any] | None:
    paths = [str(path) for path in grounding_paths]
    datasets = list(dataset_filters)
    if mode == "synthetic-only" and not paths and not datasets:
        return None
    rows = list(rows)
    groundings = [
        row["meta"]["grounding"]
        for row in rows
        if isinstance(row.get("meta"), dict) and isinstance(row["meta"].get("grounding"), dict)
    ]
    eligible_count = sum(getattr(generator, "grounding_eligible_counts", {}).values())
    actual_ratio = len(groundings) / eligible_count if eligible_count else 0.0
    return {
        "mode": mode,
        "requested_ratio": requested_ratio,
        "actual_ratio": round(actual_ratio, 4),
        "inputs": paths,
        "datasets": datasets or sorted({grounding["dataset"] for grounding in groundings}),
        "grounded_counts": dict(Counter(row.get("source", "unknown") for row in rows if isinstance(row.get("meta"), dict) and isinstance(row["meta"].get("grounding"), dict))),
        "fallback_counts": dict(getattr(generator, "grounding_fallback_counts", {})),
        "eligible_counts": dict(getattr(generator, "grounding_eligible_counts", {})),
        "attempt_counts": dict(getattr(generator, "grounding_attempt_counts", {})),
        "source_role_counts": dict(Counter(str(grounding.get("role", "unknown")) for grounding in groundings)),
        "raw_text_copied": False,
        "grounded_in_public_or_deidentified_data": True,
    }
