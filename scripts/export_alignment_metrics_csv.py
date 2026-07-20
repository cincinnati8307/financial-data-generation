#!/usr/bin/env python3
"""Export nested alignment metrics JSON to a flat CSV table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("results/pii_reranker_mmarco/alignment_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert alignment_metrics.json into one flat CSV row per "
            "alignment scope and model."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV file (default: input path with a .csv suffix)",
    )
    return parser.parse_args()


def flatten_mapping(
    value: dict[str, Any],
    *,
    prefix: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """Flatten a nested mapping using dotted column names."""
    flattened: dict[str, Any] = {}

    for key, item in value.items():
        column = f"{prefix}{separator}{key}" if prefix else key

        if isinstance(item, dict):
            flattened.update(
                flatten_mapping(item, prefix=column, separator=separator)
            )
        elif isinstance(item, (list, tuple)):
            flattened[column] = json.dumps(item, ensure_ascii=False)
        else:
            flattened[column] = item

    return flattened


def build_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one row for every top-level alignment scope and model."""
    rows: list[dict[str, Any]] = []

    for alignment_scope, models in data.items():
        # Ignore top-level metadata such as "fine_grained_note".
        if not isinstance(models, dict):
            continue

        for model_name, metrics in models.items():
            if not isinstance(metrics, dict):
                continue

            row: dict[str, Any] = {
                "alignment_scope": alignment_scope,
                "model": model_name,
            }
            row.update(flatten_mapping(metrics))
            rows.append(row)

    return rows


def collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    """Preserve first-seen column order across all rows."""
    fieldnames = ["alignment_scope", "model"]
    seen = set(fieldnames)

    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    return fieldnames


def export_csv(input_path: Path, output_path: Path) -> int:
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(
            f"Error: invalid JSON in {input_path}: "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}",
            file=sys.stderr,
        )
        return 1

    if not isinstance(data, dict):
        print("Error: the JSON root must be an object.", file=sys.stderr)
        return 1

    rows = build_rows(data)
    if not rows:
        print(
            "Error: no alignment scope/model metric objects were found.",
            file=sys.stderr,
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=collect_fieldnames(rows),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


def main() -> int:
    args = parse_args()
    output_path = args.output or args.input.with_suffix(".csv")
    return export_csv(args.input, output_path)


if __name__ == "__main__":
    raise SystemExit(main())
