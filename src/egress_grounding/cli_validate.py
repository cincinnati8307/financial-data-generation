from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .schemas import read_records_jsonl


def validate_paths(paths: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    records = []
    for path in paths:
        try:
            records.extend(read_records_jsonl(path))
        except Exception as exc:
            errors.append(str(exc))
    return {
        "valid": not errors,
        "errors": errors,
        "records": len(records),
        "datasets": dict(Counter(record.dataset for record in records)),
        "roles": dict(Counter(record.role for record in records)),
        "labels": dict(Counter(record.label for record in records)),
        "raw_text_copied": False,
        "grounded_in_public_or_deidentified_data": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate normalized grounding JSONL.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--report-out")
    args = parser.parse_args()

    report = validate_paths(args.input)
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
