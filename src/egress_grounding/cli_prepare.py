from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import adapter_names, prepare_dataset
from .schemas import write_records_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare normalized privacy-safe grounding JSONL.")
    parser.add_argument("--dataset", required=True, choices=adapter_names())
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = prepare_dataset(args.dataset, args.input, seed=args.seed, limit=args.limit)
    write_records_jsonl(args.output, records)
    manifest = {
        "dataset": args.dataset,
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "limit": args.limit,
        "records": len(records),
        "contains_real_personal_data": False,
        "raw_text_copied": False,
        "grounded_in_public_or_deidentified_data": True,
    }
    manifest_path = Path(args.output).with_suffix(Path(args.output).suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
