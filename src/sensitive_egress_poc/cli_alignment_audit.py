from __future__ import annotations

import argparse
import json

from .benchmark.audit import write_alignment_audit
from .io_utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit mixed-egress coarse alignment labels without mutating ground truth.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    summary = write_alignment_audit(rows, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
