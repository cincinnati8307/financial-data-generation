from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .io_utils import ensure_dir, write_json, write_jsonl
from .schemas import LABELS, PRIVATE_SUBTYPES, GenerationManifest
from .synthetic_generator import SyntheticFinancialGenerator


def split80(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    cut = int(len(rows) * 0.8)
    return rows[:cut], rows[cut:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/generated")
    parser.add_argument("--private", type=int, default=1200)
    parser.add_argument("--hard-negative", type=int, default=600)
    parser.add_argument("--benign", type=int, default=600)
    parser.add_argument("--mixed", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    out = Path(args.out_dir); ensure_dir(out)
    gen = SyntheticFinancialGenerator(args.seed)
    anchors = gen.generate_private(args.private) + gen.generate_hard_negatives(args.hard_negative) + gen.generate_benign(args.benign)
    gen.rng.shuffle(anchors)
    egress = gen.generate_mixed(args.mixed); gen.rng.shuffle(egress)
    at, av = split80(anchors); et, ev = split80(egress)
    write_jsonl(out / "anchors_train.jsonl", at); write_jsonl(out / "anchors_validation.jsonl", av)
    write_jsonl(out / "egress_train.jsonl", et); write_jsonl(out / "egress_validation.jsonl", ev)
    manifest = GenerationManifest("Synthetic Chinese-nuance financial-private data for egress detection.", False, LABELS, PRIVATE_SUBTYPES, {"anchors_train": len(at), "anchors_validation": len(av), "egress_train": len(et), "egress_validation": len(ev)}, args.seed)
    write_json(out / "manifest.json", manifest.to_dict())
    logging.info("Generated %s anchor and %s mixed examples", len(anchors), len(egress))

if __name__ == "__main__":
    main()
