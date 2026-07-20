from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

from egress_grounding.generation import grounding_manifest_section, grounding_split_key
from egress_grounding.store import GroundingStore

from .io_utils import ensure_dir, write_json, write_jsonl
from .schemas import LABELS, PRIVATE_SUBTYPES, GenerationManifest
from .synthetic_generator import SyntheticFinancialGenerator


def skeleton_key(row: dict) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    fallback = str(meta.get("skeleton_id") or row.get("id") or row.get("text") or "")
    return grounding_split_key(row, fallback)


def split_by_skeleton(rows: list[dict], train_ratio: float = 0.8, rng: random.Random | None = None) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[skeleton_key(row)].append(row)

    keys = list(groups.keys())
    shuffler = rng or random.Random(0)
    shuffler.shuffle(keys)

    train: list[dict] = []
    validation: list[dict] = []
    target = int(len(rows) * train_ratio)
    for key in keys:
        target_list = train if len(train) < target else validation
        target_list.extend(groups[key])

    if not validation and len(keys) > 1:
        moved_key = keys[-1]
        moved = groups[moved_key]
        train = [row for row in train if skeleton_key(row) != moved_key]
        validation.extend(moved)
    return train, validation


def split80(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    return split_by_skeleton(rows, 0.8)


def _requested_grounding_ratio(mode: str, ratio: float) -> float:
    if mode == "grounded-only":
        return 1.0
    if mode == "hybrid":
        return ratio
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/generated")
    parser.add_argument("--private", type=int, default=1200)
    parser.add_argument("--hard-negative", type=int, default=600)
    parser.add_argument("--benign", type=int, default=600)
    parser.add_argument("--mixed", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--grounding", action="append", default=[])
    parser.add_argument("--grounding-mode", choices=["synthetic-only", "hybrid", "grounded-only"], default="synthetic-only")
    parser.add_argument("--grounding-ratio", type=float, default=0.0)
    parser.add_argument("--grounding-dataset", action="append", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    out = Path(args.out_dir)
    ensure_dir(out)
    grounding_store = GroundingStore.load(args.grounding, datasets=args.grounding_dataset) if args.grounding else None
    gen = SyntheticFinancialGenerator(
        args.seed,
        grounding_store=grounding_store,
        grounding_mode=args.grounding_mode,
        grounding_ratio=args.grounding_ratio,
        grounding_datasets=set(args.grounding_dataset),
    )
    anchors = gen.generate_private(args.private) + gen.generate_hard_negatives(args.hard_negative) + gen.generate_benign(args.benign)
    egress = gen.generate_mixed(args.mixed)
    at, av = split_by_skeleton(anchors, rng=gen.rng)
    et, ev = split_by_skeleton(egress, rng=gen.rng)
    write_jsonl(out / "anchors_train.jsonl", at)
    write_jsonl(out / "anchors_validation.jsonl", av)
    write_jsonl(out / "egress_train.jsonl", et)
    write_jsonl(out / "egress_validation.jsonl", ev)
    manifest = GenerationManifest(
        "Synthetic Chinese-nuance financial-private data for egress detection.",
        False,
        LABELS,
        PRIVATE_SUBTYPES,
        {"anchors_train": len(at), "anchors_validation": len(av), "egress_train": len(et), "egress_validation": len(ev)},
        args.seed,
        grounding=grounding_manifest_section(
            rows=anchors + egress,
            mode=args.grounding_mode,
            requested_ratio=_requested_grounding_ratio(args.grounding_mode, args.grounding_ratio),
            grounding_paths=args.grounding,
            dataset_filters=args.grounding_dataset,
            generator=gen,
        ),
    )
    write_json(out / "manifest.json", manifest.to_dict())
    logging.info("Generated %s anchor and %s mixed examples", len(anchors), len(egress))


if __name__ == "__main__":
    main()
