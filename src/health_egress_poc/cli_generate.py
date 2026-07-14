from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

from sensitive_egress_poc.dataset_quality import evaluate_dataset, parse_checks
from sensitive_egress_poc.io_utils import ensure_dir, write_json, write_jsonl

from .schemas import HEALTH_SUBTYPES, LABELS, GenerationManifest
from .synthetic_generator import SyntheticHealthGenerator


def skeleton_key(row: dict) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return str(meta.get("skeleton_id") or row.get("id") or row.get("text") or "")


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
        (train if len(train) < target else validation).extend(groups[key])
    return train, validation


def remove_redundant_anchors(rows: list[dict], target_count: int) -> list[dict]:
    _, cleaned = evaluate_dataset(rows, checks=parse_checks("redundancy"))
    return cleaned[:target_count]


def generate_clean_anchors(gen: SyntheticHealthGenerator, target_private: int, target_hard_negative: int, target_benign: int) -> list[dict]:
    targets = [
        (target_private, gen.generate_private),
        (target_hard_negative, gen.generate_hard_negatives),
        (target_benign, gen.generate_benign),
    ]
    out: list[dict] = []
    for target_count, producer in targets:
        pool: list[dict] = []
        cleaned: list[dict] = []
        attempts = 0
        while len(cleaned) < target_count and attempts < 8:
            missing = target_count - len(cleaned)
            batch_size = max(50, int(missing * 1.8))
            pool.extend(producer(batch_size))
            cleaned = remove_redundant_anchors(pool, target_count)
            attempts += 1
        if len(cleaned) < target_count:
            raise RuntimeError(f"could only generate {len(cleaned)} clean rows for target {target_count}")
        out.extend(cleaned)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/health_generated")
    parser.add_argument("--private", type=int, default=1200)
    parser.add_argument("--hard-negative", type=int, default=600)
    parser.add_argument("--benign", type=int, default=600)
    parser.add_argument("--mixed", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    out = Path(args.out_dir)
    ensure_dir(out)
    gen = SyntheticHealthGenerator(args.seed)
    anchors = generate_clean_anchors(gen, args.private, args.hard_negative, args.benign)
    egress = gen.generate_mixed(args.mixed)
    at, av = split_by_skeleton(anchors, rng=gen.rng)
    et, ev = split_by_skeleton(egress, rng=gen.rng)
    write_jsonl(out / "anchors_train.jsonl", at)
    write_jsonl(out / "anchors_validation.jsonl", av)
    write_jsonl(out / "egress_train.jsonl", et)
    write_jsonl(out / "egress_validation.jsonl", ev)
    manifest = GenerationManifest("Synthetic health-private data for egress detection.", False, LABELS, HEALTH_SUBTYPES, {"anchors_train": len(at), "anchors_validation": len(av), "egress_train": len(et), "egress_validation": len(ev)}, args.seed)
    write_json(out / "manifest.json", manifest.to_dict())
    logging.info("Generated %s health anchor and %s mixed examples", len(anchors), len(egress))


if __name__ == "__main__":
    main()
