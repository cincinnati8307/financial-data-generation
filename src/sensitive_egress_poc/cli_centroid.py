from __future__ import annotations

import argparse, json, logging
from pathlib import Path
from .centroid_classifier import DEFAULT_MODEL, build_centroids, evaluate
from .io_utils import read_jsonl, write_json


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--train", required=True); p.add_argument("--validation", required=True); p.add_argument("--model", default=DEFAULT_MODEL); p.add_argument("--out", default="data/generated/centroids.json")
    a=p.parse_args(); logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    cent=build_centroids(read_jsonl(a.train), a.model); metrics=evaluate(read_jsonl(a.validation), cent)
    write_json(a.out, cent)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
if __name__=="__main__": main()
