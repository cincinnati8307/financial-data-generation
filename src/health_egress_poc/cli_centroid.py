from __future__ import annotations

import argparse
import json
import logging

from sensitive_egress_poc.io_utils import read_jsonl, write_json

from .centroid_classifier import DEFAULT_MODEL, build_centroids, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="data/health_generated/centroids.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    centroids = build_centroids(read_jsonl(args.train), args.model)
    metrics = evaluate(read_jsonl(args.validation), centroids)
    write_json(args.out, centroids)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
