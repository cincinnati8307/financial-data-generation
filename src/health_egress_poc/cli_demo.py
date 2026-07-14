from __future__ import annotations

import argparse
import json

from sensitive_egress_poc.io_utils import read_json

from .centroid_classifier import classify_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(json.dumps(classify_text(args.text, read_json(args.centroids)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
