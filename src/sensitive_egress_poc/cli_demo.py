from __future__ import annotations

import argparse, json
from .centroid_classifier import classify_text
from .io_utils import read_json


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--centroids", required=True); p.add_argument("--text", required=True)
    a=p.parse_args(); print(json.dumps(classify_text(a.text, read_json(a.centroids)), ensure_ascii=False, indent=2))
if __name__=="__main__": main()
