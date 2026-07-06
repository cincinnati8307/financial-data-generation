from __future__ import annotations

import argparse, logging
from pathlib import Path
from .io_utils import read_jsonl, write_json, write_jsonl
from .llm_augmenter import DryRunProvider, OpenAIProvider, augment_rows
from .schemas import AugmentationManifest


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--max-inputs", type=int, default=100); p.add_argument("--paraphrases-per-example", type=int, default=6); p.add_argument("--provider", choices=["dry-run","openai"], default="dry-run"); p.add_argument("--model", default="gpt-4o-mini"); p.add_argument("--include-original", action="store_true")
    a=p.parse_args(); logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    rows=read_jsonl(a.input); provider=DryRunProvider(a.model) if a.provider=="dry-run" else OpenAIProvider(a.model)
    out,reasons=augment_rows(rows, provider, a.max_inputs, a.paraphrases_per_example, a.include_original, a.model)
    write_jsonl(a.output, out)
    attempted=min(a.max_inputs, sum(1 for r in rows if r.get("label")=="financial_private"))
    manifest=AugmentationManifest(a.input,a.output,a.provider,a.model,a.max_inputs,a.paraphrases_per_example,a.include_original,attempted,len(out),sum(reasons.values()),dict(reasons))
    write_json(Path(a.output).with_suffix(".manifest.json"), manifest.to_dict())
    logging.info("Accepted %s augmented/original rows", len(out))
if __name__=="__main__": main()
