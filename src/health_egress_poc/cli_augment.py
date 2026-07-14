from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sensitive_egress_poc.io_utils import read_jsonl, write_json, write_jsonl

from .llm_augmenter import DryRunProvider, OpenAIProvider, augment_rows, estimate_augmentation_tokens
from .schemas import AugmentationManifest


def format_token_estimate(estimate: dict[str, int | str], provider: str, model: str) -> str:
    return "\n".join(["Token estimate for health augmentation:", f"  provider: {provider}", f"  model: {model}", f"  private source rows: {estimate['attempted_sources']}", f"  paraphrases per source: {estimate['paraphrases_per_example']}", f"  estimated total tokens: {estimate['estimated_total_tokens']}", f"  conservative total (+25%): {estimate['estimated_total_tokens_with_buffer']}", "  method: heuristic estimate, not provider billing truth"])


def confirm_augmentation() -> bool:
    return input("Proceed with OpenAI health augmentation? Type 'yes' to continue: ").strip().lower() in {"yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-inputs", type=int, default=100)
    parser.add_argument("--paraphrases-per-example", type=int, default=6)
    parser.add_argument("--provider", choices=["dry-run", "openai"], default="dry-run")
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    rows = read_jsonl(args.input)
    estimate = estimate_augmentation_tokens(rows, args.max_inputs, args.paraphrases_per_example)
    if args.provider == "openai" or args.estimate_only:
        print(format_token_estimate(estimate, args.provider, args.model))
    if args.estimate_only:
        return
    if args.provider == "openai" and not args.yes and not confirm_augmentation():
        logging.info("Augmentation cancelled before any OpenAI request was sent.")
        return
    provider = DryRunProvider(args.model) if args.provider == "dry-run" else OpenAIProvider(args.model)
    out, reasons = augment_rows(rows, provider, args.max_inputs, args.paraphrases_per_example, args.include_original, args.model)
    write_jsonl(args.output, out)
    manifest = AugmentationManifest(args.input, args.output, args.provider, args.model, args.max_inputs, args.paraphrases_per_example, args.include_original, int(estimate["attempted_sources"]), len(out), sum(reasons.values()), dict(reasons))
    manifest_dict = manifest.to_dict()
    manifest_dict["token_estimate"] = estimate
    write_json(Path(args.output).with_suffix(".manifest.json"), manifest_dict)
    logging.info("Accepted %s augmented/original health rows", len(out))


if __name__ == "__main__":
    main()
