from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .benchmark.audit import write_alignment_audit
from .benchmark.capid_adapter import CapidAdapter
from .benchmark.chinese_privacy_adapter import ChinesePrivacyBenchmarkModel
from .benchmark.centroid_adapter import CentroidBenchmarkModel
from .benchmark.dataset import (
    dataset_summary,
    fine_grained_alignment_rows,
    load_alignment_overrides,
    read_rows,
    row_id,
    rows_by_id,
)
from .benchmark.llm_judge import LlmJudgeModel
from .benchmark.metrics import alignment_metrics, binary_sensitivity_metrics, build_subgroup_metrics, write_subgroup_metrics_csv
from .benchmark.opf_granite_adapter import OpenAIPrivacyFilterGraniteModel
from .benchmark.pii_adapter import PiiOnlyModel
from .benchmark.pii_reranker import PiiRerankerModel
from .benchmark.plots import generate_plots
from .benchmark.runtime import summarize_method_runtime
from .benchmark.schemas import STATUS_FAILED, STATUS_SKIPPED, STATUS_UNSUPPORTED, TASK_COARSE_ALIGNMENT, TASK_FINE_ALIGNMENT, BenchmarkPrediction
from .io_utils import ensure_dir, write_json, write_jsonl

METHOD_ALIASES = {
    "centroid": "centroid",
    "pii": "pii",
    "pii_reranker": "pii_reranker",
    "capid": "capid",
    "qwen3guard": "qwen3guard",
    "shieldlm": "shieldlm",
    "llm_judge": "llm_judge",
    "opf_granite": "opf_granite",
    "opf_granite_oracle": "opf_granite_oracle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark financial-private sensitivity and coarse query-policy alignment baselines.")
    parser.add_argument("--anchor-train")
    parser.add_argument("--anchor-validation", required=True)
    parser.add_argument("--egress-train")
    parser.add_argument("--egress-validation", required=True)
    parser.add_argument("--max-anchor-validation", type=int, help="Limit Task A validation rows for smoke tests or small LLM runs.")
    parser.add_argument("--max-egress-train", type=int, help="Limit egress training rows used for alignment threshold tuning.")
    parser.add_argument("--max-egress-validation", type=int, help="Limit Task B validation rows for smoke tests or small LLM runs.")
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--methods", nargs="+", default=["centroid"], choices=sorted(METHOD_ALIASES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dir")
    parser.add_argument("--pii-backend", default="regex")
    parser.add_argument("--pii-model")
    parser.add_argument("--openai-privacy-filter-model", default="openai/privacy-filter")
    parser.add_argument("--openai-privacy-filter-threshold", type=float, default=0.5)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--granite-guardian-model", default="ibm-granite/granite-guardian-3.2-3b-a800m")
    parser.add_argument("--granite-max-new-tokens", type=int, default=20)
    parser.add_argument("--granite-load-in-4bit", action="store_true")
    parser.add_argument("--granite-trust-remote-code", action="store_true")
    parser.add_argument("--capid-model")
    parser.add_argument("--chinese-privacy-max-new-tokens", type=int, default=128)
    parser.add_argument("--qwen3guard-model", help="Override the Qwen3Guard Hugging Face model ID or local path.", default="Qwen/Qwen3Guard-Gen-8B")
    parser.add_argument("--shieldlm-model", help="ShieldLM Hugging Face model ID. Defaults to thu-coai/ShieldLM-6B-chatglm3.", default=None)
    parser.add_argument("--shieldlm-model-base", help="ShieldLM model base (qwen, baichuan, internlm, chatglm). Auto-detected from model_path if not specified.", default=None)
    parser.add_argument("--capid-base-model")
    parser.add_argument("--capid-load-in-4bit", action="store_true")
    parser.add_argument("--capid-max-new-tokens", type=int, default=512)
    parser.add_argument("--capid-trust-remote-code", action="store_true")
    parser.add_argument("--llm-provider", default="none")
    parser.add_argument("--llm-model")
    parser.add_argument("--alignment-evidence-mode", default="financial_evidence", choices=["financial_evidence", "full_outgoing_text", "highest_sensitivity_chunk"])
    parser.add_argument("--alignment-overrides", default="data/financial_generated/alignment_overrides.jsonl")
    return parser.parse_args()


def build_models(args: argparse.Namespace, egress_train: list[dict[str, Any]]) -> list[Any]:
    models = []
    for method in args.methods:
        if method == "centroid":
            models.append(
                CentroidBenchmarkModel(
                    args.centroids,
                    egress_train_rows=egress_train,
                    alignment_evidence_mode=args.alignment_evidence_mode,
                    offline=args.offline,
                )
            )
        elif method == "pii":
            pii_model = args.pii_model
            if (args.pii_backend or "").lower() == "openai_privacy_filter" and not pii_model:
                pii_model = args.openai_privacy_filter_model
            models.append(
                PiiOnlyModel(
                    backend=args.pii_backend,
                    model_id=pii_model,
                    offline=args.offline,
                    device=args.device,
                    cache_dir=args.cache_dir,
                    score_threshold=args.openai_privacy_filter_threshold,
                )
            )
        elif method == "pii_reranker":
            pii_model = args.pii_model
            if (args.pii_backend or "").lower() == "openai_privacy_filter" and not pii_model:
                pii_model = args.openai_privacy_filter_model
            models.append(
                PiiRerankerModel(
                    egress_train_rows=egress_train,
                    pii_backend=args.pii_backend,
                    pii_model=pii_model,
                    reranker_model=args.reranker_model,
                    device=args.device,
                    offline=args.offline,
                    batch_size=args.batch_size,
                    cache_dir=args.cache_dir,
                    pii_score_threshold=args.openai_privacy_filter_threshold,
                )
            )
        elif method == "opf_granite":
            models.append(
                OpenAIPrivacyFilterGraniteModel(
                    privacy_filter_model=args.openai_privacy_filter_model,
                    granite_model=args.granite_guardian_model,
                    privacy_filter_threshold=args.openai_privacy_filter_threshold,
                    device=args.device,
                    offline=args.offline,
                    cache_dir=args.cache_dir,
                    granite_max_new_tokens=args.granite_max_new_tokens,
                    granite_load_in_4bit=args.granite_load_in_4bit,
                    granite_trust_remote_code=args.granite_trust_remote_code,
                    oracle_evidence=False,
                )
            )
        elif method == "opf_granite_oracle":
            models.append(
                OpenAIPrivacyFilterGraniteModel(
                    privacy_filter_model=args.openai_privacy_filter_model,
                    granite_model=args.granite_guardian_model,
                    privacy_filter_threshold=args.openai_privacy_filter_threshold,
                    device=args.device,
                    offline=args.offline,
                    cache_dir=args.cache_dir,
                    granite_max_new_tokens=args.granite_max_new_tokens,
                    granite_load_in_4bit=args.granite_load_in_4bit,
                    granite_trust_remote_code=args.granite_trust_remote_code,
                    oracle_evidence=True,
                )
            )
        elif method == "capid":
            models.append(
                CapidAdapter(
                    model_id=args.capid_model,
                    base_model_id=args.capid_base_model,
                    device=args.device,
                    offline=args.offline,
                    cache_dir=args.cache_dir,
                    max_new_tokens=args.capid_max_new_tokens,
                    load_in_4bit=args.capid_load_in_4bit,
                    trust_remote_code=args.capid_trust_remote_code,
                )
            )
        elif method in {"qwen3guard", "shieldlm"}:
            models.append(
                ChinesePrivacyBenchmarkModel(
                    variant=method,
                    model_name=args.qwen3guard_model if method == "qwen3guard" else args.shieldlm_model,
                    model_base=args.shieldlm_model_base if method == "shieldlm" else None,
                    device=args.device,
                    offline=args.offline,
                    cache_dir=args.cache_dir,
                    max_new_tokens=args.chinese_privacy_max_new_tokens,
                )
            )
        elif method == "llm_judge":
            models.append(LlmJudgeModel(provider=args.llm_provider, model=args.llm_model, device=args.device, offline=args.offline, cache_dir=args.cache_dir))
    return models


def clone_fine_predictions(coarse_predictions: list[BenchmarkPrediction], fine_rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
    fine_by_id = rows_by_id(fine_rows)
    out: list[BenchmarkPrediction] = []
    for pred in coarse_predictions:
        if pred.task != TASK_COARSE_ALIGNMENT or pred.sample_id not in fine_by_id:
            continue
        row = fine_by_id[pred.sample_id]
        out.append(
            replace(
                pred,
                task=TASK_FINE_ALIGNMENT,
                ground_truth=row.get("_benchmark_ground_truth"),
                metadata={
                    **(pred.metadata or {}),
                    "fine_ground_truth_source": row.get("_benchmark_ground_truth_source"),
                    "fine_ground_truth_reason": row.get("_benchmark_ground_truth_reason"),
                },
            )
        )
    return out


def collect_skipped_models(predictions: list[BenchmarkPrediction]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for pred in predictions:
        if pred.status not in {STATUS_SKIPPED, STATUS_UNSUPPORTED}:
            continue
        item = by_model.setdefault(pred.model_name, {"statuses": {}, "reasons": {}})
        item["statuses"][pred.status] = item["statuses"].get(pred.status, 0) + 1
        reason = pred.error or "unknown"
        item["reasons"][reason] = item["reasons"].get(reason, 0) + 1
    return by_model


def write_errors(path: str | Path, predictions: list[BenchmarkPrediction]) -> None:
    rows = [pred.to_output_row() for pred in predictions if pred.status == STATUS_FAILED]
    write_jsonl(path, rows)


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ["SENSITIVE_EGRESS_OFFLINE"] = "1"
    out = Path(args.output_dir)
    ensure_dir(out)
    ensure_dir(out / "plots")

    anchor_train = read_rows(args.anchor_train)
    anchor_validation = read_rows(args.anchor_validation)
    egress_train = read_rows(args.egress_train)
    egress_validation = read_rows(args.egress_validation)
    if args.max_anchor_validation is not None:
        anchor_validation = anchor_validation[: args.max_anchor_validation]
    if args.max_egress_train is not None:
        egress_train = egress_train[: args.max_egress_train]
    if args.max_egress_validation is not None:
        egress_validation = egress_validation[: args.max_egress_validation]
    overrides = load_alignment_overrides(args.alignment_overrides)
    fine_rows = fine_grained_alignment_rows(egress_validation, overrides) if (overrides or egress_validation) else []

    config = {
        "anchor_train": args.anchor_train,
        "anchor_validation": args.anchor_validation,
        "egress_train": args.egress_train,
        "egress_validation": args.egress_validation,
        "max_anchor_validation": args.max_anchor_validation,
        "max_egress_train": args.max_egress_train,
        "max_egress_validation": args.max_egress_validation,
        "centroids": args.centroids,
        "methods": args.methods,
        "output_dir": args.output_dir,
        "device": args.device,
        "seed": args.seed,
        "offline": args.offline,
        "batch_size": args.batch_size,
        "cache_dir": args.cache_dir,
        "pii_backend": args.pii_backend,
        "pii_model": args.pii_model,
        "openai_privacy_filter_model": args.openai_privacy_filter_model,
        "openai_privacy_filter_threshold": args.openai_privacy_filter_threshold,
        "reranker_model": args.reranker_model,
        "granite_guardian_model": args.granite_guardian_model,
        "granite_max_new_tokens": args.granite_max_new_tokens,
        "granite_load_in_4bit": args.granite_load_in_4bit,
        "granite_trust_remote_code": args.granite_trust_remote_code,
        "capid_model": args.capid_model,
        "chinese_privacy_max_new_tokens": args.chinese_privacy_max_new_tokens,
        "qwen3guard_model": args.qwen3guard_model,
        "shieldlm_model": args.shieldlm_model,
        "shieldlm_model_base": args.shieldlm_model_base,
        "capid_base_model": args.capid_base_model,
        "capid_load_in_4bit": args.capid_load_in_4bit,
        "capid_max_new_tokens": args.capid_max_new_tokens,
        "capid_trust_remote_code": args.capid_trust_remote_code,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "alignment_evidence_mode": args.alignment_evidence_mode,
        "alignment_overrides": args.alignment_overrides,
        "manual_override_count": len(overrides),
    }
    write_json(out / "config.json", config)

    audit_summary = write_alignment_audit(egress_validation, out / "alignment_audit.csv")
    summary = dataset_summary(anchor_validation, egress_train, egress_validation, fine_rows)
    summary["anchor_train"] = {"rows": len(anchor_train)}
    summary["alignment_audit"] = audit_summary
    write_json(out / "dataset_summary.json", summary)

    models = build_models(args, egress_train)
    predictions: list[BenchmarkPrediction] = []
    runtime_metrics: dict[str, Any] = {}
    for model in models:
        model_predictions: list[BenchmarkPrediction] = []
        sensitivity = model.predict_sensitivity(anchor_validation)
        alignment = model.predict_alignment(egress_validation)
        model_predictions.extend(sensitivity)
        model_predictions.extend(alignment)
        model_predictions.extend(clone_fine_predictions(alignment, fine_rows))
        predictions.extend(model_predictions)
        runtime_metrics[model.name] = summarize_method_runtime(
            model.name,
            model_predictions,
            loading_time_s=getattr(model, "loading_time_s", None),
            extra={
                "parameter_count": getattr(model, "parameter_count", None),
                "artifact_storage_size_mb": getattr(model, "artifact_storage_size_mb", None),
                "method_config": {"class": model.__class__.__name__},
                **(getattr(model, "runtime_metadata", {}) or {}),
            },
        )
        close_model = getattr(model, "close", None)
        if callable(close_model):
            close_model()

    sensitivity_metrics = binary_sensitivity_metrics(predictions)
    coarse_alignment = alignment_metrics(predictions, TASK_COARSE_ALIGNMENT)
    fine_alignment = alignment_metrics(predictions, TASK_FINE_ALIGNMENT)
    alignment_metrics_obj = {
        "coarse_policy_alignment": coarse_alignment,
        "fine_grained_semantic_alignment": fine_alignment,
        "fine_grained_note": "Fine-grained metrics use only manual overrides and explicit subtype-constrained examples; they are never mixed with coarse policy labels.",
    }

    write_json(out / "sensitivity_metrics.json", sensitivity_metrics)
    write_json(out / "alignment_metrics.json", alignment_metrics_obj)
    write_subgroup_metrics_csv(out / "subgroup_metrics.csv", build_subgroup_metrics(predictions))
    write_jsonl(out / "predictions.jsonl", [pred.to_output_row() for pred in predictions])
    write_json(out / "runtime_metrics.json", runtime_metrics)
    write_json(out / "skipped_models.json", collect_skipped_models(predictions))
    write_errors(out / "errors.jsonl", predictions)
    created_plots = generate_plots(predictions, sensitivity_metrics, alignment_metrics_obj, runtime_metrics, out / "plots")

    print(
        json.dumps(
            {
                "output_dir": str(out),
                "methods": [model.name for model in models],
                "sensitivity_models": list(sensitivity_metrics),
                "alignment_models": list(coarse_alignment),
                "fine_grained_rows": len(fine_rows),
                "audit_summary": audit_summary,
                "plots_created": len(created_plots),
                "skipped_models": collect_skipped_models(predictions),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
