from __future__ import annotations

import time
from typing import Any

from sensitive_egress_poc.chinese_privacy import ChinesePrivacyDetector, DEFAULT_QWEN3_GUARD_MODEL, DEFAULT_SHIELDLM_MODEL

from .dataset import map_anchor_label, row_id, row_language
from .schemas import NON_SENSITIVE, SENSITIVE, STATUS_SUCCESS, TASK_COARSE_ALIGNMENT, TASK_SENSITIVITY, BenchmarkPrediction


class ChinesePrivacyBenchmarkModel:
    """Expose a Qwen3Guard or ShieldLM detector through the benchmark protocol."""

    def __init__(self, variant: str = "qwen3guard", model_name: str | None = None, **detector_kwargs: Any) -> None:
        models = {"qwen3guard": DEFAULT_QWEN3_GUARD_MODEL, "shieldlm": DEFAULT_SHIELDLM_MODEL}
        if variant not in models:
            raise ValueError(f"unknown Chinese privacy variant: {variant}")
        self.variant = variant
        started = time.perf_counter()
        self.detector = ChinesePrivacyDetector(model_name=model_name or models[variant], **detector_kwargs)
        self.loading_time_s = time.perf_counter() - started
        self.name = variant
        self.parameter_count = None
        self.artifact_storage_size_mb = None

    def predict_sensitivity(self, rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
        if self.detector.runner is None:
            reason = self.detector.init_error or "Chinese privacy model unavailable"
            return [BenchmarkPrediction.skipped(row_id(row), self.name, TASK_SENSITIVITY, reason, row=row) for row in rows]
        predictions = []
        for row in rows:
            started = time.perf_counter()
            try:
                result = self.detector.detect_privacy(str(row.get("text") or ""))
                predictions.append(BenchmarkPrediction(
                    sample_id=row_id(row), model_name=self.name, task=TASK_SENSITIVITY,
                    predicted_label=SENSITIVE if result.is_private else NON_SENSITIVE,
                    sensitivity_score=result.score, alignment_score=None, detected_entities=result.entities,
                    status=STATUS_SUCCESS, error=None,
                    metadata={"chinese_model": self.detector.model_name, "language": row_language(row), "detected_language": result.language, "model_score": result.model_score, "rule_score": result.rule_score, "centroid_score": result.centroid_score, "raw_response": result.raw_response},
                    outgoing_text=row.get("text"), financial_subtype=row.get("subtype"),
                    ground_truth=map_anchor_label(row), runtime_ms=(time.perf_counter() - started) * 1000.0,
                ))
            except Exception as exc:
                predictions.append(BenchmarkPrediction.skipped(row_id(row), self.name, TASK_SENSITIVITY, str(exc), row=row))
        return predictions

    def predict_alignment(self, rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
        return [BenchmarkPrediction.unsupported(row_id(row), self.name, TASK_COARSE_ALIGNMENT, "Chinese privacy models currently provide sensitivity detection only", row=row) for row in rows]
