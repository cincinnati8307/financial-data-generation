from __future__ import annotations

import gc
import time
from typing import Any

from sensitive_egress_poc.chinese_privacy import ChinesePrivacyDetector, DEFAULT_QWEN3_GUARD_MODEL, DEFAULT_SHIELDLM_MODEL

from .dataset import map_anchor_label, row_id, row_language
from .schemas import NON_SENSITIVE, SENSITIVE, STATUS_SUCCESS, TASK_COARSE_ALIGNMENT, TASK_SENSITIVITY, BenchmarkPrediction


class ChinesePrivacyBenchmarkModel:
    """Expose a Qwen3Guard or ShieldLM detector through the benchmark protocol."""

    def __init__(
        self,
        variant: str = "qwen3guard",
        model_name: str | None = None,
        model_base: str | None = None,
        **detector_kwargs: Any,
    ) -> None:
        models = {"qwen3guard": DEFAULT_QWEN3_GUARD_MODEL, "shieldlm": DEFAULT_SHIELDLM_MODEL}
        if variant not in models:
            raise ValueError(f"unknown Chinese privacy variant: {variant}")
        self.variant = variant
        self.name = variant
        self.model_name = model_name or models[variant]
        self.model_base = model_base
        self.detector_kwargs = detector_kwargs
        self.detector: ChinesePrivacyDetector | None = None
        self.loading_time_s = 0.0
        self.parameter_count = None
        self.artifact_storage_size_mb = None

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "chinese_model": self.model_name,
            "model_family": self.variant,
            "model_base": self.model_base,
        }

    def _ensure_detector(self) -> ChinesePrivacyDetector:
        if self.detector is None:
            started = time.perf_counter()
            self.detector = ChinesePrivacyDetector(
                model_name=self.model_name,
                model_family=self.variant,
                model_base=self.model_base,
                **self.detector_kwargs,
            )
            self.loading_time_s += time.perf_counter() - started
            runner = getattr(self.detector, "runner", None)
            self.parameter_count = getattr(runner, "parameter_count", None)
            self.artifact_storage_size_mb = getattr(runner, "artifact_storage_size_mb", None)
            if self.model_base is None:
                self.model_base = getattr(self.detector, "model_base", None)
        return self.detector

    def close(self) -> None:
        detector = self.detector
        if detector is None:
            return
        runner = getattr(detector, "runner", None)
        for attr in ("model", "tokenizer"):
            if runner is not None and hasattr(runner, attr):
                try:
                    delattr(runner, attr)
                except Exception:
                    pass
        self.detector = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def predict_sensitivity(self, rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
        detector = self._ensure_detector()
        if detector.runner is None:
            reason = detector.init_error or "Chinese privacy model unavailable"
            return [BenchmarkPrediction.skipped(row_id(row), self.name, TASK_SENSITIVITY, reason, row=row) for row in rows]
        predictions = []
        for row in rows:
            started = time.perf_counter()
            try:
                result = detector.detect_privacy(str(row.get("text") or ""))
                predictions.append(BenchmarkPrediction(
                    sample_id=row_id(row), model_name=self.name, task=TASK_SENSITIVITY,
                    predicted_label=SENSITIVE if result.is_private else NON_SENSITIVE,
                    sensitivity_score=result.score, alignment_score=None, detected_entities=result.entities,
                    status=STATUS_SUCCESS, error=None,
                    metadata={
                        "chinese_model": detector.model_name,
                        "model_family": detector.model_family,
                        "model_base": detector.model_base,
                        "language": row_language(row),
                        "detected_language": result.language,
                        "model_score": result.model_score,
                        "rule_score": result.rule_score,
                        "centroid_score": result.centroid_score,
                        "raw_response": result.raw_response,
                    },
                    outgoing_text=row.get("text"), financial_subtype=row.get("subtype"),
                    ground_truth=map_anchor_label(row), runtime_ms=(time.perf_counter() - started) * 1000.0,
                ))
            except Exception as exc:
                predictions.append(BenchmarkPrediction.skipped(row_id(row), self.name, TASK_SENSITIVITY, str(exc), row=row))
        return predictions

    def predict_alignment(self, rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
        return [BenchmarkPrediction.unsupported(row_id(row), self.name, TASK_COARSE_ALIGNMENT, "Chinese privacy models currently provide sensitivity detection only", row=row) for row in rows]
