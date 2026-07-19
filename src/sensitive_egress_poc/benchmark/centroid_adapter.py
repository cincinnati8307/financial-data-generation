from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Any

from sensitive_egress_poc.centroid_classifier import DEFAULT_MODEL, Embedder, _cos, _hash_embed, classify_text
from sensitive_egress_poc.io_utils import read_json

from .dataset import carrier_id, expected_financial, map_anchor_label, map_egress_decision, payload_format, row_id, row_language
from .schemas import (
    ALIGNED_SENSITIVE,
    MISALIGNED_SENSITIVE,
    NON_SENSITIVE,
    SENSITIVE,
    STATUS_SUCCESS,
    TASK_COARSE_ALIGNMENT,
    TASK_SENSITIVITY,
    BenchmarkPrediction,
)

ALIGNMENT_EVIDENCE_MODES = {"financial_evidence", "full_outgoing_text", "highest_sensitivity_chunk"}


class SimilarityEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, offline: bool = False) -> None:
        self.model_name = model_name
        self.offline = offline
        self.embedder = None
        if offline or model_name == "hash-only":
            return
        try:
            self.embedder = Embedder(model_name)
        except Exception:
            self.embedder = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.embedder is not None and self.embedder.model is not None:
            return self.embedder.encode(texts)
        return [_hash_embed(text) for text in texts]

    def similarity(self, left: str, right: str) -> float:
        lv, rv = self.encode([left, right])
        return float(_cos(lv, rv))


def split_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    cur = []
    separators = set("。！？!?\n")
    for ch in text:
        cur.append(ch)
        if ch in separators:
            chunk = "".join(cur).strip()
            if chunk:
                chunks.append(chunk)
            cur = []
    tail = "".join(cur).strip()
    if tail:
        chunks.append(tail)
    return chunks or [text]


def select_alignment_evidence(row: dict[str, Any], centroid_obj: dict[str, Any], mode: str) -> tuple[str, dict[str, Any] | None]:
    if mode not in ALIGNMENT_EVIDENCE_MODES:
        raise ValueError(f"unsupported alignment evidence mode: {mode}")
    if mode == "financial_evidence":
        return str(row.get("financial_evidence") or row.get("text") or ""), None
    if mode == "full_outgoing_text":
        return str(row.get("text") or ""), None

    best_text = str(row.get("financial_evidence") or row.get("text") or "")
    best_result: dict[str, Any] | None = None
    best_score = float("-inf")
    for chunk in split_chunks(str(row.get("text") or "")):
        result = classify_text(chunk, centroid_obj)
        score = float(result.get("financial_score") or 0.0)
        if score > best_score:
            best_text = chunk
            best_result = result
            best_score = score
    return best_text, best_result


def _macro_f1_for_threshold(scores: list[float], labels: list[str], threshold: float) -> tuple[float, float]:
    preds = [ALIGNED_SENSITIVE if score >= threshold else MISALIGNED_SENSITIVE for score in scores]
    f1s = []
    for label in [ALIGNED_SENSITIVE, MISALIGNED_SENSITIVE]:
        tp = sum(1 for truth, pred in zip(labels, preds) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(labels, preds) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(labels, preds) if truth == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    accuracy = sum(1 for truth, pred in zip(labels, preds) if truth == pred) / len(labels) if labels else 0.0
    return sum(f1s) / len(f1s), accuracy


def tune_similarity_threshold(scores: list[float], labels: list[str]) -> float:
    if not scores:
        return 0.20
    candidates = sorted(set(scores + [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]))
    candidates = [c - 1e-6 for c in candidates] + [c + 1e-6 for c in candidates]
    best_threshold = candidates[0]
    best_score = (-1.0, -1.0)
    for threshold in candidates:
        cur = _macro_f1_for_threshold(scores, labels, threshold)
        if cur > best_score:
            best_score = cur
            best_threshold = threshold
    return float(best_threshold)


def tune_centroid_alignment_threshold(rows: list[dict[str, Any]], centroid_obj: dict[str, Any], embedder: SimilarityEmbedder, evidence_mode: str) -> tuple[float, dict[str, Any]]:
    scores: list[float] = []
    labels: list[str] = []
    for row in rows:
        evidence, selected_result = select_alignment_evidence(row, centroid_obj, evidence_mode)
        sensitivity_result = selected_result or classify_text(evidence, centroid_obj)
        if sensitivity_result.get("predicted_label") != "financial_private":
            score = float("-inf")
        else:
            score = embedder.similarity(str(row.get("user_intent") or ""), evidence)
        scores.append(score)
        labels.append(map_egress_decision(row))
    threshold = tune_similarity_threshold(scores, labels)
    return threshold, {
        "threshold_source": "egress_train",
        "train_sample_count": len(rows),
        "candidate_score_min": min(scores) if scores else None,
        "candidate_score_max": max(scores) if scores else None,
    }


class CentroidBenchmarkModel:
    name = "centroid_query_similarity"

    def __init__(self, centroids_path: str, egress_train_rows: list[dict[str, Any]] | None = None, alignment_evidence_mode: str = "financial_evidence", offline: bool = False) -> None:
        start = time.perf_counter()
        self.centroids_path = centroids_path
        self.centroid_obj = read_json(centroids_path)
        self.alignment_evidence_mode = alignment_evidence_mode
        self.offline = offline
        self.embedder = SimilarityEmbedder(self.centroid_obj.get("model", DEFAULT_MODEL), offline=offline)
        self.alignment_threshold = 0.20
        self.threshold_metadata = {"threshold_source": "default", "train_sample_count": 0}
        if egress_train_rows:
            self.alignment_threshold, self.threshold_metadata = tune_centroid_alignment_threshold(
                egress_train_rows,
                self.centroid_obj,
                self.embedder,
                alignment_evidence_mode,
            )
        self.loading_time_s = time.perf_counter() - start
        self.parameter_count = None
        self.artifact_storage_size_mb = os.path.getsize(centroids_path) / (1024 * 1024) if os.path.exists(centroids_path) else None

    def predict_sensitivity(self, rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
        predictions: list[BenchmarkPrediction] = []
        for row in rows:
            started = time.perf_counter()
            result = classify_text(str(row.get("text") or ""), self.centroid_obj)
            predicted = SENSITIVE if result.get("predicted_label") == "financial_private" else NON_SENSITIVE
            predictions.append(
                BenchmarkPrediction(
                    sample_id=row_id(row),
                    model_name=self.name,
                    task=TASK_SENSITIVITY,
                    predicted_label=predicted,
                    sensitivity_score=float(result.get("financial_score") or 0.0),
                    alignment_score=None,
                    detected_entities=[],
                    status=STATUS_SUCCESS,
                    error=None,
                    metadata={
                        "centroid_result": result,
                        "original_label": row.get("label"),
                        "financial_subtype": row.get("subtype"),
                        "language": row_language(row),
                        "format": row.get("format"),
                        "style": row.get("style"),
                    },
                    outgoing_text=row.get("text"),
                    financial_subtype=row.get("subtype"),
                    ground_truth=map_anchor_label(row),
                    runtime_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
        return predictions

    def predict_alignment(self, rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
        predictions: list[BenchmarkPrediction] = []
        for row in rows:
            started = time.perf_counter()
            evidence, selected_result = select_alignment_evidence(row, self.centroid_obj, self.alignment_evidence_mode)
            result = selected_result or classify_text(evidence, self.centroid_obj)
            sensitivity_score = float(result.get("financial_score") or 0.0)
            if result.get("predicted_label") != "financial_private":
                predicted = NON_SENSITIVE
                alignment_score = None
            else:
                alignment_score = self.embedder.similarity(str(row.get("user_intent") or ""), evidence)
                predicted = ALIGNED_SENSITIVE if alignment_score >= self.alignment_threshold else MISALIGNED_SENSITIVE
            predictions.append(
                BenchmarkPrediction(
                    sample_id=row_id(row),
                    model_name=self.name,
                    task=TASK_COARSE_ALIGNMENT,
                    predicted_label=predicted,
                    sensitivity_score=sensitivity_score,
                    alignment_score=alignment_score,
                    detected_entities=[],
                    status=STATUS_SUCCESS,
                    error=None,
                    metadata={
                        "centroid_result": result,
                        "alignment_threshold": self.alignment_threshold,
                        "alignment_threshold_metadata": self.threshold_metadata,
                        "alignment_evidence_mode": self.alignment_evidence_mode,
                        "selected_evidence": evidence,
                        "financial_subtype": row.get("financial_subtype"),
                        "carrier_id": carrier_id(row),
                        "language": row_language(row),
                        "payload_format": payload_format(row),
                        "expected_financial": expected_financial(row),
                    },
                    user_intent=row.get("user_intent"),
                    outgoing_text=row.get("text"),
                    financial_evidence=row.get("financial_evidence"),
                    financial_subtype=row.get("financial_subtype"),
                    carrier_id=carrier_id(row),
                    ground_truth=map_egress_decision(row),
                    runtime_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
        return predictions

    def clone_prediction_for_fine_grained(self, prediction: BenchmarkPrediction, row: dict[str, Any]) -> BenchmarkPrediction:
        return replace(
            prediction,
            task="fine_grained_semantic_alignment",
            ground_truth=row.get("_benchmark_ground_truth"),
            metadata={**prediction.metadata, "fine_ground_truth_source": row.get("_benchmark_ground_truth_source"), "fine_ground_truth_reason": row.get("_benchmark_ground_truth_reason")},
        )
