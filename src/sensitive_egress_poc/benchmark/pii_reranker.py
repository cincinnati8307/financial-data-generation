from __future__ import annotations

import os
import time
from typing import Any, Protocol

from sensitive_egress_poc.centroid_classifier import _cos, _hash_embed

from .base import ModelUnavailable
from .centroid_adapter import split_chunks, tune_similarity_threshold
from .dataset import carrier_id, expected_financial, map_anchor_label, map_egress_decision, payload_format, row_id, row_language
from .pii_adapter import PiiDetector, containing_sentence, make_pii_detector, pii_entities_to_sensitivity, private_financial_entities
from .schemas import (
    ALIGNED_SENSITIVE,
    MISALIGNED_SENSITIVE,
    NON_SENSITIVE,
    STATUS_SUCCESS,
    TASK_COARSE_ALIGNMENT,
    TASK_SENSITIVITY,
    BenchmarkPrediction,
)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class RerankerScorer(Protocol):
    model_name: str

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        ...


class HashSimilarityReranker:
    model_name = "lexical_hash_similarity"

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(_cos(_hash_embed(query), _hash_embed(evidence))) for query, evidence in pairs]


class CrossEncoderReranker:
    def __init__(self, model_id: str, device: str = "auto", offline: bool = False, cache_dir: str | None = None) -> None:
        if offline and not os.path.exists(model_id):
            raise ModelUnavailable("offline mode requires --reranker-model lexical or a local model path")
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise ModelUnavailable(f"sentence-transformers CrossEncoder unavailable: {exc}") from exc
        kwargs: dict[str, Any] = {}
        if device and device != "auto":
            kwargs["device"] = device
        if cache_dir:
            kwargs["cache_folder"] = cache_dir
        if offline:
            kwargs["local_files_only"] = True
        try:
            self.model = CrossEncoder(model_id, **kwargs)
        except TypeError:
            kwargs.pop("local_files_only", None)
            if offline:
                raise ModelUnavailable("installed sentence-transformers does not support local_files_only for offline reranker loading")
            self.model = CrossEncoder(model_id, **kwargs)
        except Exception as exc:
            raise ModelUnavailable(f"failed to load reranker model {model_id}: {exc}") from exc
        self.model_name = model_id

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.model.predict([[q, e] for q, e in pairs])
        return [float(score) for score in scores]


def make_reranker(model_id: str | None, device: str = "auto", offline: bool = False, cache_dir: str | None = None) -> RerankerScorer:
    model_id = model_id or DEFAULT_RERANKER_MODEL
    if model_id in {"lexical", "hash", "hash-only", "local_hash"}:
        return HashSimilarityReranker()
    return CrossEncoderReranker(model_id, device=device, offline=offline, cache_dir=cache_dir)


def select_reranker_evidence(row: dict[str, Any], detector: PiiDetector, entities: list[dict[str, Any]] | None = None) -> tuple[str, str, list[dict[str, Any]]]:
    text = str(row.get("text") or "")
    entities = entities if entities is not None else detector.detect(text, language=row_language(row))
    private_entities = sorted(private_financial_entities(entities), key=lambda e: float(e.get("score") or 0.0), reverse=True)
    if private_entities:
        entity = private_entities[0]
        return containing_sentence(text, entity.get("start"), entity.get("end")), "detected_private_span_sentence", entities

    if row.get("financial_evidence"):
        return str(row.get("financial_evidence") or ""), "financial_evidence", entities

    best_chunk = text
    best_count = -1
    for chunk in split_chunks(text):
        chunk_entities = detector.detect(chunk, language=row_language(row))
        count = len(private_financial_entities(chunk_entities))
        if count > best_count:
            best_count = count
            best_chunk = chunk
    if best_chunk:
        return best_chunk, "highest_sensitivity_chunk", entities
    return text, "full_outgoing_text", entities


def tune_reranker_alignment_threshold(rows: list[dict[str, Any]], detector: PiiDetector, scorer: RerankerScorer, batch_size: int = 16) -> tuple[float, dict[str, Any]]:
    pairs: list[tuple[str, str]] = []
    labels: list[str] = []
    no_sensitive_indices: set[int] = set()
    for index, row in enumerate(rows):
        text = str(row.get("text") or "")
        entities = detector.detect(text, language=row_language(row))
        if not private_financial_entities(entities):
            pairs.append((str(row.get("user_intent") or ""), ""))
            no_sensitive_indices.add(index)
        else:
            evidence, _, _ = select_reranker_evidence(row, detector, entities=entities)
            pairs.append((str(row.get("user_intent") or ""), evidence))
        labels.append(map_egress_decision(row))

    scores: list[float] = []
    for start in range(0, len(pairs), max(1, batch_size)):
        batch = pairs[start : start + max(1, batch_size)]
        scores.extend(scorer.score_pairs(batch))
    for index in no_sensitive_indices:
        scores[index] = float("-inf")
    threshold = tune_similarity_threshold(scores, labels)
    return threshold, {
        "threshold_source": "egress_train",
        "train_sample_count": len(rows),
        "reranker_model": scorer.model_name,
        "no_sensitive_train_count": len(no_sensitive_indices),
    }


class PiiRerankerModel:
    name = "pii_plus_reranker"

    def __init__(
        self,
        egress_train_rows: list[dict[str, Any]] | None = None,
        pii_backend: str = "regex",
        pii_model: str | None = None,
        reranker_model: str | None = None,
        device: str = "auto",
        offline: bool = False,
        batch_size: int = 16,
        cache_dir: str | None = None,
        detector: PiiDetector | None = None,
        scorer: RerankerScorer | None = None,
    ) -> None:
        started = time.perf_counter()
        self.pii_backend = pii_backend
        self.pii_model = pii_model
        self.reranker_model = reranker_model or DEFAULT_RERANKER_MODEL
        self.batch_size = batch_size
        self.offline = offline
        self.detector_error: str | None = None
        self.reranker_error: str | None = None
        try:
            self.detector = detector or make_pii_detector(pii_backend, model_id=pii_model, offline=offline)
        except ModelUnavailable as exc:
            self.detector = None
            self.detector_error = str(exc)
        try:
            self.scorer = scorer or make_reranker(reranker_model, device=device, offline=offline, cache_dir=cache_dir)
        except ModelUnavailable as exc:
            self.scorer = None
            self.reranker_error = str(exc)
        self.alignment_threshold = 0.20
        self.threshold_metadata = {"threshold_source": "default", "train_sample_count": 0}
        if self.detector is not None and self.scorer is not None and egress_train_rows:
            self.alignment_threshold, self.threshold_metadata = tune_reranker_alignment_threshold(egress_train_rows, self.detector, self.scorer, batch_size=batch_size)
        self.loading_time_s = time.perf_counter() - started
        self.parameter_count = None
        self.artifact_storage_size_mb = None

    def predict_sensitivity(self, rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
        if self.detector is None:
            return [BenchmarkPrediction.skipped(row_id(row), self.name, TASK_SENSITIVITY, self.detector_error or "PII detector unavailable", row=row) for row in rows]
        predictions: list[BenchmarkPrediction] = []
        for row in rows:
            started = time.perf_counter()
            entities = self.detector.detect(str(row.get("text") or ""), language=row_language(row))
            label, score = pii_entities_to_sensitivity(entities)
            predictions.append(
                BenchmarkPrediction(
                    sample_id=row_id(row),
                    model_name=self.name,
                    task=TASK_SENSITIVITY,
                    predicted_label=label,
                    sensitivity_score=score,
                    alignment_score=None,
                    detected_entities=entities,
                    status=STATUS_SUCCESS,
                    error=None,
                    metadata={
                        "pii_backend": self.pii_backend,
                        "pii_model": self.pii_model,
                        "reranker_model": self.reranker_model,
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
        if self.detector is None:
            return [BenchmarkPrediction.skipped(row_id(row), self.name, TASK_COARSE_ALIGNMENT, self.detector_error or "PII detector unavailable", row=row) for row in rows]
        if self.scorer is None:
            return [BenchmarkPrediction.skipped(row_id(row), self.name, TASK_COARSE_ALIGNMENT, self.reranker_error or "reranker unavailable", row=row) for row in rows]
        predictions: list[BenchmarkPrediction] = []
        for row in rows:
            started = time.perf_counter()
            text = str(row.get("text") or "")
            entities = self.detector.detect(text, language=row_language(row))
            sensitivity_label, sensitivity_score = pii_entities_to_sensitivity(entities)
            if sensitivity_label == NON_SENSITIVE:
                predicted = NON_SENSITIVE
                alignment_score = None
                evidence = ""
                evidence_source = "none"
            else:
                evidence, evidence_source, entities = select_reranker_evidence(row, self.detector, entities=entities)
                alignment_score = self.scorer.score_pairs([(str(row.get("user_intent") or ""), evidence)])[0]
                predicted = ALIGNED_SENSITIVE if alignment_score >= self.alignment_threshold else MISALIGNED_SENSITIVE
            predictions.append(
                BenchmarkPrediction(
                    sample_id=row_id(row),
                    model_name=self.name,
                    task=TASK_COARSE_ALIGNMENT,
                    predicted_label=predicted,
                    sensitivity_score=sensitivity_score,
                    alignment_score=alignment_score,
                    detected_entities=entities,
                    status=STATUS_SUCCESS,
                    error=None,
                    metadata={
                        "pii_backend": self.pii_backend,
                        "pii_model": self.pii_model,
                        "reranker_model": self.reranker_model,
                        "alignment_threshold": self.alignment_threshold,
                        "alignment_threshold_metadata": self.threshold_metadata,
                        "evidence_source": evidence_source,
                        "selected_evidence": evidence,
                        "semantic_relevance_is_not_authorization": True,
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
