from __future__ import annotations

import time
from typing import Any

from .base import ModelUnavailable
from .dataset import carrier_id, expected_financial, map_anchor_label, map_egress_decision, payload_format, row_id, row_language
from .granite_guardian import (
    DEFAULT_GRANITE_GUARDIAN_MODEL,
    GRANITE_GUARDIAN_LANGUAGE_LIMITATION,
    GraniteGuardianRunner,
    HuggingFaceGraniteGuardianRunner,
)
from .pii_adapter import (
    DEFAULT_OPENAI_PRIVACY_FILTER_FINANCIAL_LABELS,
    DEFAULT_OPENAI_PRIVACY_FILTER_MODEL,
    PiiDetector,
    containing_sentence,
    openai_privacy_filter_entities_to_sensitivity,
    openai_privacy_filter_financial_entities,
)
from .pii_adapter import OpenAIPrivacyFilterDetector
from .schemas import (
    ALIGNED_SENSITIVE,
    MISALIGNED_SENSITIVE,
    NON_SENSITIVE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_UNSUPPORTED,
    TASK_COARSE_ALIGNMENT,
    TASK_SENSITIVITY,
    BenchmarkPrediction,
)

OPENAI_PRIVACY_FILTER_LIMITATION = (
    "OpenAI Privacy Filter detects explicit identifiers according to its fixed taxonomy. "
    "It does not natively represent semantic financial facts such as salary, account balance, "
    "debt, investment value, tax income, or transaction amount without an identifier."
)


class OpenAIPrivacyFilterGraniteModel:
    name = "openai_privacy_filter_plus_granite_guardian"

    def __init__(
        self,
        privacy_filter_model: str = DEFAULT_OPENAI_PRIVACY_FILTER_MODEL,
        granite_model: str = DEFAULT_GRANITE_GUARDIAN_MODEL,
        privacy_filter_threshold: float = 0.5,
        device: str = "auto",
        offline: bool = False,
        cache_dir: str | None = None,
        granite_max_new_tokens: int = 20,
        granite_load_in_4bit: bool = False,
        granite_trust_remote_code: bool = False,
        detector: PiiDetector | None = None,
        guardian_runner: GraniteGuardianRunner | None = None,
        oracle_evidence: bool = False,
        financial_labels: set[str] | frozenset[str] | None = None,
    ) -> None:
        started = time.perf_counter()
        self.name = "granite_guardian_oracle_evidence" if oracle_evidence else self.__class__.name
        self.privacy_filter_model = privacy_filter_model
        self.granite_model = granite_model
        self.privacy_filter_threshold = privacy_filter_threshold
        self.device = device
        self.offline = offline
        self.cache_dir = cache_dir
        self.granite_max_new_tokens = granite_max_new_tokens
        self.granite_load_in_4bit = granite_load_in_4bit
        self.granite_trust_remote_code = granite_trust_remote_code
        self.oracle_evidence = oracle_evidence
        self.financial_labels = set(financial_labels or DEFAULT_OPENAI_PRIVACY_FILTER_FINANCIAL_LABELS)
        self.detector_error: str | None = None
        self.guardian_error: str | None = None
        self.granite_invocation_count = 0
        self.granite_skipped_by_gate_count = 0

        if detector is not None:
            self.detector = detector
        elif oracle_evidence:
            self.detector = None
        else:
            try:
                self.detector = OpenAIPrivacyFilterDetector(
                    model_id=privacy_filter_model,
                    device=device,
                    cache_dir=cache_dir,
                    offline=offline,
                    score_threshold=privacy_filter_threshold,
                )
            except ModelUnavailable as exc:
                self.detector = None
                self.detector_error = str(exc)

        if guardian_runner is not None:
            self.guardian_runner = guardian_runner
            self.granite_model = guardian_runner.model_name
        else:
            try:
                self.guardian_runner = HuggingFaceGraniteGuardianRunner(
                    model_id=granite_model,
                    device=device,
                    offline=offline,
                    cache_dir=cache_dir,
                    max_new_tokens=granite_max_new_tokens,
                    load_in_4bit=granite_load_in_4bit,
                    trust_remote_code=granite_trust_remote_code,
                )
            except ModelUnavailable as exc:
                self.guardian_runner = None
                self.guardian_error = str(exc)

        self.loading_time_s = time.perf_counter() - started
        self.parameter_count = self._combined_parameter_count()
        self.artifact_storage_size_mb = self._combined_artifact_storage_size_mb()

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "components": self.component_metadata,
            "granite_invocation_count": self.granite_invocation_count,
            "granite_skipped_by_gate_count": self.granite_skipped_by_gate_count,
            "oracle_evidence": self.oracle_evidence,
            "end_to_end_detector": not self.oracle_evidence,
            "language_limitation": GRANITE_GUARDIAN_LANGUAGE_LIMITATION,
        }

    @property
    def component_metadata(self) -> dict[str, Any]:
        return {
            "privacy_filter": {
                "model": self.privacy_filter_model,
                "parameter_count": getattr(self.detector, "parameter_count", None),
                "loading_time_s": getattr(self.detector, "loading_time_s", None),
            },
            "granite_guardian": {
                "model": self.granite_model,
                "parameter_count": getattr(self.guardian_runner, "parameter_count", None),
                "loading_time_s": getattr(self.guardian_runner, "loading_time_s", None),
                "load_in_4bit": self.granite_load_in_4bit,
                "device": self.device,
                "language_limitation": GRANITE_GUARDIAN_LANGUAGE_LIMITATION,
            },
        }

    def predict_sensitivity(self, rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
        if self.oracle_evidence:
            return [self._unsupported_sensitivity(row, "oracle evidence variant is an alignment-only diagnostic and does not run Task A") for row in rows]
        if self.detector is None:
            return [BenchmarkPrediction.skipped(row_id(row), self.name, TASK_SENSITIVITY, self.detector_error or "OpenAI Privacy Filter unavailable", row=row) for row in rows]
        predictions: list[BenchmarkPrediction] = []
        for row in rows:
            started = time.perf_counter()
            text = str(row.get("text") or "")
            try:
                entities = self.detector.detect(text, language=row_language(row))
                financial_entities = openai_privacy_filter_financial_entities(entities, financial_labels=self.financial_labels)
                label, score = openai_privacy_filter_entities_to_sensitivity(entities, financial_labels=self.financial_labels)
                status = STATUS_SUCCESS
                error = None
            except Exception as exc:
                entities = []
                financial_entities = []
                label = None
                score = None
                status = STATUS_FAILED
                error = str(exc)
            predictions.append(
                BenchmarkPrediction(
                    sample_id=row_id(row),
                    model_name=self.name,
                    task=TASK_SENSITIVITY,
                    predicted_label=label,
                    sensitivity_score=score,
                    alignment_score=None,
                    detected_entities=entities,
                    status=status,
                    error=error,
                    metadata=self._sensitivity_metadata(row, entities, financial_entities),
                    outgoing_text=row.get("text"),
                    financial_subtype=row.get("subtype"),
                    ground_truth=map_anchor_label(row),
                    runtime_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
        return predictions

    def predict_alignment(self, rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
        if self.oracle_evidence:
            return self._predict_oracle_alignment(rows)
        if self.detector is None:
            return [BenchmarkPrediction.skipped(row_id(row), self.name, TASK_COARSE_ALIGNMENT, self.detector_error or "OpenAI Privacy Filter unavailable", row=row) for row in rows]
        if self.guardian_runner is None:
            return [BenchmarkPrediction.skipped(row_id(row), self.name, TASK_COARSE_ALIGNMENT, self.guardian_error or "Granite Guardian unavailable", row=row) for row in rows]

        predictions: list[BenchmarkPrediction] = []
        for row in rows:
            started = time.perf_counter()
            text = str(row.get("text") or "")
            entities: list[dict[str, Any]] = []
            financial_entities: list[dict[str, Any]] = []
            evidence = ""
            evidence_source = "none"
            guardian_result: dict[str, Any] = {}
            try:
                entities = self.detector.detect(text, language=row_language(row))
                financial_entities = openai_privacy_filter_financial_entities(entities, financial_labels=self.financial_labels)
                sensitivity_label, sensitivity_score = openai_privacy_filter_entities_to_sensitivity(entities, financial_labels=self.financial_labels)
                if sensitivity_label == NON_SENSITIVE:
                    self.granite_skipped_by_gate_count += 1
                    predictions.append(self._strict_gate_prediction(row, entities, sensitivity_score, started))
                    continue

                selected_entity = _highest_confidence_entity(financial_entities)
                evidence, evidence_source = _select_detected_evidence(text, selected_entity)
                self.granite_invocation_count += 1
                guardian_result = self.guardian_runner.score_context_relevance(str(row.get("user_intent") or ""), evidence)
                predicted = _guardian_result_to_alignment_label(guardian_result)
                alignment_score = guardian_result.get("relevance_probability")
                status = STATUS_SUCCESS
                error = None
            except Exception as exc:
                predicted = None
                sensitivity_score = None
                alignment_score = None
                status = STATUS_FAILED
                error = str(exc)
            predictions.append(
                BenchmarkPrediction(
                    sample_id=row_id(row),
                    model_name=self.name,
                    task=TASK_COARSE_ALIGNMENT,
                    predicted_label=predicted,
                    sensitivity_score=sensitivity_score,
                    alignment_score=alignment_score,
                    detected_entities=entities,
                    status=status,
                    error=error,
                    metadata={
                        **self._alignment_metadata(row),
                        "native_detected_labels": _labels(entities),
                        "financial_detected_labels": _labels(financial_entities),
                        "selected_evidence": evidence,
                        "evidence_source": evidence_source,
                        "pipeline_gate": "financial_pii_detected",
                        "granite_called": status == STATUS_SUCCESS,
                        "granite_raw_label": guardian_result.get("raw_label"),
                        "granite_risk_probability": guardian_result.get("risk_probability"),
                        "granite_relevance_probability": guardian_result.get("relevance_probability"),
                        "granite_confidence": guardian_result.get("confidence"),
                        "granite_raw_output": guardian_result.get("raw_output"),
                        "granite_context_relevance_yes_means_irrelevant": True,
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

    def _predict_oracle_alignment(self, rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
        if self.guardian_runner is None:
            return [BenchmarkPrediction.skipped(row_id(row), self.name, TASK_COARSE_ALIGNMENT, self.guardian_error or "Granite Guardian unavailable", row=row) for row in rows]
        predictions: list[BenchmarkPrediction] = []
        for row in rows:
            started = time.perf_counter()
            evidence = str(row.get("financial_evidence") or "").strip()
            if not evidence:
                predictions.append(
                    BenchmarkPrediction(
                        sample_id=row_id(row),
                        model_name=self.name,
                        task=TASK_COARSE_ALIGNMENT,
                        predicted_label=None,
                        sensitivity_score=None,
                        alignment_score=None,
                        detected_entities=[],
                        status=STATUS_SKIPPED,
                        error="oracle financial_evidence is missing or empty",
                        metadata={
                            **self._alignment_metadata(row),
                            "oracle_evidence": True,
                            "evidence_source": "dataset_financial_evidence",
                            "end_to_end_detector": False,
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
                continue
            try:
                self.granite_invocation_count += 1
                guardian_result = self.guardian_runner.score_context_relevance(str(row.get("user_intent") or ""), evidence)
                predicted = _guardian_result_to_alignment_label(guardian_result)
                status = STATUS_SUCCESS
                error = None
            except Exception as exc:
                guardian_result = {}
                predicted = None
                status = STATUS_FAILED
                error = str(exc)
            predictions.append(
                BenchmarkPrediction(
                    sample_id=row_id(row),
                    model_name=self.name,
                    task=TASK_COARSE_ALIGNMENT,
                    predicted_label=predicted,
                    sensitivity_score=None,
                    alignment_score=guardian_result.get("relevance_probability"),
                    detected_entities=[],
                    status=status,
                    error=error,
                    metadata={
                        **self._alignment_metadata(row),
                        "oracle_evidence": True,
                        "evidence_source": "dataset_financial_evidence",
                        "selected_evidence": evidence,
                        "end_to_end_detector": False,
                        "granite_called": status == STATUS_SUCCESS,
                        "granite_raw_label": guardian_result.get("raw_label"),
                        "granite_risk_probability": guardian_result.get("risk_probability"),
                        "granite_relevance_probability": guardian_result.get("relevance_probability"),
                        "granite_confidence": guardian_result.get("confidence"),
                        "granite_raw_output": guardian_result.get("raw_output"),
                        "granite_context_relevance_yes_means_irrelevant": True,
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

    def _strict_gate_prediction(
        self,
        row: dict[str, Any],
        entities: list[dict[str, Any]],
        sensitivity_score: float | None,
        started: float,
    ) -> BenchmarkPrediction:
        return BenchmarkPrediction(
            sample_id=row_id(row),
            model_name=self.name,
            task=TASK_COARSE_ALIGNMENT,
            predicted_label=NON_SENSITIVE,
            sensitivity_score=sensitivity_score,
            alignment_score=None,
            detected_entities=entities,
            status=STATUS_SUCCESS,
            error=None,
            metadata={
                **self._alignment_metadata(row),
                "native_detected_labels": _labels(entities),
                "financial_detected_labels": [],
                "selected_evidence": "",
                "evidence_source": "none",
                "pipeline_gate": "no_financial_pii_detected",
                "granite_called": False,
            },
            user_intent=row.get("user_intent"),
            outgoing_text=row.get("text"),
            financial_evidence=row.get("financial_evidence"),
            financial_subtype=row.get("financial_subtype"),
            carrier_id=carrier_id(row),
            ground_truth=map_egress_decision(row),
            runtime_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _unsupported_sensitivity(self, row: dict[str, Any], reason: str) -> BenchmarkPrediction:
        return BenchmarkPrediction(
            sample_id=row_id(row),
            model_name=self.name,
            task=TASK_SENSITIVITY,
            predicted_label=None,
            sensitivity_score=None,
            alignment_score=None,
            detected_entities=[],
            status=STATUS_UNSUPPORTED,
            error=reason,
            metadata={
                "oracle_evidence": True,
                "end_to_end_detector": False,
                "diagnostic_only": True,
            },
            outgoing_text=row.get("text"),
            financial_subtype=row.get("subtype"),
            ground_truth=map_anchor_label(row),
        )

    def _sensitivity_metadata(
        self,
        row: dict[str, Any],
        entities: list[dict[str, Any]],
        financial_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "privacy_filter_model": self.privacy_filter_model,
            "privacy_filter_threshold": self.privacy_filter_threshold,
            "native_detected_labels": _labels(entities),
            "financial_detected_labels": _labels(financial_entities),
            "financial_label_set": sorted(self.financial_labels),
            "language": row_language(row),
            "format": row.get("format"),
            "style": row.get("style"),
            "original_label": row.get("label"),
            "financial_subtype": row.get("subtype"),
            "explicit_identifier_only": True,
            "semantic_financial_detection": False,
            "limitation": OPENAI_PRIVACY_FILTER_LIMITATION,
        }

    def _alignment_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "privacy_filter_model": self.privacy_filter_model,
            "privacy_filter_threshold": self.privacy_filter_threshold,
            "granite_guardian_model": self.granite_model,
            "granite_max_new_tokens": self.granite_max_new_tokens,
            "granite_load_in_4bit": self.granite_load_in_4bit,
            "granite_trust_remote_code": self.granite_trust_remote_code,
            "financial_label_set": sorted(self.financial_labels),
            "explicit_identifier_only": True,
            "semantic_financial_detection": False,
            "openai_privacy_filter_limitation": OPENAI_PRIVACY_FILTER_LIMITATION,
            "granite_language_limitation": GRANITE_GUARDIAN_LANGUAGE_LIMITATION,
            "financial_subtype": row.get("financial_subtype"),
            "carrier_id": carrier_id(row),
            "language": row_language(row),
            "payload_format": payload_format(row),
            "expected_financial": expected_financial(row),
        }

    def _combined_parameter_count(self) -> int | None:
        counts = [
            getattr(self.detector, "parameter_count", None),
            getattr(self.guardian_runner, "parameter_count", None),
        ]
        known = [int(count) for count in counts if isinstance(count, int)]
        return sum(known) if known else None

    def _combined_artifact_storage_size_mb(self) -> float | None:
        sizes = [
            getattr(self.detector, "artifact_storage_size_mb", None),
            getattr(self.guardian_runner, "artifact_storage_size_mb", None),
        ]
        known = [float(size) for size in sizes if isinstance(size, (int, float))]
        return sum(known) if known else None


def _guardian_result_to_alignment_label(result: dict[str, Any]) -> str:
    raw_label = str(result.get("raw_label") or "").strip().title()
    if raw_label == "Yes":
        return MISALIGNED_SENSITIVE
    if raw_label == "No":
        return ALIGNED_SENSITIVE
    raise ValueError("Granite Guardian result is missing a parseable raw_label of Yes or No")


def _highest_confidence_entity(entities: list[dict[str, Any]]) -> dict[str, Any]:
    return max(entities, key=lambda entity: float(entity.get("score") if entity.get("score") is not None else -1.0))


def _select_detected_evidence(text: str, entity: dict[str, Any]) -> tuple[str, str]:
    start = _coerce_int(entity.get("start"))
    end = _coerce_int(entity.get("end"))
    if start is not None and end is not None and 0 <= start <= end <= len(text):
        return containing_sentence(text, start, end), "detected_financial_span_sentence"

    entity_text = str(entity.get("text") or "").strip()
    if entity_text:
        found = text.find(entity_text)
        if found >= 0:
            return containing_sentence(text, found, found + len(entity_text)), "detected_financial_span_sentence_text_match"

    if start is not None and 0 <= start < len(text):
        left = max(0, start - 160)
        right = min(len(text), start + 160)
        return text[left:right].strip(), "detected_financial_span_local_context"
    if entity_text:
        return entity_text, "detected_financial_span_text"
    return text[:320].strip(), "detected_financial_span_local_context"


def _labels(entities: list[dict[str, Any]]) -> list[str]:
    return sorted({str(entity.get("label") or entity.get("type") or "").upper() for entity in entities if entity.get("label") or entity.get("type")})


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None
