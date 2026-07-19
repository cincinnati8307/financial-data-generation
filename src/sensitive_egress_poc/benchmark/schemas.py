from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

STATUS_SUCCESS = "success"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_UNSUPPORTED = "unsupported"
STATUSES = {STATUS_SUCCESS, STATUS_SKIPPED, STATUS_FAILED, STATUS_UNSUPPORTED}

TASK_SENSITIVITY = "sensitivity"
TASK_COARSE_ALIGNMENT = "coarse_policy_alignment"
TASK_FINE_ALIGNMENT = "fine_grained_semantic_alignment"

SENSITIVE = "sensitive"
NON_SENSITIVE = "non_sensitive"
ALIGNED_SENSITIVE = "aligned_sensitive"
MISALIGNED_SENSITIVE = "misaligned_sensitive"


@dataclass
class BenchmarkPrediction:
    sample_id: str
    model_name: str
    predicted_label: str | None
    sensitivity_score: float | None
    alignment_score: float | None
    detected_entities: list[dict[str, Any]]
    status: str
    error: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    task: str | None = None
    user_intent: str | None = None
    outgoing_text: str | None = None
    financial_evidence: str | None = None
    financial_subtype: str | None = None
    carrier_id: str | None = None
    ground_truth: str | None = None
    runtime_ms: float | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown benchmark prediction status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["metadata"] = self.metadata or {}
        row["detected_entities"] = self.detected_entities or []
        return row

    def to_output_row(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "model_name": self.model_name,
            "task": self.task,
            "user_intent": self.user_intent,
            "outgoing_text": self.outgoing_text,
            "financial_evidence": self.financial_evidence,
            "financial_subtype": self.financial_subtype,
            "carrier_id": self.carrier_id,
            "ground_truth": self.ground_truth,
            "predicted_label": self.predicted_label,
            "sensitivity_score": self.sensitivity_score,
            "alignment_score": self.alignment_score,
            "detected_entities": self.detected_entities or [],
            "status": self.status,
            "error": self.error,
            "runtime_ms": self.runtime_ms,
            "metadata": self.metadata or {},
        }

    @classmethod
    def skipped(cls, sample_id: str, model_name: str, task: str, error: str, row: dict[str, Any] | None = None) -> "BenchmarkPrediction":
        row = row or {}
        ground_truth = row.get("_benchmark_ground_truth")
        if ground_truth is None:
            if row.get("label") == "financial_private":
                ground_truth = SENSITIVE
            elif row.get("label") in {"non_private_financial", "benign"}:
                ground_truth = NON_SENSITIVE
            elif row.get("expected_decision") == "allow":
                ground_truth = ALIGNED_SENSITIVE
            elif row.get("expected_decision") == "request_approval":
                ground_truth = MISALIGNED_SENSITIVE
        return cls(
            sample_id=sample_id,
            model_name=model_name,
            predicted_label=None,
            sensitivity_score=None,
            alignment_score=None,
            detected_entities=[],
            status=STATUS_SKIPPED,
            error=error,
            metadata={},
            task=task,
            user_intent=row.get("user_intent"),
            outgoing_text=row.get("text"),
            financial_evidence=row.get("financial_evidence"),
            financial_subtype=row.get("financial_subtype") or row.get("subtype"),
            carrier_id=(row.get("meta") or {}).get("carrier_id") if isinstance(row.get("meta"), dict) else None,
            ground_truth=ground_truth,
        )

    @classmethod
    def unsupported(cls, sample_id: str, model_name: str, task: str, error: str, row: dict[str, Any] | None = None) -> "BenchmarkPrediction":
        pred = cls.skipped(sample_id, model_name, task, error, row=row)
        pred.status = STATUS_UNSUPPORTED
        return pred
