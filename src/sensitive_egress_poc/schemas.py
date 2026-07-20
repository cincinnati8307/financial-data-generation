from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

LABELS = ["financial_private", "non_private_financial", "benign"]
PRIVATE_SUBTYPES = [
    "bank_balance",
    "transaction",
    "salary_income",
    "card_payment",
    "loan_debt",
    "invoice_receipt",
    "investment",
    "tax",
    "wallet_payment",
]


@dataclass
class JsonMixin:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SyntheticExample(JsonMixin):
    id: str
    text: str
    label: str
    subtype: str
    region: str
    language: str
    format: str
    style: str
    sensitivity_level: str
    source: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MixedEgressExample(JsonMixin):
    id: str
    user_intent: str
    text: str
    expected_categories: list[str]
    payload_labels: list[str]
    unexpected_categories: list[str]
    expected_decision: str
    format: str
    source: str
    financial_subtype: str
    financial_evidence: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationManifest(JsonMixin):
    description: str
    contains_real_personal_data: bool
    labels: list[str]
    private_subtypes: list[str]
    counts: dict[str, int]
    seed: int
    grounding: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("grounding") is None:
            data.pop("grounding", None)
        return data


@dataclass
class AugmentationManifest(JsonMixin):
    input: str
    output: str
    provider: str
    model: str
    max_inputs: int
    paraphrases_per_example: int
    include_original: bool
    attempted_sources: int
    accepted_examples: int
    rejected_examples: int
    rejection_reasons: dict[str, int]


@dataclass
class CentroidResult(JsonMixin):
    model: str
    threshold: float
    margin_threshold: float
    centroids: dict[str, list[float]]


@dataclass
class ClassificationResult(JsonMixin):
    text: str
    financial_score: float
    negative_score: float
    margin: float
    predicted_label: str
    matched_financial_subtype: str | None
    decision_hint: str
