from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

LABELS = ["health_private", "non_private_health", "benign"]
HEALTH_SUBTYPES = [
    "diagnosis_condition",
    "medication_prescription",
    "lab_result",
    "appointment_visit",
    "insurance_claim",
    "medical_bill",
    "wearable_vitals",
    "vaccination_record",
    "mental_health_note",
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
    health_subtype: str
    health_evidence: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationManifest(JsonMixin):
    description: str
    contains_real_personal_data: bool
    labels: list[str]
    private_subtypes: list[str]
    counts: dict[str, int]
    seed: int


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
