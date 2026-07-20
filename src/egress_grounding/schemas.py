from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .sanitization import find_unsafe_identifier

DOMAINS = {"financial", "health"}
ROLES = {"private_candidate", "public_negative", "distribution"}
DATASETS = {"cfpb", "banking77", "berka", "nhanes", "pubmed", "generic_jsonl"}
LABELS = {"financial_private", "health_private", "non_private_financial", "non_private_health", "benign"}


@dataclass
class GroundingRecord:
    id: str
    dataset: str
    domain: str
    role: str
    label: str
    subtype: str
    source_group_id: str
    facts: dict[str, Any] = field(default_factory=dict)
    region: str = "global"
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "GroundingRecord":
        fields = {
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "domain": row.get("domain"),
            "role": row.get("role"),
            "label": row.get("label"),
            "subtype": row.get("subtype", "*"),
            "source_group_id": row.get("source_group_id"),
            "facts": row.get("facts") or {},
            "region": row.get("region", "global"),
            "tags": row.get("tags") or [],
            "meta": row.get("meta") or {},
        }
        rec = cls(**fields)
        rec.validate()
        return rec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        required = {
            "id": self.id,
            "dataset": self.dataset,
            "domain": self.domain,
            "role": self.role,
            "label": self.label,
            "subtype": self.subtype,
            "source_group_id": self.source_group_id,
        }
        missing = [name for name, value in required.items() if value in {None, ""}]
        if missing:
            raise ValueError(f"missing_fields:{','.join(missing)}")
        if self.dataset not in DATASETS:
            raise ValueError(f"unknown_dataset:{self.dataset}")
        if self.domain not in DOMAINS:
            raise ValueError(f"unknown_domain:{self.domain}")
        if self.role not in ROLES:
            raise ValueError(f"unknown_role:{self.role}")
        if self.label not in LABELS:
            raise ValueError(f"unknown_label:{self.label}")
        if self.role == "private_candidate" and self.label not in {"financial_private", "health_private"}:
            raise ValueError("private_candidate_label_mismatch")
        if self.role == "public_negative" and self.label not in {"non_private_financial", "non_private_health", "benign"}:
            raise ValueError("public_negative_label_mismatch")
        if self.label.startswith("financial") and self.domain != "financial":
            raise ValueError("financial_label_domain_mismatch")
        if self.label.startswith("health") and self.domain != "health":
            raise ValueError("health_label_domain_mismatch")
        if not isinstance(self.facts, dict):
            raise ValueError("facts_must_be_object")
        if not isinstance(self.tags, list):
            raise ValueError("tags_must_be_list")
        unsafe_path = find_unsafe_identifier({"facts": self.facts, "meta": self.meta})
        if unsafe_path:
            raise ValueError(f"unsafe_or_raw_field:{unsafe_path}")


def record_to_json(record: GroundingRecord) -> str:
    record.validate()
    return json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)


def records_to_jsonl(records: Iterable[GroundingRecord]) -> str:
    return "\n".join(record_to_json(record) for record in records) + "\n"


def write_records_jsonl(path: str | Path, records: Iterable[GroundingRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record_to_json(record))
            f.write("\n")


def read_records_jsonl(path: str | Path) -> list[GroundingRecord]:
    records: list[GroundingRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(GroundingRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}:{exc}") from exc
    return records
