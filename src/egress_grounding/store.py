from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from .schemas import GroundingRecord, read_records_jsonl


class GroundingCoverageError(RuntimeError):
    def __init__(self, report: dict[str, object]) -> None:
        self.report = report
        super().__init__(f"insufficient_grounding_coverage:{report}")


class GroundingStore:
    def __init__(self, records: Iterable[GroundingRecord] | None = None, datasets: Iterable[str] | None = None) -> None:
        self.dataset_filter = set(datasets or [])
        self.records = [record for record in (records or []) if not self.dataset_filter or record.dataset in self.dataset_filter]
        self._records = sorted(self.records, key=lambda record: (record.dataset, record.source_group_id, record.id))

    @classmethod
    def load(cls, paths: Iterable[str | Path], datasets: Iterable[str] | None = None) -> "GroundingStore":
        records: list[GroundingRecord] = []
        for path in paths:
            records.extend(read_records_jsonl(path))
        return cls(records, datasets=datasets)

    def compatible(
        self,
        *,
        domain: str,
        label: str,
        subtype: str | None = None,
        roles: Iterable[str] | None = None,
        datasets: Iterable[str] | None = None,
    ) -> list[GroundingRecord]:
        role_set = set(roles or [])
        dataset_set = set(datasets or [])
        out: list[GroundingRecord] = []
        for record in self._records:
            if record.domain != domain or record.label != label:
                continue
            if role_set and record.role not in role_set:
                continue
            if dataset_set and record.dataset not in dataset_set:
                continue
            if subtype and record.subtype not in {subtype, "*"}:
                continue
            out.append(record)
        return out

    def sample(
        self,
        *,
        domain: str,
        label: str,
        subtype: str | None = None,
        roles: Iterable[str] | None = None,
        datasets: Iterable[str] | None = None,
        rng: random.Random | None = None,
    ) -> GroundingRecord | None:
        candidates = self.compatible(domain=domain, label=label, subtype=subtype, roles=roles, datasets=datasets)
        if not candidates:
            return None
        shuffler = rng or random.Random(0)
        return shuffler.choice(candidates)

    def counts(self) -> dict[str, dict[str, int]]:
        return {
            "datasets": dict(Counter(record.dataset for record in self.records)),
            "domains": dict(Counter(record.domain for record in self.records)),
            "roles": dict(Counter(record.role for record in self.records)),
            "labels": dict(Counter(record.label for record in self.records)),
        }

    def coverage_report(self, requests: Iterable[dict[str, object]]) -> dict[str, object]:
        items: list[dict[str, object]] = []
        for request in requests:
            candidates = self.compatible(
                domain=str(request.get("domain")),
                label=str(request.get("label")),
                subtype=str(request.get("subtype") or "*"),
                roles=request.get("roles") if isinstance(request.get("roles"), list) else None,
            )
            if not candidates:
                items.append({**request, "available": 0})
        return {"ok": not items, "missing": items, "store_counts": self.counts()}
