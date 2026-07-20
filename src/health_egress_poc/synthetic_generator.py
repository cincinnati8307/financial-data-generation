from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from egress_grounding.schemas import GroundingRecord
from egress_grounding.store import GroundingCoverageError, GroundingStore

from .filters import validate_synthetic_example
from .schemas import HEALTH_SUBTYPES, MixedEgressExample, SyntheticExample

CATALOG_PATH = Path(__file__).with_name("template_catalog.json")

FORMATS = ["natural_sentence", "key_value", "csv_row", "json", "agent_summary", "email_snippet", "chat_transcript", "ocr_fragment", "app_notification", "form_dump", "spreadsheet_row"]
STYLES = ["zh_casual", "zh_formal", "zh_en_codeswitch", "agent_summary", "email_mixed", "mobile_terse", "clinic_note"]

GROUNDING_MODES = {"synthetic-only", "hybrid", "grounded-only"}
GROUNDING_CONTEXT_KEYS = {
    "clinic",
    "hospital",
    "insurer",
    "medication",
    "condition",
    "test_name",
    "vaccine",
    "masked_ref",
    "amount",
    "lab_value",
    "heart_rate",
    "sleep_hours",
    "blood_pressure",
    "glucose",
    "date",
    "time_slot",
    "score",
}


def load_template_catalog(path: str | Path | None = None) -> dict[str, Any]:
    catalog_path = Path(path) if path is not None else CATALOG_PATH
    with catalog_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class SyntheticHealthGenerator:
    def __init__(
        self,
        seed: int = 1337,
        catalog: dict[str, Any] | None = None,
        grounding_store: GroundingStore | None = None,
        grounding_mode: str = "synthetic-only",
        grounding_ratio: float = 0.0,
        grounding_datasets: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if grounding_mode not in GROUNDING_MODES:
            raise ValueError(f"unknown_grounding_mode:{grounding_mode}")
        if not 0.0 <= grounding_ratio <= 1.0:
            raise ValueError("grounding_ratio_must_be_between_0_and_1")
        self.rng = random.Random(seed)
        self.counter = 0
        self.catalog = catalog if catalog is not None else load_template_catalog()
        self.grounding_store = grounding_store
        self.grounding_mode = grounding_mode
        self.grounding_ratio = grounding_ratio
        self.grounding_datasets = set(grounding_datasets or [])
        self.grounding_eligible_counts: Counter[str] = Counter()
        self.grounding_attempt_counts: Counter[str] = Counter()
        self.grounded_counts: Counter[str] = Counter()
        self.grounding_fallback_counts: Counter[str] = Counter()

    def _id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:06d}"

    def _pool(self, name: str) -> list[str]:
        return list(self.catalog["entity_pools"][name])

    def _choice(self, values: list[Any]) -> Any:
        return self.rng.choice(values)

    def _amount(self) -> str:
        currency = self._choice(self._pool("currencies"))
        return f"{currency} {self.rng.randint(40, 6000):,}"

    def _context(self) -> dict[str, Any]:
        return {
            "clinic": self._choice(self._pool("clinics")),
            "hospital": self._choice(self._pool("hospitals")),
            "insurer": self._choice(self._pool("insurers")),
            "medication": self._choice(self._pool("medications")),
            "condition": self._choice(self._pool("conditions")),
            "test_name": self._choice(self._pool("tests")),
            "vaccine": self._choice(self._pool("vaccines")),
            "masked_ref": self._choice(self._pool("masked_refs")),
            "amount": self._amount(),
            "lab_value": self._choice(self._pool("lab_values")),
            "heart_rate": self.rng.randint(58, 108),
            "sleep_hours": round(self.rng.uniform(4.5, 8.5), 1),
            "blood_pressure": f"{self.rng.randint(105, 150)}/{self.rng.randint(65, 95)}",
            "glucose": round(self.rng.uniform(4.2, 10.5), 1),
            "date": self._choice(self._pool("dates")),
            "time_slot": self._choice(self._pool("time_slots")),
            "score": self.rng.randint(4, 9),
        }


    def _maybe_grounding(
        self,
        *,
        label: str,
        subtype: str,
        roles: tuple[str, ...],
    ) -> GroundingRecord | None:
        key = f"{label}:{subtype}"
        if self.grounding_mode == "synthetic-only" or self.grounding_store is None:
            if self.grounding_mode == "grounded-only":
                raise GroundingCoverageError(
                    {
                        "ok": False,
                        "missing": [{"domain": "health", "label": label, "subtype": subtype, "roles": list(roles), "available": 0}],
                        "store_counts": {},
                    }
                )
            return None
        self.grounding_eligible_counts[key] += 1
        if self.grounding_mode == "hybrid" and self.rng.random() >= self.grounding_ratio:
            return None
        self.grounding_attempt_counts[key] += 1
        record = self.grounding_store.sample(
            domain="health",
            label=label,
            subtype=subtype,
            roles=roles,
            datasets=self.grounding_datasets,
            rng=self.rng,
        )
        if record is None:
            self.grounding_fallback_counts[key] += 1
            if self.grounding_mode == "grounded-only":
                report = self.grounding_store.coverage_report(
                    [{"domain": "health", "label": label, "subtype": subtype, "roles": list(roles)}]
                )
                raise GroundingCoverageError(report)
            return None
        self.grounded_counts[key] += 1
        return record

    def _grounding_context(self, record: GroundingRecord | None) -> dict[str, Any]:
        if record is None:
            return {}
        out: dict[str, Any] = {}
        for key, value in record.facts.items():
            if key not in GROUNDING_CONTEXT_KEYS or value in {None, ""}:
                continue
            out[key] = int(value) if key in {"heart_rate", "score"} and isinstance(value, (int, float)) else value
        return out

    def _grounding_meta(self, record: GroundingRecord | None, facts_used: dict[str, Any]) -> dict[str, Any] | None:
        if record is None:
            return None
        return {
            "dataset": record.dataset,
            "record_id": record.id,
            "source_group_id": record.source_group_id,
            "domain": record.domain,
            "role": record.role,
            "label": record.label,
            "subtype": record.subtype,
            "facts_used": sorted(facts_used),
            "raw_text_copied": False,
            "grounded_in_public_or_deidentified_data": True,
        }

    def _render(self, template: str, context: dict[str, Any]) -> str:
        return template.format(**context)

    def _style_text(self, text: str, style: str, label: str = "health_private") -> str:
        is_private = label == "health_private"
        is_health_public = label == "non_private_health"
        if style == "zh_casual":
            return f"随手记一下：{text}"
        if style == "zh_formal":
            prefix = "个人健康记录如下" if is_private else ("公共健康资料如下" if is_health_public else "普通备注如下")
            return f"{prefix}：{text}"
        if style == "zh_en_codeswitch":
            prefix = "Personal health note" if is_private else ("Public health note" if is_health_public else "General note")
            return f"{prefix}: {text}"
        if style == "agent_summary":
            topic = "个人健康信息" if is_private else ("公共健康或非私人健康内容" if is_health_public else "普通非健康内容")
            return f"Agent summary: 片段主题={topic}; evidence={text}"
        if style == "email_mixed":
            prefix = "private health note" if is_private else ("non-private health note" if is_health_public else "quick note")
            return f"Hi，补充一个 {prefix}：{text}"
        if style == "mobile_terse":
            prefix = "private health memo" if is_private else ("public health memo" if is_health_public else "memo")
            return f"{prefix} / {text}"
        if style == "clinic_note":
            prefix = "Clinic note excerpt" if is_private else ("Clinic/public info excerpt" if is_health_public else "Note excerpt")
            return f"{prefix}: {text}"
        return text

    def _format_text(self, text: str, fmt: str, label: str, subtype: str) -> str:
        if fmt == "key_value":
            return f"label={label}\nsubtype={subtype}\ncontent={text}"
        if fmt == "csv_row":
            return f"label,subtype,content\n{label},{subtype},{text}"
        if fmt == "json":
            return json.dumps({"label": label, "subtype": subtype, "note": text}, ensure_ascii=False)
        if fmt == "agent_summary":
            return f"Agent summary: label={label}; subtype={subtype}; evidence={text}"
        if fmt == "email_snippet":
            return f"Subject: health note\n\n{text}"
        if fmt == "chat_transcript":
            return f"user: {text}\nassistant: noted"
        if fmt == "ocr_fragment":
            return f"OCR fragment >> label={label}; subtype={subtype}; text={text}"
        if fmt == "app_notification":
            return f"app_notification | category={label} | subtype={subtype} | message={text}"
        if fmt == "form_dump":
            return f"field.label={label}\nfield.subtype={subtype}\nfield.note={text}"
        if fmt == "spreadsheet_row":
            return f"sheet=health_notes | label={label} | subtype={subtype} | value={text}"
        return text

    def _row_meta(self, kind: str, scenario_id: str, template_index: int, style: str, fmt: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        skeleton_id = f"{kind}:{scenario_id}:t{template_index}:{style}:{fmt}"
        meta = {"scenario_id": scenario_id, "template_id": f"{scenario_id}:t{template_index}", "skeleton_id": skeleton_id}
        if extra:
            meta.update(extra)
        return meta

    def _base_private(self, subtype: str, grounding: GroundingRecord | None = None) -> tuple[str, str, int, dict[str, Any], dict[str, Any]]:
        scenarios = self.catalog["private_scenarios"][subtype]
        scenario = self._choice(scenarios)
        template_index = self.rng.randrange(len(scenario["templates"]))
        context = self._context()
        grounding_context = self._grounding_context(grounding)
        context.update(grounding_context)
        text = self._render(scenario["templates"][template_index], context)
        return text, scenario["id"], template_index, scenario, grounding_context

    def private_example(self, subtype: str | None = None) -> SyntheticExample:
        subtype = subtype or self._choice(HEALTH_SUBTYPES)
        grounding = self._maybe_grounding(label="health_private", subtype=subtype, roles=("private_candidate",))
        base, scenario_id, template_index, scenario, grounding_context = self._base_private(subtype, grounding)
        style = self._choice(STYLES)
        fmt = self._choice(FORMATS)
        text = self._format_text(self._style_text(base, style, "health_private"), fmt, "health_private", subtype)
        meta_extra: dict[str, Any] = {
            "privacy_evidence": base,
            "sensitive_span": base,
            "private_cues": scenario.get("private_cues", [subtype, scenario_id]),
        }
        grounding_meta = self._grounding_meta(grounding, grounding_context)
        if grounding_meta:
            meta_extra["grounding"] = grounding_meta
        row = SyntheticExample(
            self._id("health_priv"),
            text,
            "health_private",
            subtype,
            grounding.region if grounding is not None and grounding.region not in {"", "global"} else "singapore_cn",
            language_for(text),
            fmt,
            style,
            "high",
            "grounded_synthetic" if grounding else "synthetic_template",
            self._row_meta(f"private:{subtype}", scenario_id, template_index, style, fmt, meta_extra),
        )
        ok, reason = validate_synthetic_example(asdict(row))
        if not ok:
            raise ValueError(reason)
        return row

    def _grouped_scenario(self, section: str) -> tuple[str, dict[str, Any], int]:
        groups = self.catalog[section]
        group = self._choice(list(groups.keys()))
        scenario = self._choice(groups[group])
        template_index = self.rng.randrange(len(scenario["templates"]))
        return group, scenario, template_index

    def _public_example(self, section: str, prefix: str, default_label: str, default_style: str, grounding: GroundingRecord | None = None) -> SyntheticExample:
        group, scenario, template_index = self._grouped_scenario(section)
        label = scenario.get("label", default_label)
        context = self._context()
        grounding_context = self._grounding_context(grounding)
        context.update(grounding_context)
        base = self._render(scenario["templates"][template_index], context)
        style = self._choice(STYLES if label != "benign" else ["zh_casual", "zh_formal", "zh_en_codeswitch", "email_mixed", "mobile_terse"])
        fmt = self._choice(FORMATS)
        styled = self._style_text(base, style, label) if style != default_style else base
        text = self._format_text(styled, fmt, label, "*")
        row_region = scenario.get("region", "global")
        if grounding is not None and grounding.region not in {"", "global"}:
            row_region = grounding.region
        meta_extra: dict[str, Any] = {"group": group, "non_private_reason": scenario.get("non_private_reason", f"{label}:{group}")}
        grounding_meta = self._grounding_meta(grounding, grounding_context)
        if grounding_meta:
            meta_extra["grounding"] = grounding_meta
        meta = self._row_meta(
            f"{prefix}:{group}:{label}",
            scenario["id"],
            template_index,
            style,
            fmt,
            meta_extra,
        )
        row = SyntheticExample(
            self._id(prefix),
            text,
            label,
            "*",
            row_region,
            language_for(text),
            fmt,
            style,
            "none",
            "grounded_public_negative" if grounding else "synthetic_template",
            meta,
        )
        ok, reason = validate_synthetic_example(asdict(row))
        if not ok:
            raise ValueError(reason)
        return row

    def hard_negative_example(self) -> SyntheticExample:
        grounding = self._maybe_grounding(label="non_private_health", subtype="*", roles=("public_negative",))
        return self._public_example("hard_negative_scenarios", "hard_neg", "non_private_health", "hard_negative", grounding)

    def benign_example(self) -> SyntheticExample:
        return self._public_example("benign_scenarios", "benign", "benign", "neutral")

    def mixed_egress_example(self) -> MixedEgressExample:
        carrier = self._choice(self.catalog["mixed_carriers"])
        priv = self.private_example()
        expected_health = bool(carrier.get("expected_health"))
        if expected_health:
            expected_categories = ["work", "health_private"]
            unexpected_categories: list[str] = []
            decision = "allow"
        else:
            expected_categories = ["work"]
            unexpected_categories = ["health_private"]
            decision = "request_approval"
        text = carrier["template"].replace("{payload}", priv.text)
        meta = {
            "carrier_id": carrier["id"],
            "payload_skeleton_id": priv.meta.get("skeleton_id"),
            "health_evidence": priv.text,
            "expected_health": expected_health,
            "skeleton_id": f"egress:{carrier['id']}:{priv.meta.get('skeleton_id')}",
        }
        if isinstance(priv.meta.get("grounding"), dict):
            meta["grounding"] = priv.meta["grounding"]
        return MixedEgressExample(
            self._id("egress"),
            carrier["intent"],
            text,
            expected_categories,
            ["work", "health_private"],
            unexpected_categories,
            decision,
            "mixed_egress",
            "synthetic_mixed",
            priv.subtype,
            priv.text,
            meta,
        )

    def generate_private(self, n: int) -> list[dict]:
        return [asdict(self.private_example(HEALTH_SUBTYPES[i % len(HEALTH_SUBTYPES)])) for i in range(n)]

    def generate_hard_negatives(self, n: int) -> list[dict]:
        return [asdict(self.hard_negative_example()) for _ in range(n)]

    def generate_benign(self, n: int) -> list[dict]:
        return [asdict(self.benign_example()) for _ in range(n)]

    def generate_mixed(self, n: int) -> list[dict]:
        return [asdict(self.mixed_egress_example()) for _ in range(n)]


def language_for(text: str) -> str:
    return "zh_en" if any(ch.isascii() and ch.isalpha() for ch in text) else "zh"
