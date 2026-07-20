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
from .schemas import MixedEgressExample, PRIVATE_SUBTYPES, SyntheticExample

CATALOG_PATH = Path(__file__).with_name("template_catalog.json")

FORMATS = [
    "natural_sentence",
    "key_value",
    "csv_row",
    "json",
    "agent_summary",
    "email_snippet",
    "spreadsheet_row",
    "chat_transcript",
]
STYLES = [
    "zh_casual",
    "zh_formal",
    "zh_en_codeswitch",
    "email_mixed",
    "agent_summary",
    "sg_cn_codeswitch",
    "mainland_cn",
    "mobile_terse",
    "ocr_fragment",
]

GROUNDING_MODES = {"synthetic-only", "hybrid", "grounded-only"}
GROUNDING_CONTEXT_KEYS = {
    "bank",
    "app",
    "merchant",
    "employer",
    "biller",
    "brokerage",
    "tax_agency",
    "fund",
    "loan_type",
    "expense_type",
    "account_type",
    "masked_account",
    "masked_card",
    "month",
    "weekday",
    "date",
    "amount",
    "small_amount",
    "pct",
}


def load_template_catalog(path: str | Path | None = None) -> dict[str, Any]:
    catalog_path = Path(path) if path is not None else CATALOG_PATH
    with catalog_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class SyntheticFinancialGenerator:
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

    def _region(self, scenario: dict[str, Any] | None = None) -> str:
        regions = (scenario or {}).get("regions") or ["mainland_cn", "singapore_cn"]
        if regions == ["mainland_cn", "singapore_cn"] or set(regions) == {"mainland_cn", "singapore_cn"}:
            return "singapore_cn" if self.rng.random() < 0.35 else "mainland_cn"
        return self._choice(list(regions))

    def _amount(self, subtype: str, region: str = "mainland_cn", small: bool = False) -> str:
        ranges = {
            "bank_balance": (200, 120000),
            "transaction": (20, 2500),
            "salary_income": (3000, 45000),
            "card_payment": (500, 40000),
            "loan_debt": (50000, 900000),
            "invoice_receipt": (80, 50000),
            "investment": (5000, 300000),
            "tax": (30000, 600000),
            "wallet_payment": (10, 3000),
        }
        lo, hi = (20, 6000) if small else ranges.get(subtype, (100, 50000))
        n = self.rng.randint(lo, hi)
        currencies = ["SGD", "S$", "新币"] if region == "singapore_cn" else ["人民币", "RMB", "CNY", "¥"]
        return f"{self._choice(currencies)} {n:,}"

    def _context(self, subtype: str, region: str) -> dict[str, str | int]:
        bank_pool = "singapore_banks" if region == "singapore_cn" else "mainland_banks"
        return {
            "bank": self._choice(self._pool(bank_pool)),
            "app": self._choice(self._pool("payment_apps")),
            "merchant": self._choice(self._pool("merchants")),
            "employer": self._choice(self._pool("employers")),
            "biller": self._choice(self._pool("billers")),
            "brokerage": self._choice(self._pool("brokerages")),
            "tax_agency": self._choice(self._pool("tax_agencies")),
            "fund": self._choice(self._pool("funds")),
            "loan_type": self._choice(self._pool("loan_types")),
            "expense_type": self._choice(self._pool("expense_types")),
            "account_type": self._choice(self._pool("account_types")),
            "masked_account": self._choice(self._pool("masked_accounts")),
            "masked_card": self._choice(self._pool("masked_cards")),
            "month": self._choice(self._pool("months")),
            "weekday": self._choice(self._pool("weekdays")),
            "date": self._choice(self._pool("dates")),
            "amount": self._amount(subtype, region),
            "small_amount": self._amount(subtype, region, small=True),
            "pct": self.rng.randint(3, 28),
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
                        "missing": [{"domain": "financial", "label": label, "subtype": subtype, "roles": list(roles), "available": 0}],
                        "store_counts": {},
                    }
                )
            return None
        self.grounding_eligible_counts[key] += 1
        if self.grounding_mode == "hybrid" and self.rng.random() >= self.grounding_ratio:
            return None
        self.grounding_attempt_counts[key] += 1
        record = self.grounding_store.sample(
            domain="financial",
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
                    [{"domain": "financial", "label": label, "subtype": subtype, "roles": list(roles)}]
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
            out[key] = int(value) if key == "pct" and isinstance(value, (int, float)) else value
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

    def _style_text(self, text: str, style: str, subtype: str, label: str) -> str:
        if style == "zh_casual":
            return f"随手记一下：{text}"
        if style == "zh_formal":
            return f"个人记录如下：{text}"
        if style == "zh_en_codeswitch":
            return f"Personal note: {text}"
        if style == "email_mixed":
            return f"Hi，补充一个 quick note：{text}"
        if style == "agent_summary":
            topic = "个人财务信息" if label == "financial_private" else "非私人内容"
            return f"Agent summary: 片段主题={topic}; evidence={text}"
        if style == "sg_cn_codeswitch":
            return f"FYI lah，这条记录是：{text}"
        if style == "mainland_cn":
            return f"这条备注写的是：{text}"
        if style == "mobile_terse":
            return f"备忘/{subtype}: {text}"
        if style == "ocr_fragment":
            return f"OCR fragment >> {text}"
        return text

    def _format_text(self, text: str, fmt: str, subtype: str, label: str) -> str:
        if fmt == "key_value":
            return f"label={label}\nsubtype={subtype}\ncontent={text}"
        if fmt == "csv_row":
            return f"label,subtype,content\n{label},{subtype},{text}"
        if fmt == "json":
            return json.dumps({"label": label, "subtype": subtype, "note": text}, ensure_ascii=False)
        if fmt == "agent_summary":
            return f"Agent summary: label={label}; subtype={subtype}; evidence={text}"
        if fmt == "email_snippet":
            return f"Subject: note update\n\n{text}"
        if fmt == "spreadsheet_row":
            return f"sheet=notes | label={label} | subtype={subtype} | value={text}"
        if fmt == "chat_transcript":
            return f"user: {text}\nassistant: noted"
        return text

    def _row_meta(self, kind: str, scenario_id: str, template_index: int, style: str, fmt: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        skeleton_id = f"{kind}:{scenario_id}:t{template_index}:{style}:{fmt}"
        meta = {"scenario_id": scenario_id, "template_id": f"{scenario_id}:t{template_index}", "skeleton_id": skeleton_id}
        if extra:
            meta.update(extra)
        return meta

    def _private_base(self, subtype: str, grounding: GroundingRecord | None = None) -> tuple[str, str, str, str, int, dict[str, Any]]:
        scenarios = self.catalog["private_scenarios"][subtype]
        scenario = self._choice(scenarios)
        template_index = self.rng.randrange(len(scenario["templates"]))
        region = self._region(scenario)
        context = self._context(subtype, region)
        grounding_context = self._grounding_context(grounding)
        context.update(grounding_context)
        text = self._render(scenario["templates"][template_index], context)
        if grounding is not None and grounding.region not in {"", "global"}:
            region = grounding.region
        return text, region, scenario["id"], scenario["id"], template_index, grounding_context

    def private_example(self, subtype: str | None = None) -> SyntheticExample:
        subtype = subtype or self._choice(PRIVATE_SUBTYPES)
        grounding = self._maybe_grounding(label="financial_private", subtype=subtype, roles=("private_candidate", "distribution"))
        base, region, scenario_id, _, template_index, grounding_context = self._private_base(subtype, grounding)
        style = self._choice(STYLES)
        fmt = self._choice(FORMATS)
        styled = self._style_text(base, style, subtype, "financial_private")
        text = self._format_text(styled, fmt, subtype, "financial_private")
        meta_extra: dict[str, Any] = {"region": region}
        grounding_meta = self._grounding_meta(grounding, grounding_context)
        if grounding_meta:
            meta_extra["grounding"] = grounding_meta
        meta = self._row_meta(f"private:{subtype}", scenario_id, template_index, style, fmt, meta_extra)
        row = SyntheticExample(
            self._id("fin_priv"),
            text,
            "financial_private",
            subtype,
            region,
            language_for(text),
            fmt,
            style,
            "high",
            "grounded_synthetic" if grounding else "synthetic_template",
            meta,
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

    def _public_example(self, section: str, prefix: str, default_label: str, sensitivity: str, default_style: str, grounding: GroundingRecord | None = None) -> SyntheticExample:
        group, scenario, template_index = self._grouped_scenario(section)
        region_for_entities = self._region()
        row_region = scenario.get("region", "global")
        context = self._context("transaction", region_for_entities)
        grounding_context = self._grounding_context(grounding)
        context.update(grounding_context)
        base = self._render(scenario["templates"][template_index], context)
        label = scenario.get("label", default_label)
        style = self._choice(STYLES)
        fmt = self._choice(FORMATS)
        styled = self._style_text(base, style, "*", label)
        text = self._format_text(styled, fmt, "*", label)
        if grounding is not None and grounding.region not in {"", "global"}:
            row_region = grounding.region
        meta_extra: dict[str, Any] = {"group": group}
        grounding_meta = self._grounding_meta(grounding, grounding_context)
        if grounding_meta:
            meta_extra["grounding"] = grounding_meta
        meta = self._row_meta(f"{prefix}:{group}:{label}", scenario["id"], template_index, style, fmt, meta_extra)
        row = SyntheticExample(
            self._id(prefix),
            text,
            label,
            "*",
            row_region,
            language_for(text),
            fmt,
            style if style else default_style,
            sensitivity,
            "grounded_public_negative" if grounding else "synthetic_template",
            meta,
        )
        ok, reason = validate_synthetic_example(asdict(row))
        if not ok:
            raise ValueError(reason)
        return row

    def hard_negative_example(self) -> SyntheticExample:
        grounding = self._maybe_grounding(label="non_private_financial", subtype="*", roles=("public_negative",))
        return self._public_example("hard_negative_scenarios", "hard_neg", "non_private_financial", "none", "hard_negative", grounding)

    def benign_example(self) -> SyntheticExample:
        return self._public_example("benign_scenarios", "benign", "benign", "none", "neutral")

    def mixed_egress_example(self) -> MixedEgressExample:
        carrier = self._choice(self.catalog["mixed_carriers"])
        priv = self.private_example()
        text = carrier["template"].replace("{payload}", priv.text)
        expected_financial = bool(carrier.get("expected_financial"))
        if expected_financial:
            expected_categories = ["work", "financial_private"]
            unexpected_categories: list[str] = []
            expected_decision = "allow"
        else:
            expected_categories = ["work"]
            unexpected_categories = ["financial_private"]
            expected_decision = "request_approval"
        meta = {
            "carrier_id": carrier["id"],
            "payload_skeleton_id": priv.meta.get("skeleton_id"),
            "skeleton_id": f"egress:{carrier['id']}:{priv.meta.get('skeleton_id')}",
            "expected_financial": expected_financial,
        }
        if isinstance(priv.meta.get("grounding"), dict):
            meta["grounding"] = priv.meta["grounding"]
        return MixedEgressExample(
            self._id("egress"),
            carrier["intent"],
            text,
            expected_categories,
            ["work", "financial_private"],
            unexpected_categories,
            expected_decision,
            "mixed_egress",
            "synthetic_mixed",
            priv.subtype,
            priv.text,
            meta,
        )

    def generate_private(self, n: int) -> list[dict]:
        return [asdict(self.private_example(PRIVATE_SUBTYPES[i % len(PRIVATE_SUBTYPES)])) for i in range(n)]

    def generate_hard_negatives(self, n: int) -> list[dict]:
        return [asdict(self.hard_negative_example()) for _ in range(n)]

    def generate_benign(self, n: int) -> list[dict]:
        return [asdict(self.benign_example()) for _ in range(n)]

    def generate_mixed(self, n: int) -> list[dict]:
        return [asdict(self.mixed_egress_example()) for _ in range(n)]


def language_for(text: str) -> str:
    return "zh_en" if any(ch.isascii() and ch.isalpha() for ch in text) else "zh"


def re_has_en(text: str) -> bool:
    return any(ch.isascii() and ch.isalpha() for ch in text)
