from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from sensitive_egress_poc.io_utils import read_jsonl

from .schemas import ALIGNED_SENSITIVE, MISALIGNED_SENSITIVE, NON_SENSITIVE, SENSITIVE

ANCHOR_LABEL_TO_SENSITIVITY = {
    "financial_private": SENSITIVE,
    "non_private_financial": NON_SENSITIVE,
    "benign": NON_SENSITIVE,
}

EGRESS_DECISION_TO_ALIGNMENT = {
    "allow": ALIGNED_SENSITIVE,
    "request_approval": MISALIGNED_SENSITIVE,
}

PRIVATE_SUBTYPES = {
    "bank_balance",
    "transaction",
    "salary_income",
    "card_payment",
    "loan_debt",
    "invoice_receipt",
    "investment",
    "tax",
    "wallet_payment",
}

ALL_PRIVATE_SUBTYPES = set(PRIVATE_SUBTYPES)

# Human-authored carrier semantics used only for audit and optional fine-grained subsets.
# They do not mutate the generator labels. Mismatches are excluded from fine-grained
# evaluation unless a manual override explicitly labels them.
CARRIER_ALLOWED_SUBTYPES: dict[str, set[str]] = {
    "send_tax_note": {"tax", "salary_income"},
    "send_receipt": {"invoice_receipt", "transaction", "card_payment", "wallet_payment"},
    "send_salary_to_advisor": {"salary_income"},
    "send_bank_balance_to_spouse": {"bank_balance"},
    "send_card_bill": {"card_payment"},
    "send_investment_snapshot": {"investment"},
    "send_loan_info": {"loan_debt"},
    "send_wallet_payment": {"wallet_payment", "transaction"},
    "send_transaction_dispute": {"transaction", "card_payment", "wallet_payment"},
    "send_invoice_vendor": {"invoice_receipt"},
    "send_budget_sheet": set(ALL_PRIVATE_SUBTYPES),
    "send_tax_portal": {"tax", "salary_income"},
    "send_payroll_hr": {"salary_income", "tax"},
    "send_receipt_family": {"invoice_receipt", "transaction", "card_payment", "wallet_payment"},
    "send_card_autopay": {"card_payment", "bank_balance"},
    "send_brokerage_csv": {"investment", "tax"},
}


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("sample_id") or "")


def row_meta(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta")
    return meta if isinstance(meta, dict) else {}


def carrier_id(row: dict[str, Any]) -> str | None:
    return row_meta(row).get("carrier_id")


def expected_financial(row: dict[str, Any]) -> bool | None:
    meta = row_meta(row)
    if "expected_financial" in meta:
        return bool(meta.get("expected_financial"))
    if "financial_private" in row.get("expected_categories", []):
        return True
    if "financial_private" in row.get("unexpected_categories", []):
        return False
    return None


def map_anchor_label(row: dict[str, Any]) -> str:
    label = row.get("label")
    if label not in ANCHOR_LABEL_TO_SENSITIVITY:
        raise ValueError(f"unsupported anchor label for sensitivity task: {label}")
    return ANCHOR_LABEL_TO_SENSITIVITY[label]


def map_egress_decision(row: dict[str, Any]) -> str:
    decision = row.get("expected_decision")
    if decision not in EGRESS_DECISION_TO_ALIGNMENT:
        raise ValueError(f"unsupported egress decision for alignment task: {decision}")
    return EGRESS_DECISION_TO_ALIGNMENT[decision]


def detect_language(text: str) -> str:
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    has_latin = any(ch.isascii() and ch.isalpha() for ch in text)
    if has_cjk and has_latin:
        return "zh_en"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return "unknown"


def row_language(row: dict[str, Any]) -> str:
    return str(row.get("language") or detect_language(str(row.get("text") or "")))


def payload_format(row: dict[str, Any]) -> str:
    meta = row_meta(row)
    payload_skeleton = str(meta.get("payload_skeleton_id") or "")
    if payload_skeleton:
        parts = payload_skeleton.split(":")
        if parts:
            return parts[-1]
    return str(row.get("format") or "unknown")


def read_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    return read_jsonl(path)


def load_alignment_overrides(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    overrides: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(p):
        sample_id = str(row.get("sample_id") or "")
        label = row.get("semantic_alignment_label")
        if not sample_id or label not in {ALIGNED_SENSITIVE, MISALIGNED_SENSITIVE}:
            raise ValueError(f"invalid alignment override row: {row}")
        overrides[sample_id] = row
    return overrides


def fine_grained_alignment_rows(rows: list[dict[str, Any]], overrides: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row_id(row)
        copied = deepcopy(row)
        if sample_id in overrides:
            override = overrides[sample_id]
            copied["_benchmark_ground_truth"] = override["semantic_alignment_label"]
            copied["_benchmark_ground_truth_source"] = override.get("annotator", "manual")
            copied["_benchmark_ground_truth_reason"] = override.get("reason")
            selected.append(copied)
            continue

        exp_fin = expected_financial(row)
        cid = carrier_id(row)
        subtype = row.get("financial_subtype")
        if exp_fin is False:
            copied["_benchmark_ground_truth"] = MISALIGNED_SENSITIVE
            copied["_benchmark_ground_truth_source"] = "explicit_non_financial_carrier"
            copied["_benchmark_ground_truth_reason"] = "carrier metadata has expected_financial=false"
            selected.append(copied)
            continue

        allowed = CARRIER_ALLOWED_SUBTYPES.get(str(cid)) if cid else None
        if exp_fin is True and allowed and subtype in allowed:
            copied["_benchmark_ground_truth"] = ALIGNED_SENSITIVE
            copied["_benchmark_ground_truth_source"] = "explicit_subtype_constraint"
            copied["_benchmark_ground_truth_reason"] = f"carrier allows subtype {subtype}"
            selected.append(copied)
    return selected


def count_values(rows: list[dict[str, Any]], getter) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(getter(row))] += 1
    return dict(sorted(counts.items()))


def dataset_summary(anchor_validation: list[dict[str, Any]], egress_train: list[dict[str, Any]], egress_validation: list[dict[str, Any]], fine_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    fine_rows = fine_rows or []
    return {
        "anchors_validation": {
            "rows": len(anchor_validation),
            "labels": count_values(anchor_validation, lambda r: r.get("label", "unknown")),
            "task_a_labels": count_values(anchor_validation, map_anchor_label),
            "subtypes": count_values(anchor_validation, lambda r: r.get("subtype", "unknown")),
            "languages": count_values(anchor_validation, row_language),
            "formats": count_values(anchor_validation, lambda r: r.get("format", "unknown")),
            "styles": count_values(anchor_validation, lambda r: r.get("style", "unknown")),
        },
        "egress_train": {
            "rows": len(egress_train),
            "decisions": count_values(egress_train, lambda r: r.get("expected_decision", "unknown")),
            "task_b_labels": count_values(egress_train, map_egress_decision),
            "subtypes": count_values(egress_train, lambda r: r.get("financial_subtype", "unknown")),
            "carriers": count_values(egress_train, lambda r: carrier_id(r) or "unknown"),
            "expected_financial": count_values(egress_train, lambda r: expected_financial(r)),
        },
        "egress_validation": {
            "rows": len(egress_validation),
            "decisions": count_values(egress_validation, lambda r: r.get("expected_decision", "unknown")),
            "task_b_labels": count_values(egress_validation, map_egress_decision),
            "subtypes": count_values(egress_validation, lambda r: r.get("financial_subtype", "unknown")),
            "carriers": count_values(egress_validation, lambda r: carrier_id(r) or "unknown"),
            "expected_financial": count_values(egress_validation, lambda r: expected_financial(r)),
        },
        "fine_grained_semantic_alignment": {
            "rows": len(fine_rows),
            "labels": count_values(fine_rows, lambda r: r.get("_benchmark_ground_truth", "unknown")) if fine_rows else {},
            "sources": count_values(fine_rows, lambda r: r.get("_benchmark_ground_truth_source", "unknown")) if fine_rows else {},
        },
    }


def group_key_for_row(row: dict[str, Any], field: str) -> str:
    if field == "financial_subtype":
        return str(row.get("financial_subtype") or row.get("subtype") or "unknown")
    if field == "carrier_id":
        return str(carrier_id(row) or "unknown")
    if field == "language":
        return row_language(row)
    if field == "payload_format":
        return payload_format(row)
    if field == "expected_financial":
        return str(expected_financial(row))
    if field == "original_label":
        return str(row.get("label", "unknown"))
    return str(row.get(field) or "unknown")


def rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row_id(row): row for row in rows}
