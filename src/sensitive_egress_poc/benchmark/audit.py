from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from sensitive_egress_poc.centroid_classifier import _cos, _hash_embed

from .dataset import CARRIER_ALLOWED_SUBTYPES, carrier_id, expected_financial, row_id

AUDIT_COLUMNS = [
    "sample_id",
    "user_intent",
    "financial_subtype",
    "financial_evidence",
    "carrier_id",
    "expected_decision",
    "expected_financial",
    "query_evidence_similarity",
    "audit_status",
    "audit_reason",
]

LIKELY_ALIGNED = "likely_aligned"
POSSIBLY_MISALIGNED = "possibly_misaligned"
REQUIRES_MANUAL_REVIEW = "requires_manual_review"


def query_evidence_similarity(user_intent: str, financial_evidence: str) -> float:
    if not user_intent.strip() or not financial_evidence.strip():
        return 0.0
    return float(_cos(_hash_embed(user_intent), _hash_embed(financial_evidence)))


def audit_row(row: dict[str, Any], low_similarity_threshold: float = 0.08, review_similarity_threshold: float = 0.16) -> dict[str, Any]:
    cid = carrier_id(row) or "unknown"
    subtype = str(row.get("financial_subtype") or "unknown")
    exp_fin = expected_financial(row)
    similarity = query_evidence_similarity(str(row.get("user_intent") or ""), str(row.get("financial_evidence") or ""))

    if exp_fin is False:
        status = LIKELY_ALIGNED
        reason = "carrier explicitly marks financial payload as unexpected"
    elif exp_fin is True:
        allowed = CARRIER_ALLOWED_SUBTYPES.get(cid)
        if allowed and subtype in allowed:
            status = LIKELY_ALIGNED
            reason = f"carrier allows subtype {subtype}"
        elif allowed:
            status = POSSIBLY_MISALIGNED
            reason = f"carrier allows {','.join(sorted(allowed))}; observed {subtype}"
        elif similarity < low_similarity_threshold:
            status = POSSIBLY_MISALIGNED
            reason = f"low query-evidence similarity {similarity:.4f} and no explicit carrier subtype map"
        elif similarity < review_similarity_threshold:
            status = REQUIRES_MANUAL_REVIEW
            reason = f"borderline query-evidence similarity {similarity:.4f} and no explicit carrier subtype map"
        else:
            status = LIKELY_ALIGNED
            reason = "financial carrier with moderate query-evidence similarity"
    else:
        status = REQUIRES_MANUAL_REVIEW
        reason = "missing expected_financial metadata"

    return {
        "sample_id": row_id(row),
        "user_intent": row.get("user_intent", ""),
        "financial_subtype": subtype,
        "financial_evidence": row.get("financial_evidence", ""),
        "carrier_id": cid,
        "expected_decision": row.get("expected_decision", ""),
        "expected_financial": exp_fin,
        "query_evidence_similarity": round(similarity, 6),
        "audit_status": status,
        "audit_reason": reason,
    }


def audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [audit_row(row) for row in rows]


def write_alignment_audit(rows: list[dict[str, Any]], output: str | Path) -> dict[str, Any]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = audit_rows(rows)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for row in audit:
            writer.writerow({column: row.get(column) for column in AUDIT_COLUMNS})
    return summarize_audit(audit)


def summarize_audit(audit: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(row.get("audit_status", "unknown") for row in audit)
    suspicious_by_carrier = Counter(
        row.get("carrier_id", "unknown")
        for row in audit
        if row.get("audit_status") in {POSSIBLY_MISALIGNED, REQUIRES_MANUAL_REVIEW}
    )
    return {
        "rows": len(audit),
        "status_counts": dict(status_counts),
        "suspicious_by_carrier": dict(suspicious_by_carrier),
        "possibly_misaligned_count": status_counts.get(POSSIBLY_MISALIGNED, 0),
        "requires_manual_review_count": status_counts.get(REQUIRES_MANUAL_REVIEW, 0),
    }
