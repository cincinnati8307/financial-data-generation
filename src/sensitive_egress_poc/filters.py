from __future__ import annotations

import re
from typing import Any

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.I),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.I),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*[A-Za-z0-9_\-]{16,}", re.I),
    re.compile(r"\bpassword\s*[:=]\s*\S+", re.I),
    re.compile(r"\bpasswd\s*[:=]\s*\S+", re.I),
    re.compile(r"密码\s*[:=：]\s*\S+"),
    re.compile(r"\bsession(?:id|_id| cookie)?\s*[:=]\s*[A-Za-z0-9_\-]{16,}", re.I),
]
MASKED_PATTERNS = [re.compile(r"尾号\s*\d{4}"), re.compile(r"ending\s*\d{4}", re.I), re.compile(r"\*{2,}\d{4}")]


def is_masked_card_or_account_reference(text: str) -> bool:
    return any(p.search(text) for p in MASKED_PATTERNS)


def _strip_allowed_masked(text: str) -> str:
    text = re.sub(r"尾号\s*\d{4}", "尾号 XXXX", text)
    text = re.sub(r"ending\s*\d{4}", "ending XXXX", text, flags=re.I)
    text = re.sub(r"\*{2,}\d{4}", "****XXXX", text)
    return text


def looks_like_full_card_or_account_number(text: str) -> bool:
    scrubbed = _strip_allowed_masked(text)
    compact = re.sub(r"[ -]", "", scrubbed)
    if re.search(r"\b\d{12,19}\b", compact):
        return True
    if re.search(r"(?:账号|账户|account|card|卡号)[^\n]{0,12}\b\d{8,}\b", scrubbed, re.I):
        return True
    return False


def contains_disallowed_secret_like_content(text: str) -> bool:
    if any(p.search(text) for p in SECRET_PATTERNS):
        return True
    return looks_like_full_card_or_account_number(text)


def normalize_text_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = normalize_text_for_dedup(str(row.get("text", "")))
        if key and key not in seen:
            seen.add(key)
            out.append(row)
    return out


def validate_synthetic_example(row: dict[str, Any]) -> tuple[bool, str]:
    required = {"id", "text", "label", "region", "language", "format", "style", "source"}
    missing = required - row.keys()
    if missing:
        return False, f"missing_fields:{','.join(sorted(missing))}"
    if row["label"] not in {"financial_private", "non_private_financial", "benign"}:
        return False, "wrong_label"
    if contains_disallowed_secret_like_content(str(row["text"])):
        return False, "secret_like_content"
    return True, "ok"


def validate_augmented_row(row: dict[str, Any], source_row: dict[str, Any]) -> tuple[bool, str]:
    ok, reason = validate_synthetic_example(row)
    if not ok:
        return ok, reason
    if row.get("label") != "financial_private":
        return False, "wrong_label"
    if row.get("subtype") != source_row.get("subtype"):
        return False, "subtype_mismatch"
    if row.get("parent_id") != source_row.get("id"):
        return False, "parent_mismatch"
    return True, "ok"
