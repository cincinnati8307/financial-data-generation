from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Any

RAW_TEXT_KEYS = {
    "abstract",
    "abstract_text",
    "complaint_what_happened",
    "consumer_complaint_narrative",
    "narrative",
    "raw",
    "raw_text",
    "source_text",
    "text",
    "title",
    "utterance",
}

MASKED_REFERENCE_PATTERNS = [
    re.compile(r"\*{2,}\d{4}"),
    re.compile(r"\bending\s+\d{4}\b", re.I),
    re.compile(r"尾号\s*\d{4}"),
]

UNSAFE_IDENTIFIER_PATTERNS = [
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b"),
    re.compile(r"\b(?:account|acct|card|loan|client|patient|mrn|medical record|seqn|id|账号|账户|卡号|客户|病历号|患者编号)\s*[:#=：-]?\s*\d{5,}\b", re.I),
    re.compile(r"\b\d{12,19}\b"),
    re.compile(r"\b[STFG]\d{7}[A-Z]\b"),
    re.compile(r"\b\d{17}[0-9Xx]\b"),
    re.compile(r"\b(?:api[_-]?key|secret|token|password|passwd)\s*[:=：]\s*\S+", re.I),
]


def hash_id(value: Any, *, salt: str = "egress_grounding", prefix: str = "src") -> str:
    raw = f"{salt}:{value}".encode("utf-8", errors="ignore")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:16]}"


def strip_allowed_masked_references(text: str) -> str:
    scrubbed = text
    for pattern in MASKED_REFERENCE_PATTERNS:
        scrubbed = pattern.sub("MASKED_REF", scrubbed)
    return scrubbed


def contains_unsafe_identifier(text: str) -> bool:
    scrubbed = strip_allowed_masked_references(text)
    return any(pattern.search(scrubbed) for pattern in UNSAFE_IDENTIFIER_PATTERNS)


def find_unsafe_identifier(obj: Any, path: str = "$") -> str | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            item_path = f"{path}.{key}"
            if str(key).lower() in RAW_TEXT_KEYS:
                return item_path
            found = find_unsafe_identifier(value, item_path)
            if found:
                return found
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found = find_unsafe_identifier(value, f"{path}[{i}]")
            if found:
                return found
    elif isinstance(obj, str) and contains_unsafe_identifier(obj):
        return path
    return None


def safe_str(value: Any, *, max_len: int = 96) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text[:max_len]


def rounded_number(value: Any, *, step: float = 1.0, digits: int | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    if step > 0:
        number = round(number / step) * step
    if digits is not None:
        number = round(number, digits)
    return number


def perturbed_number(value: Any, rng: random.Random, *, pct: float = 0.03, step: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    jitter = number * rng.uniform(-pct, pct)
    return rounded_number(number + jitter, step=step)


def bucket_number(value: Any, *, buckets: list[tuple[float, str]]) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    for upper, label in buckets:
        if number <= upper:
            return label
    return buckets[-1][1] if buckets else "unknown"


def safe_month(value: Any) -> str:
    text = str(value or "")
    match = re.search(r"(?:^|\D)(\d{4})[-/]?(\d{1,2})(?:\D|$)", text)
    if match:
        month = max(1, min(12, int(match.group(2))))
        return f"{month}月"
    match = re.search(r"(?:^|\D)(\d{1,2})(?:\D|$)", text)
    if match:
        month = max(1, min(12, int(match.group(1))))
        return f"{month}月"
    return "本月"
