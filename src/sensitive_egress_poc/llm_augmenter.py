from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from typing import Any

from .filters import deduplicate_rows, normalize_text_for_dedup, validate_augmented_row

STYLES = [
    "zh_casual",
    "zh_formal",
    "zh_en_codeswitch",
    "email_mixed",
    "agent_summary",
    "key_value",
    "csv_row",
    "json_note",
    "sg_cn_codeswitch",
    "mainland_cn",
]

STYLE_FORMATS = {
    "agent_summary": "agent_summary",
    "key_value": "key_value",
    "csv_row": "csv_row",
    "json_note": "json",
}

STYLE_ALIASES = {
    "json": "json_note",
    "json_like": "json_note",
    "json_note": "json_note",
    "csv": "csv_row",
    "csv_like": "csv_row",
    "sg_codeswitch": "sg_cn_codeswitch",
    "singapore_cn": "sg_cn_codeswitch",
    "mainland": "mainland_cn",
    "mainland_chinese": "mainland_cn",
}

STYLE_INSTRUCTIONS = {
    "zh_casual": "casual Chinese chat note",
    "zh_formal": "formal personal finance record",
    "zh_en_codeswitch": "Chinese-English code-switching note",
    "email_mixed": "short mixed Chinese-English email snippet",
    "agent_summary": "agent-style summary with evidence",
    "key_value": "key-value style note",
    "csv_row": "CSV-like single-row note",
    "json_note": "JSON-like note written as text",
    "sg_cn_codeswitch": "Singapore Chinese code-switching phrasing",
    "mainland_cn": "Mainland Chinese phrasing",
}


class CandidateParseError(ValueError):
    pass



def financial_private_sources(rows: list[dict[str, Any]], max_inputs: int) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("label") == "financial_private"][:max_inputs]


def estimate_text_tokens(text: str) -> int:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    without_cjk = "".join(" " if "\u4e00" <= ch <= "\u9fff" else ch for ch in text)
    words = re.findall(r"[A-Za-z0-9_]+", without_cjk)
    punctuation = re.findall(r"[^\sA-Za-z0-9_]", without_cjk)
    estimated = cjk + int(len(words) * 1.3) + int(len(punctuation) * 0.5)
    return max(1, estimated)


def estimate_completion_tokens(row: dict[str, Any], paraphrases_per_example: int) -> int:
    source_tokens = estimate_text_tokens(str(row.get("text", "")))
    per_candidate = source_tokens + 40
    json_overhead = 12 * paraphrases_per_example
    return max(1, paraphrases_per_example * per_candidate + json_overhead)


def estimate_augmentation_tokens(
    rows: list[dict[str, Any]],
    max_inputs: int,
    paraphrases_per_example: int,
) -> dict[str, int | str]:
    sources = financial_private_sources(rows, max_inputs)
    prompt_tokens = sum(estimate_text_tokens(build_prompt(row, paraphrases_per_example)) for row in sources)
    completion_tokens = sum(estimate_completion_tokens(row, paraphrases_per_example) for row in sources)
    total = prompt_tokens + completion_tokens
    return {
        "method": "heuristic_cjk_word_estimate",
        "attempted_sources": len(sources),
        "paraphrases_per_example": paraphrases_per_example,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_completion_tokens": completion_tokens,
        "estimated_total_tokens": total,
        "estimated_total_tokens_with_buffer": int(total * 1.25),
    }


class LLMProvider:
    name = "base"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class DryRunProvider(LLMProvider):
    name = "dry-run"

    def __init__(self, model: str = "dry-run") -> None:
        self.model = model

    def generate_candidates(self, source: dict[str, Any], n: int) -> list[dict[str, Any]]:
        candidates = []
        for i in range(n):
            style = STYLES[i % len(STYLES)]
            candidates.append({"text": _dry_run_text(source, style, i), "style": style})
        return candidates

    def generate_rows(self, source: dict[str, Any], n: int) -> list[dict[str, Any]]:
        return [
            _aug_row(source, c["text"], c["style"], i, self.name, self.model)
            for i, c in enumerate(self.generate_candidates(source, n))
        ]

    def generate(self, prompt: str) -> str:
        return ""


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use provider=openai") from exc
        self.client = OpenAI()
        self.model = model

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content or ""


def _dry_run_text(source: dict[str, Any], style: str, index: int) -> str:
    text = source["text"]
    subtype = source["subtype"]
    variants = {
        "zh_casual": f"随手记一下：{text}",
        "zh_formal": f"个人财务记录如下：{text}",
        "zh_en_codeswitch": f"Personal finance note: {text}",
        "email_mixed": f"Hi，补充一个 private finance note：{text}",
        "agent_summary": f"Agent summary: 该片段包含个人财务信息；evidence={text}",
        "key_value": f"style=key_value\nsubtype={subtype}\nvalue={text}",
        "csv_row": f"style,subtype,value\ncsv_row,{subtype},{text}",
        "json_note": json.dumps({"style": "json_note", "subtype": subtype, "note": text}, ensure_ascii=False),
        "sg_cn_codeswitch": f"Finance note lah：{text}",
        "mainland_cn": f"这是一条个人财务备忘：{text}",
    }
    value = variants[style]
    if index >= len(STYLES):
        value = f"{value}\nvariant={index // len(STYLES) + 1}"
    return value


def _normalize_style(style: Any, index: int) -> str | None:
    if not style:
        return STYLES[index % len(STYLES)]
    normalized = str(style).strip().lower().replace("-", "_")
    normalized = STYLE_ALIASES.get(normalized, normalized)
    return normalized if normalized in STYLES else None


def _aug_row(source: dict[str, Any], text: str, style: str, i: int, provider: str, model: str) -> dict[str, Any]:
    return {
        "id": f"aug_{source['id']}_{i}",
        "text": text,
        "label": "financial_private",
        "subtype": source["subtype"],
        "region": source.get("region", "mainland_cn"),
        "language": source.get("language", "zh"),
        "format": STYLE_FORMATS.get(style, "natural_sentence"),
        "style": style,
        "sensitivity_level": "high",
        "source": "llm_paraphrase",
        "parent_id": source["id"],
        "meta": {
            "augmentation_model": model,
            "augmentation_provider": provider,
            "original_text": source["text"],
        },
    }


def build_prompt(row: dict[str, Any], n: int) -> str:
    style_lines = "\n".join(f"- {name}: {desc}" for name, desc in STYLE_INSTRUCTIONS.items())
    source = json.dumps(row, ensure_ascii=False)
    return f"""Return JSON only: an array of {n} objects, each with exactly these keys: text, style.
Do not return dataset rows. Do not include id, label, subtype, parent_id, source, region, language, format, sensitivity_level, or meta.
Allowed styles:
{style_lines}

Requirements:
- Preserve the same financial_private meaning and subtype: {row['subtype']}.
- Preserve visible synthetic amounts, institutions/apps/merchants, and masked last-4 references from the source when present.
- Vary wording, surrounding context, and format according to the requested style.
- Do not introduce real names, account numbers, card numbers, phone numbers, IDs, API keys, private keys, passwords, OAuth tokens, cookies, or other secrets.
- Do not copy the source text unchanged.
- If a candidate would violate a rule, omit it instead of adding extra fields.

Source row:
{source}
"""


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"```(?:jsonl?|JSONL?)?\s*(.*?)```", stripped, flags=re.S)
    if match:
        return match.group(1).strip()
    return stripped


def parse_candidate_objects(text: str) -> list[dict[str, Any]]:
    body = _strip_markdown_fence(text)
    if not body:
        return []
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        rows = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CandidateParseError("parse_error") from exc
        return rows
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise CandidateParseError("parse_error")


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    return parse_candidate_objects(text)


def _candidate_to_row(
    source: dict[str, Any],
    candidate: dict[str, Any],
    index: int,
    provider_name: str,
    model: str,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(candidate, dict):
        return None, "wrong_candidate_type"

    text = candidate.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "missing_text"
    text = text.strip()

    style = _normalize_style(candidate.get("style"), index)
    if style is None:
        return None, "unknown_style"

    if normalize_text_for_dedup(text) == normalize_text_for_dedup(str(source.get("text", ""))):
        return None, "unchanged_text"

    row = _aug_row(source, text, style, index, provider_name, model)
    ok, reason = validate_augmented_row(row, source)
    if not ok:
        return None, reason
    return row, "ok"


def _provider_candidates(provider: LLMProvider, row: dict[str, Any], n: int) -> list[dict[str, Any]]:
    if isinstance(provider, DryRunProvider):
        return provider.generate_candidates(row, n)
    return parse_candidate_objects(provider.generate(build_prompt(row, n)))


def augment_rows(
    rows: list[dict[str, Any]],
    provider: LLMProvider,
    max_inputs: int,
    paraphrases_per_example: int,
    include_original: bool,
    model: str,
) -> tuple[list[dict[str, Any]], Counter]:
    out = []
    reasons = Counter()
    sources = financial_private_sources(rows, max_inputs)
    provider_name = getattr(provider, "name", provider.__class__.__name__)

    for row in sources:
        if include_original:
            out.append(row)
        try:
            candidates = _provider_candidates(provider, row, paraphrases_per_example)
        except CandidateParseError:
            reasons["parse_error"] += 1
            continue
        except Exception as exc:
            logging.exception("Augmentation failed for %s", row.get("id"))
            reasons[type(exc).__name__] += 1
            continue

        for i, candidate in enumerate(candidates):
            aug_row, reason = _candidate_to_row(row, candidate, i, provider_name, model)
            if aug_row is None:
                reasons[reason] += 1
            else:
                out.append(aug_row)

    before = len(out)
    out = deduplicate_rows(out)
    reasons["duplicate"] += before - len(out)
    return out, reasons
