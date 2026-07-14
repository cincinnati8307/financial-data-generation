from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from typing import Any

from .filters import deduplicate_rows, normalize_text_for_dedup, validate_augmented_row

STYLES = ["zh_casual", "zh_formal", "zh_en_codeswitch", "agent_summary", "key_value", "email_mixed", "clinic_note"]
STYLE_FORMATS = {"agent_summary": "agent_summary", "key_value": "key_value"}


class CandidateParseError(ValueError):
    pass


class LLMProvider:
    name = "base"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class DryRunProvider(LLMProvider):
    name = "dry-run"

    def __init__(self, model: str = "dry-run") -> None:
        self.model = model

    def generate_candidates(self, source: dict[str, Any], n: int) -> list[dict[str, Any]]:
        out = []
        for i in range(n):
            style = STYLES[i % len(STYLES)]
            text = source["text"]
            variants = {
                "zh_casual": f"随手记一下这条健康备注：{text}",
                "zh_formal": f"个人健康记录如下：{text}",
                "zh_en_codeswitch": f"Personal health note: {text}",
                "agent_summary": f"Agent summary: 该片段包含个人健康信息；evidence={text}",
                "key_value": f"style=key_value\nsubtype={source['subtype']}\nvalue={text}",
                "email_mixed": f"Hi，补充一个 private health note：{text}",
                "clinic_note": f"Clinic note excerpt: {text}",
            }
            out.append({"text": variants[style], "style": style})
        return out

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


def health_private_sources(rows: list[dict[str, Any]], max_inputs: int) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("label") == "health_private"][:max_inputs]


def estimate_text_tokens(text: str) -> int:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    without_cjk = "".join(" " if "\u4e00" <= ch <= "\u9fff" else ch for ch in text)
    words = re.findall(r"[A-Za-z0-9_]+", without_cjk)
    punctuation = re.findall(r"[^\sA-Za-z0-9_]", without_cjk)
    return max(1, cjk + int(len(words) * 1.3) + int(len(punctuation) * 0.5))


def estimate_augmentation_tokens(rows: list[dict[str, Any]], max_inputs: int, paraphrases_per_example: int) -> dict[str, int | str]:
    sources = health_private_sources(rows, max_inputs)
    prompt_tokens = sum(estimate_text_tokens(build_prompt(row, paraphrases_per_example)) for row in sources)
    completion_tokens = sum(paraphrases_per_example * (estimate_text_tokens(str(row.get("text", ""))) + 52) for row in sources)
    total = prompt_tokens + completion_tokens
    return {"method": "heuristic_cjk_word_estimate", "attempted_sources": len(sources), "paraphrases_per_example": paraphrases_per_example, "estimated_prompt_tokens": prompt_tokens, "estimated_completion_tokens": completion_tokens, "estimated_total_tokens": total, "estimated_total_tokens_with_buffer": int(total * 1.25)}


def _aug_row(source: dict[str, Any], text: str, style: str, i: int, provider: str, model: str) -> dict[str, Any]:
    source_meta = source.get("meta") if isinstance(source.get("meta"), dict) else {}
    return {
        "id": f"aug_{source['id']}_{i}",
        "text": text,
        "label": "health_private",
        "subtype": source["subtype"],
        "region": source.get("region", "singapore_cn"),
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
            "source_skeleton_id": source_meta.get("skeleton_id"),
            "privacy_evidence": source_meta.get("privacy_evidence", source["text"]),
            "sensitive_span": source_meta.get("sensitive_span", source["text"]),
            "private_cues": source_meta.get("private_cues", [source.get("subtype", "health_private")]),
        },
    }


def build_prompt(row: dict[str, Any], n: int) -> str:
    return f"""Return JSON only: an array of {n} objects, each with exactly these keys: text, style.
Do not return full dataset rows or invent identifiers.
Allowed styles: {', '.join(STYLES)}.
Preserve the same health_private meaning and subtype: {row['subtype']}.
Preserve visible synthetic masked patient references such as MRN ****5678 when present.
Do not introduce real names, full medical record numbers, national IDs, phone numbers, account numbers, credentials, tokens, passwords, or other secrets.
Do not copy the source text unchanged.
Source row:
{json.dumps(row, ensure_ascii=False)}
"""


def _strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:jsonl?|JSONL?)?\s*(.*?)```", text.strip(), flags=re.S)
    return match.group(1).strip() if match else text.strip()


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


def _candidate_to_row(source: dict[str, Any], candidate: dict[str, Any], index: int, provider_name: str, model: str) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(candidate, dict):
        return None, "wrong_candidate_type"
    text = candidate.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "missing_text"
    style = str(candidate.get("style") or STYLES[index % len(STYLES)]).strip()
    if style not in STYLES:
        return None, "unknown_style"
    if normalize_text_for_dedup(text) == normalize_text_for_dedup(str(source.get("text", ""))):
        return None, "unchanged_text"
    row = _aug_row(source, text.strip(), style, index, provider_name, model)
    ok, reason = validate_augmented_row(row, source)
    return (row, "ok") if ok else (None, reason)


def augment_rows(rows: list[dict[str, Any]], provider: LLMProvider, max_inputs: int, paraphrases_per_example: int, include_original: bool, model: str) -> tuple[list[dict[str, Any]], Counter]:
    out = []
    reasons = Counter()
    provider_name = getattr(provider, "name", provider.__class__.__name__)
    for source in health_private_sources(rows, max_inputs):
        if include_original:
            out.append(source)
        try:
            candidates = provider.generate_candidates(source, paraphrases_per_example) if isinstance(provider, DryRunProvider) else parse_candidate_objects(provider.generate(build_prompt(source, paraphrases_per_example)))
        except CandidateParseError:
            reasons["parse_error"] += 1
            continue
        except Exception as exc:
            logging.exception("Augmentation failed for %s", source.get("id"))
            reasons[type(exc).__name__] += 1
            continue
        for i, candidate in enumerate(candidates):
            row, reason = _candidate_to_row(source, candidate, i, provider_name, model)
            if row is None:
                reasons[reason] += 1
            else:
                out.append(row)
    before = len(out)
    out = deduplicate_rows(out)
    reasons["duplicate"] += before - len(out)
    return out, reasons
