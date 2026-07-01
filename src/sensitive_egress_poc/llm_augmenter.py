from __future__ import annotations

import json, logging, os
from collections import Counter
from typing import Any

from .filters import deduplicate_rows, validate_augmented_row

STYLES = ["zh_casual", "zh_formal", "zh_en_codeswitch", "agent_summary", "key_value", "email_mixed"]

class LLMProvider:
    name = "base"
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class DryRunProvider(LLMProvider):
    name = "dry-run"
    def __init__(self, model: str = "dry-run") -> None:
        self.model = model
    def generate_rows(self, source: dict[str, Any], n: int) -> list[dict[str, Any]]:
        rows=[]
        for i, style in enumerate(STYLES[:n]):
            text = source["text"]
            variants = {
                "zh_casual": f"随手记一下：{text}",
                "zh_formal": f"个人财务记录如下：{text}",
                "zh_en_codeswitch": f"Personal finance note: {text}",
                "agent_summary": f"Agent summary: 该片段包含个人财务信息；evidence={text}",
                "key_value": f"style={style}\nsubtype={source['subtype']}\nvalue={text}",
                "email_mixed": f"Hi，补充一个 private finance note：{text}",
            }
            rows.append(_aug_row(source, variants[style], style, i, self.name, self.model))
        return rows
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
        self.client = OpenAI(); self.model = model
    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=[{"role":"user","content":prompt}], temperature=0.4)
        return resp.choices[0].message.content or ""

def _aug_row(source: dict[str, Any], text: str, style: str, i: int, provider: str, model: str) -> dict[str, Any]:
    return {"id": f"aug_{source['id']}_{i}", "text": text, "label": "financial_private", "subtype": source["subtype"], "region": source.get("region","mainland_cn"), "language": source.get("language","zh"), "format": "key_value" if style=="key_value" else ("agent_summary" if style=="agent_summary" else "natural_sentence"), "style": style, "sensitivity_level": "high", "source": "llm_paraphrase", "parent_id": source["id"], "meta": {"augmentation_model": model, "augmentation_provider": provider, "original_text": source["text"]}}

def build_prompt(row: dict[str, Any], n: int) -> str:
    return f"Return JSONL only with {n} paraphrases preserving label financial_private, subtype {row['subtype']}, amounts/banks/merchants/card last4. Forbid real names, account numbers, card numbers, phones, IDs, API keys, private keys, passwords, OAuth tokens, cookies. Styles: {', '.join(STYLES)}. Source JSON: {json.dumps(row, ensure_ascii=False)}"

def parse_jsonl(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]

def augment_rows(rows: list[dict[str, Any]], provider: LLMProvider, max_inputs: int, paraphrases_per_example: int, include_original: bool, model: str) -> tuple[list[dict[str, Any]], Counter]:
    out=[]; reasons=Counter(); sources=[r for r in rows if r.get("label")=="financial_private"][:max_inputs]
    for row in sources:
        try:
            candidates = provider.generate_rows(row, paraphrases_per_example) if isinstance(provider, DryRunProvider) else parse_jsonl(provider.generate(build_prompt(row, paraphrases_per_example)))
            if include_original: out.append(row)
            for c in candidates:
                ok, reason = validate_augmented_row(c, row)
                if ok: out.append(c)
                else: reasons[reason]+=1
        except Exception as exc:
            logging.exception("Augmentation failed for %s", row.get("id")); reasons[type(exc).__name__]+=1
    before=len(out); out=deduplicate_rows(out); reasons["duplicate"] += before-len(out)
    return out, reasons
