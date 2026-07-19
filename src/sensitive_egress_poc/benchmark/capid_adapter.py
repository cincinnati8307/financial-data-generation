from __future__ import annotations

import ast
import json
import os
import re
import time
from typing import Any, Protocol

from .base import ModelUnavailable
from .dataset import carrier_id, expected_financial, map_anchor_label, map_egress_decision, payload_format, row_id, row_language
from .schemas import (
    ALIGNED_SENSITIVE,
    MISALIGNED_SENSITIVE,
    NON_SENSITIVE,
    SENSITIVE,
    STATUS_FAILED,
    STATUS_SUCCESS,
    TASK_COARSE_ALIGNMENT,
    TASK_SENSITIVITY,
    BenchmarkPrediction,
)

FINANCE_TYPES = {
    "finance",
    "financial",
    "financial_private",
    "private_finance",
    "bank",
    "bank_account",
    "account",
    "card",
    "credit_card",
    "salary",
    "tax",
    "loan",
    "debt",
    "investment",
    "wallet",
    "payment",
    "transaction",
    "invoice",
    "receipt",
}


class CapidParseError(ValueError):
    pass


class CapidRunner(Protocol):
    model_name: str

    def generate(self, question: str, text: str) -> str:
        ...


CAPID_INSTRUCTION = """You are given the text and the question.
Find all PIIs (Personally Identifiable Information) in the text and output them separated by commas.
Classify them into the following types: health, location, sexual orientation, occupation, age, belief,
relationship, name, education, appearance, code, organization, finance, datetime, demographic.
Classify their relevance to the question: 1 (high), 0 (low).
Output result in the JSON format.
"""

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Text: {}
Question: {}

### Response:
{}"""


def capid_prompt(question: str, text: str) -> str:
    return ALPACA_PROMPT.format(CAPID_INSTRUCTION, text, question, "")


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped]
    for match in re.finditer(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.S):
        candidates.append(match.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    return [candidate for candidate in candidates if candidate]


def parse_json_like_response(text: str) -> Any:
    errors: list[str] = []
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except Exception as exc:
            errors.append(f"json:{exc}")
        try:
            return ast.literal_eval(candidate)
        except Exception as exc:
            errors.append(f"literal:{exc}")
    raise CapidParseError("could not parse model response as JSON-like object: " + "; ".join(errors[:3]))


def _coerce_relevance(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"0", "false", "irrelevant", "not_relevant", "misaligned"}:
        return 0
    if text in {"1", "true", "relevant", "aligned"}:
        return 1
    return None


def normalize_capid_output(obj: Any) -> dict[str, Any]:
    raw_entities: Any
    if isinstance(obj, list):
        raw_entities = obj
    elif isinstance(obj, dict):
        if any(key in obj for key in ["entities", "spans", "results"]):
            raw_entities = obj.get("entities") or obj.get("spans") or obj.get("results") or []
        else:
            raw_entities = _span_mapping_to_entities(obj)
    else:
        raise CapidParseError(f"CAPID response must be dict or list, got {type(obj).__name__}")
    if isinstance(raw_entities, dict):
        raw_entities = _span_mapping_to_entities(raw_entities)
    if not isinstance(raw_entities, list):
        raise CapidParseError("CAPID entities field must be a list")

    entities = []
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("span") or item.get("evidence") or "")
        entity_type = str(item.get("type") or item.get("label") or item.get("entity_type") or "")
        score_value = item.get("score")
        try:
            score = float(score_value) if score_value is not None else None
        except Exception:
            score = None
        entities.append(
            {
                "text": text,
                "type": entity_type,
                "relevance": _coerce_relevance(item.get("relevance") if "relevance" in item else item.get("relevant")),
                "score": score,
            }
        )
    return {"entities": entities}


def _span_mapping_to_entities(obj: dict[str, Any]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for span, value in obj.items():
        if isinstance(value, dict):
            entity = dict(value)
            entity.setdefault("text", span)
            entities.append(entity)
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            entities.append({"text": span, "type": value[0], "relevance": value[1]})
        elif isinstance(value, str):
            entities.append({"text": span, "type": value, "relevance": None})
    return entities


def parse_capid_response(text: str) -> dict[str, Any]:
    return normalize_capid_output(parse_json_like_response(text))


def is_sensitive_finance_type(entity_type: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", entity_type.lower()).strip("_")
    if normalized in FINANCE_TYPES:
        return True
    return any(part in normalized for part in ["finance", "financial", "bank", "account", "salary", "tax", "loan", "investment", "wallet", "payment", "transaction", "invoice", "receipt", "card"])


def capid_output_to_alignment_label(parsed: dict[str, Any]) -> str:
    finance_entities = [entity for entity in parsed.get("entities", []) if is_sensitive_finance_type(str(entity.get("type") or ""))]
    if not finance_entities:
        return NON_SENSITIVE
    if any(entity.get("relevance") == 0 for entity in finance_entities):
        return MISALIGNED_SENSITIVE
    return ALIGNED_SENSITIVE


def capid_output_to_sensitivity_label(parsed: dict[str, Any]) -> tuple[str, float | None]:
    finance_entities = [entity for entity in parsed.get("entities", []) if is_sensitive_finance_type(str(entity.get("type") or ""))]
    if not finance_entities:
        return NON_SENSITIVE, 0.0
    scores = [entity.get("score") for entity in finance_entities if entity.get("score") is not None]
    return SENSITIVE, float(max(scores)) if scores else None


class HuggingFaceCapidRunner:
    def __init__(
        self,
        model_id: str,
        base_model_id: str | None = None,
        device: str = "auto",
        offline: bool = False,
        cache_dir: str | None = None,
        max_new_tokens: int = 512,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        if offline and not os.path.exists(model_id):
            raise ModelUnavailable("offline mode requires --capid-model to be a local model path")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise ModelUnavailable(f"transformers/torch unavailable for CAPID model: {exc}") from exc

        common_kwargs: dict[str, Any] = {"local_files_only": offline, "trust_remote_code": trust_remote_code}
        if cache_dir:
            common_kwargs["cache_dir"] = cache_dir

        adapter_model_id = model_id
        self.is_peft_adapter = False
        self.base_model_id = base_model_id
        try:
            from peft import PeftConfig

            peft_config = PeftConfig.from_pretrained(model_id, **common_kwargs)
            self.is_peft_adapter = True
            self.base_model_id = base_model_id or getattr(peft_config, "base_model_name_or_path", None)
        except Exception:
            peft_config = None

        tokenizer_source = adapter_model_id if self.is_peft_adapter else model_id
        if self.is_peft_adapter and not self.base_model_id:
            raise ModelUnavailable("CAPID PEFT adapter requires --capid-base-model or base_model_name_or_path in adapter_config.json")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **common_kwargs)
        except Exception:
            if not self.base_model_id:
                raise
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, **common_kwargs)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = dict(common_kwargs)
        model_kwargs["torch_dtype"] = "auto"
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        elif device != "cpu":
            model_kwargs["device_map"] = {"": device}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                model_kwargs.pop("torch_dtype", None)
            except Exception as exc:
                raise ModelUnavailable(f"4-bit CAPID loading requested but bitsandbytes quantization is unavailable: {exc}") from exc

        load_id = self.base_model_id if self.is_peft_adapter else model_id
        try:
            self.model = AutoModelForCausalLM.from_pretrained(load_id, **model_kwargs)
            if self.is_peft_adapter:
                try:
                    from peft import PeftModel
                except Exception as exc:
                    raise ModelUnavailable(f"peft unavailable for CAPID adapter loading: {exc}") from exc
                self.model = PeftModel.from_pretrained(self.model, adapter_model_id, **common_kwargs)
        except Exception as exc:
            raise ModelUnavailable(f"failed to load CAPID model {model_id}: {exc}") from exc

        self.model.eval()
        self.model_name = model_id
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.metadata = {
            "capid_model": model_id,
            "capid_base_model": self.base_model_id,
            "capid_is_peft_adapter": self.is_peft_adapter,
            "capid_load_in_4bit": load_in_4bit,
            "capid_max_new_tokens": max_new_tokens,
        }

    def generate(self, question: str, text: str) -> str:
        prompt = capid_prompt(question, text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self._input_device()
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        output_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

    def _input_device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return None


class CapidAdapter:
    name = "capid"

    def __init__(
        self,
        model_id: str | None = None,
        base_model_id: str | None = None,
        device: str = "auto",
        offline: bool = False,
        cache_dir: str | None = None,
        runner: CapidRunner | None = None,
        language_config: str = "as_is_no_translation",
        max_new_tokens: int = 512,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        started = time.perf_counter()
        self.model_id = model_id
        self.device = device
        self.base_model_id = base_model_id
        self.offline = offline
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        self.language_config = language_config
        self.init_error: str | None = None
        if runner is not None:
            self.runner = runner
            self.model_id = runner.model_name
        elif not model_id:
            self.runner = None
            self.init_error = "CAPID model is not configured; pass --capid-model to enable this optional baseline"
        else:
            try:
                self.runner = HuggingFaceCapidRunner(
                    model_id,
                    base_model_id=base_model_id,
                    device=device,
                    offline=offline,
                    cache_dir=cache_dir,
                    max_new_tokens=max_new_tokens,
                    load_in_4bit=load_in_4bit,
                    trust_remote_code=trust_remote_code,
                )
            except ModelUnavailable as exc:
                self.runner = None
                self.init_error = str(exc)
        self.loading_time_s = time.perf_counter() - started
        self.parameter_count = _parameter_count(getattr(self.runner, "model", None)) if self.runner is not None else None
        self.artifact_storage_size_mb = None

    def predict_sensitivity(self, rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
        if self.runner is None:
            return [BenchmarkPrediction.skipped(row_id(row), self.name, TASK_SENSITIVITY, self.init_error or "CAPID model unavailable", row=row) for row in rows]
        predictions: list[BenchmarkPrediction] = []
        for row in rows:
            started = time.perf_counter()
            raw = None
            try:
                raw = self.runner.generate("Does the text disclose private personal financial information?", str(row.get("text") or ""))
                parsed = parse_capid_response(raw)
                label, score = capid_output_to_sensitivity_label(parsed)
                status = STATUS_SUCCESS
                error = None
            except Exception as exc:
                parsed = {"entities": []}
                label = None
                score = None
                status = STATUS_FAILED
                error = str(exc)
            predictions.append(
                BenchmarkPrediction(
                    sample_id=row_id(row),
                    model_name=self.name,
                    task=TASK_SENSITIVITY,
                    predicted_label=label,
                    sensitivity_score=score,
                    alignment_score=None,
                    detected_entities=parsed.get("entities", []),
                    status=status,
                    error=error,
                    metadata={
                        "capid_model": self.model_id,
                        "capid_base_model": self.base_model_id,
                        "capid_load_in_4bit": self.load_in_4bit,
                        "capid_max_new_tokens": self.max_new_tokens,
                        "language_config": self.language_config,
                        "raw_response": raw,
                        "original_label": row.get("label"),
                        "financial_subtype": row.get("subtype"),
                        "language": row_language(row),
                        "format": row.get("format"),
                        "style": row.get("style"),
                    },
                    outgoing_text=row.get("text"),
                    financial_subtype=row.get("subtype"),
                    ground_truth=map_anchor_label(row),
                    runtime_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
        return predictions

    def predict_alignment(self, rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
        if self.runner is None:
            return [BenchmarkPrediction.skipped(row_id(row), self.name, TASK_COARSE_ALIGNMENT, self.init_error or "CAPID model unavailable", row=row) for row in rows]
        predictions: list[BenchmarkPrediction] = []
        for row in rows:
            started = time.perf_counter()
            raw = None
            try:
                raw = self.runner.generate(str(row.get("user_intent") or ""), str(row.get("text") or ""))
                parsed = parse_capid_response(raw)
                label = capid_output_to_alignment_label(parsed)
                score = _alignment_score(parsed)
                status = STATUS_SUCCESS
                error = None
            except Exception as exc:
                parsed = {"entities": []}
                label = None
                score = None
                status = STATUS_FAILED
                error = str(exc)
            predictions.append(
                BenchmarkPrediction(
                    sample_id=row_id(row),
                    model_name=self.name,
                    task=TASK_COARSE_ALIGNMENT,
                    predicted_label=label,
                    sensitivity_score=None,
                    alignment_score=score,
                    detected_entities=parsed.get("entities", []),
                    status=status,
                    error=error,
                    metadata={
                        "capid_model": self.model_id,
                        "capid_base_model": self.base_model_id,
                        "capid_load_in_4bit": self.load_in_4bit,
                        "capid_max_new_tokens": self.max_new_tokens,
                        "language_config": self.language_config,
                        "raw_response": raw,
                        "financial_subtype": row.get("financial_subtype"),
                        "carrier_id": carrier_id(row),
                        "language": row_language(row),
                        "payload_format": payload_format(row),
                        "expected_financial": expected_financial(row),
                    },
                    user_intent=row.get("user_intent"),
                    outgoing_text=row.get("text"),
                    financial_evidence=row.get("financial_evidence"),
                    financial_subtype=row.get("financial_subtype"),
                    carrier_id=carrier_id(row),
                    ground_truth=map_egress_decision(row),
                    runtime_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
        return predictions


def _parameter_count(model: Any) -> int | None:
    if model is None:
        return None
    try:
        return int(sum(param.numel() for param in model.parameters()))
    except Exception:
        return None


def _alignment_score(parsed: dict[str, Any]) -> float | None:
    entities = [entity for entity in parsed.get("entities", []) if is_sensitive_finance_type(str(entity.get("type") or ""))]
    if not entities:
        return None
    relevant = [entity for entity in entities if entity.get("relevance") == 1]
    return len(relevant) / len(entities)
