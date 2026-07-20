from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from typing import Any, Protocol

from .base import ModelUnavailable

DEFAULT_GRANITE_GUARDIAN_MODEL = "ibm-granite/granite-guardian-3.2-3b-a800m"
GRANITE_GUARDIAN_LANGUAGE_LIMITATION = "Granite Guardian 3.2 is trained and tested primarily in English; samples are passed as-is without translation."


class GraniteGuardianParseError(ValueError):
    pass


class GraniteGuardianRunner(Protocol):
    model_name: str

    def score_context_relevance(self, user_query: str, evidence: str) -> dict[str, Any]:
        ...


def parse_granite_guardian_output(
    text: str,
    risk_probability: float | None = None,
    relevance_probability: float | None = None,
) -> dict[str, Any]:
    match = re.match(r"\s*(Yes|No)\b", text, flags=re.I)
    if not match:
        raise GraniteGuardianParseError("could not parse Granite Guardian decision token as Yes or No")
    raw_label = match.group(1).title()
    confidence_match = re.search(r"<confidence>\s*(High|Medium|Low)\s*</confidence>", text, flags=re.I)
    confidence = confidence_match.group(1).title() if confidence_match else None
    if risk_probability is not None and relevance_probability is None:
        relevance_probability = 1.0 - risk_probability
    elif relevance_probability is not None and risk_probability is None:
        risk_probability = 1.0 - relevance_probability
    return {
        "risk_detected": raw_label == "Yes",
        "raw_label": raw_label,
        "risk_probability": risk_probability,
        "relevance_probability": relevance_probability,
        "confidence": confidence,
        "raw_output": text,
    }


def _single_token_id(tokenizer: Any, text: str) -> int | None:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    except Exception:
        return None
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def _first_batch_row(logits: Any) -> Any:
    shape = getattr(logits, "shape", None)
    if shape is not None and len(shape) == 2:
        return logits[0]
    if isinstance(logits, (list, tuple)) and logits and isinstance(logits[0], (list, tuple)):
        return logits[0]
    return logits


def _logit_at(row: Any, token_id: int) -> float:
    value = row[token_id]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def decision_probabilities_from_logits(tokenizer: Any, first_step_logits: Any) -> tuple[float | None, float | None]:
    yes_id = _single_token_id(tokenizer, "Yes")
    no_id = _single_token_id(tokenizer, "No")
    if yes_id is None or no_id is None:
        return None, None
    try:
        row = _first_batch_row(first_step_logits)
        yes_logit = _logit_at(row, yes_id)
        no_logit = _logit_at(row, no_id)
    except Exception:
        return None, None
    max_logit = max(yes_logit, no_logit)
    yes_exp = math.exp(yes_logit - max_logit)
    no_exp = math.exp(no_logit - max_logit)
    denom = yes_exp + no_exp
    if denom <= 0:
        return None, None
    risk_probability = yes_exp / denom
    return risk_probability, 1.0 - risk_probability


class HuggingFaceGraniteGuardianRunner:
    def __init__(
        self,
        model_id: str = DEFAULT_GRANITE_GUARDIAN_MODEL,
        device: str = "auto",
        offline: bool = False,
        cache_dir: str | None = None,
        max_new_tokens: int = 20,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        started = time.perf_counter()
        self.model_name = model_id
        self.model_id = model_id
        self.device = device
        self.offline = offline
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise ModelUnavailable(f"transformers unavailable for Granite Guardian: {exc}") from exc

        common_kwargs: dict[str, Any] = {"local_files_only": offline, "trust_remote_code": trust_remote_code}
        if cache_dir:
            common_kwargs["cache_dir"] = cache_dir

        model_kwargs: dict[str, Any] = dict(common_kwargs)
        model_kwargs["torch_dtype"] = "auto"
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        elif device != "cpu":
            model_kwargs["device_map"] = {"": device}
        if load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                model_kwargs.pop("torch_dtype", None)
            except Exception as exc:
                raise ModelUnavailable(f"4-bit Granite Guardian loading requested but bitsandbytes is unavailable: {exc}") from exc

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except Exception as exc:
            mode = "local/offline" if offline else "Hugging Face"
            raise ModelUnavailable(f"failed to load Granite Guardian model {model_id} from {mode}: {exc}") from exc

        try:
            self.model.eval()
        except Exception:
            pass
        self.loading_time_s = time.perf_counter() - started
        self.parameter_count = _parameter_count(getattr(self, "model", None))
        self.artifact_storage_size_mb = None
        self.metadata = {
            "granite_guardian_model": model_id,
            "granite_load_in_4bit": load_in_4bit,
            "granite_max_new_tokens": max_new_tokens,
            "granite_trust_remote_code": trust_remote_code,
            "device": device,
            "offline": offline,
            "language_limitation": GRANITE_GUARDIAN_LANGUAGE_LIMITATION,
        }

    def score_context_relevance(self, user_query: str, evidence: str) -> dict[str, Any]:
        messages = [
            {
                "role": "context",
                "content": evidence,
            },
            {
                "role": "user",
                "content": user_query,
            },
        ]
        guardian_config = {"risk_name": "context_relevance"}
        encoded = self.tokenizer.apply_chat_template(
            messages,
            guardian_config=guardian_config,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        encoded = self._move_to_input_device(encoded)
        generation_kwargs: dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": self.max_new_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        if isinstance(encoded, Mapping):
            output = self.model.generate(**encoded, **generation_kwargs)
        else:
            output = self.model.generate(encoded, **generation_kwargs)
        raw_output = self._decode_generated(output, encoded)
        risk_probability, relevance_probability = None, None
        scores = getattr(output, "scores", None)
        if scores:
            risk_probability, relevance_probability = decision_probabilities_from_logits(self.tokenizer, scores[0])
        return parse_granite_guardian_output(
            raw_output,
            risk_probability=risk_probability,
            relevance_probability=relevance_probability,
        )

    def _move_to_input_device(self, input_ids: Any) -> Any:
        try:
            device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        except Exception:
            device = None
        if device is not None and isinstance(input_ids, Mapping):
            return {key: value.to(device) if hasattr(value, "to") else value for key, value in input_ids.items()}
        if device is not None and hasattr(input_ids, "to"):
            return input_ids.to(device)
        return input_ids

    def _decode_generated(self, output: Any, input_ids: Any) -> str:
        sequences = getattr(output, "sequences", output)
        sequence = sequences[0]
        prompt_input_ids = input_ids.get("input_ids") if isinstance(input_ids, Mapping) else input_ids
        prompt_len = int(getattr(prompt_input_ids, "shape", [0, 0])[-1])
        generated_tokens = sequence[prompt_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def _parameter_count(model: Any) -> int | None:
    if model is None:
        return None
    try:
        return int(sum(param.numel() for param in model.parameters()))
    except Exception:
        return None
