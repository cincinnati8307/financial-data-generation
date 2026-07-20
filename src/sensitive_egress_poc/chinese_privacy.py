"""Language-aware privacy detection for Simplified and Traditional Chinese text.

The module deliberately keeps heavyweight model dependencies optional.  Model loading only
happens when a detector is constructed without an injected runner, so applications can use
the deterministic Chinese rules in constrained/offline deployments and tests can supply a
small runner without downloading a model.
"""
from __future__ import annotations

import ast
import importlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

DEFAULT_QWEN3_GUARD_MODEL = "Qwen/Qwen3Guard-Gen-8B"
DEFAULT_SHIELDLM_MODEL = "thu-coai/ShieldLM-6B-chatglm3"
DEFAULT_SHIELDLM_MODEL_QWEN = "thu-coai/ShieldLM-14B-qwen"
DEFAULT_SHIELDLM_MODEL_BAICHUAN = "thu-coai/ShieldLM-13B-baichuan2"
DEFAULT_SHIELDLM_MODEL_INTERNLM = "thu-coai/ShieldLM-7B-internlm2"
DEFAULT_SHIELDLM_MODEL_CHATGLM = "thu-coai/ShieldLM-6B-chatglm3"

_CHINESE_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_PRIVATE_FINANCE = re.compile(
    r"(?:我的|本人|个人).{0,12}(?:账户|账号|银行卡|信用卡|余额|工资|薪资|贷款|税单)|"
    r"(?:账户|账号|银行卡|信用卡)(?:号码|号|尾号)?.{0,10}(?:\*{2,}|\d{4,})|"
    r"(?:余额|工资|薪资|还款|贷款|个税|报税|交易记录).{0,12}(?:人民币|元|CNY|RMB|¥|\d)",
    re.I,
)
_MASKED_ACCOUNT = re.compile(r"(?:账号|账户|银行卡|卡号|信用卡)(?:号码|号|尾号)?\s*[:：]?\s*(?:\*{2,}\s*)?\d{4}", re.I)


class PrivacyRunner(Protocol):
    model_name: str

    def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class DetectionResult:
    is_private: bool
    score: float
    language: str
    entities: list[dict[str, Any]] = field(default_factory=list)
    model_score: float | None = None
    rule_score: float = 0.0
    centroid_score: float | None = None
    raw_response: str | None = None


def detect_chinese_language(text: str, minimum_ratio: float = 0.15) -> bool:
    """Return whether Chinese characters make up a meaningful part of *text*."""
    visible = [char for char in text if not char.isspace() and not char.isdigit()]
    return bool(visible) and sum(bool(_CHINESE_CHARACTER.match(char)) for char in visible) / len(visible) >= minimum_ratio


def chinese_privacy_prompt(text: str) -> str:
    return (
        "你是隐私泄露检测器。判断文本是否包含私人金融信息（账户、银行卡、工资、余额、"
        "贷款、税务或交易记录），而不是公开财经新闻。只输出 JSON："
        '{"private": true|false, "score": 0到1, "entities": [{"text": "", "type": ""}]}。\n'
        f"文本：{text}"
    )


def parse_privacy_response(response: str) -> tuple[bool, float | None, list[dict[str, Any]]]:
    """Parse a tolerant JSON response from a Chinese guard model."""
    candidates = [response.strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", response, flags=re.I | re.S)
    candidates.extend(item.strip() for item in fenced)
    first, last = response.find("{"), response.rfind("}")
    if first >= 0 and last > first:
        candidates.append(response[first : last + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if not isinstance(value, dict):
            continue
        private = value.get("private", value.get("is_private", value.get("sensitive", False)))
        if isinstance(private, str):
            private = private.strip().lower() in {"true", "1", "yes", "是", "敏感"}
        score_value = value.get("score", value.get("confidence"))
        try:
            score = max(0.0, min(1.0, float(score_value))) if score_value is not None else None
        except (TypeError, ValueError):
            score = None
        entities = value.get("entities", value.get("spans", []))
        return bool(private), score, list(entities) if isinstance(entities, list) else []
    raise ValueError("model response did not contain a privacy JSON object")


class HuggingFacePrivacyRunner:
    """Generic causal-LM runner for Qwen3Guard and ShieldLM checkpoints following the official example pattern."""

    def __init__(self, model_name: str, *, device: str = "auto", offline: bool = False, cache_dir: str | None = None, max_new_tokens: int = 128, trust_remote_code: bool = True) -> None:
        try:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError("Chinese privacy models require transformers and torch") from exc
        
        # Determine model_base from model_name following the official ShieldLM pattern
        self.model_base = self._determine_model_base(model_name)
        
        # Prepare kwargs following the official Qwen3Guard-Gen example
        tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
        if offline:
            tokenizer_kwargs["local_files_only"] = offline
            
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
            "trust_remote_code": trust_remote_code
        }
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
        if offline:
            model_kwargs["local_files_only"] = offline
        
        try:
            # Load tokenizer following the Qwen3Guard-Gen example (no padding_side specified)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            
            # Load model following the exact Qwen3Guard-Gen example approach
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            self.model.eval()
            
        except Exception as exc:
            raise RuntimeError(f"failed to load Chinese privacy model {model_name}: {exc}") from exc
        
        self.torch = importlib.import_module("torch")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
    
    def _determine_model_base(self, model_name: str) -> str:
        """Determine the model base (qwen, baichuan, internlm, chatglm) from the model name."""
        model_name_lower = model_name.lower()
        if "qwen" in model_name_lower:
            return "qwen"
        elif "baichuan" in model_name_lower:
            return "baichuan"
        elif "internlm" in model_name_lower:
            return "internlm"
        elif "chatglm" in model_name_lower:
            return "chatglm"
        else:
            # Default fallback
            return "qwen"

    def generate(self, prompt: str) -> str:
        # Follow the Qwen3Guard-Gen example for model inputs
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        
        # Generate following the Qwen3Guard-Gen example pattern
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens
        )
        
        # Decode only the new generated tokens (following Qwen3Guard-Gen example)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)


class ChinesePrivacyDetector:
    """Ensembles Chinese rules, an optional guard model, and optional centroid score."""

    def __init__(self, model_name: str = DEFAULT_QWEN3_GUARD_MODEL, *, runner: PrivacyRunner | None = None, centroid_scorer: Callable[[str], float] | None = None, threshold: float = 0.5, device: str = "auto", offline: bool = False, cache_dir: str | None = None, max_new_tokens: int = 128) -> None:
        self.model_name = model_name
        self.centroid_scorer = centroid_scorer
        self.threshold = threshold
        self.init_error: str | None = None
        if runner is not None:
            self.runner = runner
        else:
            try:
                self.runner = HuggingFacePrivacyRunner(model_name, device=device, offline=offline, cache_dir=cache_dir, max_new_tokens=max_new_tokens)
            except RuntimeError as exc:
                self.runner = None
                self.init_error = str(exc)

    def detect_privacy(self, text: str) -> DetectionResult:
        language = "zh" if detect_chinese_language(text) else "non_zh"
        matches = list(_PRIVATE_FINANCE.finditer(text)) + list(_MASKED_ACCOUNT.finditer(text))
        entities = [{"text": match.group(0), "type": "PRIVATE_FINANCE", "start": match.start(), "end": match.end(), "score": 0.8} for match in matches]
        rule_score = 0.8 if entities else 0.0
        model_score: float | None = None
        raw_response: str | None = None
        if language == "zh" and self.runner is not None:
            raw_response = self.runner.generate(chinese_privacy_prompt(text))
            private, model_score, model_entities = parse_privacy_response(raw_response)
            if model_score is None:
                model_score = 1.0 if private else 0.0
            entities.extend(item for item in model_entities if isinstance(item, dict))
        centroid_score = self.centroid_scorer(text) if self.centroid_scorer is not None else None
        scores = [rule_score]
        if model_score is not None:
            scores.append(model_score)
        if centroid_score is not None:
            scores.append(max(0.0, min(1.0, float(centroid_score))))
        score = max(scores)
        return DetectionResult(score >= self.threshold, score, language, entities, model_score, rule_score, centroid_score, raw_response)

    def detect_batch(self, texts: list[str]) -> list[DetectionResult]:
        return [self.detect_privacy(text) for text in texts]


def create_app(detector: ChinesePrivacyDetector):
    """Create an optional FastAPI endpoint without making FastAPI a core dependency."""
    try:
        fastapi = importlib.import_module("fastapi")
    except ImportError as exc:
        raise RuntimeError("API serving requires fastapi; install it to use create_app") from exc
    app = fastapi.FastAPI(title="Chinese Privacy Detector")

    @app.post("/detect")
    def detect(payload: dict[str, str]) -> dict[str, Any]:
        started = time.perf_counter()
        result = detector.detect_privacy(payload.get("text", ""))
        return {**result.__dict__, "latency_ms": (time.perf_counter() - started) * 1000.0}

    @app.post("/detect/batch")
    def detect_batch(payload: dict[str, list[str]]) -> list[dict[str, Any]]:
        return [result.__dict__ for result in detector.detect_batch(payload.get("texts", []))]

    return app
