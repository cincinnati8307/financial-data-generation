"""Language-aware privacy detection for Simplified and Traditional Chinese text.

The module deliberately keeps heavyweight model dependencies optional. Model loading only
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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

DEFAULT_QWEN3_GUARD_MODEL = "Qwen/Qwen3Guard-Gen-8B"
DEFAULT_SHIELDLM_MODEL = "thu-coai/ShieldLM-6B-chatglm3"
DEFAULT_SHIELDLM_MODEL_QWEN = "thu-coai/ShieldLM-14B-qwen"
DEFAULT_SHIELDLM_MODEL_BAICHUAN = "thu-coai/ShieldLM-13B-baichuan2"
DEFAULT_SHIELDLM_MODEL_INTERNLM = "thu-coai/ShieldLM-7B-internlm2"
DEFAULT_SHIELDLM_MODEL_CHATGLM = "thu-coai/ShieldLM-6B-chatglm3"
MODEL_FAMILY_QWEN3_GUARD = "qwen3guard"
MODEL_FAMILY_SHIELDLM = "shieldlm"

_CHINESE_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_PRIVATE_FINANCE = re.compile(
    r"(?:我的|本人|个人).{0,12}(?:账户|账号|银行卡|信用卡|余额|工资|薪资|贷款|税单)|"
    r"(?:账户|账号|银行卡|信用卡)(?:号码|号|尾号)?.{0,10}(?:\*{2,}|\d{4,})|"
    r"(?:余额|工资|薪资|还款|贷款|个税|报税|交易记录).{0,12}(?:人民币|元|CNY|RMB|¥|\d)",
    re.I,
)
_MASKED_ACCOUNT = re.compile(r"(?:账号|账户|银行卡|卡号|信用卡)(?:号码|号|尾号)?\s*[:：]?\s*(?:\*{2,}\s*)?\d{4}", re.I)
_QWEN_SAFETY = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.I)
_QWEN_CATEGORY = re.compile(r"(PII|Personally Identifiable Information|None)", re.I)
_SHIELDLM_ANSWER = re.compile(r"\[(?:答案|Answer)\]\s*([^\n\r\[]+)", re.I)
_SHIELDLM_PRIVACY_RULES = (
    "1. 如果B的回复包含个人或他人的私人金融信息，应判为不安全，包括账户、账号、"
    "银行卡、信用卡、工资、薪资、余额、贷款、债务、税务、交易记录、投资持仓或付款信息。\n"
    "2. 即使号码被部分遮盖，只要能看出是个人账户、卡号、余额、交易或债务等私人金融事实，也应判为不安全。\n"
    "3. 如果B的回复只是公开财经新闻、通用金融知识、虚构占位符或不涉及个人财务事实的内容，应判为安全。\n"
    "4. 中英混合、简体中文和繁体中文内容都必须按照以上规则判断。"
)


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
    """Backward-compatible alias for the Qwen3Guard privacy prompt."""
    return qwen3guard_privacy_prompt(text)


def qwen3guard_privacy_prompt(text: str) -> str:
    return (
        "你是隐私泄露检测器。判断文本是否包含未经授权披露的私人金融信息（账户、银行卡、"
        "工资、余额、贷款、税务或交易记录），而不是公开财经新闻。"
        "请按 Qwen3Guard 标签格式输出：\n"
        "Safety: Safe|Unsafe|Controversial\n"
        "Categories: PII|None\n"
        f"文本：{text}"
    )


def shieldlm_privacy_prompt(text: str, model_base: str = "chatglm") -> str:
    ipt = (
        "您是一位文本安全专家，具有很强的语言理解能力。以下是一段A和B之间的对话，"
        "您需要检查B的回复是否安全。您需要对回复进行详细的分析，并考虑其可能具有的潜在含义。"
        "以下是若干条您在分析过程中必须严格遵循的规则：\n"
        f"{_SHIELDLM_PRIVACY_RULES}\n"
        "您需要提供一个总体评估的答案，表明B的反应是安全、不安全还是有争议的（从安全的角度），"
        "在提供您的详细解释之前。您必须严格按照以下形式进行回复：\n"
        "[答案] 你的答案\n"
        "[分析] 你的详细分析\n\n"
        "A: 请检查B的回复是否泄露私人金融信息。\n"
        f"B: {text}"
    )
    return _add_shieldlm_model_prompt(ipt, model_base)


def _add_shieldlm_model_prompt(prompt: str, model_base: str) -> str:
    if model_base in {"qwen", "internlm"}:
        return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    if model_base == "baichuan":
        return prompt
    if model_base == "chatglm":
        return f"[gMASK]sop<|user|> \n {prompt}<|assistant|> \n"
    raise ValueError(f"unsupported ShieldLM model base: {model_base}")


def normalize_model_family(model_name: str, model_family: str | None = None) -> str:
    if model_family:
        family = model_family.strip().lower().replace("-", "").replace("_", "")
        if family in {"qwen3guard", "qwen"}:
            return MODEL_FAMILY_QWEN3_GUARD
        if family == "shieldlm":
            return MODEL_FAMILY_SHIELDLM
        raise ValueError(f"unsupported Chinese privacy model family: {model_family}")
    if "shieldlm" in model_name.lower():
        return MODEL_FAMILY_SHIELDLM
    return MODEL_FAMILY_QWEN3_GUARD


def determine_shieldlm_model_base(model_name: str, model_base: str | None = None) -> str:
    if model_base:
        base = model_base.strip().lower()
        if base not in {"qwen", "baichuan", "internlm", "chatglm"}:
            raise ValueError(f"unsupported ShieldLM model base: {model_base}")
        return base
    lowered = model_name.lower()
    if "baichuan" in lowered:
        return "baichuan"
    if "internlm" in lowered:
        return "internlm"
    if "chatglm" in lowered:
        return "chatglm"
    return "qwen"


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


def parse_qwen3guard_response(response: str) -> tuple[bool, float | None, list[dict[str, Any]]]:
    try:
        return parse_privacy_response(response)
    except ValueError:
        pass
    safety = _QWEN_SAFETY.search(response)
    if not safety:
        raise ValueError("Qwen3Guard response did not contain a Safety label")
    label = safety.group(1).lower()
    categories = [match.lower() for match in _QWEN_CATEGORY.findall(response)]
    has_pii = any(category in {"pii", "personally identifiable information"} for category in categories)
    if label == "unsafe":
        return True, 0.95 if has_pii or not categories else 0.85, []
    if label == "controversial":
        return True, 0.6, []
    return False, 0.05, []


def parse_shieldlm_response(response: str) -> tuple[bool, float | None, list[dict[str, Any]]]:
    answer = _SHIELDLM_ANSWER.search(response)
    lines = response.strip().splitlines()
    label_text = answer.group(1).strip() if answer else (lines[0].strip() if lines else "")
    label = _normalize_guard_label(label_text)
    if label == "unsafe":
        return True, 0.95, []
    if label == "controversial":
        return True, 0.6, []
    if label == "safe":
        return False, 0.05, []
    raise ValueError("ShieldLM response did not contain a safe/unsafe/controversial answer")


def parse_model_privacy_response(response: str, model_family: str) -> tuple[bool, float | None, list[dict[str, Any]]]:
    if model_family == MODEL_FAMILY_SHIELDLM:
        return parse_shieldlm_response(response)
    if model_family == MODEL_FAMILY_QWEN3_GUARD:
        return parse_qwen3guard_response(response)
    return parse_privacy_response(response)


def _normalize_guard_label(label_text: str) -> str | None:
    compact = re.sub(r"\s+", "", label_text).lower()
    if "不安全" in compact or re.search(r"\bunsafe\b", label_text, flags=re.I):
        return "unsafe"
    if "有争议" in compact or "controversial" in compact:
        return "controversial"
    if "安全" in compact or re.search(r"\bsafe\b", label_text, flags=re.I):
        return "safe"
    return None


def _patch_tokenizer_padding_for_transformers(tokenizer: Any) -> None:
    original_pad = getattr(tokenizer, "_pad", None)
    if original_pad is None or getattr(tokenizer, "_sensitive_egress_padding_patched", False):
        return

    def compat_pad(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("padding_side", None)
        return original_pad(*args, **kwargs)

    tokenizer._pad = compat_pad
    tokenizer._sensitive_egress_padding_patched = True


def _patch_remote_model_class_for_transformers(config: Any, model_name: str, common_kwargs: dict[str, Any]) -> None:
    auto_map = getattr(config, "auto_map", None) or {}
    class_reference = auto_map.get("AutoModelForCausalLM")
    if not class_reference:
        return
    try:
        dynamic_module_utils = importlib.import_module("transformers.dynamic_module_utils")
        get_class = dynamic_module_utils.get_class_from_dynamic_module
        model_class = get_class(
            class_reference,
            model_name,
            cache_dir=common_kwargs.get("cache_dir"),
            local_files_only=bool(common_kwargs.get("local_files_only")),
        )
    except Exception:
        return
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}
    if not hasattr(model_class, "_extract_past_from_model_output"):
        def extract_past_from_model_output(self: Any, outputs: Any, standardize_cache_format: bool = False) -> Any:
            if hasattr(outputs, "past_key_values"):
                return outputs.past_key_values
            if isinstance(outputs, dict):
                return outputs.get("past_key_values")
            if isinstance(outputs, (tuple, list)) and len(outputs) > 1:
                return outputs[1]
            return None

        model_class._extract_past_from_model_output = extract_past_from_model_output


class HuggingFacePrivacyRunner:
    """Hugging Face runner for Qwen3Guard and ShieldLM Chinese privacy checkpoints."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        offline: bool = False,
        cache_dir: str | None = None,
        max_new_tokens: int = 128,
        trust_remote_code: bool = True,
        model_family: str | None = None,
        model_base: str | None = None,
    ) -> None:
        try:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError("Chinese privacy models require transformers and torch") from exc

        self.model_name = model_name
        self.model_family = normalize_model_family(model_name, model_family)
        self.model_base = determine_shieldlm_model_base(model_name, model_base) if self.model_family == MODEL_FAMILY_SHIELDLM else None
        self.max_new_tokens = max_new_tokens
        self.device = device
        common_kwargs: dict[str, Any] = {"local_files_only": offline, "trust_remote_code": trust_remote_code}
        if cache_dir:
            common_kwargs["cache_dir"] = cache_dir

        tokenizer_kwargs = dict(common_kwargs)
        if self.model_family == MODEL_FAMILY_SHIELDLM:
            tokenizer_kwargs["padding_side"] = "left"

        has_cuda = torch.cuda.is_available()
        model_kwargs: dict[str, Any] = dict(common_kwargs)
        if self.model_family == MODEL_FAMILY_SHIELDLM and has_cuda:
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = "auto"
        if device == "auto" and has_cuda:
            model_kwargs["device_map"] = "auto"
        elif device != "auto" and device != "cpu":
            model_kwargs["device_map"] = {"": device}

        if self.model_family == MODEL_FAMILY_SHIELDLM:
            try:
                config = transformers.AutoConfig.from_pretrained(model_name, **common_kwargs)
                if not hasattr(config, "max_length"):
                    config.max_length = getattr(config, "seq_length", 8192)
                if not hasattr(config, "num_hidden_layers"):
                    config.num_hidden_layers = getattr(config, "num_layers", None)
                _patch_remote_model_class_for_transformers(config, model_name, common_kwargs)
                model_kwargs["config"] = config
            except Exception as exc:
                raise RuntimeError(f"failed to load ShieldLM config {model_name}: {exc}") from exc

        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            if self.model_family == MODEL_FAMILY_SHIELDLM:
                _patch_tokenizer_padding_for_transformers(self.tokenizer)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            if self.model_family == MODEL_FAMILY_SHIELDLM:
                if hasattr(self.model.config, "max_length"):
                    try:
                        delattr(self.model.config, "max_length")
                    except Exception:
                        self.model.config.max_length = None
                self.model.config.use_cache = False
                if hasattr(self.model, "generation_config"):
                    self.model.generation_config.use_cache = False
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(f"failed to load Chinese privacy model {model_name}: {exc}") from exc

        if getattr(self.tokenizer, "eos_token", None) is None:
            try:
                self.tokenizer.eos_token = "<|endoftext|>"
            except Exception:
                pass
        if getattr(self.tokenizer, "pad_token", None) is None:
            try:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception:
                pass
        self.parameter_count = _parameter_count(self.model)
        self.artifact_storage_size_mb = None

    def generate(self, prompt: str) -> str:
        prompt_text = self._format_prompt(prompt)
        if self.model_family == MODEL_FAMILY_SHIELDLM:
            model_inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True)
        else:
            model_inputs = self.tokenizer([prompt_text], return_tensors="pt", truncation=True, padding=True)
        model_inputs = self._move_to_input_device(model_inputs)
        generation_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        if self.model_family == MODEL_FAMILY_SHIELDLM:
            generation_kwargs["use_cache"] = False
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        generated_ids = self.model.generate(**model_inputs, **generation_kwargs)
        sequence = generated_ids[0]
        prompt_len = int(model_inputs["input_ids"].shape[-1])
        output_ids = sequence[prompt_len:].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def _format_prompt(self, prompt: str) -> str:
        if self.model_family != MODEL_FAMILY_QWEN3_GUARD or not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def _move_to_input_device(self, model_inputs: Any) -> Any:
        try:
            device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        except Exception:
            device = None
        if device is not None and isinstance(model_inputs, Mapping):
            return {key: value.to(device) if hasattr(value, "to") else value for key, value in model_inputs.items()}
        if device is not None and hasattr(model_inputs, "to"):
            return model_inputs.to(device)
        return model_inputs


class ChinesePrivacyDetector:
    """Ensembles Chinese rules, an optional guard model, and optional centroid score."""

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN3_GUARD_MODEL,
        *,
        runner: PrivacyRunner | None = None,
        centroid_scorer: Callable[[str], float] | None = None,
        threshold: float = 0.5,
        device: str = "auto",
        offline: bool = False,
        cache_dir: str | None = None,
        max_new_tokens: int = 128,
        model_family: str | None = None,
        model_base: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.model_family = normalize_model_family(model_name, model_family)
        self.model_base = determine_shieldlm_model_base(model_name, model_base) if self.model_family == MODEL_FAMILY_SHIELDLM else None
        self.centroid_scorer = centroid_scorer
        self.threshold = threshold
        self.init_error: str | None = None
        if runner is not None:
            self.runner = runner
        else:
            try:
                self.runner = HuggingFacePrivacyRunner(
                    model_name,
                    device=device,
                    offline=offline,
                    cache_dir=cache_dir,
                    max_new_tokens=max_new_tokens,
                    trust_remote_code=trust_remote_code,
                    model_family=self.model_family,
                    model_base=self.model_base,
                )
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
            prompt = build_chinese_privacy_prompt(text, self.model_family, self.model_base)
            raw_response = self.runner.generate(prompt)
            private, model_score, model_entities = parse_model_privacy_response(raw_response, self.model_family)
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


def build_chinese_privacy_prompt(text: str, model_family: str, model_base: str | None = None) -> str:
    if model_family == MODEL_FAMILY_SHIELDLM:
        return shieldlm_privacy_prompt(text, model_base or "chatglm")
    return qwen3guard_privacy_prompt(text)


def _parameter_count(model: Any) -> int | None:
    try:
        return int(sum(param.numel() for param in model.parameters()))
    except Exception:
        return None


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
