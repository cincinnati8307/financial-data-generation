from __future__ import annotations

import re
import time
from typing import Any, Protocol

from .base import ModelUnavailable
from .dataset import carrier_id, map_anchor_label, payload_format, row_id, row_language
from .schemas import NON_SENSITIVE, SENSITIVE, STATUS_SUCCESS, TASK_COARSE_ALIGNMENT, TASK_SENSITIVITY, BenchmarkPrediction

PRIVATE_FINANCIAL_ENTITY_LABELS = {
    "ACCOUNT_NUMBER",
    "BANK_ACCOUNT",
    "CARD_NUMBER",
    "CREDIT_CARD_NUMBER",
    "FINANCIAL_ACCOUNT_NUMBER",
    "FINANCIAL_ID",
    "PRIVATE_FINANCE",
    "PRIVATE_FINANCIAL_INFORMATION",
    "SALARY",
    "TAX_FINANCE",
    "LOAN_DEBT",
    "INVESTMENT_ACCOUNT",
    "WALLET_PAYMENT",
}

MONEY_LABELS = {"MONEY", "AMOUNT", "CURRENCY"}

DEFAULT_OPENAI_PRIVACY_FILTER_MODEL = "openai/privacy-filter"
OPENAI_PRIVACY_FILTER_LABEL_MAPPING = {
    "account_number": "ACCOUNT_NUMBER",
    "private_address": "PRIVATE_ADDRESS",
    "private_email": "PRIVATE_EMAIL",
    "private_person": "PRIVATE_PERSON",
    "private_phone": "PRIVATE_PHONE",
    "private_url": "PRIVATE_URL",
    "private_date": "PRIVATE_DATE",
    "secret": "SECRET",
}
DEFAULT_OPENAI_PRIVACY_FILTER_FINANCIAL_LABELS = {"ACCOUNT_NUMBER"}


class PiiDetector(Protocol):
    name: str

    def detect(self, text: str, language: str | None = None) -> list[dict[str, Any]]:
        ...


def normalize_entity(text: str, label: str, score: float | None = None, start: int | None = None, end: int | None = None) -> dict[str, Any]:
    return {
        "text": text,
        "label": label.upper(),
        "score": score,
        "start": start,
        "end": end,
    }


def is_private_financial_entity(entity: dict[str, Any]) -> bool:
    label = str(entity.get("label") or entity.get("type") or "").upper()
    return label in PRIVATE_FINANCIAL_ENTITY_LABELS


def private_financial_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entity for entity in entities if is_private_financial_entity(entity)]


def is_openai_privacy_filter_financial_entity(
    entity: dict[str, Any],
    financial_labels: set[str] | frozenset[str] | None = None,
) -> bool:
    labels = financial_labels or DEFAULT_OPENAI_PRIVACY_FILTER_FINANCIAL_LABELS
    label = str(entity.get("label") or entity.get("type") or "").upper()
    return label in labels


def openai_privacy_filter_financial_entities(
    entities: list[dict[str, Any]],
    financial_labels: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    return [entity for entity in entities if is_openai_privacy_filter_financial_entity(entity, financial_labels=financial_labels)]


def openai_privacy_filter_entities_to_sensitivity(
    entities: list[dict[str, Any]],
    financial_labels: set[str] | frozenset[str] | None = None,
) -> tuple[str, float | None]:
    private = openai_privacy_filter_financial_entities(entities, financial_labels=financial_labels)
    if not private:
        return NON_SENSITIVE, 0.0
    scores = [entity.get("score") for entity in private if entity.get("score") is not None]
    return SENSITIVE, float(max(scores)) if scores else 1.0


def pii_entities_to_sensitivity(entities: list[dict[str, Any]]) -> tuple[str, float | None]:
    private = private_financial_entities(entities)
    if not private:
        return NON_SENSITIVE, 0.0
    scores = [entity.get("score") for entity in private if entity.get("score") is not None]
    return SENSITIVE, float(max(scores)) if scores else 1.0


def containing_sentence(text: str, start: int | None, end: int | None) -> str:
    if start is None or end is None or start < 0 or end < start:
        return text
    left = max(text.rfind("。", 0, start), text.rfind("\n", 0, start), text.rfind(".", 0, start), text.rfind(";", 0, start))
    right_candidates = [pos for pos in [text.find("。", end), text.find("\n", end), text.find(".", end), text.find(";", end)] if pos != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right + 1].strip() or text


class RegexPiiDetector:
    name = "regex"

    money_pattern = re.compile(r"(?:SGD|S\$|RMB|CNY|人民币|新币|¥)\s*[0-9][0-9,]*(?:\.[0-9]+)?|[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:SGD|RMB|CNY)", re.I)
    masked_pattern = re.compile(r"尾号\s*\d{4}|(?:card|account)\s+ending\s*\d{4}|(?:账号|账户|卡尾号|银行卡尾号)\s*\*{0,4}\s*\d{4}|\*{2,}\d{4}", re.I)
    full_card_pattern = re.compile(r"\b(?:\d[ -]?){12,19}\b")
    private_terms = re.compile(
        r"我的|个人|账户|余额|bank balance|available balance|工资|薪资|salary|payroll|bonus|扣款|账单|card payment|minimum payment|房贷|贷款|mortgage|loan|发票|收据|报销|invoice|receipt|股票账户|投资账户|portfolio|brokerage|fund|个税|报税|tax|支付宝|微信支付|PayNow|GrabPay|DBS|OCBC|UOB|浦发银行|招商银行|工商银行|建设银行",
        re.I,
    )
    public_finance_terms = re.compile(r"公开|新闻|article|说明文|定义|教程|政策|inflation|通胀|央行|central bank|company revenue|公司营收|股价|stock price", re.I)

    def detect(self, text: str, language: str | None = None) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        for match in self.money_pattern.finditer(text):
            entities.append(normalize_entity(match.group(0), "MONEY", 0.80, match.start(), match.end()))
        for match in self.masked_pattern.finditer(text):
            window = text[max(0, match.start() - 30) : min(len(text), match.end() + 30)]
            label = "CARD_NUMBER" if re.search(r"card|卡", window, re.I) else "ACCOUNT_NUMBER"
            entities.append(normalize_entity(match.group(0), label, 0.93, match.start(), match.end()))
        for match in self.full_card_pattern.finditer(text):
            compact = re.sub(r"\D", "", match.group(0))
            if 12 <= len(compact) <= 19:
                entities.append(normalize_entity(match.group(0), "CREDIT_CARD_NUMBER", 0.98, match.start(), match.end()))

        has_money = bool(self.money_pattern.search(text))
        has_private_context = bool(self.private_terms.search(text))
        if has_money and has_private_context and not self._looks_public_finance_only(text):
            span = self._best_private_span(text)
            if span:
                start, end = span
                entities.append(normalize_entity(text[start:end], "PRIVATE_FINANCE", 0.74, start, end))
        return _dedupe_entities(entities)

    def _looks_public_finance_only(self, text: str) -> bool:
        if not self.public_finance_terms.search(text):
            return False
        if re.search(r"我的|个人|尾号|\*{2,}|salary|工资|余额|账户", text, re.I):
            return False
        return True

    def _best_private_span(self, text: str) -> tuple[int, int] | None:
        money = self.money_pattern.search(text)
        context = self.private_terms.search(text)
        if not money or not context:
            return None
        start = max(0, min(money.start(), context.start()) - 24)
        end = min(len(text), max(money.end(), context.end()) + 24)
        sentence = containing_sentence(text, start, end)
        sentence_start = text.find(sentence)
        if sentence_start >= 0:
            return sentence_start, sentence_start + len(sentence)
        return start, end


def _clean_openai_privacy_filter_label(label: str) -> str:
    normalized = label.strip()
    if "-" in normalized and normalized[:2].upper() in {"B-", "I-"}:
        normalized = normalized[2:]
    normalized = normalized.lower().replace(" ", "_").replace("-", "_")
    return OPENAI_PRIVACY_FILTER_LABEL_MAPPING.get(normalized, normalized.upper())


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def normalize_openai_privacy_filter_entity(item: dict[str, Any], source_text: str | None = None) -> dict[str, Any]:
    raw_label = str(item.get("entity_group") or item.get("entity") or item.get("label") or item.get("type") or "")
    label = _clean_openai_privacy_filter_label(raw_label)
    start = _coerce_int(item.get("start"))
    end = _coerce_int(item.get("end"))
    raw_text = str(item.get("word") or item.get("text") or "")

    if source_text is not None and start is not None and end is not None and 0 <= start <= end <= len(source_text):
        span_text = source_text[start:end]
    else:
        span_text = raw_text

    leading_ws = len(span_text) - len(span_text.lstrip())
    if leading_ws:
        span_text = span_text[leading_ws:]
        if start is not None:
            start += leading_ws

    if not span_text and raw_text:
        span_text = raw_text.lstrip()

    return normalize_entity(span_text, label, _coerce_float(item.get("score")), start, end)


def normalize_openai_privacy_filter_outputs(
    items: list[dict[str, Any]],
    source_text: str | None = None,
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    entities = []
    for item in items:
        entity = normalize_openai_privacy_filter_entity(item, source_text=source_text)
        score = entity.get("score")
        if score is not None and float(score) < score_threshold:
            continue
        entities.append(entity)
    return _dedupe_entities(entities)


def _pipeline_device(device: str) -> int:
    selected = (device or "auto").lower()
    if selected == "cpu":
        return -1
    if selected.startswith("cuda"):
        if ":" in selected:
            try:
                return int(selected.split(":", 1)[1])
            except Exception:
                return 0
        return 0
    if selected == "auto":
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1
    try:
        return int(selected)
    except Exception:
        return -1


class OpenAIPrivacyFilterDetector:
    name = "openai_privacy_filter"

    def __init__(
        self,
        model_id: str = DEFAULT_OPENAI_PRIVACY_FILTER_MODEL,
        device: str = "auto",
        cache_dir: str | None = None,
        offline: bool = False,
        score_threshold: float = 0.5,
    ) -> None:
        started = time.perf_counter()
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        self.offline = offline
        self.score_threshold = score_threshold
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
        except Exception as exc:
            raise ModelUnavailable(f"transformers unavailable for OpenAI Privacy Filter: {exc}") from exc

        kwargs: dict[str, Any] = {"local_files_only": offline}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
            self.model = AutoModelForTokenClassification.from_pretrained(model_id, **kwargs)
            self.pipeline = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=_pipeline_device(device),
            )
        except Exception as exc:
            mode = "local/offline" if offline else "Hugging Face"
            raise ModelUnavailable(f"failed to load OpenAI Privacy Filter model {model_id} from {mode}: {exc}") from exc
        self.loading_time_s = time.perf_counter() - started
        self.parameter_count = _parameter_count(getattr(self, "model", None))
        self.artifact_storage_size_mb = None

    def detect(self, text: str, language: str | None = None) -> list[dict[str, Any]]:
        del language
        try:
            raw = self.pipeline(text)
        except Exception as exc:
            raise ModelUnavailable(f"OpenAI Privacy Filter inference failed: {exc}") from exc
        return normalize_openai_privacy_filter_outputs(list(raw), source_text=text, score_threshold=self.score_threshold)


class PresidioPiiDetector:
    name = "presidio"

    def __init__(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
        except Exception as exc:
            raise ModelUnavailable(f"presidio backend unavailable: {exc}") from exc
        self.analyzer = AnalyzerEngine()

    def detect(self, text: str, language: str | None = None) -> list[dict[str, Any]]:
        lang = "en" if not language or language.startswith("zh") else language[:2]
        try:
            results = self.analyzer.analyze(text=text, language=lang)
        except Exception:
            results = self.analyzer.analyze(text=text, language="en")
        entities = []
        for result in results:
            label = _map_external_label(getattr(result, "entity_type", ""))
            start = int(getattr(result, "start", 0))
            end = int(getattr(result, "end", 0))
            entities.append(normalize_entity(text[start:end], label, float(getattr(result, "score", 0.0)), start, end))
        return entities


class GlinerPiiDetector:
    name = "gliner"

    def __init__(self, model_id: str, offline: bool = False) -> None:
        if offline and not model_id.startswith("/"):
            raise ModelUnavailable("offline mode requires --pii-model to be a local GLiNER path")
        try:
            from gliner import GLiNER
        except Exception as exc:
            raise ModelUnavailable(f"gliner backend unavailable: {exc}") from exc
        try:
            self.model = GLiNER.from_pretrained(model_id)
        except Exception as exc:
            raise ModelUnavailable(f"failed to load GLiNER model {model_id}: {exc}") from exc
        self.labels = [
            "account number",
            "credit card number",
            "private financial information",
            "salary",
            "tax record",
            "loan debt",
            "investment account",
            "wallet payment",
        ]

    def detect(self, text: str, language: str | None = None) -> list[dict[str, Any]]:
        results = self.model.predict_entities(text, self.labels, threshold=0.35)
        entities = []
        for item in results:
            start = int(item.get("start", 0))
            end = int(item.get("end", 0))
            label = _map_external_label(str(item.get("label") or item.get("type") or ""))
            entities.append(normalize_entity(text[start:end], label, float(item.get("score", 0.0)), start, end))
        return entities


def _map_external_label(label: str) -> str:
    upper = label.upper().replace(" ", "_").replace("-", "_")
    mapping = {
        "CREDIT_CARD": "CREDIT_CARD_NUMBER",
        "CREDIT_DEBIT_NUMBER": "CREDIT_CARD_NUMBER",
        "BANK_ACCOUNT": "BANK_ACCOUNT",
        "FINANCIAL_ACCOUNT_NUMBER": "FINANCIAL_ACCOUNT_NUMBER",
        "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
        "PRIVATE_FINANCIAL_INFORMATION": "PRIVATE_FINANCE",
        "PRIVATE_FINANCE": "PRIVATE_FINANCE",
        "TAX_RECORD": "TAX_FINANCE",
        "LOAN_DEBT": "LOAN_DEBT",
        "INVESTMENT_ACCOUNT": "INVESTMENT_ACCOUNT",
        "WALLET_PAYMENT": "WALLET_PAYMENT",
    }
    return mapping.get(upper, upper)


def _dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, str]] = set()
    out = []
    for entity in entities:
        key = (entity.get("start"), entity.get("end"), str(entity.get("label")))
        if key not in seen:
            seen.add(key)
            out.append(entity)
    return out


def _parameter_count(model: Any) -> int | None:
    if model is None:
        return None
    try:
        return int(sum(param.numel() for param in model.parameters()))
    except Exception:
        return None


def make_pii_detector(
    backend: str = "regex",
    model_id: str | None = None,
    offline: bool = False,
    device: str = "auto",
    cache_dir: str | None = None,
    score_threshold: float | None = None,
) -> PiiDetector:
    backend = (backend or "regex").lower()
    if backend == "regex":
        return RegexPiiDetector()
    if backend == "presidio":
        return PresidioPiiDetector()
    if backend == "gliner":
        return GlinerPiiDetector(model_id or "urchade/gliner_multi_pii-v1", offline=offline)
    if backend == "openai_privacy_filter":
        return OpenAIPrivacyFilterDetector(
            model_id or DEFAULT_OPENAI_PRIVACY_FILTER_MODEL,
            device=device,
            cache_dir=cache_dir,
            offline=offline,
            score_threshold=0.5 if score_threshold is None else score_threshold,
        )
    if backend == "openmed_privacy_filter":
        raise ModelUnavailable("openmed_privacy_filter backend is not installed in this repository; select regex, presidio, gliner, or openai_privacy_filter")
    raise ModelUnavailable(f"unknown PII backend: {backend}")


class PiiOnlyModel:
    name = "pii_only"

    def __init__(
        self,
        backend: str = "regex",
        model_id: str | None = None,
        offline: bool = False,
        detector: PiiDetector | None = None,
        device: str = "auto",
        cache_dir: str | None = None,
        score_threshold: float | None = None,
    ) -> None:
        started = time.perf_counter()
        self.backend = backend
        self.model_id = model_id
        self.offline = offline
        self.device = device
        self.cache_dir = cache_dir
        self.score_threshold = score_threshold
        self.init_error: str | None = None
        try:
            self.detector = detector or make_pii_detector(backend, model_id=model_id, offline=offline, device=device, cache_dir=cache_dir, score_threshold=score_threshold)
        except ModelUnavailable as exc:
            self.detector = None
            self.init_error = str(exc)
        self.loading_time_s = time.perf_counter() - started
        self.parameter_count = _parameter_count(getattr(self.detector, "model", None)) if self.detector is not None else None
        self.artifact_storage_size_mb = getattr(self.detector, "artifact_storage_size_mb", None) if self.detector is not None else None

    def predict_sensitivity(self, rows: list[dict[str, Any]]) -> list[BenchmarkPrediction]:
        if self.detector is None:
            return [BenchmarkPrediction.skipped(row_id(row), self.name, TASK_SENSITIVITY, self.init_error or "PII detector unavailable", row=row) for row in rows]
        predictions: list[BenchmarkPrediction] = []
        for row in rows:
            started = time.perf_counter()
            entities = self.detector.detect(str(row.get("text") or ""), language=row_language(row))
            label, score = pii_entities_to_sensitivity(entities)
            predictions.append(
                BenchmarkPrediction(
                    sample_id=row_id(row),
                    model_name=self.name,
                    task=TASK_SENSITIVITY,
                    predicted_label=label,
                    sensitivity_score=score,
                    alignment_score=None,
                    detected_entities=entities,
                    status=STATUS_SUCCESS,
                    error=None,
                    metadata={
                        "pii_backend": self.backend,
                        "pii_model": self.model_id,
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
        return [BenchmarkPrediction.unsupported(row_id(row), self.name, TASK_COARSE_ALIGNMENT, "pure PII detection does not evaluate query alignment", row=row) for row in rows]
