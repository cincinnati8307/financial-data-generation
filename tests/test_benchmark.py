import json
import math
import sys
from collections import UserDict
from types import SimpleNamespace

import pytest

from sensitive_egress_poc.benchmark.audit import audit_row
from sensitive_egress_poc.benchmark.base import ModelUnavailable
from sensitive_egress_poc.benchmark.capid_adapter import CapidAdapter, CapidParseError, capid_prompt, parse_capid_response
from sensitive_egress_poc.benchmark.centroid_adapter import CentroidBenchmarkModel
from sensitive_egress_poc.benchmark.dataset import (
    ALIGNED_SENSITIVE,
    MISALIGNED_SENSITIVE,
    fine_grained_alignment_rows,
    load_alignment_overrides,
    map_anchor_label,
    map_egress_decision,
)
from sensitive_egress_poc.benchmark.metrics import alignment_metrics
from sensitive_egress_poc.benchmark.granite_guardian import (
    GraniteGuardianParseError,
    HuggingFaceGraniteGuardianRunner,
    decision_probabilities_from_logits,
    parse_granite_guardian_output,
)
from sensitive_egress_poc.benchmark.opf_granite_adapter import OpenAIPrivacyFilterGraniteModel
from sensitive_egress_poc.benchmark.pii_adapter import (
    OpenAIPrivacyFilterDetector,
    PiiOnlyModel,
    RegexPiiDetector,
    is_openai_privacy_filter_financial_entity,
    normalize_entity,
    normalize_openai_privacy_filter_outputs,
    openai_privacy_filter_entities_to_sensitivity,
    pii_entities_to_sensitivity,
)
from sensitive_egress_poc.benchmark.pii_reranker import PiiRerankerModel, tune_reranker_alignment_threshold
from sensitive_egress_poc.benchmark.schemas import (
    ALIGNED_SENSITIVE as LABEL_ALIGNED,
    MISALIGNED_SENSITIVE as LABEL_MISALIGNED,
    NON_SENSITIVE,
    SENSITIVE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_UNSUPPORTED,
    TASK_COARSE_ALIGNMENT,
    BenchmarkPrediction,
)
from sensitive_egress_poc.io_utils import read_jsonl, write_json, write_jsonl
from sensitive_egress_poc.benchmark.llm_judge import render_judge_prompt


def anchor(label="financial_private", text="我的 DBS 账户余额是 SGD 1,200。", row_id="a1"):
    return {
        "id": row_id,
        "text": text,
        "label": label,
        "subtype": "bank_balance" if label == "financial_private" else "*",
        "language": "zh_en",
        "format": "natural_sentence",
        "style": "zh_casual",
        "meta": {},
    }


def egress(decision="allow", subtype="bank_balance", carrier="send_bank_balance_to_spouse", text="private high"):
    expected_financial = decision == "allow"
    return {
        "id": f"e_{decision}_{subtype}_{carrier}",
        "user_intent": "把账户余额发给家人做预算。",
        "text": text,
        "financial_evidence": text,
        "financial_subtype": subtype,
        "expected_decision": decision,
        "expected_categories": ["work", "financial_private"] if expected_financial else ["work"],
        "unexpected_categories": [] if expected_financial else ["financial_private"],
        "format": "mixed_egress",
        "meta": {"carrier_id": carrier, "expected_financial": expected_financial},
    }


def test_centroid_adapter_preserves_classifier_output(monkeypatch, tmp_path):
    centroids_path = tmp_path / "centroids.json"
    write_json(centroids_path, {"model": "hash-only", "centroids": {}, "threshold": 0.55, "margin_threshold": 0.05})
    expected_classifier_result = {"predicted_label": "financial_private", "financial_score": 0.77, "margin": 0.2}

    def fake_classify(text, centroid_obj, threshold=None, margin_threshold=None):
        assert text == "我的 DBS 账户余额是 SGD 1,200。"
        return expected_classifier_result

    monkeypatch.setattr("sensitive_egress_poc.benchmark.centroid_adapter.classify_text", fake_classify)
    model = CentroidBenchmarkModel(str(centroids_path))

    prediction = model.predict_sensitivity([anchor()])[0]

    assert prediction.status == STATUS_SUCCESS
    assert prediction.predicted_label == SENSITIVE
    assert prediction.sensitivity_score == 0.77
    assert prediction.metadata["centroid_result"] is expected_classifier_result


def test_anchor_labels_map_to_task_a():
    assert map_anchor_label(anchor("financial_private")) == SENSITIVE
    assert map_anchor_label(anchor("non_private_financial")) == NON_SENSITIVE
    assert map_anchor_label(anchor("benign")) == NON_SENSITIVE


def test_allow_maps_to_aligned_sensitive():
    assert map_egress_decision(egress("allow")) == ALIGNED_SENSITIVE


def test_request_approval_maps_to_misaligned_sensitive():
    assert map_egress_decision(egress("request_approval")) == MISALIGNED_SENSITIVE


def test_pii_entities_convert_to_message_sensitivity():
    label, score = pii_entities_to_sensitivity([normalize_entity("****9012", "ACCOUNT_NUMBER", 0.93, 12, 20)])

    assert label == SENSITIVE
    assert score == 0.93


def test_money_entity_alone_is_not_private_financial():
    entities = RegexPiiDetector().detect("公开报道：公司营收 SGD 1,200，股价上涨。")
    label, score = pii_entities_to_sensitivity(entities)

    assert any(entity["label"] == "MONEY" for entity in entities)
    assert label == NON_SENSITIVE
    assert score == 0.0


def test_openai_privacy_filter_label_normalization_and_leading_whitespace():
    text = "My account ****9012 belongs to Alice."
    entities = normalize_openai_privacy_filter_outputs(
        [
            {"entity_group": "account_number", "word": " ****9012", "start": 10, "end": 19, "score": 0.96},
            {"entity_group": "private_person", "word": "Alice", "start": 31, "end": 36, "score": 0.91},
            {"entity_group": "secret", "word": "token", "start": None, "end": None, "score": 0.80},
        ],
        source_text=text,
        score_threshold=0.5,
    )

    assert [entity["label"] for entity in entities] == ["ACCOUNT_NUMBER", "PRIVATE_PERSON", "SECRET"]
    assert entities[0]["text"] == "****9012"
    assert entities[0]["start"] == 11
    assert entities[0]["end"] == 19


def test_openai_privacy_filter_financial_policy_defaults_to_account_number_only():
    assert is_openai_privacy_filter_financial_entity(normalize_entity("****9012", "ACCOUNT_NUMBER")) is True
    assert is_openai_privacy_filter_financial_entity(normalize_entity("Alice", "PRIVATE_PERSON")) is False
    assert is_openai_privacy_filter_financial_entity(normalize_entity("2026-01-01", "PRIVATE_DATE")) is False
    assert is_openai_privacy_filter_financial_entity(normalize_entity("SGD 1,200", "MONEY")) is False


def test_openai_privacy_filter_private_person_alone_is_not_financial_sensitive():
    label, score = openai_privacy_filter_entities_to_sensitivity([normalize_entity("Alice", "PRIVATE_PERSON", 0.91, 0, 5)])

    assert label == NON_SENSITIVE
    assert score == 0.0


def test_openai_privacy_filter_currency_amount_without_account_number_is_not_sensitive():
    label, score = openai_privacy_filter_entities_to_sensitivity([normalize_entity("SGD 1,200", "MONEY", 0.99, 0, 9)])

    assert label == NON_SENSITIVE
    assert score == 0.0


class FakeOpenAIPrivacyFilterDetector:
    name = "fake-opf"
    model_id = "fake-opf"
    loading_time_s = 0.0
    parameter_count = 0
    artifact_storage_size_mb = 0.0

    def __init__(self, mode="account_when_present"):
        self.mode = mode
        self.calls = []

    def detect(self, text, language=None):
        self.calls.append((text, language))
        if self.mode == "person_only":
            start = text.index("Alice") if "Alice" in text else 0
            return [normalize_entity("Alice", "PRIVATE_PERSON", 0.92, start, start + 5)]
        if self.mode == "money_only":
            return [normalize_entity("SGD 1,200", "MONEY", 0.90, 0, 9)]
        if "****9012" in text:
            start = text.index("****9012")
            return [normalize_entity("****9012", "ACCOUNT_NUMBER", 0.96, start, start + len("****9012"))]
        return []


class FakeGraniteGuardianRunner:
    model_name = "fake-granite"
    loading_time_s = 0.0
    parameter_count = 0
    artifact_storage_size_mb = 0.0

    def __init__(self, raw_label="No"):
        self.raw_label = raw_label
        self.calls = []

    def score_context_relevance(self, user_query, evidence):
        self.calls.append((user_query, evidence))
        risk = self.raw_label == "Yes"
        return {
            "risk_detected": risk,
            "raw_label": self.raw_label,
            "risk_probability": 0.83 if risk else 0.17,
            "relevance_probability": 0.17 if risk else 0.83,
            "confidence": "High",
            "raw_output": f"{self.raw_label}<confidence>High</confidence>",
        }


def test_opf_granite_task_a_detects_account_number_only():
    model = OpenAIPrivacyFilterGraniteModel(detector=FakeOpenAIPrivacyFilterDetector(), guardian_runner=FakeGraniteGuardianRunner())
    rows = [
        anchor(text="Account ****9012 should be blocked.", row_id="a_account"),
        anchor(label="benign", text="No identifier here.", row_id="a_benign"),
    ]

    predictions = model.predict_sensitivity(rows)

    assert predictions[0].predicted_label == SENSITIVE
    assert predictions[0].sensitivity_score == 0.96
    assert predictions[0].metadata["financial_detected_labels"] == ["ACCOUNT_NUMBER"]
    assert predictions[1].predicted_label == NON_SENSITIVE


def test_opf_granite_task_a_private_person_does_not_make_sample_sensitive():
    model = OpenAIPrivacyFilterGraniteModel(detector=FakeOpenAIPrivacyFilterDetector("person_only"), guardian_runner=FakeGraniteGuardianRunner())
    prediction = model.predict_sensitivity([anchor(text="Alice mentioned a public budget article.")])[0]

    assert prediction.status == STATUS_SUCCESS
    assert prediction.predicted_label == NON_SENSITIVE
    assert prediction.metadata["native_detected_labels"] == ["PRIVATE_PERSON"]
    assert prediction.metadata["financial_detected_labels"] == []


def test_opf_granite_task_a_money_without_account_number_is_non_sensitive():
    model = OpenAIPrivacyFilterGraniteModel(detector=FakeOpenAIPrivacyFilterDetector("money_only"), guardian_runner=FakeGraniteGuardianRunner())
    prediction = model.predict_sensitivity([anchor(label="non_private_financial", text="Public revenue was SGD 1,200.")])[0]

    assert prediction.status == STATUS_SUCCESS
    assert prediction.predicted_label == NON_SENSITIVE
    assert prediction.metadata["financial_detected_labels"] == []


def test_strict_opf_granite_gate_skips_granite_when_no_financial_pii():
    detector = FakeOpenAIPrivacyFilterDetector("person_only")
    runner = FakeGraniteGuardianRunner("No")
    model = OpenAIPrivacyFilterGraniteModel(detector=detector, guardian_runner=runner)

    prediction = model.predict_alignment([egress(text="Alice wrote a public status update.")])[0]

    assert prediction.status == STATUS_SUCCESS
    assert prediction.predicted_label == NON_SENSITIVE
    assert prediction.alignment_score is None
    assert runner.calls == []
    assert prediction.metadata["granite_called"] is False
    assert prediction.metadata["pipeline_gate"] == "no_financial_pii_detected"
    assert model.granite_skipped_by_gate_count == 1


def test_strict_opf_granite_yes_maps_to_misaligned_sensitive_without_oracle_fallback():
    runner = FakeGraniteGuardianRunner("Yes")
    model = OpenAIPrivacyFilterGraniteModel(detector=FakeOpenAIPrivacyFilterDetector(), guardian_runner=runner)
    row = egress("request_approval", text="Work update. Account ****9012 has a payment due.", carrier="work_note_tail")
    row["financial_evidence"] = "ORACLE EVIDENCE MUST NOT BE USED"

    prediction = model.predict_alignment([row])[0]

    assert prediction.status == STATUS_SUCCESS
    assert prediction.predicted_label == LABEL_MISALIGNED
    assert prediction.alignment_score == 0.17
    assert runner.calls
    assert runner.calls[0][1] != row["financial_evidence"]
    assert "Account ****9012 has a payment due." in runner.calls[0][1]
    assert prediction.metadata["granite_raw_label"] == "Yes"
    assert prediction.metadata["granite_context_relevance_yes_means_irrelevant"] is True


def test_strict_opf_granite_no_maps_to_aligned_sensitive():
    runner = FakeGraniteGuardianRunner("No")
    model = OpenAIPrivacyFilterGraniteModel(detector=FakeOpenAIPrivacyFilterDetector(), guardian_runner=runner)
    row = egress("allow", text="Please include account ****9012 in the budget note.")

    prediction = model.predict_alignment([row])[0]

    assert prediction.status == STATUS_SUCCESS
    assert prediction.predicted_label == LABEL_ALIGNED
    assert prediction.alignment_score == 0.83
    assert prediction.metadata["granite_raw_label"] == "No"


def test_granite_context_relevance_mapping_is_negative_semantics():
    yes_result = {"raw_label": "Yes"}
    no_result = {"raw_label": "No"}

    assert OpenAIPrivacyFilterGraniteModel(detector=FakeOpenAIPrivacyFilterDetector(), guardian_runner=FakeGraniteGuardianRunner("Yes")).predict_alignment(
        [egress("request_approval", text="Account ****9012 should not go here.")]
    )[0].predicted_label == LABEL_MISALIGNED
    assert OpenAIPrivacyFilterGraniteModel(detector=FakeOpenAIPrivacyFilterDetector(), guardian_runner=FakeGraniteGuardianRunner("No")).predict_alignment(
        [egress("allow", text="Account ****9012 should go here.")]
    )[0].predicted_label == LABEL_ALIGNED
    assert yes_result["raw_label"] == "Yes"
    assert no_result["raw_label"] == "No"


class RaisingDetector:
    name = "raising-detector"

    def detect(self, text, language=None):
        raise AssertionError("oracle mode must not run the privacy filter detector")


def test_oracle_opf_granite_uses_financial_evidence_without_detector_gate():
    runner = FakeGraniteGuardianRunner("No")
    model = OpenAIPrivacyFilterGraniteModel(detector=RaisingDetector(), guardian_runner=runner, oracle_evidence=True)
    row = egress("allow", text="No account number in outgoing text.")
    row["financial_evidence"] = "Account ****9012 appears only in oracle evidence."

    prediction = model.predict_alignment([row])[0]

    assert prediction.model_name == "granite_guardian_oracle_evidence"
    assert prediction.status == STATUS_SUCCESS
    assert prediction.predicted_label == LABEL_ALIGNED
    assert runner.calls == [(row["user_intent"], row["financial_evidence"])]
    assert prediction.metadata["oracle_evidence"] is True
    assert prediction.metadata["evidence_source"] == "dataset_financial_evidence"
    assert prediction.metadata["end_to_end_detector"] is False


def test_oracle_opf_granite_task_a_is_unsupported():
    model = OpenAIPrivacyFilterGraniteModel(detector=RaisingDetector(), guardian_runner=FakeGraniteGuardianRunner(), oracle_evidence=True)
    prediction = model.predict_sensitivity([anchor()])[0]

    assert prediction.status == STATUS_UNSUPPORTED
    assert prediction.predicted_label is None
    assert prediction.metadata["diagnostic_only"] is True


def test_granite_output_parser_handles_decisions_confidence_whitespace_and_suffixes():
    assert parse_granite_guardian_output("Yes")["raw_label"] == "Yes"
    assert parse_granite_guardian_output("No")["risk_detected"] is False
    assert parse_granite_guardian_output("  Yes<confidence>High</confidence>")["confidence"] == "High"
    parsed = parse_granite_guardian_output("\nNo <confidence>Low</confidence> trailing text")
    assert parsed["raw_label"] == "No"
    assert parsed["confidence"] == "Low"


class MalformedGraniteRunner:
    model_name = "malformed-granite"

    def score_context_relevance(self, user_query, evidence):
        raise GraniteGuardianParseError("could not parse Granite Guardian decision token as Yes or No")


def test_malformed_granite_output_is_marked_failed():
    model = OpenAIPrivacyFilterGraniteModel(detector=FakeOpenAIPrivacyFilterDetector(), guardian_runner=MalformedGraniteRunner())
    prediction = model.predict_alignment([egress(text="Account ****9012 should be checked.")])[0]

    assert prediction.status == STATUS_FAILED
    assert prediction.predicted_label is None
    assert "could not parse" in prediction.error


def test_granite_probability_uses_single_yes_no_token_logits():
    class SingleTokenTokenizer:
        def encode(self, text, add_special_tokens=False):
            return {"Yes": [1], "No": [2]}[text]

    risk, relevance = decision_probabilities_from_logits(SingleTokenTokenizer(), [[0.0, 2.0, 0.0]])

    expected = math.exp(2.0) / (math.exp(2.0) + math.exp(0.0))
    assert risk == pytest.approx(expected)
    assert relevance == pytest.approx(1.0 - expected)


def test_granite_probability_remains_none_when_decisions_are_not_single_tokens():
    class MultiTokenTokenizer:
        def encode(self, text, add_special_tokens=False):
            return {"Yes": [1, 2], "No": [3]}[text]

    risk, relevance = decision_probabilities_from_logits(MultiTokenTokenizer(), [[0.0, 2.0, 0.0, 1.0]])
    parsed = parse_granite_guardian_output("Yes", risk_probability=risk, relevance_probability=relevance)

    assert risk is None
    assert relevance is None
    assert parsed["raw_label"] == "Yes"
    assert parsed["risk_probability"] is None
    assert parsed["relevance_probability"] is None


def test_granite_runner_handles_dict_chat_template_outputs():
    class FakeShape:
        shape = (1, 3)

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 9

        def apply_chat_template(self, messages, guardian_config, add_generation_prompt, return_tensors):
            assert messages[-1]["role"] == "user"
            assert guardian_config == {"risk_name": "context_relevance"}
            assert add_generation_prompt is True
            assert return_tensors == "pt"
            return UserDict({"input_ids": FakeShape(), "attention_mask": FakeShape()})

        def encode(self, text, add_special_tokens=False):
            return {"Yes": [1], "No": [2]}[text]

        def decode(self, tokens, skip_special_tokens=True):
            assert tokens == [2]
            return "No <confidence>Low</confidence>"

    class FakeModel:
        def __init__(self):
            self.calls = []

        def parameters(self):
            return iter(())

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            assert "input_ids" in kwargs
            assert "attention_mask" in kwargs
            return SimpleNamespace(sequences=[[7, 8, 9, 2]], scores=[[[0.0, 0.0, 2.0]]])

    runner = HuggingFaceGraniteGuardianRunner.__new__(HuggingFaceGraniteGuardianRunner)
    runner.tokenizer = FakeTokenizer()
    runner.model = FakeModel()
    runner.max_new_tokens = 20

    result = runner.score_context_relevance("send account details", "Account ****9012")

    assert result["raw_label"] == "No"
    assert result["risk_detected"] is False
    assert result["confidence"] == "Low"
    assert result["relevance_probability"] > result["risk_probability"]
    assert runner.model.calls


def test_openai_privacy_filter_offline_mode_passes_local_files_only(monkeypatch):
    calls = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("tokenizer", model_id, kwargs))
            if kwargs.get("local_files_only") is not True:
                raise AssertionError("offline loading must pass local_files_only=True")
            return cls()

    class FakeTokenClassifier:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            if kwargs.get("local_files_only") is not True:
                raise AssertionError("offline loading must pass local_files_only=True")
            return cls()

        def parameters(self):
            return []

    def fake_pipeline(*args, **kwargs):
        return lambda text: []

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizer, AutoModelForTokenClassification=FakeTokenClassifier, pipeline=fake_pipeline),
    )

    detector = OpenAIPrivacyFilterDetector(model_id="fake-opf", offline=True, device="cpu")

    assert detector.detect("hello") == []
    assert [call[0] for call in calls] == ["tokenizer", "model"]


def test_granite_guardian_offline_mode_passes_local_files_only(monkeypatch):
    calls = []

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("tokenizer", model_id, kwargs))
            if kwargs.get("local_files_only") is not True:
                raise AssertionError("offline loading must pass local_files_only=True")
            return cls()

    class FakeCausalLM:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            if kwargs.get("local_files_only") is not True:
                raise AssertionError("offline loading must pass local_files_only=True")
            return cls()

        def eval(self):
            return None

        def parameters(self):
            return []

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizer, AutoModelForCausalLM=FakeCausalLM),
    )

    runner = HuggingFaceGraniteGuardianRunner(model_id="fake-granite", offline=True)

    assert runner.model_name == "fake-granite"
    assert [call[0] for call in calls] == ["tokenizer", "model"]


def test_missing_opf_model_is_skipped_not_zero_scored(monkeypatch):
    def unavailable_detector(*args, **kwargs):
        raise ModelUnavailable("missing local OpenAI Privacy Filter files")

    monkeypatch.setattr("sensitive_egress_poc.benchmark.opf_granite_adapter.OpenAIPrivacyFilterDetector", unavailable_detector)
    model = OpenAIPrivacyFilterGraniteModel(guardian_runner=FakeGraniteGuardianRunner(), offline=True)

    prediction = model.predict_sensitivity([anchor()])[0]

    assert prediction.status == STATUS_SKIPPED
    assert prediction.predicted_label is None
    assert prediction.sensitivity_score is None
    assert "missing local OpenAI Privacy Filter files" in prediction.error


class FakeDetector:
    name = "fake"

    def detect(self, text, language=None):
        if "private" not in text:
            return []
        start = text.index("private")
        return [normalize_entity(text[start:], "PRIVATE_FINANCE", 0.9, start, len(text))]


class FakeScorer:
    model_name = "fake-scorer"

    def __init__(self):
        self.calls = []

    def score_pairs(self, pairs):
        self.calls.extend(pairs)
        scores = []
        for _, evidence in pairs:
            if "high" in evidence:
                scores.append(0.9)
            elif "validation" in evidence:
                scores.append(0.4)
            else:
                scores.append(0.1)
        return scores


def test_reranker_thresholds_are_tuned_only_on_egress_training_data():
    train = [egress("allow", text="private high"), egress("request_approval", carrier="work_note_tail", text="private low")]
    validation = [egress("allow", text="private validation")]
    scorer = FakeScorer()

    model = PiiRerankerModel(egress_train_rows=train, detector=FakeDetector(), scorer=scorer)

    assert model.threshold_metadata["threshold_source"] == "egress_train"
    assert model.threshold_metadata["train_sample_count"] == 2
    assert all("validation" not in evidence for _, evidence in scorer.calls)

    model.predict_alignment(validation)
    assert any("validation" in evidence for _, evidence in scorer.calls)


def test_capid_json_parsing_handles_valid_fenced_python_and_malformed():
    valid = parse_capid_response('{"entities": [{"text": "x", "type": "finance", "relevance": 0, "score": null}]}')
    fenced = parse_capid_response('```json\n{"entities": [{"text": "x", "type": "finance", "relevance": 1}]}\n```')
    python_style = parse_capid_response("Here: {'entities': [{'text': 'x', 'type': 'finance', 'relevance': 0}]}")

    assert valid["entities"][0]["relevance"] == 0
    assert fenced["entities"][0]["relevance"] == 1
    assert python_style["entities"][0]["type"] == "finance"
    with pytest.raises(CapidParseError):
        parse_capid_response("not parseable")


def test_capid_parser_handles_span_mapping_output():
    parsed = parse_capid_response('{"My salary is SGD 7,820": {"type": "finance", "relevance": 0}}')

    assert parsed == {"entities": [{"text": "My salary is SGD 7,820", "type": "finance", "relevance": 0, "score": None}]}


def test_capid_prompt_uses_model_card_alpaca_format():
    prompt = capid_prompt("Can you help with a budget?", "My salary is SGD 7,820")

    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "Text: My salary is SGD 7,820" in prompt
    assert "Question: Can you help with a budget?" in prompt


def test_capid_adapter_keeps_lora_runtime_config_with_mock_runner():
    class GoodRunner:
        model_name = "mock-capid"

        def generate(self, question, text):
            return '{"entities": [{"text": "salary", "type": "finance", "relevance": 1}]}'

    model = CapidAdapter(
        model_id="adapter",
        base_model_id="base",
        runner=GoodRunner(),
        max_new_tokens=32,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    prediction = model.predict_alignment([egress()])[0]

    assert prediction.status == STATUS_SUCCESS
    assert prediction.predicted_label == LABEL_ALIGNED
    assert prediction.metadata["capid_base_model"] == "base"
    assert prediction.metadata["capid_load_in_4bit"] is True
    assert prediction.metadata["capid_max_new_tokens"] == 32


def test_llm_judge_prompt_renders_literal_json_schema():
    prompt = render_judge_prompt("send it", "message", "evidence")

    assert '"label": "..."' in prompt
    assert "send it" in prompt
    assert "message" in prompt
    assert "evidence" in prompt


def test_unconfigured_capid_model_is_skipped_without_download():
    prediction = CapidAdapter(model_id=None, offline=True).predict_sensitivity([anchor()])[0]

    assert prediction.status == STATUS_SKIPPED
    assert "CAPID model is not configured" in prediction.error


def test_failed_and_unavailable_baselines_are_marked_correctly():
    unavailable = PiiOnlyModel(backend="gliner", offline=True)
    skipped = unavailable.predict_sensitivity([anchor()])[0]
    assert skipped.status == STATUS_SKIPPED
    assert "offline mode" in skipped.error

    class BadRunner:
        model_name = "bad"

        def generate(self, question, text):
            return "not json"

    failed = CapidAdapter(runner=BadRunner()).predict_alignment([egress()])[0]
    assert failed.status == STATUS_FAILED
    assert failed.predicted_label is None


def pred(truth, guessed):
    return BenchmarkPrediction(
        sample_id=f"{truth}_{guessed}",
        model_name="m",
        task=TASK_COARSE_ALIGNMENT,
        predicted_label=guessed,
        sensitivity_score=None,
        alignment_score=None,
        detected_entities=[],
        status=STATUS_SUCCESS,
        error=None,
        metadata={},
        ground_truth=truth,
    )


def test_leakage_rate_is_calculated_correctly():
    metrics = alignment_metrics(
        [
            pred(LABEL_MISALIGNED, LABEL_ALIGNED),
            pred(LABEL_MISALIGNED, NON_SENSITIVE),
            pred(LABEL_MISALIGNED, LABEL_MISALIGNED),
            pred(LABEL_ALIGNED, LABEL_ALIGNED),
        ]
    )["m"]

    assert metrics["leakage_rate"] == pytest.approx(2 / 3)


def test_false_block_rate_is_calculated_correctly():
    metrics = alignment_metrics(
        [
            pred(LABEL_ALIGNED, LABEL_MISALIGNED),
            pred(LABEL_ALIGNED, LABEL_ALIGNED),
            pred(LABEL_MISALIGNED, LABEL_MISALIGNED),
        ]
    )["m"]

    assert metrics["false_block_rate"] == pytest.approx(1 / 2)


def test_manual_alignment_overrides_do_not_mutate_original_jsonl_files(tmp_path):
    data_path = tmp_path / "egress.jsonl"
    override_path = tmp_path / "alignment_overrides.jsonl"
    original = [egress("allow", subtype="investment", carrier="send_tax_note")]
    write_jsonl(data_path, original)
    write_jsonl(
        override_path,
        [
            {
                "sample_id": original[0]["id"],
                "semantic_alignment_label": "misaligned_sensitive",
                "annotator": "manual",
                "reason": "investment payload is not a tax note",
            }
        ],
    )

    rows = read_jsonl(data_path)
    selected = fine_grained_alignment_rows(rows, load_alignment_overrides(override_path))

    assert selected[0]["_benchmark_ground_truth"] == "misaligned_sensitive"
    assert "_benchmark_ground_truth" not in rows[0]
    assert "_benchmark_ground_truth" not in read_jsonl(data_path)[0]


def test_alignment_audit_flags_suspicious_carrier_subtype_combinations():
    result = audit_row(egress("allow", subtype="investment", carrier="send_tax_note"))

    assert result["audit_status"] == "possibly_misaligned"
    assert "observed investment" in result["audit_reason"]


def test_offline_mode_skips_remote_reranker_without_attempting_model_loads():
    model = PiiRerankerModel(egress_train_rows=[], detector=FakeDetector(), reranker_model="BAAI/bge-reranker-v2-m3", offline=True)
    prediction = model.predict_alignment([egress()])[0]

    assert model.reranker_error.startswith("offline mode requires")
    assert prediction.status == STATUS_SKIPPED
