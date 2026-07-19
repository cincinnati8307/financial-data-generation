import json

import pytest

from sensitive_egress_poc.benchmark.audit import audit_row
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
from sensitive_egress_poc.benchmark.pii_adapter import PiiOnlyModel, RegexPiiDetector, normalize_entity, pii_entities_to_sensitivity
from sensitive_egress_poc.benchmark.pii_reranker import PiiRerankerModel, tune_reranker_alignment_threshold
from sensitive_egress_poc.benchmark.schemas import (
    ALIGNED_SENSITIVE as LABEL_ALIGNED,
    MISALIGNED_SENSITIVE as LABEL_MISALIGNED,
    NON_SENSITIVE,
    SENSITIVE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
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
