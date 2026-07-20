from sensitive_egress_poc.chinese_privacy import ChinesePrivacyDetector, detect_chinese_language, parse_privacy_response
from sensitive_egress_poc.benchmark.chinese_privacy_adapter import ChinesePrivacyBenchmarkModel
from sensitive_egress_poc.benchmark.schemas import NON_SENSITIVE, SENSITIVE, STATUS_SUCCESS


class FakeRunner:
    model_name = "fake-guard"

    def generate(self, prompt: str) -> str:
        assert "隐私泄露检测器" in prompt
        return '{"private": true, "score": 0.91, "entities": [{"text": "账户尾号1234", "type": "account"}]}'


def test_chinese_detector_ensembles_model_and_rules():
    detector = ChinesePrivacyDetector(runner=FakeRunner())
    result = detector.detect_privacy("请汇总我的账户尾号1234和余额人民币800元")
    assert result.language == "zh"
    assert result.is_private is True
    assert result.score == 0.91
    assert result.model_score == 0.91
    assert result.rule_score == 0.8
    assert len(result.entities) >= 2


def test_non_chinese_text_does_not_call_chinese_model():
    detector = ChinesePrivacyDetector(runner=FakeRunner())
    result = detector.detect_privacy("Public market news only.")
    assert result.language == "non_zh"
    assert result.is_private is False
    assert result.model_score is None


def test_response_parser_and_language_detection():
    assert detect_chinese_language("繁體中文與 English") is True
    private, score, entities = parse_privacy_response("```json\n{'private': '是', 'score': '0.7', 'entities': []}\n```")
    assert (private, score, entities) == (True, 0.7, [])


def test_benchmark_adapter_uses_chinese_detector():
    model = ChinesePrivacyBenchmarkModel(variant="qwen3guard", runner=FakeRunner())
    prediction = model.predict_sensitivity([{"id": "zh-1", "text": "我的账户尾号1234", "label": "financial_private"}])[0]
    assert prediction.status == STATUS_SUCCESS
    assert prediction.predicted_label == SENSITIVE
    assert prediction.ground_truth == SENSITIVE
    assert model.predict_alignment([{"id": "zh-1", "text": "我的账户尾号1234"}])[0].predicted_label is None
