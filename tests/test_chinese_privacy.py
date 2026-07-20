from sensitive_egress_poc.chinese_privacy import (
    ChinesePrivacyDetector,
    DEFAULT_SHIELDLM_MODEL,
    detect_chinese_language,
    parse_privacy_response,
    parse_qwen3guard_response,
    parse_shieldlm_response,
)
from sensitive_egress_poc.benchmark.chinese_privacy_adapter import ChinesePrivacyBenchmarkModel
from sensitive_egress_poc.benchmark.schemas import NON_SENSITIVE, SENSITIVE, STATUS_SUCCESS


class FakeRunner:
    model_name = "fake-guard"

    def generate(self, prompt: str) -> str:
        assert "隐私泄露检测器" in prompt
        return '{"private": true, "score": 0.91, "entities": [{"text": "账户尾号1234", "type": "account"}]}'


class StaticRunner:
    model_name = "static-guard"

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


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


def test_qwen3guard_native_response_parser():
    private, score, entities = parse_qwen3guard_response("Safety: Unsafe\nCategories: PII")
    assert (private, score, entities) == (True, 0.95, [])
    private, score, entities = parse_qwen3guard_response("Safety: Safe\nCategories: None")
    assert (private, score, entities) == (False, 0.05, [])


def test_qwen3guard_detector_accepts_native_safety_output():
    runner = StaticRunner("Safety: Unsafe\nCategories: PII")
    detector = ChinesePrivacyDetector(runner=runner, model_family="qwen3guard")
    result = detector.detect_privacy("我的银行卡尾号7788需要核对")
    assert result.is_private is True
    assert result.model_score == 0.95
    assert "Qwen3Guard" in runner.prompts[0]


def test_shieldlm_response_parser():
    assert parse_shieldlm_response("[答案] 不安全\n[分析] 包含私人金融信息。") == (True, 0.95, [])
    assert parse_shieldlm_response("[答案] 安全\n[分析] 公开新闻。") == (False, 0.05, [])
    assert parse_shieldlm_response("[答案] 有争议\n[分析] 需要上下文。") == (True, 0.6, [])


def test_shieldlm_chatglm_prompt_and_unsafe_detection():
    runner = StaticRunner("[答案] 不安全\n[分析] 包含私人金融信息。")
    detector = ChinesePrivacyDetector(
        model_name=DEFAULT_SHIELDLM_MODEL,
        model_family="shieldlm",
        model_base="chatglm",
        runner=runner,
    )
    result = detector.detect_privacy("普通中文内容")
    assert result.is_private is True
    assert result.model_score == 0.95
    assert runner.prompts[0].startswith("[gMASK]sop<|user|>")
    assert "必须严格遵循的规则" in runner.prompts[0]


def test_shieldlm_safe_output_is_non_private_without_rule_hit():
    runner = StaticRunner("[答案] 安全\n[分析] 只是公开市场新闻。")
    detector = ChinesePrivacyDetector(
        model_name=DEFAULT_SHIELDLM_MODEL,
        model_family="shieldlm",
        model_base="chatglm",
        runner=runner,
    )
    result = detector.detect_privacy("今天讨论公开市场新闻")
    assert result.is_private is False
    assert result.score == 0.05
    assert result.rule_score == 0.0


def test_benchmark_adapter_uses_chinese_detector():
    model = ChinesePrivacyBenchmarkModel(variant="qwen3guard", runner=FakeRunner())
    prediction = model.predict_sensitivity([{"id": "zh-1", "text": "我的账户尾号1234", "label": "financial_private"}])[0]
    assert prediction.status == STATUS_SUCCESS
    assert prediction.predicted_label == SENSITIVE
    assert prediction.ground_truth == SENSITIVE
    assert prediction.metadata["model_family"] == "qwen3guard"
    assert model.predict_alignment([{"id": "zh-1", "text": "我的账户尾号1234"}])[0].predicted_label is None


def test_benchmark_adapter_passes_shieldlm_model_base():
    runner = StaticRunner("[答案] 安全\n[分析] 只是公开市场新闻。")
    model = ChinesePrivacyBenchmarkModel(variant="shieldlm", model_base="chatglm", runner=runner)
    prediction = model.predict_sensitivity([{"id": "zh-2", "text": "今天讨论公开市场新闻", "label": "benign"}])[0]
    assert prediction.status == STATUS_SUCCESS
    assert prediction.predicted_label == NON_SENSITIVE
    assert prediction.metadata["model_family"] == "shieldlm"
    assert prediction.metadata["model_base"] == "chatglm"
