import json

from sensitive_egress_poc.filters import contains_disallowed_secret_like_content, looks_like_full_card_or_account_number, validate_synthetic_example, validate_augmented_row
from sensitive_egress_poc.llm_augmenter import DryRunProvider, LLMProvider, augment_rows
from sensitive_egress_poc.cli_generate import split_by_skeleton
from sensitive_egress_poc.schemas import PRIVATE_SUBTYPES
from sensitive_egress_poc.synthetic_generator import SyntheticFinancialGenerator, load_template_catalog


class StaticProvider(LLMProvider):
    name = "static"

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.payload


def test_generator_creates_valid_private_examples_without_full_numbers():
    gen = SyntheticFinancialGenerator(seed=1)
    rows = gen.generate_private(30)
    assert rows
    for row in rows:
        ok, reason = validate_synthetic_example(row)
        assert ok, reason
        assert row["label"] == "financial_private"
        assert not looks_like_full_card_or_account_number(row["text"])


def test_hard_negatives_non_private_labels():
    gen = SyntheticFinancialGenerator(seed=2)
    rows = gen.generate_hard_negatives(30)
    assert {r["label"] for r in rows} <= {"non_private_financial", "benign"}


def test_dry_run_augmentation_produces_valid_llm_rows():
    gen = SyntheticFinancialGenerator(seed=3)
    source = gen.generate_private(1)[0]
    rows, reasons = augment_rows([source], DryRunProvider(), 1, 6, False, "dry-run")
    assert rows
    assert reasons["duplicate"] == 0
    for row in rows:
        ok, reason = validate_augmented_row(row, source)
        assert ok, reason
        assert row["source"] == "llm_paraphrase"
        assert row["parent_id"] == source["id"]


def test_openai_style_candidates_are_wrapped_locally():
    gen = SyntheticFinancialGenerator(seed=4)
    source = gen.generate_private(1)[0]
    payload = json.dumps(
        [
            {"text": f"Hi，整理一下这条 private finance note：{source['text']}", "style": "email_mixed"},
            {"text": f"JSON-like personal finance memo: {source['text']}", "style": "json"},
        ],
        ensure_ascii=False,
    )
    provider = StaticProvider(payload)

    rows, reasons = augment_rows([source], provider, 1, 2, True, "fake-model")

    assert "Do not return dataset rows" in provider.prompt
    assert reasons["duplicate"] == 0
    assert rows[0] == source
    assert len(rows) == 3
    for row in rows[1:]:
        ok, reason = validate_augmented_row(row, source)
        assert ok, reason
        assert row["label"] == "financial_private"
        assert row["subtype"] == source["subtype"]
        assert row["parent_id"] == source["id"]
        assert row["source"] == "llm_paraphrase"
        assert row["meta"]["augmentation_provider"] == "static"
        assert row["meta"]["augmentation_model"] == "fake-model"
    assert rows[2]["style"] == "json_note"
    assert rows[2]["format"] == "json"


def test_fenced_jsonl_candidates_report_rejection_reasons():
    gen = SyntheticFinancialGenerator(seed=5)
    source = gen.generate_private(1)[0]
    valid_text = f"改写后的个人财务备注：{source['text']}"
    lines = [
        {"text": source["text"], "style": "zh_casual"},
        {"style": "zh_formal"},
        {"text": "未知风格文本", "style": "not_a_style"},
        {"text": "password: abcdefghijklmnopqrstuvwxyz", "style": "zh_formal"},
        {"text": valid_text, "style": "zh_casual"},
        {"text": valid_text, "style": "zh_formal"},
    ]
    payload = "```jsonl\n" + "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n```"

    rows, reasons = augment_rows([source], StaticProvider(payload), 1, 6, False, "fake-model")

    assert [row["text"] for row in rows] == [valid_text]
    assert reasons["unchanged_text"] == 1
    assert reasons["missing_text"] == 1
    assert reasons["unknown_style"] == 1
    assert reasons["secret_like_content"] == 1
    assert reasons["duplicate"] == 1


def test_bad_provider_json_is_counted_as_parse_error():
    gen = SyntheticFinancialGenerator(seed=6)
    source = gen.generate_private(1)[0]

    rows, reasons = augment_rows([source], StaticProvider("not json"), 1, 1, False, "fake-model")

    assert rows == []
    assert reasons["parse_error"] == 1



def test_template_catalog_has_required_coverage():
    catalog = load_template_catalog()
    assert "entity_pools" in catalog
    assert set(PRIVATE_SUBTYPES) == set(catalog["private_scenarios"])
    for subtype in PRIVATE_SUBTYPES:
        assert len(catalog["private_scenarios"][subtype]) >= 8
    assert len(catalog["mixed_carriers"]) >= 30
    assert catalog["hard_negative_scenarios"]
    assert catalog["benign_scenarios"]


def test_generator_anchor_diversity_targets():
    gen = SyntheticFinancialGenerator(seed=1337)
    private_rows = gen.generate_private(1000)
    negative_rows = gen.generate_hard_negatives(500) + gen.generate_benign(500)

    assert len({r["meta"]["skeleton_id"] for r in private_rows}) >= 300
    assert len({r["meta"]["skeleton_id"] for r in negative_rows}) >= 150
    assert {r["subtype"] for r in private_rows} == set(PRIVATE_SUBTYPES)
    assert len({r["style"] for r in private_rows}) >= 7
    assert len({r["format"] for r in private_rows}) >= 7

    for row in private_rows + negative_rows:
        ok, reason = validate_synthetic_example(row)
        assert ok, reason
        assert "skeleton_id" in row["meta"]
        assert not contains_disallowed_secret_like_content(row["text"])


def test_split_by_skeleton_keeps_anchor_skeletons_disjoint():
    gen = SyntheticFinancialGenerator(seed=2024)
    anchors = gen.generate_private(400) + gen.generate_hard_negatives(200) + gen.generate_benign(200)

    train, validation = split_by_skeleton(anchors, rng=gen.rng)

    assert train
    assert validation
    train_skeletons = {r["meta"]["skeleton_id"] for r in train}
    validation_skeletons = {r["meta"]["skeleton_id"] for r in validation}
    assert train_skeletons.isdisjoint(validation_skeletons)
    assert len(train) + len(validation) == len(anchors)


def test_mixed_egress_diversity_and_decisions():
    gen = SyntheticFinancialGenerator(seed=99)
    rows = gen.generate_mixed(300)

    assert len({r["meta"]["skeleton_id"] for r in rows}) >= 120
    assert len({r["meta"]["carrier_id"] for r in rows}) >= 25
    assert {r["expected_decision"] for r in rows} == {"allow", "request_approval"}

    for row in rows:
        assert row["payload_labels"] == ["work", "financial_private"]
        assert "skeleton_id" in row["meta"]
        assert row["financial_evidence"] in row["text"]
        assert not contains_disallowed_secret_like_content(row["text"])
        if row["expected_decision"] == "allow":
            assert row["unexpected_categories"] == []
            assert "financial_private" in row["expected_categories"]
        else:
            assert row["unexpected_categories"] == ["financial_private"]
