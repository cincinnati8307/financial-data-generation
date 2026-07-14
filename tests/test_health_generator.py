import json

from health_egress_poc.centroid_classifier import _hash_embed, classify_text
from health_egress_poc.cli_generate import generate_clean_anchors, split_by_skeleton
from health_egress_poc.filters import (
    contains_disallowed_identifier_or_secret,
    validate_augmented_row,
    validate_synthetic_example,
)
from health_egress_poc.llm_augmenter import DryRunProvider, LLMProvider, augment_rows
from health_egress_poc.schemas import HEALTH_SUBTYPES
from health_egress_poc.synthetic_generator import FORMATS, SyntheticHealthGenerator, load_template_catalog
from sensitive_egress_poc.dataset_quality import evaluate_dataset, parse_checks


class StaticProvider(LLMProvider):
    name = "static"

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.payload


def test_health_template_catalog_has_required_coverage():
    catalog = load_template_catalog()

    assert "entity_pools" in catalog
    assert set(HEALTH_SUBTYPES) == set(catalog["private_scenarios"])
    for subtype in HEALTH_SUBTYPES:
        assert len(catalog["private_scenarios"][subtype]) >= 8
        assert sum(len(scenario["templates"]) for scenario in catalog["private_scenarios"][subtype]) >= 12
    assert sum(len(group) for group in catalog["hard_negative_scenarios"].values()) >= 30
    assert sum(len(group) for group in catalog["benign_scenarios"].values()) >= 20
    assert len(catalog["mixed_carriers"]) >= 30
    assert sum(1 for carrier in catalog["mixed_carriers"] if carrier["expected_health"]) >= 10
    assert sum(1 for carrier in catalog["mixed_carriers"] if not carrier["expected_health"]) >= 10
    assert len(FORMATS) >= 10


def test_generator_creates_valid_private_health_examples_without_full_identifiers():
    gen = SyntheticHealthGenerator(seed=1)
    rows = gen.generate_private(45)

    assert rows
    assert {r["subtype"] for r in rows} == set(HEALTH_SUBTYPES)
    for row in rows:
        ok, reason = validate_synthetic_example(row)
        assert ok, reason
        assert row["label"] == "health_private"
        assert "skeleton_id" in row["meta"]
        assert row["meta"]["privacy_evidence"]
        assert row["meta"]["sensitive_span"] in row["text"]
        assert row["meta"]["private_cues"]
        assert not contains_disallowed_identifier_or_secret(row["text"])


def test_health_identifier_filter_allows_masked_refs_but_rejects_full_identifiers():
    assert not contains_disallowed_identifier_or_secret("Patient ref MRN ****5678 has a lab result.")
    assert not contains_disallowed_identifier_or_secret("患者编号尾号 9012 的复诊记录需要更新。")

    assert contains_disallowed_identifier_or_secret("MRN 123456789 has a lab result.")
    assert contains_disallowed_identifier_or_secret("patient id: 998877665544 has a visit note.")
    assert contains_disallowed_identifier_or_secret("联系电话 +65 9123 4567")
    assert contains_disallowed_identifier_or_secret("S1234567A")


def test_hard_negatives_use_non_private_labels():
    gen = SyntheticHealthGenerator(seed=2)
    rows = gen.generate_hard_negatives(40)

    assert {r["label"] for r in rows} <= {"non_private_health", "benign"}
    for row in rows:
        ok, reason = validate_synthetic_example(row)
        assert ok, reason
        assert row["meta"].get("non_private_reason")
        if row["label"] != "health_private":
            assert "补充一个 private health note" not in row["text"]
            assert "Personal health note:" not in row["text"]
            assert "个人健康记录如下" not in row["text"]


def test_dry_run_health_augmentation_produces_valid_rows():
    gen = SyntheticHealthGenerator(seed=3)
    source = gen.generate_private(1)[0]

    rows, reasons = augment_rows([source], DryRunProvider(), 1, 5, False, "dry-run")

    assert rows
    assert reasons["duplicate"] == 0
    for row in rows:
        ok, reason = validate_augmented_row(row, source)
        assert ok, reason
        assert row["source"] == "llm_paraphrase"
        assert row["parent_id"] == source["id"]
        assert row["label"] == "health_private"
        assert row["meta"].get("privacy_evidence")
        assert row["meta"].get("source_skeleton_id") == source["meta"]["skeleton_id"]


def test_openai_style_health_candidates_are_wrapped_locally():
    gen = SyntheticHealthGenerator(seed=4)
    source = gen.generate_private(1)[0]
    payload = json.dumps(
        [
            {"text": f"Hi，整理一下这条 private health note：{source['text']}", "style": "email_mixed"},
            {"text": f"Clinic note excerpt, rewritten: {source['text']}", "style": "clinic_note"},
        ],
        ensure_ascii=False,
    )
    provider = StaticProvider(payload)

    rows, reasons = augment_rows([source], provider, 1, 2, True, "fake-model")

    assert "Preserve the same health_private meaning" in provider.prompt
    assert reasons["duplicate"] == 0
    assert rows[0] == source
    assert len(rows) == 3
    for row in rows[1:]:
        ok, reason = validate_augmented_row(row, source)
        assert ok, reason
        assert row["subtype"] == source["subtype"]
        assert row["meta"]["augmentation_provider"] == "static"
        assert row["meta"]["augmentation_model"] == "fake-model"


def test_health_candidate_rejections_and_deduplication_are_counted():
    gen = SyntheticHealthGenerator(seed=5)
    source = gen.generate_private(1)[0]
    valid_text = f"改写后的个人健康备注：{source['text']}"
    lines = [
        {"text": source["text"], "style": "zh_casual"},
        {"style": "zh_formal"},
        {"text": "未知风格文本", "style": "not_a_style"},
        {"text": "MRN 123456789 has a result", "style": "clinic_note"},
        {"text": valid_text, "style": "zh_casual"},
        {"text": valid_text, "style": "zh_formal"},
    ]
    payload = "```jsonl\n" + "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n```"

    rows, reasons = augment_rows([source], StaticProvider(payload), 1, 6, False, "fake-model")

    assert [row["text"] for row in rows] == [valid_text]
    assert reasons["unchanged_text"] == 1
    assert reasons["missing_text"] == 1
    assert reasons["unknown_style"] == 1
    assert reasons["identifier_or_secret_like_content"] == 1
    assert reasons["duplicate"] == 1


def test_health_generator_anchor_diversity_targets():
    gen = SyntheticHealthGenerator(seed=1337)
    private_rows = gen.generate_private(600)
    negative_rows = gen.generate_hard_negatives(250) + gen.generate_benign(250)

    assert len({r["meta"]["skeleton_id"] for r in private_rows}) >= 350
    assert len({r["meta"]["skeleton_id"] for r in negative_rows}) >= 180
    assert {r["subtype"] for r in private_rows} == set(HEALTH_SUBTYPES)
    assert len({r["style"] for r in private_rows}) >= 7
    assert len({r["format"] for r in private_rows}) >= 10

    for row in private_rows + negative_rows:
        ok, reason = validate_synthetic_example(row)
        assert ok, reason
        assert "skeleton_id" in row["meta"]
        assert not contains_disallowed_identifier_or_secret(row["text"])


def test_health_cli_clean_anchor_generation_removes_redundant_rows():
    gen = SyntheticHealthGenerator(seed=42)

    rows = generate_clean_anchors(gen, target_private=40, target_hard_negative=20, target_benign=20)
    report, cleaned = evaluate_dataset(rows, checks=parse_checks("redundancy"))

    assert len(rows) == 80
    assert report["summary"]["rejected_rows"] == 0
    assert report["redundancy"]["exact_duplicate_count"] == 0
    assert report["redundancy"]["near_duplicate_count"] == 0
    assert len(cleaned) == len(rows)


def test_health_split_by_skeleton_keeps_anchor_skeletons_disjoint():
    gen = SyntheticHealthGenerator(seed=2024)
    anchors = gen.generate_private(300) + gen.generate_hard_negatives(150) + gen.generate_benign(150)

    train, validation = split_by_skeleton(anchors, rng=gen.rng)

    assert train
    assert validation
    train_skeletons = {r["meta"]["skeleton_id"] for r in train}
    validation_skeletons = {r["meta"]["skeleton_id"] for r in validation}
    assert train_skeletons.isdisjoint(validation_skeletons)
    assert len(train) + len(validation) == len(anchors)


def test_health_mixed_egress_diversity_and_decisions():
    gen = SyntheticHealthGenerator(seed=99)
    rows = gen.generate_mixed(250)

    assert len({r["meta"]["skeleton_id"] for r in rows}) >= 150
    assert len({r["meta"]["carrier_id"] for r in rows}) >= 25
    assert {r["expected_decision"] for r in rows} == {"allow", "request_approval"}

    for row in rows:
        assert row["payload_labels"] == ["work", "health_private"]
        assert "skeleton_id" in row["meta"]
        assert row["health_evidence"] in row["text"]
        assert row["meta"].get("health_evidence") == row["health_evidence"]
        assert not contains_disallowed_identifier_or_secret(row["text"])
        if row["expected_decision"] == "allow":
            assert row["unexpected_categories"] == []
            assert "health_private" in row["expected_categories"]
        else:
            assert row["unexpected_categories"] == ["health_private"]


def test_health_centroid_classifier_flags_obvious_health_private_text_with_hash_centroids():
    centroid_obj = {
        "model": "hash-only",
        "threshold": 0.55,
        "margin_threshold": 0.05,
        "centroids": {
            "health_private.lab_result": _hash_embed("Lab portal shows HbA1c=7.1, ref MRN ****5678."),
            "benign.*": _hash_embed("The deployment finished and the document was updated."),
        },
    }

    result = classify_text("Lab portal shows HbA1c=7.1, ref MRN ****5678.", centroid_obj)
    public_result = classify_text("公开文章解释高血压常见风险因素。", centroid_obj)

    assert result["predicted_label"] == "health_private"
    assert result["decision_hint"] == "request_approval_if_user_intent_does_not_expect_health_data"
    assert public_result["health_score"] < 0.75


def test_shared_dataset_quality_accepts_health_labels_in_dry_run_judge():
    rows = [
        {
            "id": "h1",
            "text": "Lab portal shows HbA1c=7.1, ref MRN ****5678.",
            "label": "health_private",
            "subtype": "lab_result",
            "region": "singapore_cn",
            "language": "zh_en",
            "format": "natural_sentence",
            "style": "clinic_note",
            "sensitivity_level": "high",
            "source": "synthetic_template",
            "meta": {"skeleton_id": "health_private:lab_result:clinic_note"},
        },
        {
            "id": "h2",
            "text": "公开文章解释高血压常见风险因素。",
            "label": "non_private_health",
            "subtype": "*",
            "region": "global",
            "language": "zh",
            "format": "natural_sentence",
            "style": "zh_formal",
            "sensitivity_level": "none",
            "source": "synthetic_template",
            "meta": {"skeleton_id": "health_public:condition"},
        },
    ]

    report, cleaned = evaluate_dataset(rows, checks=parse_checks("llm_realism,safety"), provider="dry-run", sample_size=2)

    assert report["summary"]["accepted_rows"] == 2
    assert {judgment["action"] for judgment in report["llm_realism"]["judgments"]} == {"pass"}
    assert [row["id"] for row in cleaned] == ["h1", "h2"]
