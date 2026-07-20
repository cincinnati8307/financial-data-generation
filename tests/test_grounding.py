import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from egress_grounding.cli_prepare import main as prepare_main
from egress_grounding.cli_validate import main as validate_main
from egress_grounding.registry import prepare_dataset
from egress_grounding.schemas import GroundingRecord, write_records_jsonl
from egress_grounding.store import GroundingCoverageError, GroundingStore
from health_egress_poc.synthetic_generator import SyntheticHealthGenerator
from sensitive_egress_poc.centroid_classifier import build_centroids
from sensitive_egress_poc.cli_generate import main as financial_generate_main
from sensitive_egress_poc.cli_generate import split_by_skeleton
from sensitive_egress_poc.dataset_quality import evaluate_dataset, parse_checks
from sensitive_egress_poc.io_utils import read_json, read_jsonl
from sensitive_egress_poc.synthetic_generator import SyntheticFinancialGenerator

FIXTURES = Path(__file__).parent / "fixtures" / "grounding"


def serialized(records):
    return "\n".join(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) for record in records)


def test_grounding_adapters_map_artificial_fixtures_without_raw_text_or_ids():
    cfpb = prepare_dataset("cfpb", FIXTURES / "cfpb_sample.csv", seed=1)
    banking = prepare_dataset("banking77", FIXTURES / "banking77_sample.csv", seed=1)
    berka = prepare_dataset("berka", FIXTURES / "berka", seed=1)
    nhanes = prepare_dataset("nhanes", FIXTURES / "nhanes_sample.csv", seed=1)
    pubmed = prepare_dataset("pubmed", FIXTURES / "pubmed_sample.xml", seed=1)
    generic = prepare_dataset("generic_jsonl", FIXTURES / "generic_valid.jsonl", seed=1)

    assert {record.dataset for record in cfpb + banking + berka + nhanes + pubmed + generic} == {
        "cfpb",
        "banking77",
        "berka",
        "nhanes",
        "pubmed",
        "generic_jsonl",
    }
    assert {record.role for record in banking} == {"private_candidate", "public_negative"}
    assert {record.role for record in berka} == {"distribution"}
    assert {record.label for record in nhanes} == {"health_private"}
    assert {record.label for record in pubmed} == {"non_private_health"}

    text = serialized(cfpb + banking + berka + nhanes + pubmed)
    assert "must never be copied" not in text
    assert "artificial utterance should not be copied" not in text
    assert "111111" not in text
    assert "900001" not in text
    assert "100001" not in text
    assert "artificial public abstract" not in text


def test_generic_jsonl_rejects_raw_text_field():
    with pytest.raises(ValueError, match="unsafe_or_raw_field"):
        prepare_dataset("generic_jsonl", FIXTURES / "generic_invalid_raw_text.jsonl", seed=1)


def test_prepare_and_validate_clis_write_manifest_and_fail_invalid(tmp_path, monkeypatch):
    output = tmp_path / "cfpb.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grounding_prepare",
            "--dataset",
            "cfpb",
            "--input",
            str(FIXTURES / "cfpb_sample.csv"),
            "--output",
            str(output),
            "--limit",
            "1",
        ],
    )
    prepare_main()

    assert len(read_jsonl(output)) == 1
    manifest = read_json(output.with_suffix(".jsonl.manifest.json"))
    assert manifest["dataset"] == "cfpb"
    assert manifest["contains_real_personal_data"] is False
    assert manifest["raw_text_copied"] is False

    report = tmp_path / "validate.json"
    monkeypatch.setattr(sys, "argv", ["grounding_validate", "--input", str(output), "--report-out", str(report)])
    validate_main()
    assert read_json(report)["valid"] is True

    monkeypatch.setattr(sys, "argv", ["grounding_validate", "--input", str(FIXTURES / "generic_invalid_raw_text.jsonl")])
    with pytest.raises(SystemExit):
        validate_main()


def wildcard_financial_store() -> GroundingStore:
    return GroundingStore(
        [
            GroundingRecord(
                id="fin_all",
                dataset="generic_jsonl",
                domain="financial",
                role="private_candidate",
                label="financial_private",
                subtype="*",
                source_group_id="grp_fin_all",
                facts={"bank": "DBS", "amount": "SGD 4,200", "masked_account": "account ****4321", "account_type": "savings account"},
                region="singapore_cn",
                tags=["fixture"],
                meta={"privacy_transform": "fixture"},
            )
        ]
    )


def test_synthetic_only_behavior_is_unchanged_when_store_is_supplied():
    store = GroundingStore.load([FIXTURES / "generic_valid.jsonl"])

    baseline = SyntheticFinancialGenerator(seed=11).generate_private(8)
    with_store = SyntheticFinancialGenerator(seed=11, grounding_store=store, grounding_mode="synthetic-only", grounding_ratio=1.0).generate_private(8)

    assert with_store == baseline


def test_financial_hybrid_grounding_metadata_and_public_negative_source():
    store = GroundingStore.load([FIXTURES / "generic_valid.jsonl"])
    gen = SyntheticFinancialGenerator(seed=12, grounding_store=store, grounding_mode="hybrid", grounding_ratio=1.0)

    private = asdict(gen.private_example("bank_balance"))
    negative = asdict(gen.hard_negative_example())

    assert private["source"] == "grounded_synthetic"
    assert private["meta"]["grounding"]["dataset"] == "generic_jsonl"
    assert private["meta"]["grounding"]["raw_text_copied"] is False
    assert "SGD 4,200" in private["text"]
    assert negative["source"] == "grounded_public_negative"
    assert negative["meta"]["grounding"]["role"] == "public_negative"


def test_health_hybrid_grounding_metadata():
    store = GroundingStore.load([FIXTURES / "generic_valid.jsonl"])
    gen = SyntheticHealthGenerator(seed=13, grounding_store=store, grounding_mode="hybrid", grounding_ratio=1.0)

    row = asdict(gen.private_example("lab_result"))

    assert row["source"] == "grounded_synthetic"
    assert row["meta"]["grounding"]["domain"] == "health"
    assert row["meta"]["grounding"]["facts_used"]
    assert "6.1" in row["text"]


def test_grounded_only_reports_insufficient_coverage():
    store = GroundingStore.load([FIXTURES / "generic_valid.jsonl"], datasets=["nhanes"])
    gen = SyntheticFinancialGenerator(seed=14, grounding_store=store, grounding_mode="grounded-only", grounding_ratio=1.0)

    with pytest.raises(GroundingCoverageError) as exc:
        gen.private_example("bank_balance")

    assert exc.value.report["missing"]


def test_grounded_generation_is_deterministic_for_fixed_seed():
    store = wildcard_financial_store()
    left = SyntheticFinancialGenerator(seed=15, grounding_store=store, grounding_mode="hybrid", grounding_ratio=1.0).generate_mixed(5)
    right = SyntheticFinancialGenerator(seed=15, grounding_store=store, grounding_mode="hybrid", grounding_ratio=1.0).generate_mixed(5)

    assert left == right
    assert all("grounding" in row["meta"] for row in left)


def test_mixed_egress_inherits_private_payload_grounding():
    gen = SyntheticFinancialGenerator(seed=16, grounding_store=wildcard_financial_store(), grounding_mode="hybrid", grounding_ratio=1.0)

    row = asdict(gen.mixed_egress_example())

    assert row["source"] == "synthetic_mixed"
    assert row["meta"]["grounding"]["source_group_id"] == "grp_fin_all"
    assert row["financial_evidence"] in row["text"]


def test_grounding_source_group_isolated_train_validation_split():
    rows = [
        {"id": "a", "text": "one", "meta": {"skeleton_id": "s1", "grounding": {"dataset": "generic_jsonl", "source_group_id": "same"}}},
        {"id": "b", "text": "two", "meta": {"skeleton_id": "s2", "grounding": {"dataset": "generic_jsonl", "source_group_id": "same"}}},
        {"id": "c", "text": "three", "meta": {"skeleton_id": "s3"}},
        {"id": "d", "text": "four", "meta": {"skeleton_id": "s4"}},
    ]

    train, validation = split_by_skeleton(rows, train_ratio=0.5)

    train_groups = {row["meta"].get("grounding", {}).get("source_group_id") for row in train if row["meta"].get("grounding")}
    validation_groups = {row["meta"].get("grounding", {}).get("source_group_id") for row in validation if row["meta"].get("grounding")}
    assert train_groups.isdisjoint(validation_groups)


def test_quality_and_centroid_accept_grounded_rows():
    store = GroundingStore.load([FIXTURES / "generic_valid.jsonl"])
    gen = SyntheticFinancialGenerator(seed=17, grounding_store=store, grounding_mode="hybrid", grounding_ratio=1.0)
    private = asdict(gen.private_example("bank_balance"))
    benign = gen.generate_benign(1)[0]

    report, cleaned = evaluate_dataset([private, benign], checks=parse_checks("safety"))
    centroids = build_centroids(cleaned, model_name="hash-only")

    assert report["summary"]["accepted_rows"] == 2
    assert "financial_private.bank_balance" in centroids["centroids"]


def test_financial_generation_cli_accepts_multiple_grounding_inputs(tmp_path, monkeypatch):
    records = GroundingStore.load([FIXTURES / "generic_valid.jsonl"]).records
    first = tmp_path / "private.jsonl"
    second = tmp_path / "negative.jsonl"
    write_records_jsonl(first, [record for record in records if record.label == "financial_private"])
    write_records_jsonl(second, [record for record in records if record.label == "non_private_financial"])

    out_dir = tmp_path / "generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "financial_generate",
            "--out-dir",
            str(out_dir),
            "--private",
            "1",
            "--hard-negative",
            "1",
            "--benign",
            "0",
            "--mixed",
            "0",
            "--seed",
            "18",
            "--grounding",
            str(first),
            "--grounding",
            str(second),
            "--grounding-mode",
            "hybrid",
            "--grounding-ratio",
            "1.0",
        ],
    )
    financial_generate_main()

    manifest = read_json(out_dir / "manifest.json")
    assert manifest["grounding"]["inputs"] == [str(first), str(second)]
    assert manifest["grounding"]["grounded_counts"]["grounded_synthetic"] == 1
    assert manifest["grounding"]["grounded_counts"]["grounded_public_negative"] == 1
    assert manifest["contains_real_personal_data"] is False
