import json
import sys

from sensitive_egress_poc.cli_augment import main
from sensitive_egress_poc.io_utils import read_json, read_jsonl, write_jsonl
from sensitive_egress_poc.llm_augmenter import estimate_augmentation_tokens
from sensitive_egress_poc.synthetic_generator import SyntheticFinancialGenerator


def private_rows(n=2):
    return SyntheticFinancialGenerator(seed=123).generate_private(n)


def test_token_estimate_counts_selected_private_sources():
    rows = private_rows(3) + SyntheticFinancialGenerator(seed=4).generate_benign(2)

    estimate = estimate_augmentation_tokens(rows, max_inputs=2, paraphrases_per_example=4)

    assert estimate["attempted_sources"] == 2
    assert estimate["paraphrases_per_example"] == 4
    assert estimate["estimated_prompt_tokens"] > 0
    assert estimate["estimated_completion_tokens"] > 0
    assert estimate["estimated_total_tokens"] == estimate["estimated_prompt_tokens"] + estimate["estimated_completion_tokens"]
    assert estimate["estimated_total_tokens_with_buffer"] >= estimate["estimated_total_tokens"]


def test_cli_estimate_only_does_not_require_openai_key_or_write_output(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "anchors.jsonl"
    output_path = tmp_path / "augmented.jsonl"
    write_jsonl(input_path, private_rows(2))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli_augment",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--provider",
            "openai",
            "--estimate-only",
            "--max-inputs",
            "2",
            "--paraphrases-per-example",
            "3",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert "Token estimate for augmentation" in captured.out
    assert "private source rows: 2" in captured.out
    assert not output_path.exists()
    assert not output_path.with_suffix(".manifest.json").exists()


def test_cli_openai_cancel_happens_before_output_or_provider_init(tmp_path, monkeypatch):
    input_path = tmp_path / "anchors.jsonl"
    output_path = tmp_path / "augmented.jsonl"
    write_jsonl(input_path, private_rows(1))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("builtins.input", lambda _: "no")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli_augment",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--provider",
            "openai",
            "--max-inputs",
            "1",
        ],
    )

    main()

    assert not output_path.exists()
    assert not output_path.with_suffix(".manifest.json").exists()


def test_cli_dry_run_manifest_includes_token_estimate(tmp_path, monkeypatch):
    input_path = tmp_path / "anchors.jsonl"
    output_path = tmp_path / "augmented.jsonl"
    write_jsonl(input_path, private_rows(1))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli_augment",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--provider",
            "dry-run",
            "--paraphrases-per-example",
            "2",
        ],
    )

    main()

    assert len(read_jsonl(output_path)) == 2
    manifest = read_json(output_path.with_suffix(".manifest.json"))
    assert manifest["attempted_sources"] == 1
    assert manifest["token_estimate"]["attempted_sources"] == 1
    assert manifest["token_estimate"]["estimated_total_tokens"] > 0
