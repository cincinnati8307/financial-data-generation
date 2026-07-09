import json
import sys

from sensitive_egress_poc.dataset_quality import (
    evaluate_dataset,
    estimate_llm_judge_tokens,
    main,
    parse_checks,
    text_similarity,
)
from sensitive_egress_poc.io_utils import read_json, read_jsonl, write_jsonl


def row(row_id, text, label="financial_private", subtype="bank_balance", source="synthetic_template", meta=None):
    return {
        "id": row_id,
        "text": text,
        "label": label,
        "subtype": subtype,
        "region": "singapore_cn",
        "language": "zh_en",
        "format": "natural_sentence",
        "style": "zh_casual",
        "sensitivity_level": "high" if label == "financial_private" else "none",
        "source": source,
        "meta": meta or {},
    }


def test_similarity_detects_near_duplicate_text():
    assert text_similarity("我的 DBS 账户余额是 SGD 1,200。", "我的DBS账户余额是SGD 1,200") >= 0.9


def test_quality_checker_rejects_redundant_and_unsafe_rows():
    rows = [
        row("r1", "我的 DBS 账户余额是 SGD 1,200。", meta={"skeleton_id": "s1"}),
        row("r2", "我的 DBS 账户余额是 SGD 1,200。", meta={"skeleton_id": "s1"}),
        row("r3", "我的DBS账户余额是SGD 1,200", meta={"skeleton_id": "s3"}),
        row(
            "r4",
            "我的 DBS 账户余额是 SGD 1,200。",
            source="llm_paraphrase",
            meta={"original_text": "我的 DBS 账户余额是 SGD 1,200。"},
        ),
        row("r5", "password: abcdefghijklmnopqrstuvwxyz", label="benign", subtype="*"),
    ]

    report, cleaned = evaluate_dataset(rows, checks=parse_checks("redundancy,safety"), near_duplicate_threshold=0.9)

    rejected = {item["id"]: set(item["reasons"]) for item in report["rejected_rows"]}
    assert "exact_duplicate" in rejected["r2"]
    assert "near_duplicate" in rejected["r3"]
    assert "unchanged_from_original" in rejected["r4"]
    assert "secret_like_content" in rejected["r5"]
    assert [r["id"] for r in cleaned] == ["r1"]
    assert report["recommended_action"] == "fail"


def test_self_bleu_report_has_grouped_metrics():
    rows = [row(f"p{i}", f"我的 DBS 账户余额是 SGD {1000+i}。", meta={"scenario_id": "bank_balance"}) for i in range(6)]
    rows += [row(f"b{i}", f"会议纪要第 {i} 条。", label="benign", subtype="*", meta={"scenario_id": "meeting"}) for i in range(6)]

    report, _ = evaluate_dataset(rows, checks=parse_checks("self_bleu"), self_bleu_sample_size=20)

    assert report["self_bleu"]["overall"]["count"] == 12
    assert "financial_private" in report["self_bleu"]["groups"]["label"]
    assert "benign" in report["self_bleu"]["groups"]["label"]
    assert "bank_balance" in report["self_bleu"]["groups"]["meta.scenario_id"]


def test_dry_run_llm_judge_can_fail_unrealistic_rows():
    rows = [row("bad", "{}", label="benign", subtype="*")]

    report, cleaned = evaluate_dataset(rows, checks=parse_checks("llm_realism"), provider="dry-run", sample_size=1)

    assert report["llm_realism"]["enabled"] is True
    assert report["llm_realism"]["judgments"][0]["action"] == "fail"
    assert report["rejected_rows"][0]["reasons"] == ["llm_realism_fail"]
    assert cleaned == []


def test_llm_judge_token_estimate_shape():
    estimate = estimate_llm_judge_tokens([row("r1", "我的 DBS 账户余额是 SGD 1,200。")], sample_size=1)

    assert estimate["sample_size"] == 1
    assert estimate["estimated_prompt_tokens"] > 0
    assert estimate["estimated_completion_tokens"] > 0
    assert estimate["estimated_total_tokens_with_buffer"] >= estimate["estimated_total_tokens"]


def test_cli_writes_report_and_clean_output(tmp_path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    report_path = tmp_path / "report.json"
    clean_path = tmp_path / "clean.jsonl"
    write_jsonl(input_path, [row("r1", "我的 DBS 账户余额是 SGD 1,200。"), row("r2", "我的 DBS 账户余额是 SGD 1,200。")])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_quality",
            "--input",
            str(input_path),
            "--report-out",
            str(report_path),
            "--clean-output",
            str(clean_path),
            "--checks",
            "redundancy,safety",
        ],
    )

    main()

    report = read_json(report_path)
    cleaned = read_jsonl(clean_path)
    assert report["summary"]["input_rows"] == 2
    assert report["summary"]["accepted_rows"] == 1
    assert [r["id"] for r in cleaned] == ["r1"]


def test_cli_estimate_only_does_not_require_openai_key_or_write_outputs(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "input.jsonl"
    report_path = tmp_path / "report.json"
    clean_path = tmp_path / "clean.jsonl"
    write_jsonl(input_path, [row("r1", "我的 DBS 账户余额是 SGD 1,200。")])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_quality",
            "--input",
            str(input_path),
            "--report-out",
            str(report_path),
            "--clean-output",
            str(clean_path),
            "--checks",
            "llm_realism",
            "--provider",
            "openai",
            "--estimate-only",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert "Token estimate for LLM dataset quality judging" in captured.out
    assert not report_path.exists()
    assert not clean_path.exists()
