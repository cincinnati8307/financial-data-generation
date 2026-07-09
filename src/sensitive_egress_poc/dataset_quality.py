from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import sacrebleu
except Exception:  # pragma: no cover - exercised only when optional dependency is absent
    sacrebleu = None

from .filters import contains_disallowed_secret_like_content, normalize_text_for_dedup
from .io_utils import read_jsonl, write_json, write_jsonl

GROUP_KEYS = ["label", "subtype", "style", "format", "region", "source", "meta.scenario_id"]
DEFAULT_CHECKS = ["redundancy", "self_bleu", "safety"]
LLM_REALISM_CHECK = "llm_realism"


@dataclass
class DuplicateCluster:
    representative: int
    duplicate: int
    score: float
    reason: str


def row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("id") or f"row_{index}")


def row_text(row: dict[str, Any]) -> str:
    return str(row.get("text", ""))


def nested_get(row: dict[str, Any], key: str, default: str = "unknown") -> str:
    cur: Any = row
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return str(cur if cur not in {None, ""} else default)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def simple_tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())
    return words or list(normalize_text_for_dedup(text))


def char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = re.sub(r"\s+", "", normalize_text_for_dedup(text))
    if len(compact) <= n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def text_similarity(left: str, right: str) -> float:
    return jaccard(char_ngrams(left), char_ngrams(right))


def row_quality_rank(row: dict[str, Any]) -> tuple[int, int, int]:
    source_rank = {"synthetic_template": 3, "llm_paraphrase": 2}.get(str(row.get("source", "")), 1)
    has_skeleton = int(bool((row.get("meta") or {}).get("skeleton_id"))) if isinstance(row.get("meta"), dict) else 0
    return (source_rank, has_skeleton, len(row_text(row)))


def choose_representative(rows: list[dict[str, Any]], indices: list[int]) -> int:
    return max(indices, key=lambda i: (row_quality_rank(rows[i]), -i))


def add_rejection(rejections: dict[int, set[str]], index: int, reason: str) -> None:
    rejections.setdefault(index, set()).add(reason)


def find_exact_duplicates(rows: list[dict[str, Any]], rejections: dict[int, set[str]]) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        key = normalize_text_for_dedup(row_text(row))
        if key:
            groups[key].append(i)

    duplicates = []
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        rep = choose_representative(rows, indices)
        for i in indices:
            if i == rep:
                continue
            add_rejection(rejections, i, "exact_duplicate")
            duplicates.append({"id": row_id(rows[i], i), "index": i, "representative_id": row_id(rows[rep], rep), "reason": "exact_duplicate"})
    return duplicates


def find_skeleton_duplicates(rows: list[dict[str, Any]], rejections: dict[int, set[str]]) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        skeleton_id = meta.get("skeleton_id")
        if skeleton_id:
            groups[str(skeleton_id)].append(i)

    duplicates = []
    for skeleton_id, indices in groups.items():
        if len(indices) < 2:
            continue
        rep = choose_representative(rows, indices)
        for i in indices:
            if i == rep:
                continue
            add_rejection(rejections, i, "skeleton_duplicate")
            duplicates.append(
                {
                    "id": row_id(rows[i], i),
                    "index": i,
                    "representative_id": row_id(rows[rep], rep),
                    "skeleton_id": skeleton_id,
                    "reason": "skeleton_duplicate",
                }
            )
    return duplicates


def find_near_duplicates(rows: list[dict[str, Any]], rejections: dict[int, set[str]], threshold: float = 0.92) -> list[dict[str, Any]]:
    signatures = [char_ngrams(row_text(row)) for row in rows]
    representatives: list[int] = []
    duplicate_items: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        sig = signatures[i]
        if not sig:
            continue
        best_rep = None
        best_score = 0.0
        sig_len = len(sig)
        for rep in representatives:
            rep_sig = signatures[rep]
            rep_len = len(rep_sig)
            if not rep_sig:
                continue
            if min(sig_len, rep_len) / max(sig_len, rep_len) < threshold:
                continue
            score = jaccard(sig, rep_sig)
            if score > best_score:
                best_score = score
                best_rep = rep
        if best_rep is None or best_score < threshold:
            representatives.append(i)
            continue

        current_better = row_quality_rank(row) > row_quality_rank(rows[best_rep])
        duplicate_index = best_rep if current_better else i
        representative_index = i if current_better else best_rep
        if current_better:
            representatives[representatives.index(best_rep)] = i
        add_rejection(rejections, duplicate_index, "near_duplicate")
        duplicate_items.append(
            {
                "id": row_id(rows[duplicate_index], duplicate_index),
                "index": duplicate_index,
                "representative_id": row_id(rows[representative_index], representative_index),
                "similarity": round(best_score, 4),
                "reason": "near_duplicate",
            }
        )
    return duplicate_items


def find_augmented_source_redundancy(rows: list[dict[str, Any]], rejections: dict[int, set[str]], threshold: float = 0.92) -> list[dict[str, Any]]:
    items = []
    for i, row in enumerate(rows):
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        original = meta.get("original_text")
        if not original:
            continue
        current = row_text(row)
        if normalize_text_for_dedup(current) == normalize_text_for_dedup(str(original)):
            reason = "unchanged_from_original"
            score = 1.0
        else:
            score = text_similarity(current, str(original))
            if score < threshold:
                continue
            reason = "too_close_to_original"
        add_rejection(rejections, i, reason)
        items.append({"id": row_id(row, i), "index": i, "similarity": round(score, 4), "reason": reason})
    return items


def find_safety_issues(rows: list[dict[str, Any]], rejections: dict[int, set[str]]) -> list[dict[str, Any]]:
    issues = []
    for i, row in enumerate(rows):
        if contains_disallowed_secret_like_content(row_text(row)):
            add_rejection(rejections, i, "secret_like_content")
            issues.append({"id": row_id(row, i), "index": i, "reason": "secret_like_content"})
    return issues


def sentence_bleu(candidate: str, references: list[str]) -> float:
    if not references:
        return 0.0
    if sacrebleu is not None:
        return sacrebleu.sentence_bleu(candidate, references).score / 100.0

    cand_tokens = simple_tokenize(candidate)
    ref_tokens = set(token for ref in references for token in simple_tokenize(ref))
    if not cand_tokens or not ref_tokens:
        return 0.0
    return sum(1 for tok in cand_tokens if tok in ref_tokens) / len(cand_tokens)


def deterministic_sample(values: list[str], max_samples: int, seed: int) -> list[str]:
    if max_samples <= 0 or len(values) <= max_samples:
        return list(values)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(values)), max_samples))
    return [values[i] for i in indices]


def compute_self_bleu(samples: list[str], max_samples: int = 200, seed: int = 1337) -> float:
    sampled = deterministic_sample(samples, max_samples, seed)
    if len(sampled) < 2:
        return 0.0
    scores = []
    for i, sent in enumerate(sampled):
        references = sampled[:i] + sampled[i + 1 :]
        scores.append(sentence_bleu(sent, references))
    return sum(scores) / len(scores)


def group_rows(data: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        groups[nested_get(item, key)].append(item)
    return groups


def evaluate_self_bleu(
    data: list[dict[str, Any]],
    group_keys: Iterable[str] = GROUP_KEYS,
    min_group_size: int = 5,
    max_samples: int = 200,
    seed: int = 1337,
    high_threshold: float = 0.75,
) -> dict[str, Any]:
    texts = [row_text(row) for row in data if row_text(row)]
    overall = compute_self_bleu(texts, max_samples=max_samples, seed=seed) if len(texts) >= 2 else 0.0
    groups_report: dict[str, dict[str, Any]] = {}
    for key in group_keys:
        key_report: dict[str, Any] = {}
        for group, rows in group_rows(data, key).items():
            group_texts = [row_text(row) for row in rows if row_text(row)]
            if len(group_texts) < min_group_size:
                continue
            score = compute_self_bleu(group_texts, max_samples=max_samples, seed=seed)
            key_report[group] = {"count": len(group_texts), "self_bleu": round(score, 4), "flag": score >= high_threshold}
        groups_report[key] = key_report
    return {
        "overall": {"count": len(texts), "self_bleu": round(overall, 4), "flag": overall >= high_threshold},
        "groups": groups_report,
        "high_threshold": high_threshold,
        "max_samples": max_samples,
    }


def estimate_text_tokens(text: str) -> int:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    without_cjk = "".join(" " if "\u4e00" <= ch <= "\u9fff" else ch for ch in text)
    words = re.findall(r"[A-Za-z0-9_]+", without_cjk)
    punctuation = re.findall(r"[^\sA-Za-z0-9_]", without_cjk)
    return max(1, cjk + int(len(words) * 1.3) + int(len(punctuation) * 0.5))


def build_judge_prompt(row: dict[str, Any]) -> str:
    row_json = json.dumps(row, ensure_ascii=False)
    return f"""Judge this synthetic dataset row. Return JSON only with keys: id, realism_score, label_correct, subtype_correct, privacy_safe, is_real_paraphrase, action, reasons.

Scoring rules:
- realism_score: integer 1-5, where 5 looks like plausible user/agent text and 1 looks artificial or broken.
- label_correct: whether text matches the label.
- subtype_correct: true for non-private rows, or whether financial_private subtype fits the text.
- privacy_safe: false if there is a full account/card number, credential, token, password, real-looking ID, phone, or other secret.
- is_real_paraphrase: null unless meta.original_text exists; then true only if text is a real paraphrase rather than a copy or tiny wrapper.
- action: pass, review, or fail.
- reasons: short list of reason strings.

Row:
{row_json}
"""


def estimate_llm_judge_tokens(rows: list[dict[str, Any]], sample_size: int) -> dict[str, int | str]:
    sample = rows[:sample_size] if sample_size > 0 else []
    prompt_tokens = sum(estimate_text_tokens(build_judge_prompt(row)) for row in sample)
    completion_tokens = 140 * len(sample)
    total = prompt_tokens + completion_tokens
    return {
        "method": "heuristic_cjk_word_estimate",
        "sample_size": len(sample),
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_completion_tokens": completion_tokens,
        "estimated_total_tokens": total,
        "estimated_total_tokens_with_buffer": int(total * 1.25),
    }


def format_token_estimate(estimate: dict[str, int | str], provider: str, model: str) -> str:
    return "\n".join(
        [
            "Token estimate for LLM dataset quality judging:",
            f"  provider: {provider}",
            f"  model: {model}",
            f"  sampled rows: {estimate['sample_size']}",
            f"  estimated prompt tokens: {estimate['estimated_prompt_tokens']}",
            f"  estimated completion tokens: {estimate['estimated_completion_tokens']}",
            f"  estimated total tokens: {estimate['estimated_total_tokens']}",
            f"  conservative total (+25%): {estimate['estimated_total_tokens_with_buffer']}",
            "  method: heuristic estimate, not provider billing truth",
        ]
    )


def dry_run_judge(row: dict[str, Any], index: int) -> dict[str, Any]:
    text = row_text(row)
    label = row.get("label")
    subtype = row.get("subtype")
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    original = meta.get("original_text")
    privacy_safe = not contains_disallowed_secret_like_content(text)
    realism_score = 4 if len(text.strip()) >= 12 and "{}" not in text else 2
    label_correct = bool(label in {"financial_private", "non_private_financial", "benign"} or row.get("expected_decision"))
    subtype_correct = True if label != "financial_private" else bool(subtype and subtype != "*")
    if original is None:
        is_real_paraphrase = None
    else:
        is_real_paraphrase = normalize_text_for_dedup(text) != normalize_text_for_dedup(str(original)) and text_similarity(text, str(original)) < 0.98
    action = "pass" if privacy_safe and realism_score >= 3 and label_correct and subtype_correct and is_real_paraphrase is not False else "fail"
    reasons = [] if action == "pass" else ["dry_run_quality_rule_failed"]
    return {
        "id": row_id(row, index),
        "index": index,
        "realism_score": realism_score,
        "label_correct": label_correct,
        "subtype_correct": subtype_correct,
        "privacy_safe": privacy_safe,
        "is_real_paraphrase": is_real_paraphrase,
        "action": action,
        "reasons": reasons,
    }


class OpenAIJudge:
    def __init__(self, model: str) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use provider=openai") from exc
        self.client = OpenAI()
        self.model = model

    def judge(self, row: dict[str, Any], index: int) -> dict[str, Any]:
        resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": build_judge_prompt(row)}])
        content = resp.choices[0].message.content or "{}"
        try:
            obj = json.loads(strip_json_fence(content))
        except json.JSONDecodeError:
            return {"id": row_id(row, index), "index": index, "action": "review", "reasons": ["judge_parse_error"], "raw": content}
        obj.setdefault("id", row_id(row, index))
        obj["index"] = index
        obj.setdefault("action", "review")
        obj.setdefault("reasons", [])
        return obj


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.S | re.I)
    return match.group(1).strip() if match else stripped


def deterministic_row_sample(rows: list[tuple[int, dict[str, Any]]], sample_size: int, seed: int) -> list[tuple[int, dict[str, Any]]]:
    if sample_size <= 0:
        return []
    if len(rows) <= sample_size:
        return list(rows)
    rng = random.Random(seed)
    positions = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[pos] for pos in positions]


def run_llm_realism_judge(
    rows: list[tuple[int, dict[str, Any]]],
    provider: str,
    model: str,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    sampled = deterministic_row_sample(rows, sample_size, seed)
    if provider == "dry-run":
        judgments = [dry_run_judge(row, i) for i, row in sampled]
    elif provider == "openai":
        judge = OpenAIJudge(model)
        judgments = [judge.judge(row, i) for i, row in sampled]
    else:
        raise ValueError(f"unsupported provider:{provider}")

    actions = Counter(str(j.get("action", "review")) for j in judgments)
    realism_scores = [float(j.get("realism_score", 0)) for j in judgments if isinstance(j.get("realism_score"), (int, float))]
    pass_count = actions.get("pass", 0)
    return {
        "enabled": True,
        "provider": provider,
        "model": model,
        "sampled_rows": len(sampled),
        "aggregate": {
            "pass_rate": round(pass_count / len(judgments), 4) if judgments else 0.0,
            "actions": dict(actions),
            "average_realism_score": round(sum(realism_scores) / len(realism_scores), 4) if realism_scores else 0.0,
        },
        "judgments": judgments,
    }


def deterministic_quality_checks(
    rows: list[dict[str, Any]],
    near_duplicate_threshold: float,
    original_similarity_threshold: float,
    checks: set[str],
) -> tuple[dict[str, Any], dict[int, set[str]]]:
    rejections: dict[int, set[str]] = {}
    redundancy = {
        "exact_duplicates": [],
        "near_duplicates": [],
        "skeleton_duplicates": [],
        "augmentation_source_redundancy": [],
    }
    if "redundancy" in checks:
        redundancy["exact_duplicates"] = find_exact_duplicates(rows, rejections)
        redundancy["skeleton_duplicates"] = find_skeleton_duplicates(rows, rejections)
        redundancy["near_duplicates"] = find_near_duplicates(rows, rejections, threshold=near_duplicate_threshold)
        redundancy["augmentation_source_redundancy"] = find_augmented_source_redundancy(
            rows, rejections, threshold=original_similarity_threshold
        )
    safety_issues = find_safety_issues(rows, rejections) if "safety" in checks else []
    report = {
        "redundancy": {
            "exact_duplicate_count": len(redundancy["exact_duplicates"]),
            "near_duplicate_count": len(redundancy["near_duplicates"]),
            "skeleton_duplicate_count": len(redundancy["skeleton_duplicates"]),
            "augmentation_source_redundancy_count": len(redundancy["augmentation_source_redundancy"]),
            "items": {
                "exact_duplicates": redundancy["exact_duplicates"][:100],
                "near_duplicates": redundancy["near_duplicates"][:100],
                "skeleton_duplicates": redundancy["skeleton_duplicates"][:100],
                "augmentation_source_redundancy": redundancy["augmentation_source_redundancy"][:100],
            },
        },
        "safety": {"secret_like_count": len(safety_issues), "items": safety_issues[:100]},
    }
    return report, rejections


def rejected_rows_report(rows: list[dict[str, Any]], rejections: dict[int, set[str]]) -> list[dict[str, Any]]:
    return [
        {"id": row_id(rows[i], i), "index": i, "reasons": sorted(reasons)}
        for i, reasons in sorted(rejections.items())
        if reasons
    ]


def clean_rows(rows: list[dict[str, Any]], rejections: dict[int, set[str]]) -> list[dict[str, Any]]:
    return [row for i, row in enumerate(rows) if i not in rejections]


def recommended_action(report: dict[str, Any], rejections: dict[int, set[str]], row_count: int) -> str:
    rejection_rate = len(rejections) / max(1, row_count)
    llm = report.get("llm_realism", {})
    llm_actions = ((llm.get("aggregate") or {}).get("actions") or {}) if isinstance(llm, dict) else {}
    if report.get("safety", {}).get("secret_like_count", 0) > 0 or rejection_rate >= 0.2 or llm_actions.get("fail", 0) > 0:
        return "fail"
    if rejections or llm_actions.get("review", 0) > 0:
        return "review"
    return "pass"


def evaluate_dataset(
    rows: list[dict[str, Any]],
    checks: set[str] | None = None,
    near_duplicate_threshold: float = 0.92,
    original_similarity_threshold: float = 0.92,
    self_bleu_sample_size: int = 200,
    sample_size: int = 50,
    provider: str = "dry-run",
    model: str = "gpt-5-nano",
    seed: int = 1337,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_checks = checks or set(DEFAULT_CHECKS)
    deterministic_report, rejections = deterministic_quality_checks(
        rows, near_duplicate_threshold, original_similarity_threshold, selected_checks
    )
    report: dict[str, Any] = {
        "summary": {
            "input_rows": len(rows),
            "checks": sorted(selected_checks),
            "label_counts": dict(Counter(str(row.get("label", "unknown")) for row in rows)),
            "source_counts": dict(Counter(str(row.get("source", "unknown")) for row in rows)),
        },
        **deterministic_report,
    }

    if "self_bleu" in selected_checks:
        report["self_bleu"] = evaluate_self_bleu(rows, max_samples=self_bleu_sample_size, seed=seed)

    if LLM_REALISM_CHECK in selected_checks:
        judge_candidates = [(i, row) for i, row in enumerate(rows) if i not in rejections]
        report["llm_realism"] = run_llm_realism_judge(judge_candidates, provider, model, sample_size, seed)
        for judgment in report["llm_realism"]["judgments"]:
            if str(judgment.get("action", "review")) == "fail":
                original_index = judgment.get("index")
                if isinstance(original_index, int):
                    add_rejection(rejections, original_index, "llm_realism_fail")
    else:
        report["llm_realism"] = {"enabled": False}

    report["rejected_rows"] = rejected_rows_report(rows, rejections)
    report["summary"]["rejected_rows"] = len(report["rejected_rows"])
    report["summary"]["accepted_rows"] = len(rows) - len(report["rejected_rows"])
    report["recommended_action"] = recommended_action(report, rejections, len(rows))
    return report, clean_rows(rows, rejections)


def parse_checks(value: str) -> set[str]:
    checks = {part.strip() for part in value.split(",") if part.strip()}
    allowed = {"redundancy", "self_bleu", "llm_realism", "safety"}
    unknown = checks - allowed
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown checks: {', '.join(sorted(unknown))}")
    return checks


def confirm() -> bool:
    answer = input("Proceed with OpenAI LLM-as-judge quality check? Type 'yes' to continue: ").strip().lower()
    return answer in {"yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report-out")
    parser.add_argument("--clean-output")
    parser.add_argument("--checks", default=parse_checks(",".join(DEFAULT_CHECKS)), type=parse_checks)
    parser.add_argument("--provider", choices=["dry-run", "openai"], default="dry-run")
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--self-bleu-sample-size", type=int, default=200)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument("--original-similarity-threshold", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true")
    args = parser.parse_args()

    rows = load_dataset(args.input)
    if LLM_REALISM_CHECK in args.checks:
        estimate = estimate_llm_judge_tokens(rows, args.sample_size)
        print(format_token_estimate(estimate, args.provider, args.model))
        if args.estimate_only:
            return
        if args.provider == "openai" and not args.yes and not confirm():
            print("Quality check cancelled before any OpenAI request was sent.")
            return
    elif args.estimate_only:
        print("No LLM-as-judge check selected; no token estimate is needed.")
        return

    report, cleaned = evaluate_dataset(
        rows,
        checks=args.checks,
        near_duplicate_threshold=args.near_duplicate_threshold,
        original_similarity_threshold=args.original_similarity_threshold,
        self_bleu_sample_size=args.self_bleu_sample_size,
        sample_size=args.sample_size,
        provider=args.provider,
        model=args.model,
        seed=args.seed,
    )

    if args.report_out:
        write_json(args.report_out, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.clean_output:
        write_jsonl(args.clean_output, cleaned)


if __name__ == "__main__":
    main()
