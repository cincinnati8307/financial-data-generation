from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schemas import (
    ALIGNED_SENSITIVE,
    MISALIGNED_SENSITIVE,
    NON_SENSITIVE,
    SENSITIVE,
    STATUS_SUCCESS,
    TASK_COARSE_ALIGNMENT,
    TASK_SENSITIVITY,
    BenchmarkPrediction,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def generate_plots(predictions: list[BenchmarkPrediction], sensitivity_metrics: dict[str, Any], alignment_metrics: dict[str, Any], runtime_metrics: dict[str, Any], output_dir: str | Path) -> list[str]:
    plt = _load_pyplot()
    if plt is None:
        return []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    created.extend(_plot_confusion_matrices(plt, predictions, out))
    created.extend(_plot_pr_curves(plt, predictions, out))
    created.extend(_plot_alignment_rates(plt, alignment_metrics, out))
    created.extend(_plot_f1_by_subtype(plt, predictions, out))
    created.extend(_plot_runtime(plt, runtime_metrics, out))
    return created


def _plot_confusion_matrices(plt, predictions: list[BenchmarkPrediction], out: Path) -> list[str]:
    created: list[str] = []
    grouped: dict[tuple[str, str], list[BenchmarkPrediction]] = defaultdict(list)
    for pred in predictions:
        if pred.status == STATUS_SUCCESS and pred.ground_truth and pred.predicted_label:
            grouped[(pred.task or "unknown", pred.model_name)].append(pred)
    for (task, model_name), preds in grouped.items():
        if task == TASK_SENSITIVITY:
            labels = [SENSITIVE, NON_SENSITIVE]
        elif "alignment" in task:
            labels = [ALIGNED_SENSITIVE, MISALIGNED_SENSITIVE, NON_SENSITIVE]
        else:
            continue
        matrix = [[0 for _ in labels] for _ in labels]
        label_index = {label: i for i, label in enumerate(labels)}
        for pred in preds:
            truth = pred.ground_truth if pred.ground_truth in label_index else NON_SENSITIVE
            guessed = pred.predicted_label if pred.predicted_label in label_index else NON_SENSITIVE
            matrix[label_index[truth]][label_index[guessed]] += 1
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"{model_name} {task}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = out / f"{_safe_name(task)}_{_safe_name(model_name)}_confusion_matrix.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(str(path))
    return created


def _plot_pr_curves(plt, predictions: list[BenchmarkPrediction], out: Path) -> list[str]:
    created: list[str] = []
    grouped: dict[str, list[BenchmarkPrediction]] = defaultdict(list)
    for pred in predictions:
        if pred.task == TASK_SENSITIVITY and pred.status == STATUS_SUCCESS and pred.sensitivity_score is not None:
            grouped[pred.model_name].append(pred)
    for model_name, preds in grouped.items():
        if len({p.ground_truth for p in preds}) < 2:
            continue
        try:
            from sklearn.metrics import precision_recall_curve

            y = [1 if p.ground_truth == SENSITIVE else 0 for p in preds]
            scores = [float(p.sensitivity_score or 0.0) for p in preds]
            precision, recall, _ = precision_recall_curve(y, scores)
        except Exception:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision)
        ax.set_title(f"Sensitivity PR: {model_name}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        path = out / f"sensitivity_{_safe_name(model_name)}_precision_recall.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(str(path))
    return created


def _plot_alignment_rates(plt, alignment_metrics: dict[str, Any], out: Path) -> list[str]:
    coarse = alignment_metrics.get(TASK_COARSE_ALIGNMENT, {}) if TASK_COARSE_ALIGNMENT in alignment_metrics else alignment_metrics
    scored = {name: m for name, m in coarse.items() if m.get("status") == "scored"}
    if not scored:
        return []
    names = list(scored.keys())
    leakage = [scored[name].get("leakage_rate") or 0.0 for name in names]
    false_block = [scored[name].get("false_block_rate") or 0.0 for name in names]
    x = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    ax.bar([i - 0.2 for i in x], leakage, width=0.4, label="Leakage rate")
    ax.bar([i + 0.2 for i in x], false_block, width=0.4, label="False-block rate")
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Coarse Policy Alignment Rates")
    ax.legend()
    fig.tight_layout()
    path = out / "alignment_leakage_false_block_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [str(path)]


def _simple_f1(preds: list[BenchmarkPrediction], positive_label: str) -> float | None:
    tp = sum(1 for p in preds if p.ground_truth == positive_label and p.predicted_label == positive_label)
    fp = sum(1 for p in preds if p.ground_truth != positive_label and p.predicted_label == positive_label)
    fn = sum(1 for p in preds if p.ground_truth == positive_label and p.predicted_label != positive_label)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _plot_f1_by_subtype(plt, predictions: list[BenchmarkPrediction], out: Path) -> list[str]:
    created: list[str] = []
    for task, positive in [(TASK_SENSITIVITY, SENSITIVE), (TASK_COARSE_ALIGNMENT, MISALIGNED_SENSITIVE)]:
        grouped: dict[str, dict[str, list[BenchmarkPrediction]]] = defaultdict(lambda: defaultdict(list))
        for pred in predictions:
            if pred.task == task and pred.status == STATUS_SUCCESS:
                subtype = str(pred.financial_subtype or pred.metadata.get("financial_subtype") or "unknown")
                grouped[pred.model_name][subtype].append(pred)
        for model_name, subtype_preds in grouped.items():
            values = [(subtype, _simple_f1(preds, positive)) for subtype, preds in sorted(subtype_preds.items())]
            values = [(name, score) for name, score in values if score is not None]
            if not values:
                continue
            names = [v[0] for v in values]
            scores = [float(v[1]) for v in values]
            fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.8), 4))
            ax.bar(range(len(names)), scores)
            ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
            ax.set_ylim(0, 1)
            ax.set_title(f"{task} F1 by subtype: {model_name}")
            fig.tight_layout()
            path = out / f"{_safe_name(task)}_{_safe_name(model_name)}_f1_by_subtype.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(str(path))
    return created


def _plot_runtime(plt, runtime_metrics: dict[str, Any], out: Path) -> list[str]:
    if not runtime_metrics:
        return []
    names = list(runtime_metrics.keys())
    latencies = [runtime_metrics[name].get("mean_latency_ms") or 0.0 for name in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    ax.bar(range(len(names)), latencies)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Runtime Comparison")
    fig.tight_layout()
    path = out / "runtime_mean_latency_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [str(path)]
