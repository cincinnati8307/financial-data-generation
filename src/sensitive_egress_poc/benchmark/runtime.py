from __future__ import annotations

import math
import os
import resource
import time
from statistics import median
from typing import Any

from .schemas import BenchmarkPrediction, STATUS_SUCCESS


def perf_counter() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def current_cpu_memory_mb() -> float | None:
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    # Linux reports KB; macOS reports bytes. This workspace is Linux, but keep sane fallback.
    if rss > 10_000_000_000:
        return rss / (1024 * 1024)
    return rss / 1024


def current_gpu_memory_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        return None


def parameter_count(model: Any) -> int | None:
    obj = getattr(model, "model", None) or getattr(model, "scorer", None) or model
    try:
        params = obj.parameters()
    except Exception:
        return None
    try:
        return int(sum(p.numel() for p in params))
    except Exception:
        return None


def directory_size_mb(path: str | os.PathLike[str] | None) -> float | None:
    if not path:
        return None
    total = 0
    try:
        for base, _, files in os.walk(path):
            for name in files:
                fp = os.path.join(base, name)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    except Exception:
        return None
    return total / (1024 * 1024)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def summarize_method_runtime(model_name: str, predictions: list[BenchmarkPrediction], loading_time_s: float | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    pred_for_model = [p for p in predictions if p.model_name == model_name]
    latencies = [float(p.runtime_ms) for p in pred_for_model if p.runtime_ms is not None]
    total_inference_s = sum(latencies) / 1000.0 if latencies else None
    successful = sum(1 for p in pred_for_model if p.status == STATUS_SUCCESS)
    return {
        "model_name": model_name,
        "loading_time_s": loading_time_s,
        "total_inference_time_s": total_inference_s,
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "examples_per_second": (successful / total_inference_s) if total_inference_s and total_inference_s > 0 else None,
        "successful_predictions": successful,
        "total_predictions": len(pred_for_model),
        "peak_cpu_memory_mb": current_cpu_memory_mb(),
        "peak_gpu_memory_mb": current_gpu_memory_mb(),
        "parameter_count": (extra or {}).get("parameter_count"),
        "artifact_storage_size_mb": (extra or {}).get("artifact_storage_size_mb"),
        "metadata": extra or {},
    }
