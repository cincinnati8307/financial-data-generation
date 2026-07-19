from __future__ import annotations

from typing import Protocol

from .schemas import BenchmarkPrediction


class ModelUnavailable(RuntimeError):
    pass


class BenchmarkModel(Protocol):
    name: str
    loading_time_s: float

    def predict_sensitivity(self, rows: list[dict]) -> list[BenchmarkPrediction]:
        ...

    def predict_alignment(self, rows: list[dict]) -> list[BenchmarkPrediction]:
        ...
