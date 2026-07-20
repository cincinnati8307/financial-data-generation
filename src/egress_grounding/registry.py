from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .schemas import GroundingRecord

Adapter = Callable[[Path, int, int | None], list[GroundingRecord]]


def _load_adapters() -> dict[str, Adapter]:
    from . import adapters

    return {
        "cfpb": adapters.prepare_cfpb,
        "banking77": adapters.prepare_banking77,
        "berka": adapters.prepare_berka,
        "nhanes": adapters.prepare_nhanes,
        "pubmed": adapters.prepare_pubmed,
        "generic_jsonl": adapters.prepare_generic_jsonl,
    }


def adapter_names() -> list[str]:
    return sorted(_load_adapters())


def get_adapter(name: str) -> Adapter:
    adapters = _load_adapters()
    if name not in adapters:
        raise KeyError(f"unknown_adapter:{name}")
    return adapters[name]


def prepare_dataset(dataset: str, input_path: str | Path, *, seed: int = 1337, limit: int | None = None) -> list[GroundingRecord]:
    return get_adapter(dataset)(Path(input_path), seed, limit)
