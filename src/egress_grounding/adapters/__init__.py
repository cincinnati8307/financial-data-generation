from __future__ import annotations

import csv
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from egress_grounding.overlap import has_suspicious_overlap
from egress_grounding.sanitization import hash_id, perturbed_number, rounded_number, safe_month, safe_str
from egress_grounding.schemas import GroundingRecord


def _limit(records: list[GroundingRecord], limit: int | None) -> list[GroundingRecord]:
    return records if limit is None else records[: max(0, limit)]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|") if sample.strip() else csv.excel
        return [dict(row) for row in csv.DictReader(f, dialect=dialect)]


def _read_jsonish_rows(path: Path) -> list[dict[str, Any]]:
    text = _read_text(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "data", "examples"):
            if isinstance(payload.get(key), list):
                return [row for row in payload[key] if isinstance(row, dict)]
    return []


def _first(row: dict[str, Any], names: Iterable[str]) -> str:
    lowered = {str(k).lower().strip(): v for k, v in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in {None, ""}:
            return str(value)
    return ""


def _record(
    *,
    dataset: str,
    domain: str,
    role: str,
    label: str,
    subtype: str,
    group_seed: Any,
    facts: dict[str, Any],
    region: str = "global",
    tags: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> GroundingRecord:
    group = hash_id(group_seed, salt=f"{dataset}:group", prefix="grp")
    record = GroundingRecord(
        id=hash_id(group_seed, salt=f"{dataset}:record", prefix=dataset),
        dataset=dataset,
        domain=domain,
        role=role,
        label=label,
        subtype=subtype,
        source_group_id=group,
        facts={k: v for k, v in facts.items() if v not in {None, ""}},
        region=region,
        tags=tags or [],
        meta={"privacy_transform": "hashed_ids_and_fact_bundles", **(meta or {})},
    )
    record.validate()
    return record


def _financial_subtype_from_text(product: str, issue: str = "") -> str:
    text = f"{product} {issue}".lower()
    if any(token in text for token in ("mortgage", "loan", "debt", "student", "auto")):
        return "loan_debt"
    if any(token in text for token in ("credit card", "prepaid card", "card")):
        return "card_payment"
    if any(token in text for token in ("checking", "savings", "deposit", "bank account", "balance")):
        return "bank_balance"
    if any(token in text for token in ("money transfer", "virtual currency", "wallet", "payment")):
        return "wallet_payment"
    if any(token in text for token in ("tax", "income")):
        return "tax"
    return "transaction"


def _banking_private_subtype(intent: str) -> str | None:
    text = intent.lower().replace("_", " ")
    if "balance" in text:
        return "bank_balance"
    if "card" in text or "cash withdrawal" in text or "cash withdraw" in text:
        return "card_payment"
    if "transfer" in text or "beneficiary" in text or "standing order" in text or "direct debit" in text:
        return "transaction"
    if "loan" in text:
        return "loan_debt"
    return None


def prepare_cfpb(input_path: Path, seed: int, limit: int | None = None) -> list[GroundingRecord]:
    records: list[GroundingRecord] = []
    for index, row in enumerate(_read_csv_rows(input_path)):
        complaint_id = _first(row, ["Complaint ID", "complaint_id", "id"]) or f"row-{index}"
        product = safe_str(_first(row, ["Product", "product"]))
        subproduct = safe_str(_first(row, ["Sub-product", "sub_product", "subproduct"]))
        issue = safe_str(_first(row, ["Issue", "issue"]))
        state = safe_str(_first(row, ["State", "state"]), max_len=24)
        date = _first(row, ["Date received", "date_received", "date"])
        narrative = _first(row, ["Consumer complaint narrative", "complaint_what_happened", "narrative"])
        facts = {
            "account_type": subproduct or product or "consumer account",
            "expense_type": issue or "consumer finance issue",
            "loan_type": product or "consumer loan",
            "month": safe_month(date),
        }
        if narrative and has_suspicious_overlap(" ".join(str(v) for v in facts.values()), [narrative]):
            continue
        subtype = _financial_subtype_from_text(product, issue)
        records.append(
            _record(
                dataset="cfpb",
                domain="financial",
                role="private_candidate",
                label="financial_private",
                subtype=subtype,
                group_seed=complaint_id,
                facts=facts,
                region="us" if state else "global",
                tags=["categorical", "complaint"],
                meta={"source_id_hash": hash_id(complaint_id, salt="cfpb:complaint_id", prefix="cid")},
            )
        )
    return _limit(records, limit)


def prepare_banking77(input_path: Path, seed: int, limit: int | None = None) -> list[GroundingRecord]:
    rows = _read_csv_rows(input_path) if input_path.suffix.lower() in {".csv", ".tsv"} else _read_jsonish_rows(input_path)
    records: list[GroundingRecord] = []
    for index, row in enumerate(rows):
        intent = safe_str(_first(row, ["category", "intent", "label", "class"]), max_len=64)
        if not intent:
            continue
        utterance = _first(row, ["text", "utterance", "query", "example"])
        private_subtype = _banking_private_subtype(intent)
        if private_subtype:
            label = "financial_private"
            role = "private_candidate"
            subtype = private_subtype
            facts = {
                "expense_type": intent.replace("_", " "),
                "account_type": "banking app account",
                "app": "banking support app",
            }
        else:
            label = "non_private_financial"
            role = "public_negative"
            subtype = "*"
            facts = {
                "expense_type": intent.replace("_", " "),
                "merchant": "banking support topic",
                "app": "public banking FAQ",
            }
        if utterance and has_suspicious_overlap(" ".join(str(v) for v in facts.values()), [utterance], english_span=7):
            continue
        records.append(
            _record(
                dataset="banking77",
                domain="financial",
                role=role,
                label=label,
                subtype=subtype,
                group_seed=f"{intent}:{index}",
                facts=facts,
                tags=["intent"],
                meta={"intent": intent},
            )
        )
    return _limit(records, limit)


def _numeric_values(rows: list[dict[str, str]], names: Iterable[str]) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = _first(row, names)
        try:
            out.append(float(str(value).replace(",", "")))
        except (TypeError, ValueError):
            continue
    return out


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _top_value(rows: list[dict[str, str]], names: Iterable[str], default: str) -> str:
    values = [safe_str(_first(row, names), max_len=40) for row in rows if _first(row, names)]
    return Counter(values).most_common(1)[0][0] if values else default


def _read_berka_tables(input_path: Path) -> dict[str, list[dict[str, str]]]:
    paths = sorted(input_path.glob("*.asc")) if input_path.is_dir() else [input_path]
    tables: dict[str, list[dict[str, str]]] = {}
    for path in paths:
        tables[path.stem.lower()] = _read_csv_rows(path)
    return tables


def prepare_berka(input_path: Path, seed: int, limit: int | None = None) -> list[GroundingRecord]:
    rng = random.Random(seed)
    tables = _read_berka_tables(input_path)
    records: list[GroundingRecord] = []
    account_rows = tables.get("account", [])
    if account_rows:
        records.append(
            _record(
                dataset="berka",
                domain="financial",
                role="distribution",
                label="financial_private",
                subtype="bank_balance",
                group_seed=f"account:{len(account_rows)}",
                facts={
                    "account_type": _top_value(account_rows, ["frequency"], "statement account"),
                    "amount": f"CNY {int(rounded_number(perturbed_number(len(account_rows) * 850, rng, pct=0.02, step=100), step=100)):,}",
                    "month": "本月",
                },
                tags=["aggregate"],
                meta={"aggregate_rows": len(account_rows), "source_table": "account"},
            )
        )
    trans_rows = tables.get("trans", []) or tables.get("transaction", [])
    if trans_rows:
        amount = _median(_numeric_values(trans_rows, ["amount", "balance"]))
        records.append(
            _record(
                dataset="berka",
                domain="financial",
                role="distribution",
                label="financial_private",
                subtype="transaction",
                group_seed=f"trans:{len(trans_rows)}:{_top_value(trans_rows, ['operation', 'type'], 'transaction')}",
                facts={
                    "expense_type": _top_value(trans_rows, ["operation", "type", "k_symbol"], "transaction"),
                    "amount": f"CNY {int(rounded_number(perturbed_number(amount or 1000, rng, pct=0.03, step=50), step=50)):,}",
                    "merchant": "bank transaction category",
                    "month": "本月",
                },
                tags=["aggregate"],
                meta={"aggregate_rows": len(trans_rows), "source_table": "trans"},
            )
        )
    loan_rows = tables.get("loan", [])
    if loan_rows:
        amount = _median(_numeric_values(loan_rows, ["amount"]))
        records.append(
            _record(
                dataset="berka",
                domain="financial",
                role="distribution",
                label="financial_private",
                subtype="loan_debt",
                group_seed=f"loan:{len(loan_rows)}",
                facts={
                    "loan_type": "consumer loan",
                    "amount": f"CNY {int(rounded_number(perturbed_number(amount or 50000, rng, pct=0.03, step=500), step=500)):,}",
                    "month": "本月",
                },
                tags=["aggregate"],
                meta={"aggregate_rows": len(loan_rows), "source_table": "loan"},
            )
        )
    return _limit(records, limit)


def _load_nhanes_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".xpt":
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - depends on optional pandas
            raise RuntimeError("pandas is required for NHANES XPT input") from exc
        return [dict(row) for row in pd.read_sas(path, format="xport").to_dict(orient="records")]
    return _read_csv_rows(path)


def _float(row: dict[str, Any], names: Iterable[str]) -> float | None:
    value = _first(row, names)
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return number


def _plausible(number: float | None, low: float, high: float) -> bool:
    return number is not None and low <= number <= high


def prepare_nhanes(input_path: Path, seed: int, limit: int | None = None) -> list[GroundingRecord]:
    rng = random.Random(seed)
    records: list[GroundingRecord] = []
    for index, row in enumerate(_load_nhanes_rows(input_path)):
        seqn = _first(row, ["SEQN", "seqn", "participant_id"]) or f"row-{index}"
        systolic = _float(row, ["BPXSY1", "BPXSY2", "systolic"])
        diastolic = _float(row, ["BPXDI1", "BPXDI2", "diastolic"])
        glucose = _float(row, ["LBXGLU", "LBDGLUSI", "glucose"])
        a1c = _float(row, ["LBXGH", "HbA1c", "hba1c"])
        facts: dict[str, Any] = {"masked_ref": f"ref ending {hash_id(seqn, salt='nhanes:ref', prefix='')[-4:]}"}
        subtype = "lab_result"
        if _plausible(a1c, 3.0, 16.0):
            facts.update({"test_name": "HbA1c", "lab_value": f"{rounded_number(perturbed_number(a1c, rng, pct=0.01, step=0.1), step=0.1, digits=1):.1f}"})
        elif _plausible(glucose, 40.0, 500.0):
            mmol = glucose / 18.0 if glucose > 30 else glucose
            facts.update({"test_name": "glucose", "glucose": round(rounded_number(perturbed_number(mmol, rng, pct=0.02, step=0.1), step=0.1, digits=1), 1), "lab_value": f"{mmol:.1f}"})
        if _plausible(systolic, 70.0, 240.0) and _plausible(diastolic, 40.0, 140.0):
            facts["blood_pressure"] = f"{int(rounded_number(perturbed_number(systolic, rng, pct=0.01, step=2), step=2))}/{int(rounded_number(perturbed_number(diastolic, rng, pct=0.01, step=2), step=2))}"
            if "test_name" not in facts:
                facts["test_name"] = "blood pressure"
                facts["lab_value"] = facts["blood_pressure"]
                subtype = "wearable_vitals"
        if "test_name" not in facts:
            continue
        records.append(
            _record(
                dataset="nhanes",
                domain="health",
                role="private_candidate",
                label="health_private",
                subtype=subtype,
                group_seed=seqn,
                facts=facts,
                region="us",
                tags=["deidentified", "rounded_clinical"],
                meta={"source_id_hash": hash_id(seqn, salt="nhanes:seqn", prefix="seq")},
            )
        )
    return _limit(records, limit)


def _topic_from_health_text(text: str) -> dict[str, str]:
    lower = text.lower()
    if any(term in lower for term in ("vaccine", "vaccination", "booster", "immunization")):
        return {"vaccine": "public vaccine topic", "condition": "infectious disease prevention"}
    if any(term in lower for term in ("diabetes", "glucose", "hba1c")):
        return {"test_name": "HbA1c", "condition": "diabetes public research"}
    if any(term in lower for term in ("blood pressure", "hypertension")):
        return {"test_name": "blood pressure", "condition": "hypertension public research"}
    if any(term in lower for term in ("depression", "anxiety", "mental")):
        return {"condition": "mental health public research"}
    return {"condition": "public health research", "test_name": "population study"}


def prepare_pubmed(input_path: Path, seed: int, limit: int | None = None) -> list[GroundingRecord]:
    records: list[GroundingRecord] = []
    if input_path.suffix.lower() == ".jsonl":
        rows = _read_jsonish_rows(input_path)
        for index, row in enumerate(rows):
            title = _first(row, ["title", "article_title"])
            abstract = _first(row, ["abstract", "abstract_text"])
            pmid = _first(row, ["pmid", "id"]) or f"row-{index}"
            facts = _topic_from_health_text(f"{title} {abstract}")
            if has_suspicious_overlap(" ".join(facts.values()), [title, abstract]):
                continue
            records.append(
                _record(
                    dataset="pubmed",
                    domain="health",
                    role="public_negative",
                    label="non_private_health",
                    subtype="*",
                    group_seed=pmid,
                    facts=facts,
                    tags=["public_literature"],
                    meta={"source_id_hash": hash_id(pmid, salt="pubmed:pmid", prefix="pmid")},
                )
            )
            if limit is not None and len(records) >= limit:
                return records
        return records

    for _, elem in ET.iterparse(input_path, events=("end",)):
        if elem.tag.split("}")[-1] != "PubmedArticle":
            continue
        pmid = elem.findtext(".//PMID") or f"article-{len(records)}"
        title = "".join(elem.findtext(".//ArticleTitle") or "")
        abstract = " ".join(text.text or "" for text in elem.findall(".//AbstractText"))
        facts = _topic_from_health_text(f"{title} {abstract}")
        if not has_suspicious_overlap(" ".join(facts.values()), [title, abstract]):
            records.append(
                _record(
                    dataset="pubmed",
                    domain="health",
                    role="public_negative",
                    label="non_private_health",
                    subtype="*",
                    group_seed=pmid,
                    facts=facts,
                    tags=["public_literature"],
                    meta={"source_id_hash": hash_id(pmid, salt="pubmed:pmid", prefix="pmid")},
                )
            )
        elem.clear()
        if limit is not None and len(records) >= limit:
            break
    return records


def prepare_generic_jsonl(input_path: Path, seed: int, limit: int | None = None) -> list[GroundingRecord]:
    records: list[GroundingRecord] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(GroundingRecord.from_dict(json.loads(line)))
            if limit is not None and len(records) >= limit:
                break
    return records
