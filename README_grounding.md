# Real-World Grounding Layer

The `egress_grounding` package converts local public or deidentified datasets into normalized JSONL fact bundles. Financial and health generators can optionally sample those bundles, but they still render rows through local synthetic templates. Grounded examples are synthetic rows with provenance in `meta.grounding`; raw narratives, utterances, abstracts, participant rows, and source identifiers are not copied into generated examples.

Default generation remains `synthetic-only`.

## Architecture

- `src/egress_grounding/schemas.py`: `GroundingRecord` schema, enum validation, JSONL read/write.
- `store.py`: load, filter, index, and sample records by domain, label, subtype, role, and dataset.
- `registry.py` and `adapters/`: offline dataset conversion for CFPB, Banking77, Berka, NHANES, PubMed, and already-normalized generic JSONL.
- `sanitization.py`: hashed IDs, unsafe identifier checks, and numeric rounding/perturbation helpers.
- `overlap.py`: English token-span and CJK character-span source-copy checks.
- `cli_prepare.py`: convert local source files to normalized grounding JSONL.
- `cli_validate.py`: validate normalized JSONL and fail non-zero on invalid records.

## Normalized Record Shape

```json
{
  "id": "generic_fin_1",
  "dataset": "generic_jsonl",
  "domain": "financial",
  "role": "private_candidate",
  "label": "financial_private",
  "subtype": "bank_balance",
  "source_group_id": "grp_generic_fin_1",
  "facts": {
    "bank": "DBS",
    "amount": "SGD 4,200",
    "masked_account": "account ****4321"
  },
  "region": "singapore_cn",
  "tags": ["fixture"],
  "meta": {"privacy_transform": "fact_bundle"}
}
```

Valid domains are `financial` and `health`. Valid roles are `private_candidate`, `public_negative`, and `distribution`. The generator only consumes facts whose keys match existing template context fields.

## Data Layout

Manual acquisition only; no automatic downloads are performed.

```text
data/grounding/raw/        # local source files, ignored by git
data/grounding/processed/  # normalized JSONL, ignored by git
```

## Prepare Datasets

CFPB CSV:

```bash
PYTHONPATH=src uv run python -m egress_grounding.cli_prepare \
  --dataset cfpb \
  --input data/grounding/raw/cfpb_complaints.csv \
  --output data/grounding/processed/cfpb.jsonl
```

Banking77 CSV/JSON/JSONL:

```bash
PYTHONPATH=src uv run python -m egress_grounding.cli_prepare \
  --dataset banking77 \
  --input data/grounding/raw/banking77.csv \
  --output data/grounding/processed/banking77.jsonl
```

Berka directory containing `.asc` files:

```bash
PYTHONPATH=src uv run python -m egress_grounding.cli_prepare \
  --dataset berka \
  --input data/grounding/raw/berka \
  --output data/grounding/processed/berka.jsonl
```

NHANES CSV or XPT:

```bash
PYTHONPATH=src uv run python -m egress_grounding.cli_prepare \
  --dataset nhanes \
  --input data/grounding/raw/nhanes_lab.csv \
  --output data/grounding/processed/nhanes.jsonl
```

PubMed XML or JSONL:

```bash
PYTHONPATH=src uv run python -m egress_grounding.cli_prepare \
  --dataset pubmed \
  --input data/grounding/raw/pubmed.xml \
  --output data/grounding/processed/pubmed.jsonl
```

Validate one or more normalized files:

```bash
PYTHONPATH=src uv run python -m egress_grounding.cli_validate \
  --input data/grounding/processed/cfpb.jsonl \
  --input data/grounding/processed/banking77.jsonl \
  --report-out data/grounding/processed/grounding_validation.json
```

## Generate Grounded Data

Financial hybrid generation:

```bash
PYTHONPATH=src uv run python -m sensitive_egress_poc.cli_generate \
  --out-dir data/financial_generated \
  --private 1000 \
  --hard-negative 500 \
  --benign 500 \
  --mixed 300 \
  --grounding data/grounding/processed/cfpb.jsonl \
  --grounding data/grounding/processed/banking77.jsonl \
  --grounding data/grounding/processed/berka.jsonl \
  --grounding-mode hybrid \
  --grounding-ratio 0.35
```

Health hybrid generation:

```bash
PYTHONPATH=src uv run python -m health_egress_poc.cli_generate \
  --out-dir data/health_generated \
  --private 1000 \
  --hard-negative 500 \
  --benign 500 \
  --mixed 300 \
  --grounding data/grounding/processed/nhanes.jsonl \
  --grounding data/grounding/processed/pubmed.jsonl \
  --grounding-mode hybrid \
  --grounding-ratio 0.35
```

Use `--grounding-mode grounded-only` when you want missing compatible coverage to fail generation. Use repeatable `--grounding-dataset DATASET` to restrict loaded records by dataset name.

## Privacy Transformations

- CFPB: hashes complaint IDs and keeps categorical product/issue/month features only. Narratives are discarded.
- Banking77: maps intents to private candidates or public financial hard negatives. Utterances are discarded.
- Berka: emits aggregate distribution records from `.asc` files. Raw account/client/transaction/loan IDs are not retained.
- NHANES: hashes/discards `SEQN`, extracts alias-based clinical variables, filters implausible values, and rounds/perturbs fact values.
- PubMed: streams XML/JSONL into public-health hard-negative topics. Titles and abstracts are not retained.
- Generic JSONL: strictly validates already-normalized fact bundles and rejects raw-text fields or unsafe identifiers.

## Splitting And Manifests

Grounded rows use this split key:

```text
grounding:{dataset}:{source_group_id}
```

Rows without grounding keep the existing `meta.skeleton_id` fallback. This keeps variants from the same grounded source group in one split.

Generated manifests may include a `grounding` section with mode, requested and actual ratios, datasets, grounded counts, fallback counts, source-role counts, `raw_text_copied=false`, and `grounded_in_public_or_deidentified_data=true`. `contains_real_personal_data` remains `false`.

## Quality, Centroids, Evaluation

Run the same local quality checks, build centroids, and evaluate on source-group-isolated validation data:

```bash
PYTHONPATH=src uv run python -m sensitive_egress_poc.dataset_quality \
  --input data/financial_generated/anchors_train.jsonl \
  --report-out data/financial_generated/quality_report.json \
  --checks redundancy,self_bleu,safety

PYTHONPATH=src uv run python -m sensitive_egress_poc.cli_centroid \
  --train data/financial_generated/anchors_train.jsonl \
  --validation data/financial_generated/anchors_validation.jsonl
```

For health, use `health_egress_poc.cli_centroid` with the generated health paths.

## Licensing And Access Notes

You are responsible for acquiring datasets under their own licenses and access terms. Keep raw files local, do not commit them, and review whether public/deidentified datasets are appropriate for your evaluation. This layer is not a de-identification system for arbitrary private data.

## Limitations

Grounding improves scenario coverage and hard negatives, but generated rows are still synthetic. The adapter checks reject obvious raw text and identifiers, not every possible privacy risk. Review normalized JSONL before generation, especially when adding a new generic source.
