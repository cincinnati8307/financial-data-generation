# health-egress-poc

Parallel proof-of-concept for **synthetic health-private data generation** and **centroid-based sensitive health egress detection**. It follows the same workflow as the financial PoC, but the labels, templates, validators, and lexical hints are health-domain specific.

> **Synthetic-only warning:** generated examples are fabricated templates and optional paraphrases. They are not real patient data and should not be treated as representative of real users without further validation.

## Labels and Private Subtypes

Labels:

- `health_private`
- `non_private_health`
- `benign`

Private health subtypes:

- `diagnosis_condition`
- `medication_prescription`
- `lab_result`
- `appointment_visit`
- `insurance_claim`
- `medical_bill`
- `wearable_vitals`
- `vaccination_record`
- `mental_health_note`

The generator is catalog-driven from `src/health_egress_poc/template_catalog.json`. That file contains entity pools, private health scenarios, hard negatives, benign scenarios, and mixed-egress carriers. The catalog currently targets at least 8 scenarios and 12 templates per private subtype, 30+ hard-negative scenarios, 20+ benign scenarios, and 30+ mixed-egress carriers split across allowed and unexpected health-private sends.

Generated rows include Simplified Chinese, Chinese-English code-switching, Singapore/China health context, natural text, key-value, JSON, CSV-like rows, agent summaries, emails, chat snippets, OCR fragments, app notifications, form dumps, and spreadsheet-like rows. Full MRNs, full patient IDs, phone numbers, national IDs, credentials, tokens, and passwords are rejected. Masked synthetic references such as `MRN ****5678`, `patient ID ****1234`, and `患者编号尾号 9012` are allowed.

Private rows include traceability metadata in `meta.privacy_evidence`, `meta.sensitive_span`, and `meta.private_cues`. Public-health and benign rows include `meta.non_private_reason` so hard negatives can be audited instead of treated as generic negatives.

## Generate Synthetic Health Data

```bash
uv run python -m health_egress_poc.cli_generate \
  --out-dir data/health_generated \
  --private 1200 \
  --hard-negative 600 \
  --benign 600 \
  --mixed 400
```

Outputs:

- `data/health_generated/anchors_train.jsonl`
- `data/health_generated/anchors_validation.jsonl`
- `data/health_generated/egress_train.jsonl`
- `data/health_generated/egress_validation.jsonl`
- `data/health_generated/manifest.json`

The train/validation split groups by `meta.skeleton_id`, so rows created from the same template skeleton do not leak across splits.


## Optional Real-World Grounding

Grounding is opt-in. The default health generator remains synthetic-only. For the full architecture, privacy transformations, generic JSONL schema, licensing notes, and limitations, see [README_grounding.md](README_grounding.md).

Workflow:

1. Acquire public or deidentified health datasets manually.
2. Place them under `data/grounding/raw/`.
3. Prepare normalized grounding JSONL under `data/grounding/processed/`.
4. Validate grounding records.
5. Generate hybrid anchors and mixed egress.
6. Run dataset quality checks.
7. Build centroids.
8. Evaluate on source-group-isolated validation data.

Prepare health grounding files:

```bash
PYTHONPATH=src uv run python -m egress_grounding.cli_prepare --dataset nhanes --input data/grounding/raw/nhanes_lab.csv --output data/grounding/processed/nhanes.jsonl
PYTHONPATH=src uv run python -m egress_grounding.cli_prepare --dataset pubmed --input data/grounding/raw/pubmed.xml --output data/grounding/processed/pubmed.jsonl
PYTHONPATH=src uv run python -m egress_grounding.cli_validate --input data/grounding/processed/nhanes.jsonl --input data/grounding/processed/pubmed.jsonl
```

Generate grounded health data:

```bash
PYTHONPATH=src uv run python -m health_egress_poc.cli_generate \
  --out-dir data/health_generated \
  --private 1000 --hard-negative 500 --benign 500 --mixed 300 \
  --grounding data/grounding/processed/nhanes.jsonl \
  --grounding data/grounding/processed/pubmed.jsonl \
  --grounding-mode hybrid \
  --grounding-ratio 0.35
```

Then run quality checks and centroids as usual:

```bash
PYTHONPATH=src uv run python -m sensitive_egress_poc.dataset_quality \
  --input data/health_generated/anchors_train.jsonl \
  --report-out data/health_generated/quality_report.json \
  --checks redundancy,self_bleu,safety

PYTHONPATH=src uv run python -m health_egress_poc.cli_centroid \
  --train data/health_generated/anchors_train.jsonl \
  --validation data/health_generated/anchors_validation.jsonl \
  --out data/health_generated/centroids.json
```

Grounded rows remain synthetic and use `source=grounded_synthetic` or `source=grounded_public_negative`; provenance is stored only under `meta.grounding`. The split keeps all rows with the same `grounding:{dataset}:{source_group_id}` in one split.

## Optional Health LLM Augmentation

Dry-run augmentation is deterministic and does not call external APIs:

```bash
uv run python -m health_egress_poc.cli_augment \
  --input data/health_generated/anchors_train.jsonl \
  --output data/health_generated/anchors_train_augmented.jsonl \
  --max-inputs 20 \
  --paraphrases-per-example 3 \
  --provider dry-run \
  --include-original
```

OpenAI augmentation prints a heuristic token estimate before requests. Use `--estimate-only` to preview spend without sending data, or `--yes` for non-interactive runs:

```bash
uv run python -m health_egress_poc.cli_augment \
  --input data/health_generated/anchors_train.jsonl \
  --output data/health_generated/anchors_train_augmented.jsonl \
  --max-inputs 100 \
  --paraphrases-per-example 3 \
  --provider openai \
  --model gpt-5-nano \
  --include-original \
  --estimate-only
```

```bash
uv run python -m health_egress_poc.cli_augment \
  --input data/health_generated/anchors_train.jsonl \
  --output data/health_generated/anchors_train_augmented.jsonl \
  --max-inputs 100 \
  --paraphrases-per-example 3 \
  --provider openai \
  --model gpt-5-nano \
  --include-original
```

The augmenter only paraphrases `health_private` source rows. Local validators preserve the subtype and parent ID, reject unchanged text, remove duplicate text, and reject full identifiers or secrets introduced by the LLM.

## Quality Check

The shared quality checker can be reused for health datasets. For health data, prioritize redundancy, self-BLEU, and sampled LLM realism checks:

```bash
uv run python -m sensitive_egress_poc.dataset_quality \
  --input data/health_generated/anchors_train_augmented.jsonl \
  --report-out data/health_generated/anchors_train_augmented_quality.json \
  --clean-output data/health_generated/anchors_train_augmented_clean.jsonl \
  --checks redundancy,self_bleu,llm_realism,safety \
  --provider dry-run \
  --sample-size 50
```

For OpenAI LLM-as-judge, estimate token usage first:

```bash
uv run python -m sensitive_egress_poc.dataset_quality \
  --input data/health_generated/anchors_train_augmented.jsonl \
  --checks redundancy,self_bleu,llm_realism,safety \
  --provider openai \
  --model gpt-5-nano \
  --sample-size 50 \
  --estimate-only
```

Then run with confirmation or `--yes`:

```bash
uv run python -m sensitive_egress_poc.dataset_quality \
  --input data/health_generated/anchors_train_augmented.jsonl \
  --report-out data/health_generated/anchors_train_augmented_quality.json \
  --clean-output data/health_generated/anchors_train_augmented_clean.jsonl \
  --checks redundancy,self_bleu,llm_realism,safety \
  --provider openai \
  --model gpt-5-nano \
  --sample-size 50
```

## Build Centroids and Evaluate

Train on the clean augmented anchors if available. Otherwise use `anchors_train.jsonl`:

```bash
uv run python -m health_egress_poc.cli_centroid \
  --train data/health_generated/anchors_train_augmented_clean.jsonl \
  --validation data/health_generated/anchors_validation.jsonl \
  --out data/health_generated/centroids.json
```

The default model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. If it is unavailable, the classifier falls back to a deterministic local hash embedder for offline smoke runs. You can also pass `--model hash-only` for explicit offline testing.

## Demo Classification

```bash
uv run python -m health_egress_poc.cli_demo \
  --centroids data/health_generated/centroids.json \
  --text "Lab portal shows HbA1c=7.1, ref MRN ****5678."
```

For mixed egress:

```bash
uv run python -m health_egress_poc.cli_demo \
  --centroids data/health_generated/centroids.json \
  --text "Sprint notes: release ready. Clipboard: Lab portal shows HbA1c=7.1, ref MRN ****5678."
```

## Recommended Workflow

1. Expand or review `src/health_egress_poc/template_catalog.json` when changing domain coverage.
2. Generate template data with `health_egress_poc.cli_generate`.
3. Inspect diversity with the quality checker using `redundancy,self_bleu`.
4. Augment only the `health_private` anchor rows when you need more language variation; 2-4 paraphrases per strong source is usually enough.
5. Run quality checks again with redundancy, self-BLEU, and sampled LLM realism.
6. Train centroids on the cleaned anchor set and evaluate on the untouched validation set.
7. Use mixed egress rows to test whether policy decisions request approval when health-private payloads appear in unrelated outbound text.

## Limitations

- Synthetic health-private examples are useful for PoC development but are not a substitute for consented, privacy-safe evaluation corpora.
- Validators block obvious full identifiers and secrets, but they are not a complete PHI de-identification system.
- LLM-as-judge realism is sampled and subjective; use it as a quality signal, not as ground truth.
- Embedding-centroid detection can miss subtle, obfuscated, or novel health-private content and can false-positive on public health discussion.
