# Sensitive Egress Privacy Detection Framework

A research framework for **synthetic Chinese-nuance private data generation** and **embedding-centroid sensitive data detection**. The project is intended for research on reducing sensitive data leakage by AI agents running on personal devices.

> **Synthetic-only warning:** generated examples are fabricated templates and paraphrases. They are not real leaked data and should not be treated as representative of real users without further validation.

## Threat model summary

The PoC models an AI agent that is asked to perform a benign task, such as sending work notes, while an outgoing chunk unexpectedly contains private financial data. The detector compares outgoing text against financial subtype centroids and negative/benign centroids to decide whether the chunk should require approval.

## Generated labels and subtypes

Labels:

- `financial_private`
- `non_private_financial`
- `benign`

Private financial subtypes:

- `bank_balance`
- `transaction`
- `salary_income`
- `card_payment`
- `loan_debt`
- `invoice_receipt`
- `investment`
- `tax`
- `wallet_payment`

The generator includes Simplified Chinese, Chinese-English code-switching, Mainland China and Singapore Chinese context, key-value/JSON/CSV-like rows, agent summaries, and mixed email/chat styles. Full account and card numbers are forbidden; only masked references such as `尾号 1234`, `card ending 1234`, and `账号 ****5678` are allowed.


## Health domain variant

A parallel health-private data workflow is available under `src/health_egress_poc`. It generates synthetic `health_private`, `non_private_health`, and `benign` anchor rows plus mixed-egress examples for diagnosis, medication, lab result, appointment, insurance, bill, wearable vitals, vaccination, and mental-health notes. See [README_health.md](README_health.md) for the health-specific commands and safety notes.

## Installation

```bash
pip install -r requirements.txt
```

For local source-layout execution without installation, run commands with `PYTHONPATH=src` or install the project in editable mode.


## Generated Data Locations

Domain-separated outputs live under `data/`:

- `data/financial_generated/` for financial-private data.
- `data/health_generated/` for health-private data.
- `data/generated/` is the legacy financial output path kept for compatibility.

## Generate synthetic data

```bash
python -m sensitive_egress_poc.cli_generate \
  --out-dir data/financial_generated \
  --private 1000 \
  --hard-negative 500 \
  --benign 500 \
  --mixed 300
```

Outputs:

- `data/financial_generated/anchors_train.jsonl`
- `data/financial_generated/anchors_validation.jsonl`
- `data/financial_generated/egress_train.jsonl`
- `data/financial_generated/egress_validation.jsonl`
- `data/financial_generated/manifest.json`


## Optional Real-World Grounding

Grounding is opt-in. The default generator remains synthetic-only. For the full architecture, privacy transformations, generic JSONL schema, licensing notes, and limitations, see [README_grounding.md](README_grounding.md).

Workflow:

1. Acquire public or deidentified datasets manually.
2. Place them under `data/grounding/raw/`.
3. Prepare normalized grounding JSONL under `data/grounding/processed/`.
4. Validate grounding records.
5. Generate hybrid anchors and mixed egress.
6. Run dataset quality checks.
7. Build centroids.
8. Evaluate on source-group-isolated validation data.

Prepare financial grounding files:

```bash
PYTHONPATH=src uv run python -m egress_grounding.cli_prepare --dataset cfpb --input data/grounding/raw/cfpb_complaints.csv --output data/grounding/processed/cfpb.jsonl
PYTHONPATH=src uv run python -m egress_grounding.cli_prepare --dataset banking77 --input data/grounding/raw/banking77.csv --output data/grounding/processed/banking77.jsonl
PYTHONPATH=src uv run python -m egress_grounding.cli_prepare --dataset berka --input data/grounding/raw/berka --output data/grounding/processed/berka.jsonl
PYTHONPATH=src uv run python -m egress_grounding.cli_validate --input data/grounding/processed/cfpb.jsonl --input data/grounding/processed/banking77.jsonl --input data/grounding/processed/berka.jsonl
```

Generate grounded financial data:

```bash
PYTHONPATH=src uv run python -m sensitive_egress_poc.cli_generate \
  --out-dir data/financial_generated \
  --private 1000 --hard-negative 500 --benign 500 --mixed 300 \
  --grounding data/grounding/processed/cfpb.jsonl \
  --grounding data/grounding/processed/banking77.jsonl \
  --grounding data/grounding/processed/berka.jsonl \
  --grounding-mode hybrid \
  --grounding-ratio 0.35
```

Then run quality checks and centroids as usual:

```bash
PYTHONPATH=src uv run python -m sensitive_egress_poc.dataset_quality \
  --input data/financial_generated/anchors_train.jsonl \
  --report-out data/financial_generated/quality_report.json \
  --checks redundancy,self_bleu,safety

PYTHONPATH=src uv run python -m sensitive_egress_poc.cli_centroid \
  --train data/financial_generated/anchors_train.jsonl \
  --validation data/financial_generated/anchors_validation.jsonl
```

Grounded rows remain synthetic and use `source=grounded_synthetic` or `source=grounded_public_negative`; provenance is stored only under `meta.grounding`. The split keeps all rows with the same `grounding:{dataset}:{source_group_id}` in one split.

## Optional LLM augmentation

Dry-run augmentation is deterministic and does not call external APIs:

```bash
python -m sensitive_egress_poc.cli_augment \
  --input data/financial_generated/anchors_train.jsonl \
  --output data/financial_generated/anchors_train_augmented.jsonl \
  --max-inputs 20 \
  --provider dry-run \
  --include-original
```

OpenAI augmentation reads `OPENAI_API_KEY` from the environment and uses the official OpenAI Python SDK. Before sending requests, the CLI prints a heuristic token estimate and asks for confirmation. Use `--estimate-only` to preview without running, or `--yes` for non-interactive runs:

```bash
python -m sensitive_egress_poc.cli_augment \
  --input data/financial_generated/anchors_train.jsonl \
  --output data/financial_generated/anchors_train_augmented.jsonl \
  --max-inputs 100 \
  --paraphrases-per-example 6 \
  --provider openai \
  --model gpt-5-nano \
  --include-original \
  --estimate-only
```

```bash
python -m sensitive_egress_poc.cli_augment \
  --input data/financial_generated/anchors_train.jsonl \
  --output data/financial_generated/anchors_train_augmented.jsonl \
  --max-inputs 100 \
  --paraphrases-per-example 6 \
  --provider openai \
  --model gpt-5-nano \
  --include-original
```

The prompt asks for lightweight paraphrase candidates only. Local validators construct full rows, preserve the financial subtype and visible synthetic masked attributes, and reject real personal identifiers, secrets, tokens, passwords, and full financial identifiers.

## Build centroids and evaluate

```bash
python -m sensitive_egress_poc.cli_centroid \
  --train data/financial_generated/anchors_train_augmented.jsonl \
  --validation data/financial_generated/anchors_validation.jsonl
```

The default embedding model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. If `sentence-transformers` or the model is unavailable, the code falls back to a deterministic local hash embedder so the dry-run path still executes end-to-end.

Centroids are saved to:

```text
data/financial_generated/centroids.json
```

## Run demo classification

```bash
python -m sensitive_egress_poc.cli_demo \
  --centroids data/financial_generated/centroids.json \
  --text "我 DBS 账户里还剩 SGD 4,200。"
```

For a mixed-egress example:

```bash
python -m sensitive_egress_poc.cli_demo \
  --centroids data/financial_generated/centroids.json \
  --text "会议纪要：周五部署新版本。附带备注：我 DBS 账户里还剩 SGD 4,200。"
```

## Dataset schema

`SyntheticExample` rows contain fields such as:

```json
{
  "id": "fin_priv_xxx",
  "text": "我 DBS 账户里还剩 SGD 4,200。",
  "label": "financial_private",
  "subtype": "bank_balance",
  "region": "singapore_cn",
  "language": "zh_en",
  "format": "natural_sentence",
  "style": "zh_en_codeswitch",
  "sensitivity_level": "high",
  "source": "synthetic_template",
  "meta": {}
}
```

`MixedEgressExample` rows contain user intent, mixed outgoing text, expected categories, payload labels, unexpected categories, expected decision, financial subtype, and financial evidence.

## Benchmark against query-aware privacy baselines

This repository includes a reproducible benchmark for comparing the existing centroid detector with ready-to-use privacy baselines. It evaluates two separate tasks:

- **Task A: financial sensitivity detection** maps outgoing text to `sensitive` or `non_sensitive`. Anchor labels are mapped as `financial_private -> sensitive`, while `non_private_financial` and `benign` are mapped to `non_sensitive`.
- **Task B: coarse policy alignment** maps mixed-egress rows to `aligned_sensitive` or `misaligned_sensitive` using the existing `expected_decision` field. This is intentionally called `coarse_policy_alignment`, not strict semantic alignment, because current carrier metadata treats any financial payload as aligned when `expected_financial=true`.

Supported benchmark methods:

- `centroid`: wraps the existing centroid classifier for Task A and adds `centroid_query_similarity` for Task B without changing centroid construction.
- `pii`: local PII/private-financial detection only. It does not treat a plain `MONEY` entity as private financial data and reports Task B as unsupported.
- `pii_reranker`: PII/private-financial detection followed by query-evidence relevance scoring. Semantic relevance is useful evidence, but it is not equivalent to user authorization.
- `capid`: optional CAPID-compatible query-aware model that receives `question=user_intent` and `text=outgoing text`. Public CAPID checkpoints may be LoRA adapters and can require a compatible base model and Hugging Face access.
- `llm_judge`: optional prompted multilingual privacy judge. It is disabled unless an explicit local Hugging Face or OpenAI-compatible provider is selected.
- `qwen3guard`: optional Qwen3Guard-Gen-8B Chinese sensitivity detector, with deterministic Chinese privacy rules as an ensemble signal.
- `shieldlm`: optional ShieldLM-6B-ChatGLM3 Chinese sensitivity detector, using the same Chinese-aware contract.
- `opf_granite`: strict composed baseline using OpenAI Privacy Filter followed by Granite Guardian only when an account-number span is detected.
- `opf_granite_oracle`: Granite Guardian diagnostic using dataset `financial_evidence` for Task B only. It is not an end-to-end privacy detector and reports Task A as unsupported.

Install the existing project dependencies first:

```bash
pip install -r requirements.txt
```

Optional benchmark extras are separated so heavyweight model packages are not part of the minimum installation:

```bash
pip install -r requirements-benchmark.txt
```

Lightweight centroid-only run:

```bash
python -m sensitive_egress_poc.cli_benchmark \
  --anchor-validation data/financial_generated/anchors_validation.jsonl \
  --egress-train data/financial_generated/egress_train.jsonl \
  --egress-validation data/financial_generated/egress_validation.jsonl \
  --centroids data/financial_generated/centroids.json \
  --methods centroid \
  --output-dir results/centroid_only
```

Complete benchmark command. Optional methods skip cleanly when dependencies, local models, or API credentials are unavailable:

Chinese-model benchmark examples. The models are downloaded unless `--offline` is set; the commands below use the shared Hugging Face cache under `/mnt/data/lambang/.cache/huggingface`. Running both methods in one command is supported, and each Chinese model is released after its benchmark slice to reduce peak memory use.

```bash
HF_HOME=/mnt/data/lambang/.cache/huggingface \
HF_HUB_CACHE=/mnt/data/lambang/.cache/huggingface/hub \
python -m sensitive_egress_poc.cli_benchmark \
  --anchor-validation data/financial_generated/anchors_validation.jsonl \
  --egress-validation data/financial_generated/egress_validation.jsonl \
  --centroids data/financial_generated/centroids.json \
  --methods qwen3guard shieldlm \
  --qwen3guard-model Qwen/Qwen3Guard-Gen-8B \
  --shieldlm-model thu-coai/ShieldLM-6B-chatglm3 \
  --shieldlm-model-base chatglm \
  --cache-dir /mnt/data/lambang/.cache/huggingface/hub \
  --output-dir results/chinese_privacy_benchmark
```

Use `--offline` for an already-downloaded ShieldLM smoke run:

```bash
HF_HOME=/mnt/data/lambang/.cache/huggingface \
HF_HUB_CACHE=/mnt/data/lambang/.cache/huggingface/hub \
python -m sensitive_egress_poc.cli_benchmark \
  --anchor-validation data/financial_generated/anchors_validation.jsonl \
  --egress-validation data/financial_generated/egress_validation.jsonl \
  --centroids data/financial_generated/centroids.json \
  --methods shieldlm \
  --shieldlm-model thu-coai/ShieldLM-6B-chatglm3 \
  --shieldlm-model-base chatglm \
  --cache-dir /mnt/data/lambang/.cache/huggingface/hub \
  --max-anchor-validation 1 \
  --max-egress-validation 1 \
  --offline \
  --output-dir results/shieldlm_smoke
```

The Chinese detectors expose `detect_privacy(text)` and `detect_batch(texts)` in `sensitive_egress_poc.chinese_privacy`. They run guard-model inference only for Chinese or mixed Chinese text, ensemble it with conservative Chinese financial privacy rules, and retain the raw model response in benchmark metadata for auditing. `create_app(detector)` optionally provides `/detect` and `/detect/batch` FastAPI endpoints when FastAPI is installed.

```bash
python -m sensitive_egress_poc.cli_benchmark \
  --anchor-train data/financial_generated/anchors_train_augmented_clean.jsonl \
  --anchor-validation data/financial_generated/anchors_validation.jsonl \
  --egress-train data/financial_generated/egress_train.jsonl \
  --egress-validation data/financial_generated/egress_validation.jsonl \
  --centroids data/financial_generated/centroids.json \
  --methods centroid pii pii_reranker capid llm_judge \
  --output-dir results/financial_benchmark \
  --device auto \
  --seed 42
```

Use offline mode to prevent model downloads and rely only on local files/caches:

```bash
python -m sensitive_egress_poc.cli_benchmark \
  --anchor-validation data/financial_generated/anchors_validation.jsonl \
  --egress-train data/financial_generated/egress_train.jsonl \
  --egress-validation data/financial_generated/egress_validation.jsonl \
  --centroids data/financial_generated/centroids.json \
  --methods centroid pii pii_reranker capid llm_judge \
  --output-dir results/offline_benchmark \
  --offline
```

For small API-backed smoke tests, cap the validation rows explicitly:

```bash
python -m sensitive_egress_poc.cli_benchmark \
  --anchor-validation data/financial_generated/anchors_validation.jsonl \
  --egress-validation data/financial_generated/egress_validation.jsonl \
  --centroids data/financial_generated/centroids.json \
  --methods llm_judge \
  --llm-provider openai-compatible \
  --llm-model gpt-5-nano \
  --max-anchor-validation 2 \
  --max-egress-validation 2 \
  --output-dir results/llm_judge_smoke
```

CAPID LoRA smoke run. This model is documented as English-only and may require Hugging Face access to the Llama base model:

```bash
python -m sensitive_egress_poc.cli_benchmark \
  --anchor-validation data/financial_generated/anchors_validation.jsonl \
  --egress-validation data/financial_generated/egress_validation.jsonl \
  --centroids data/financial_generated/centroids.json \
  --methods capid \
  --capid-model ponoma16/capid-llama8b-lora \
  --capid-base-model unsloth/Meta-Llama-3.1-8B-bnb-4bit \
  --capid-load-in-4bit \
  --capid-max-new-tokens 256 \
  --max-anchor-validation 1 \
  --max-egress-validation 1 \
  --output-dir results/capid_smoke
```

### OpenAI Privacy Filter + Granite Guardian

`opf_granite` composes two Hugging Face models:

1. OpenAI Privacy Filter detects explicit privacy spans according to a fixed taxonomy.
2. The strict composed baseline only continues to Granite Guardian when an `ACCOUNT_NUMBER` span is detected.
3. Granite Guardian evaluates whether the detected evidence is relevant to the user query using its `context_relevance` risk.

A Granite result of `Yes` means irrelevance risk was detected and maps to `misaligned_sensitive`. A result of `No` means the evidence is relevant and maps to `aligned_sensitive`.

OpenAI Privacy Filter native labels are normalized as:

```text
account_number  -> ACCOUNT_NUMBER
private_address -> PRIVATE_ADDRESS
private_email   -> PRIVATE_EMAIL
private_person  -> PRIVATE_PERSON
private_phone   -> PRIVATE_PHONE
private_url     -> PRIVATE_URL
private_date    -> PRIVATE_DATE
secret          -> SECRET
```

By default, only `ACCOUNT_NUMBER` is treated as financial-sensitive for this composed baseline. `PRIVATE_PERSON`, `PRIVATE_DATE`, `PRIVATE_EMAIL`, `PRIVATE_PHONE`, `PRIVATE_ADDRESS`, `PRIVATE_URL`, `SECRET`, and arbitrary currency or `MONEY` spans do not establish private financial information by themselves.

OpenAI Privacy Filter detects explicit identifiers according to its fixed taxonomy. It does not natively represent semantic financial facts such as salary, account balance, debt, investment value, tax income, or transaction amount without an identifier.

Limitations:

- OpenAI Privacy Filter is primarily an explicit PII detector.
- Its native taxonomy does not include semantic salary, balance, debt, investment, invoice, tax, or transaction-fact labels.
- Granite Guardian 3.2 is trained and tested primarily in English; samples are passed as-is and are not translated automatically.
- Relevance is not identical to authorization or necessity.
- The strict baseline may have low recall on the current semantic-financial dataset.
- The oracle-evidence variant is diagnostic only and must not be compared as a complete detector.

Strict smoke command:

```bash
python -m sensitive_egress_poc.cli_benchmark \
  --anchor-validation data/financial_generated/anchors_validation.jsonl \
  --egress-train data/financial_generated/egress_train.jsonl \
  --egress-validation data/financial_generated/egress_validation.jsonl \
  --centroids data/financial_generated/centroids.json \
  --methods opf_granite \
  --openai-privacy-filter-model openai/privacy-filter \
  --granite-guardian-model ibm-granite/granite-guardian-3.2-3b-a800m \
  --max-anchor-validation 2 \
  --max-egress-validation 2 \
  --output-dir results/opf_granite_smoke
```

Oracle diagnostic command:

```bash
python -m sensitive_egress_poc.cli_benchmark \
  --anchor-validation data/financial_generated/anchors_validation.jsonl \
  --egress-train data/financial_generated/egress_train.jsonl \
  --egress-validation data/financial_generated/egress_validation.jsonl \
  --centroids data/financial_generated/centroids.json \
  --methods opf_granite_oracle \
  --granite-guardian-model ibm-granite/granite-guardian-3.2-3b-a800m \
  --max-egress-validation 10 \
  --output-dir results/granite_oracle_diagnostic
```

Full query-aware comparison command:

```bash
python -m sensitive_egress_poc.cli_benchmark \
  --anchor-validation data/financial_generated/anchors_validation.jsonl \
  --egress-train data/financial_generated/egress_train.jsonl \
  --egress-validation data/financial_generated/egress_validation.jsonl \
  --centroids data/financial_generated/centroids.json \
  --methods centroid capid opf_granite opf_granite_oracle \
  --capid-model ponoma16/capid-llama8b-lora \
  --capid-base-model unsloth/Meta-Llama-3.1-8B-bnb-4bit \
  --capid-load-in-4bit \
  --openai-privacy-filter-model openai/privacy-filter \
  --granite-guardian-model ibm-granite/granite-guardian-3.2-3b-a800m \
  --output-dir results/query_aware_comparison
```

Run the alignment audit separately:

```bash
python -m sensitive_egress_poc.cli_alignment_audit \
  --input data/financial_generated/egress_validation.jsonl \
  --output results/alignment_audit.csv
```

The benchmark writes:

```text
results/<run_name>/
  config.json
  dataset_summary.json
  sensitivity_metrics.json
  alignment_metrics.json
  subgroup_metrics.csv
  predictions.jsonl
  runtime_metrics.json
  alignment_audit.csv
  skipped_models.json
  errors.jsonl
  plots/
```

Every prediction row includes the sample id, model name, task, user intent, outgoing text, financial evidence, subtype, carrier id, ground truth, prediction, scores, detected entities, status, error, and per-row runtime.

For Task B, **leakage rate** is the fraction of `misaligned_sensitive` examples predicted as `aligned_sensitive` or `non_sensitive`. **False-block rate** is the fraction of `aligned_sensitive` examples predicted as `misaligned_sensitive`.

Manual fine-grained alignment overrides can be added at:

```text
data/financial_generated/alignment_overrides.jsonl
```

Each row should look like:

```json
{"sample_id":"egress_000001","semantic_alignment_label":"aligned_sensitive","annotator":"manual","reason":"The tax-income evidence is necessary for the tax filing request."}
```

Fine-grained semantic metrics are reported separately from coarse policy metrics and use only manual overrides or explicitly subtype-constrained carrier examples. The benchmark never uses model-generated labels as ground truth and never mutates the original JSONL files.

Reproducibility notes:

- Keep validation files untouched; thresholds are tuned only with `egress_train.jsonl`.
- Use `--seed` to record the run seed in `config.json`.
- Use `--offline` when benchmark runs must not download models.
- Record `--pii-backend`, `--pii-model`, `--reranker-model`, `--capid-model`, `--capid-base-model`, `--llm-provider`, `--llm-model`, `--openai-privacy-filter-model`, `--openai-privacy-filter-threshold`, `--granite-guardian-model`, `--granite-max-new-tokens`, `--granite-load-in-4bit`, and `--granite-trust-remote-code` in the run config.

Benchmark limitations:

- Synthetic examples are useful for controlled comparison but do not replace evaluation on consented, privacy-safe real-world corpora.
- `coarse_policy_alignment` inherits category-level generator labels: a carrier with `expected_financial=true` may still contain a semantically unrelated financial subtype.
- The audit flags suspicious carrier/subtype pairs for review but does not silently relabel them.
- PII and DLP baselines may miss private financial facts that do not contain explicit account, card, or identifier-like spans.
- The strict OpenAI Privacy Filter + Granite Guardian baseline may miss semantic financial facts unless OpenAI Privacy Filter first detects a supported financial identifier.

## Limitations

- Template-generated data is useful for PoC iteration but does not replace real-world evaluation with consented, privacy-safe corpora.
- Embedding-centroid detection can miss obfuscated or novel sensitive content and can false-positive on nearby financial language.
- The fallback hash embedder is for offline execution only and is not a quality substitute for multilingual semantic embeddings.
- Policy decisions should combine semantic detection with allowlists, user intent checks, structured secret scanners, and user approval flows.
- LLM augmentation quality depends on the selected model and prompt adherence; all augmented rows are still filtered by local validators.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture, design principles, and component overview
- **[docs/COMPONENTS.md](docs/COMPONENTS.md)** - Detailed component documentation and code structure  
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Development guidelines, coding standards, and debugging
- **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** - Contribution guidelines and workflow
- **[docs/TECHNICAL_DEBT.md](docs/TECHNICAL_DEBT.md)** - Known issues, technical debt, and improvement areas

### Getting Started

1. **Installation**: See Installation section above
2. **Quick Start**: Generate synthetic data and test detection
3. **Understanding the System**: Read ARCHITECTURE.md for system design
4. **Development**: Follow DEVELOPMENT.md for coding guidelines
5. **Contributing**: Review CONTRIBUTING.md before submitting changes

## Tests

```bash
pytest
```
