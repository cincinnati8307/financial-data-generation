# sensitive-egress-financial-poc

Proof-of-concept for **synthetic Chinese-nuance private financial data generation** and **embedding-centroid sensitive financial data detection**. The project is intended for research on reducing sensitive data leakage by AI agents running on personal devices.

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

## Limitations

- Template-generated data is useful for PoC iteration but does not replace real-world evaluation with consented, privacy-safe corpora.
- Embedding-centroid detection can miss obfuscated or novel sensitive content and can false-positive on nearby financial language.
- The fallback hash embedder is for offline execution only and is not a quality substitute for multilingual semantic embeddings.
- Policy decisions should combine semantic detection with allowlists, user intent checks, structured secret scanners, and user approval flows.
- LLM augmentation quality depends on the selected model and prompt adherence; all augmented rows are still filtered by local validators.

## Tests

```bash
pytest
```
