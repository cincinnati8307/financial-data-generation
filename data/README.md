# Generated Datasets

The generated datasets are separated by domain:

- `financial_generated/`: financial-private data.
- `health_generated/`: health-private data.
- `generated/`: legacy financial output path kept for compatibility.

Recommended cleaned anchor training files:

- `financial_generated/anchors_train_augmented_clean.jsonl`
- `health_generated/anchors_train_clean.jsonl`

Validation and mixed-egress files stay domain-specific in the same directories:

- `anchors_validation.jsonl`
- `egress_train.jsonl`
- `egress_validation.jsonl`
- `quality_report.json`
- `manifest.json`
