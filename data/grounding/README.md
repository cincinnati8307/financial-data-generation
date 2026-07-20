# Grounding Data Area

Place manually acquired public or deidentified source datasets under `data/grounding/raw/`, then write normalized privacy-safe JSONL files under `data/grounding/processed/`.

Raw and processed data files in these directories are ignored by git. Only this README and `.gitkeep` placeholders are tracked.

Use:

```bash
PYTHONPATH=src python -m egress_grounding.cli_prepare --dataset cfpb --input data/grounding/raw/cfpb.csv --output data/grounding/processed/cfpb.jsonl
PYTHONPATH=src python -m egress_grounding.cli_validate --input data/grounding/processed/cfpb.jsonl
```
