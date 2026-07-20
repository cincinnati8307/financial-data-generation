# Sensitive Egress Privacy Detection Framework

Research framework for detecting sensitive data leakage in AI agent communications using synthetic data generation and embedding-based detection.

## Overview

This project addresses the challenge of AI agents inadvertently leaking private information (financial, health) when performing看似 benign tasks. The framework generates synthetic multilingual data and develops detection mechanisms based on semantic embeddings and policy alignment.

### Key Features

- **Multilingual Synthetic Data Generation**: Chinese-English mixed-language content with regional variations (Mainland China, Singapore)
- **Embedding-Centroid Detection**: Semantic similarity-based classification of sensitive content
- **Multi-Domain Support**: Financial and health privacy domains with extensible architecture
- **Benchmarking Suite**: Comprehensive evaluation against privacy baselines (PII detection, CAPID, LLM judges, etc.)
- **Privacy-Preserving Grounding**: Optional integration with public/deidentified datasets for improved realism

## Architecture

### Core Components

```
src/
├── sensitive_egress_poc/      # Financial domain implementation
│   ├── synthetic_generator.py # Template-based synthetic data generation
│   ├── centroid_classifier.py # Embedding-based detection
│   ├── cli_*.py               # Command-line interfaces
│   └── benchmark/             # Evaluation framework
│
├── health_egress_poc/         # Health domain implementation
│   ├── synthetic_generator.py # Health-specific data generation
│   └── ...                    # (mirrors financial structure)
│
├── egress_grounding/          # Grounding data integration
│   ├── cli_prepare.py         # Dataset preparation pipeline
│   ├── cli_validate.py        # Quality validation
│   └── store.py               # Grounding data management
│
└── utils/                     # Shared utilities
    └── vis.py                 # Visualization tools
```

### Data Flow

1. **Generation**: Create synthetic sensitive/non-sensitive examples
2. **Grounding** (optional): Enrich with public/deidentified data patterns
3. **Enhancement**: LLM augmentation for diversity
4. **Detection**: Build embedding centroids and classify content
5. **Evaluation**: Benchmark against privacy detection baselines

## Quick Start

### Installation

```bash
# Basic dependencies
pip install -r requirements.txt

# Benchmark dependencies (optional)
pip install -r requirements-benchmark.txt
```

### Generate Financial Data

```bash
python -m sensitive_egress_poc.cli_generate \
  --out-dir data/financial_generated \
  --private 1000 --hard-negative 500 --benign 500 --mixed 300
```

### Build Detection Centroids

```bash
python -m sensitive_egress_poc.cli_centroid \
  --train data/financial_generated/anchors_train.jsonl \
  --validation data/financial_generated/anchors_validation.jsonl
```

### Test Classification

```bash
python -m sensitive_egress_poc.cli_demo \
  --centroids data/financial_generated/centroids.json \
  --text "我 DBS 账户里还剩 SGD 4,200。"
```

## Domain-Specific Guides

### Financial Domain

- **Labels**: `financial_private`, `non_private_financial`, `benign`
- **Subtypes**: bank_balance, transaction, salary_income, card_payment, loan_debt, invoice_receipt, investment, tax, wallet_payment
- **Documentation**: See this README and [README_grounding.md](README_grounding.md)

### Health Domain

- **Labels**: `health_private`, `non_private_health`, `benign`
- **Subtypes**: diagnosis, medication, lab_result, appointment, insurance, bill, wearable_vitals, vaccination, mental_health
- **Documentation**: See [README_health.md](README_health.md)

## Threat Model

The system models an AI agent that performs benign tasks (sending work notes, scheduling meetings) while potentially including sensitive information in outgoing communications. The detector analyzes outgoing text chunks and flags content that contains sensitive financial or health information requiring user approval.

**Key Assumptions**:
- Agent operates with benign intent but may include sensitive context
- Sensitive data appears in natural language rather than structured formats
- Detection should preserve usefulness of legitimate communications
- False positives (blocking legitimate content) are preferable to false negatives (allowing leaks)

## Grounding Architecture

Grounding provides an optional path to improve synthetic data realism by integrating patterns from public/deidentified datasets:

1. **Data Collection**: Acquire public financial/health datasets manually
2. **Normalization**: Convert to unified JSONL schema under `data/grounding/processed/`
3. **Hybrid Generation**: Mix synthetic patterns with grounded data
4. **Provenance Tracking**: Grounded rows clearly labeled in metadata

**Safety**: Grounded content remains synthetic with provenance tracking; no raw text copying from source datasets.

See [README_grounding.md](README_grounding.md) for detailed architecture and safety considerations.

## Evaluation and Benchmarking

### Benchmark Tasks

- **Task A (Sensitivity Detection)**: Classify content as `sensitive` or `non_sensitive`
- **Task B (Policy Alignment)**: Determine if sensitive content in mixed communications is aligned with user intent

### Supported Methods

- **Centroid**: Semantic embedding similarity (baseline)
- **PII Detection**: Entity-based private information detection
- **PII Reranker**: PII + query-evidence relevance scoring
- **CAPID**: Query-aware privacy model (English-only)
- **LLM Judge**: Multilingual privacy evaluation model
- **OPF + Granite**: Composed baseline using OpenAI Privacy Filter + Granite Guardian

### Running Benchmarks

```bash
# Full benchmark suite
python -m sensitive_egress_poc.cli_benchmark \
  --anchor-validation data/financial_generated/anchors_validation.jsonl \
  --egress-train data/financial_generated/egress_train.jsonl \
  --egress-validation data/financial_generated/egress_validation.jsonl \
  --centroids data/financial_generated/centroids.json \
  --methods centroid pii pii_reranker capid llm_judge \
  --output-dir results/financial_benchmark
```

## Current Limitations and Future Directions

### Known Limitations

- **Code Duplication**: ~95% duplicate code between financial and health implementations
- **Synthetic Bias**: Template-generated data may not capture real-world complexity
- **Detection Precision**: Embedding-based methods can miss obfuscated threats
- **Scalability**: Adding new privacy domains requires significant duplication

### Planned Improvements

1. **Unified Core Framework**: Extract common logic into shared components
2. **Plugin Architecture**: Domain-specific behavior via configuration/plugins
3. **Enhanced Validation**: More sophisticated quality checking for generated data
4. **Multi-Modal Support**: Extend beyond text to structured data, images
5. **Real-World Evaluation**: Partner with organizations for consented real-world data testing

## Research Use Cases

- **Pre-computation Risk Assessment**: Evaluate AI agents for data leakage patterns
- **Privacy Policy Compliance**: Verify systems respect sensitive content rules
- **Retrieval-Augmented Generation Safety**: Protect against RAG system leaks
- **Cross-Border Data Transfer**: Monitor for unauthorized sensitive data movement

## Contributing

### Development Guidelines

- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation for new features
- Consider both domains when making changes

### Running Tests

```bash
pytest
```

## Safety and Ethics

**Important**: This framework generates synthetic data for research purposes only. Generated examples are templates and paraphrases, not real leaked data. They should not be treated as representative of real users without validation.

**Research Integrity**: 
- Use only for legitimate privacy research
- Must complement real-world evaluation with consented corpora
- Intended for defensive security research only
- Avoid adversarial applications that target vulnerabilities

## License and Citation

See project license file. When using this framework for research, please cite appropriately according to your institution's guidelines.

## Contact and Support

For questions, issues, or research collaboration inquiries, please use the project's issue tracker or contact mechanisms provided in your organization.

---

**Note**: This framework is actively under development. APIs and interfaces may change as improvements are made.