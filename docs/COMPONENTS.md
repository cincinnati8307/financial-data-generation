# Code Structure and Components

This document provides a detailed overview of the codebase structure, component relationships, and key implementation details.

## Directory Structure

```
financial-data-generation/
├── src/
│   ├── sensitive_egress_poc/          # Financial domain implementation
│   │   ├── __init__.py
│   │   ├── synthetic_generator.py     # Core data generation logic
│   │   ├── centroid_classifier.py     # Embedding-based detection
│   │   ├── schemas.py                 # Data models and validation
│   │   ├── filters.py                 # Content validation and filtering
│   │   ├── llm_augmenter.py           # LLM-based data augmentation
│   │   ├── dataset_quality.py         # Quality assessment metrics
│   │   ├── io_utils.py                # I/O utilities
│   │   ├── template_catalog.json      # Financial templates and entity pools
│   │   ├── cli_generate.py            # Data generation CLI
│   │   ├── cli_centroid.py            # Centroid building CLI
│   │   ├── cli_augment.py             # Augmentation CLI
│   │   ├── cli_demo.py                # Classification demo CLI
│   │   ├── cli_benchmark.py           # Benchmark evaluation CLI
│   │   ├── cli_alignment_audit.py     # Policy alignment audit CLI
│   │   └── benchmark/                 # Benchmark implementations
│   │       ├── __init__.py
│   │       ├── methods.py             # Baseline method implementations
│   │       ├── metrics.py             # Evaluation metrics
│   │       ├── tasks.py               # Task definitions
│   │       └── run.py                 # Benchmark orchestration
│   │
│   ├── health_egress_poc/             # Health domain implementation
│   │   ├── __init__.py
│   │   ├── synthetic_generator.py     # Health-specific generation (95% duplicate)
│   │   ├── centroid_classifier.py     # Same as financial implementation
│   │   ├── schemas.py                 # Health-specific schemas
│   │   ├── filters.py                 # Same as financial implementation
│   │   ├── llm_augmenter.py           # Same as financial implementation
│   │   ├── template_catalog.json      # Health templates and entities
│   │   ├── improved_health_privacy_catalog_v2.json
│   │   ├── cli_generate.py            # Health generation CLI
│   │   ├── cli_centroid.py            # Same as financial implementation
│   │   ├── cli_augment.py             # Same as financial implementation
│   │   └── cli_demo.py                # Same as financial implementation
│   │
│   ├── egress_grounding/              # Grounding data integration
│   │   ├── __init__.py
│   │   ├── schemas.py                 # Grounding data schemas
│   │   ├── store.py                   # Grounding data storage and sampling
│   │   ├── registry.py                # Dataset registration
│   │   ├── generation.py              # Grounding-aware generation
│   │   ├── sanitization.py            # Data sanitization utilities
│   │   ├── overlap.py                 # Dataset overlap detection
│   │   ├── cli_prepare.py             # Dataset preparation CLI
│   │   ├── cli_validate.py            # Data validation CLI
│   │   └── adapters/                  # Dataset adapters
│   │       ├── __init__.py
│   │       ├── cfpb.py                # CFPB complaints adapter
│   │       ├── banking77.py           # Banking77 dataset adapter
│   │       └── berka.py               # Berka banking dataset adapter
│   │
│   └── utils/                         # Shared utilities
│       └── vis.py                     # Visualization tools
│
├── data/                              # Generated and source data
│   ├── financial_generated/           # Financial domain outputs
│   ├── health_generated/              # Health domain outputs
│   ├── generated/                     # Legacy financial output path
│   └── grounding/                     # Grounding dataset storage
│       ├── raw/                       # Original datasets
│       └── processed/                 # Processed JSONL files
│
├── results/                           # Benchmark results
│   └── [run_name]/
│       ├── config.json                # Run configuration
│       ├── dataset_summary.json       # Dataset statistics
│       ├── sensitivity_metrics.json   # Task A metrics
│       ├── alignment_metrics.json     # Task B metrics
│       ├── subgroup_metrics.csv       # Subtype-level metrics
│       ├── predictions.jsonl          # Per-example predictions
│       ├── runtime_metrics.json       # Performance metrics
│       ├── alignment_audit.csv        # Alignment audit results
│       ├── skipped_models.json        # Models that couldn't run
│       └── plots/                     # Visual results
│
├── tests/                             # Unit and integration tests
│   ├── test_generator.py
│   ├── test_filters.py
│   ├── test_centroid.py
│   ├── test_augment.py
│   ├── test_dataset_quality.py
│   ├── test_benchmark.py
│   ├── test_grounding.py
│   └── test_io.py
│
├── docs/                              # Additional documentation
│   └── COMPONENTS.md                  # This file
│
├── README.md                          # Main project README
├── README_grounding.md                # Grounding-specific documentation
├── README_health.md                   # Health domain documentation
├── ARCHITECTURE.md                    # Architecture overview
├── pyproject.toml                     # Project configuration
├── requirements.txt                   # Core dependencies
└── requirements-benchmark.txt         # Benchmark-specific dependencies
```

## Core Components

### 1. Data Generation (`synthetic_generator.py`)

**Purpose**: Generate synthetic private data examples with controlled variations.

**Key Classes**:
- `SyntheticFinancialGenerator`: Financial domain data generator
- `SyntheticHealthGenerator`: Health domain data generator (substantially duplicate)

**Core Functionality**:
```python
class SyntheticFinancialGenerator:
    def __init__(self, seed, grounding_store, grounding_mode, grounding_ratio, grounding_datasets)
    def private_example(self, subtype) -> SyntheticExample
    def hard_negative_example(self) -> SyntheticExample
    def benign_example(self) -> SyntheticExample
    def mixed_egress_example(self) -> MixedEgressExample
    def generate_private(self, n) -> List[dict]
    def generate_hard_negatives(self, n) -> List[dict]
    def generate_benign(self, n) -> List[dict]
    def generate_mixed(self, n) -> List[dict]
```

**Data Generation Flow**:
1. Template selection from catalog
2. Entity pool sampling (banks, merchants, etc.)
3. Context construction with regions and amounts
4. Optional grounding from public datasets
5. Text styling (casual, formal, code-switch, etc.)
6. Formatting (natural sentence, JSON, CSV, etc.)
7. Validation against safety rules
8. Metadata construction

**Templates**: JSON files containing:
- Entity pools (banks, merchants, medications, etc.)
- Scenario templates with placeholders
- Mixed carrier templates for egress simulation
- Regional and style variations

### 2. Detection and Classification (`centroid_classifier.py`)

**Purpose**: Build semantic embedding centroids for content classification.

**Key Classes**:
- `CentroidClassifier`: Base classifier (identical in both domains)

**Core Functionality**:
```python
class CentroidClassifier:
    def __init__(self, embedding_model)
    def fit(self, anchor_examples) -> None
    def predict(self, text) -> dict
    def get_similarity_scores(self, text) -> dict
    def save(self, path) -> None
    def load(path) -> 'CentroidClassifier'
```

**Embedding Models**:
- Primary: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Fallback: Deterministic hash-based embedder for offline use

**Classification Logic**:
1. Convert text to embedding vector
2. Calculate similarity to each centroid
3. Return label with highest similarity
4. Provide confidence scores and nearest neighbors

### 3. Data Validation (`filters.py`)

**Purpose**: Ensure generated content meets safety and quality standards.

**Key Functions**:
```python
def validate_synthetic_example(row: dict) -> tuple[bool, str]:
    # Comprehensive validation including:
    # - No full account/card numbers (only masked references)
    # - No real personal identifiers
    # - No secrets/passwords/tokens
    # - Proper subtype consistency
    # - Minimum text length
    # - Required field presence
```

**Validation Rules**:
- Block full financial identifiers (account numbers, cards)
- Allow only masked references (e.g., "尾号 1234", "****5678")
- Detect and block email addresses, phone numbers, addresses
- Block common secret patterns (API keys, tokens, passwords)
- Ensure content consistency with subtype

### 4. LLM Augmentation (`llm_augmenter.py`)

**Purpose**: Use LLMs to generate paraphrased variations for data diversity.

**Key Classes**:
- `LLMAugmenter`: Base augmentation interface (identical in both domains)

**Augmentation Process**:
1. Extract sensitive content from synthetic examples
2. Send to LLM for lightweight paraphrasing
3. Validate paraphrases locally
4. Reconstruct complete examples with preserved metadata
5. Apply strict filters to prevent realistic identifiers

**Safety**: 
- Only paraphrases sentence structure, not facts
- Preserves entity types (banks, subtypes, regions)
- Local validation rejects any real identifiers
- OpenAI API with confirmation prompts and cost estimates

### 5. Dataset Quality (`dataset_quality.py`)

**Purpose**: Assess quality and characteristics of generated datasets.

**Quality Checks**:
- **Redundancy**: Near-duplicate detection using embeddings
- **Self BLEU**: Measure of template diversity
- **Safety**: Comprehensive content safety analysis

```python
def evaluate_dataset(rows: List[dict], checks: List[str]) -> tuple[dict, List[dict]]
    # Returns quality metrics and cleaned dataset
```

### 6. Benchmark Framework (`benchmark/`)

**Purpose**: Compare detection methods against privacy baselines.

**Components**:

**Methods** (`methods.py`):
- `centroid_method`: Embedding similarity classification
- `pii_method`: PII/entity detection
- `pii_reranker_method`: PII + relevance scoring
- `capid_method`: Query-aware privacy model
- `llm_judge_method`: Prompted LLM evaluation
- `opf_granite_method`: OPF + Granite Guardian composition

**Tasks** (`tasks.py`):
- `sensitivity_detection_task`: Classify sensitive vs non-sensitive
- `coarse_policy_alignment_task`: Policy alignment assessment

**Metrics** (`metrics.py`):
- Accuracy, precision, recall, F1
- False positive/negative rates
- Leakage rate, false-block rate
- Per-subtype performance
- Runtime performance

**Orchestration** (`run.py`):
- Manages multiple methods and tasks
- Handles missing dependencies gracefully
- Produces comprehensive result reports

### 7. Grounding Integration (`egress_grounding/`)

**Purpose**: Integrate public/deidentified dataset patterns for realism.

**Key Components**:

**Store** (`store.py`):
```python
class GroundingStore:
    def load(paths: List[str]) -> 'GroundingStore'
    def sample(domain, label, subtype, roles, datasets, rng) -> GroundingRecord | None
    def coverage_report(requests) -> dict
```

**Data Flow**:
1. Raw datasets → Adapter normalization → Processed JSONL
2. Validation ensures schema compliance
3. Generation optionally samples from grounding data
4. Provenance tracked in metadata

**Safety**:
- No raw text copying
- Template-based rendering with controlled entities
- Provenance metadata for traceability
- Source group isolation in data splits

## Data Workflow

### Generation Pipeline

```
1. Initialize Generator
   ├── Load template catalog
   ├── Setup grounding store (optional)
   └── Configure random seed and parameters

2. Generate Examples
   ├── Private examples (sensitive content)
   ├── Hard negatives (non-private financial/health)
   ├── Benign examples (non-sensitive content)
   └── Mixed egress (sensitive content in benign context)

3. Split Data
   ├── 80% training set
   ├── 20% validation set
   └── Group by skeleton to prevent leakage

4. Output Files
   ├── anchors_train.jsonl
   ├── anchors_validation.jsonl
   ├── egress_train.jsonl
   ├── egress_validation.jsonl
   └── manifest.json
```

### Enhancement Pipeline

```
1. Load Generated Data
   └── anchors_train.jsonl

2. LLM Augmentation (optional)
   ├── Extract sensitive spans
   ├── Request paraphrases from LLM
   ├── Validate paraphrases locally
   ├── Reject any realistic identifiers
   └── Reconstruct with original metadata

3. Quality Assessment
   ├── Check for redundancy
   ├── Measure diversity (self-BLEU)
   └── Validate safety constraints

4. Output Enhanced Data
   └── anchors_train_augmented.jsonl
```

### Detection Pipeline

```
1. Build Centroids
   ├── Load training data
   ├── Generate embeddings
   ├── Calculate centroid vectors
   ├── Optimize thresholds on validation
   └── Save model artifacts

2. Classification
   ├── Load centroids
   ├── Embed incoming text
   ├── Calculate similarities
   └── Return predictions with scores
```

### Evaluation Pipeline

```
1. Run Benchmarks
   ├── Load test data
   ├── Apply each detection method
   ├── Collect predictions and metrics
   └── Generate comprehensive reports

2. Analyze Results
   ├── Method comparison
   ├── Subtype-level performance
   ├── Error analysis
   └── Runtime profiling
```

## Schema Definitions

### Synthetic Example Schema

```python
{
    "id": str,                    # Unique identifier
    "text": str,                  # Generated content
    "label": str,                 # financial_private/health_private/etc.
    "subtype": str,               # Specific subtype
    "region": str,                # singapore_cn/mainland_cn/global
    "language": str,              # zh/zh_en
    "format": str,                # natural_sentence/json/csv/etc.
    "style": str,                 # zh_casual/formal/etc.
    "sensitivity_level": str,     # high/none
    "source": str,                # synthetic_template/grounded_synthetic
    "meta": {                     # Extended metadata
        "scenario_id": str,
        "template_id": str,
        "skeleton_id": str,
        "grounding": {...}        # Optional grounding metadata
    }
}
```

### Mixed Egress Example Schema

```python
{
    "id": str,
    "user_intent": str,           # User's stated intent
    "text": str,                  # Mixed outgoing text
    "expected_categories": list,  # Expected content categories
    "payload_labels": list,       # Actual content present
    "unexpected_categories": list, # Categories that shouldn't appear
    "expected_decision": str,     # allow/request_approval
    "domain": str,                # mixed_egress
    "source": str,                # synthetic_mixed
    "financial_subtype": str,     # For financial domain
    "health_subtype": str,        # For health domain
    "financial_evidence": str,    # Sensitive content extracted
    "meta": {
        "carrier_id": str,
        "payload_skeleton_id": str,
        "skeleton_id": str,
        "expected_financial": bool, # Financial domain
        "expected_health": bool      # Health domain
    }
}
```

## Common Utilities

### IO Utilities (`io_utils.py`)

- `ensure_dir()`: Create directories safely
- `write_json()`: Write JSON files
- `write_jsonl()`: Write JSONL files
- `read_json()`: Read JSON files
- `read_jsonl()`: Read JSONL files

### Visualization (`vis.py`)

- Performance plot generation
- Comparison charts
- Error analysis visualizations

## Known Issues and Development Notes

### Code Duplication

The financial and health implementations share ~95% of code but are maintained separately:

**Duplicate Files**:
- `centroid_classifier.py` (identical)
- `filters.py` (identical)
- `llm_augmenter.py` (identical)
- `cli_centroid.py`, `cli_augment.py`, `cli_demo.py` (identical)
- `synthetic_generator.py` (similar structure, domain-specific content)

**Maintenance Burden**:
- Bug fixes must be applied twice
- New features require dual implementation
- Inconsistent updates across domains
- Higher testing overhead

### Performance Considerations

- **Embedding Generation**: Can be slow for large datasets; consider batching
- **Memory Usage**: Centroid classifier loads all examples into memory
- **LLM Augmentation**: Requires API calls; costs and latency should be monitored
- **Benchmark Scaling**: Some methods require significant compute resources

### Testing Gaps

- Limited integration tests for end-to-end workflows
- Missing tests for grounding data preparation
- Insufficient coverage of benchmark methods
- No regression tests for schema evolution

## Extension Points

### Adding New Privacy Domains

1. **Domain Configuration**: Create new template catalog with:
   - Entity pools specific to domain
   - Privacy-sensitive scenarios
   - Regional variations
   - Style variants

2. **Generator Implementation**: Extend `SyntheticGenerator` base with:
   - Domain-specific subtypes
   - Context construction logic
   - Validation rules

3. **CLI and Integration**: Provide domain-specific CLI entrypoints

### New Detection Methods

Implement standard interface in `benchmark/methods.py`:
```python
def custom_method(config, examples, centroids):
    # Validate dependencies
    # Apply detection logic
    # Return predictions and metadata
    return predictions, method_info
```

## Dependencies

### Core Requirements
- `numpy`: Numerical operations
- `scikit-learn`: Machine learning utilities
- `sentence-transformers`: Multilingual embeddings
- `openai`: LLM augmentation
- `pytest`: Testing framework
- `sacrebleu`: Diversity measurement
- `tqdm`: Progress bars

### Benchmark Requirements
- `transformers`: Hugging Face models
- `torch`: Deep learning framework
- Additional model-specific packages (see requirements-benchmark.txt)

## Configuration Files

### Pyproject.toml
```toml
[project]
name = "sensitive-egress-financial-poc"
version = "0.1.0"
description = "Synthetic data generation and detection framework"
requires-python = ">=3.10"

[tool.pytest.ini_options]
pythonpath = ["src"]
```

### Template Catalogs
JSON files containing:
- `entity_pools`: Dictionaries of entity categories
- `private_scenarios`: Sensitive content scenarios
- `hard_negative_scenarios`: Non-private but related scenarios
- `benign_scenarios`: Clear non-sensitive scenarios
- `mixed_carriers`: Egress simulation templates

## Development Workflow

1. **Setup Development Environment**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-benchmark.txt  # optional
   ```

2. **Run Tests**
   ```bash
   pytest
   ```

3. **Generate Test Data**
   ```bash
   python -m sensitive_egress_poc.cli_generate --private 10 --benign 10
   ```

4. **Run Quality Checks**
   ```bash
   python -m sensitive_egress_poc.dataset_quality --input data/financial_generated/anchors_train.jsonl
   ```

5. **Build Detection Model**
   ```bash
   python -m sensitive_egress_poc.cli_centroid --train ...
   ```

6. **Evaluate Benchmarks**
   ```bash
   python -m sensitive_egress_poc.cli_benchmark --methods centroid ...
   ```

This documentation should help developers understand the codebase structure and contribute effectively to the project.