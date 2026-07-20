# Development Guide

This guide provides practical information for developers working on the Sensitive Egress Privacy Detection Framework.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git for version control
- Basic understanding of NLP and privacy concepts

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd financial-data-generation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install benchmark dependencies (optional)
pip install -r requirements-benchmark.txt

# Run tests to verify setup
pytest
```

### Development Workflow

1. **Create feature branch**: `git checkout -b feature/my-feature`
2. **Make changes**: Edit code following existing patterns
3. **Test thoroughly**: Run relevant tests and quality checks
4. **Documentation**: Update relevant docs if behavior changes
5. **Commit**: Use clear, descriptive commit messages
6. **Push and PR**: Submit pull request with description

## Coding Standards

### Code Style

Follow these general conventions:

- **Imports**: Use `from __future__ import annotations` at top of files
- **Type Hints**: Include type hints for function signatures
- **Documentation**: Use docstrings for complex functions
- **Naming**: Follow PEP 8 with descriptive names
- **Error Handling**: Use specific exception types with clear messages

### File Organization

```
# Standard import order
from __future__ import annotations

# Standard library imports
import argparse
import logging
from pathlib import Path

# Third-party imports
from sentence_transformers import SentenceTransformer

# Local imports
from sensitive_egress_poc.synthetic_generator import SyntheticFinancialGenerator
```

### Common Patterns

#### Data Validation

```python
def validate_data(data: dict) -> tuple[bool, str]:
    """Validate data structure and content
    
    Returns:
        (is_valid, error_message) tuple
    """
    required_fields = ['id', 'text', 'label', 'subtype']
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Additional validation logic
    if not data['text'].strip():
        return False, "Text cannot be empty"
    
    return True, ""
```

#### Random Seed Management

```python
# Consistent random behavior
rng = random.Random(seed)

# Use rng instead of random module
value = rng.choice(values)
random_value = rng.random()
```

#### Error Handling

```python
def risky_operation():
    try:
        result = perform_operation()
        if not validate_result(result):
            raise ValueError(f"Invalid result: {result}")
        return result
    except Exception as e:
        logging.error(f"Operation failed: {e}")
        raise  # Re-raise for caller to handle
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_generator.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_generator.py -k test_private_example

# Verbose output
pytest -v
```

### Writing Tests

```python
import pytest
from sensitive_egress_poc.synthetic_generator import SyntheticFinancialGenerator

class TestSyntheticGenerator:
    def test_initialization(self):
        """Test generator initialization"""
        gen = SyntheticFinancialGenerator(seed=42)
        assert gen.counter == 0
        assert len(gen.catalog) > 0
    
    def test_private_example_generation(self):
        """Test private example generation"""
        gen = SyntheticFinancialGenerator(seed=42)
        example = gen.private_example(subtype="bank_balance")
        
        assert example.label == "financial_private"
        assert example.subtype == "bank_balance"
        assert len(example.text) > 0
        assert example.sensitivity_level == "high"
    
    def test_batch_generation(self):
        """Test batch generation consistency"""
        gen = SyntheticFinancialGenerator(seed=42)
        examples1 = gen.generate_private(10)
        gen2 = SyntheticFinancialGenerator(seed=42)
        examples2 = gen2.generate_private(10)
        
        assert examples1 == examples2
```

### Fixtures

Create reusable test fixtures in `conftest.py`:

```python
import pytest

@pytest.fixture
def sample_generator():
    """Provide a generator instance for testing"""
    return SyntheticFinancialGenerator(seed=42)

@pytest.fixture
def sample_data(tmp_path):
    """Provide sample data files"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create test data files
    ...
    
    return data_dir
```

## Debugging

### Common Debugging Approaches

#### Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Use logging instead of print
logging.debug(f"Processing data point: {data_index}")
logging.info(f"Generated {len(examples)} examples")
logging.warning(f"Unusual condition detected: {condition}")
logging.error(f"Processing failed: {error}")
```

#### Interactive Debugging

```python
# Add debug breakpoints
import pdb; pdb.set_trace()

# Or use IPython debugger
import IPython; IPython.embed()

# For specific debugging
def debug_function(data):
    print(f"Input: {data}")
    result = process(data)
    print(f"Output: {result}")
    return result
```

#### Common Issues and Solutions

**Issue**: Import errors with custom modules
```bash
# Solution: Set PYTHONPATH
export PYTHONPATH=src
# Or use: pip install -e .
```

**Issue**: Model download failures
```python
# Solution: Use offline mode or check network
# Set transformers offline mode
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

**Issue**: Memory issues with large datasets
```python
# Solution: Process in batches
BATCH_SIZE = 1000
for i in range(0, len(data), BATCH_SIZE):
    batch = data[i:i+BATCH_SIZE]
    process_batch(batch)
```

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run your code
    result = your_function()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Show top 10 functions
    
    return result
```

### Optimization Techniques

#### Vectorization

```python
# Slow: Python loop
results = []
for text in texts:
    embedding = model.encode(text)
    results.append(embedding)

# Fast: Batch processing
results = model.encode(texts, batch_size=32, show_progress_bar=True)
```

#### Memory Management

```python
# Process data in chunks to reduce memory usage
def process_large_file(filepath, chunk_size=1000):
    with open(filepath) as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield process_chunk(chunk)
                chunk = []
        if chunk:
            yield process_chunk(chunk)
```

#### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(param):
    # Expensive computation
    return result
```

## Documentation

### Docstring Format

```python
def generate_private_examples(
    count: int,
    subtype: str | None = None,
    seed: int = 42
) -> list[dict]:
    """Generate private financial examples with specified subtypes.
    
    Args:
        count: Number of examples to generate
        subtype: Specific subtype to generate, or random if None
        seed: Random seed for reproducibility
        
    Returns:
        List of generated examples as dictionaries
        
    Raises:
        ValueError: If count is negative or subtype is invalid
        
    Example:
        >>> gen = SyntheticFinancialGenerator(seed=42)
        >>> examples = gen.generate_private(10, "bank_balance")
        >>> len(examples)
        10
    """
    # Implementation
    pass
```

### Keeping Documentation Updated

- Update README.md when changing API interfaces
- Update COMPONENTS.md when modifying file structure
- Add inline comments for complex logic
- Maintain examples in docstrings

## Troubleshooting Common Development Issues

### Import Path Issues

**Problem**: Can't import modules from src/

**Solution**:
```bash
export PYTHONPATH=src
# Or use absolute imports in tests
```

### Test Infrastructure

**Problem**: Tests fail due to missing test data

**Solution**: Generate fixtures or use temporary directories
```python
import tempfile
import pytest

def test_with_temp_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test data")
        f.flush()
        
        # Run test with file
        result = process_file(f.name)
        
        # Cleanup happens automatically
```

### Virtual Environment Issues

**Problem**: Dependencies not installing correctly

**Solution**: 
```bash
# Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Clean install
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Code Review Guidelines

When reviewing code:

1. **Functionality**: Does it work as intended?
2. **Testing**: Are tests comprehensive enough?
3. **Error Handling**: Are edge cases handled?
4. **Performance**: Is it efficient enough?
5. **Readability**: Is the code clear and maintainable?
6. **Documentation**: Are changes documented?
7. **Security**: Are there any security concerns?
8. **Consistency**: Does it follow project conventions?

## Release Checklist

Before releasing new version:

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Version number updated in pyproject.toml
- [ ] Changelog updated
- [ ] Dependencies reviewed
- [ ] Performance validated
- [ ] Security audit if needed
- [ ] Test on clean environment
- [ ] Git tag created
- [ ] Release notes prepared

## Contributing Guidelines

### Before Submitting

1. **Run tests locally**: Ensure all tests pass
2. **Lint code**: Apply code style checks
3. **Update docs**: Document significant changes
4. **Examples tested**: Keep working examples current
5. **Backward compatibility**: Consider impact on existing users

### Pull Request Process

1. **Clear description**: Explain what and why
2. **Related issues**: Reference issue numbers
3. **Testing**: Describe test coverage
4. **Breaking changes**: Highlight any
5. **Documentation**: Link to updated docs
6. **Review**: Address feedback promptly

## Additional Resources

### Learning Resources

- [Python Type Hints Guide](https://docs.python.org/3/library/typing.html)
- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Pytest Tutorial](https://docs.pytest.org/)

### Project References

- [ARCHITECTURE.md](../ARCHITECTURE.md): System design and components
- [COMPONENTS.md](COMPONENTS.md): Detailed component documentation
- [README.md](../README.md): Project overview and usage

## Getting Help

### Questions?

- Check existing documentation
- Search existing issues
- Create new issue with details
- Contact maintainers via project channels

### Issue Template

```markdown
**Description:** Brief description of the issue

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:** What should happen

**Actual Behavior:** What actually happens

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11]
- Dependencies: [relevant versions]

**Additional Context:** Any other relevant information
```

This development guide should help new contributors get started quickly and existing contributors maintain consistent quality.