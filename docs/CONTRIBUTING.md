# Contribution Guide

This guide provides information for contributors who want to help improve the Sensitive Egress Privacy Detection Framework.

## How to Contribute

We welcome contributions in many forms:

- **Code improvements**: Bug fixes, new features, performance enhancements
- **Documentation**: Better docs, examples, tutorials
- **Testing**: More test coverage, integration tests
- **Research ideas**: Novel detection methods, evaluation approaches
- **Bug reports**: Issue reports with detailed information

## Getting Started

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd financial-data-generation

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-benchmark.txt  # optional

# Run tests to verify
pytest
```

### Understanding the Codebase

Before contributing, familiarize yourself with:

1. **ARCHITECTURE.md** - System design and architecture
2. **COMPONENTS.md** - Detailed component documentation  
3. **TECHNICAL_DEBT.md** - Known issues and improvement areas
4. **DEVELOPMENT.md** - Development guidelines and coding standards

## Contribution Workflow

### 1. Find or Create an Issue

- Check existing issues to avoid duplication
- For new work, create an issue describing the problem or feature
- Get feedback from maintainers before starting work

### 2. Set Up Development Environment

- Create a feature branch: `git checkout -b feature/your-feature`
- Ensure tests pass: `pytest`
- Set up pre-commit hooks if available

### 3. Make Your Changes

- Follow the coding standards in DEVELOPMENT.md
- Write tests for new functionality
- Update relevant documentation
- Keep changes focused and minimal

### 4. Test Thoroughly

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_generator.py

# Run with coverage
pytest --cov=src --cov-report=html

# Check code quality
# (add linters as configured in project)
```

### 5. Submit Your Contribution

1. **Commit your changes**: Use clear, descriptive commit messages
2. **Push to your fork**: `git push origin feature/your-feature`
3. **Create Pull Request**: Include description, testing notes, and related issues
4. **Respond to feedback**: Address reviewer comments promptly

## Types of Contributions

### Bug Fixes

For bug fixes, include:

- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Fix implementation
- Tests that verify the fix

```markdown
## Description
Fixes #123 - Dataset quality checks fail on empty input

## Testing
Added test case `test_empty_dataset_quality` that verifies proper handling of empty datasets
```

### New Features

For new features, include:

- Feature description and motivation
- Usage examples
- Implementation details
- Tests for new functionality
- Documentation updates

### Documentation

For documentation improvements:

- Clear description of what's being improved
- Accuracy of information
- Example code that works
- Consistency with existing style

### Research Contributions

For research contributions:

- Clear research question or hypothesis
- Methodology description
- Experimental setup and results
- Discussion of implications
- Suggestions for future work

## Code Review Process

### What Reviewers Look For

- **Functionality**: Does it work as intended?
- **Testing**: Are tests comprehensive enough?
- **Error Handling**: Are edge cases handled properly?
- **Performance**: Is it efficient and doesn't regress performance?
- **Code Quality**: Is it clear, maintainable, and follows standards?
- **Documentation**: Are changes documented appropriately?
- **Security**: Are there security considerations?
- **Backward Compatibility**: Does it break existing functionality?

### Responding to Reviews

- Address reviewer comments constructively
- Explain your reasoning when appropriate
- Split large changes into smaller, reviewable parts
- Update commit history if requested (squash, rebase)

## Development Guidelines

### Coding Standards

- **Type hints**: Include for function signatures
- **Docstrings**: Use consistent format for functions and classes
- **Error handling**: Use specific exceptions with clear messages
- **Testing**: Write tests before or with code changes
- **Style**: Follow PEP 8 and existing code patterns

### Git Workflow

#### Branch Naming

```
feature/new-feature-name
fix/issue-description
docs/documentation-improvement
refactor/cleanup-description
test/test-coverage-addition
```

#### Commit Messages

```
type(scope): subject

# Detailed explanation

- Closes #issue_number
```

Example:
```
feat(generator): add support for new financial subtype

Implemented bank_loan subtype with proper validation and example
generation. Added entity pools for loan types and payment terms.

- Closes #42
```

## Testing Guidelines

### What to Test

- **Core functionality**: Main paths through the system
- **Edge cases**: Boundary conditions and unusual inputs  
- **Error handling**: Proper error messages and recovery
- **Integration**: Component interactions
- **Performance**: No regressions in critical paths

### Test Structure

```python
import pytest

class TestYourComponent:
    def test_basic_functionality(self):
        """Test basic usage scenario"""
        pass
    
    def test_edge_case(self):
        """Test edge case conditions"""
        pass
    
    def test_error_handling(self):
        """Test error handling"""
        pass
    
    def test_integration(self):
        """Test component integration"""
        pass
```

### Continuous Integration

The project should have CI that runs on every pull request:

```bash
# Typical CI checks
pytest                           # Run all tests
pytest --cov=src                 # Coverage checks
pre-commit                       # Code quality checks
```

## Special Considerations

### Domain-Specific Code

When making changes that affect both financial and health domains:

- Ensure changes apply to both implementations
- Test in both domains
- Document domain-specific behavior
- Consider refactoring to reduce duplication

### Safety and Ethics

This framework deals with sensitive data:

- Never include real personally identifiable information
- Consider privacy implications of changes
- Validate that no realistic sensitive data is generated
- Ensure detection methods respect privacy norms

### Performance Impact

Performance is critical for research workflows:

- Profile code before and after changes
- Add benchmarks for critical paths
- Monitor memory usage
- Consider scalability to large datasets

## Documentation Standards

### Docstring Format

```python
def function_name(param1: type, param2: type) -> return_type:
    """Brief description of what the function does.
    
    Extended description if needed. Can span multiple lines.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: Description of when this error occurs
        
    Example:
        >>> function_name(1, 2)
        expected_result
    """
    pass
```

### Documentation Updates

When making changes:

- Update README.md if user-facing behavior changes
- Update COMPONENTS.md if file structure changes  
- Update DEVELOPMENT.md if development workflow changes
- Update TECHNICAL_DEBT.md if resolving issues
- Add new docs for major features

## Release Process

### Versioning

Follow semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Release Checklist

Before release:

- [ ] All tests passing
- [ ] Coverage meets threshold (%)
- [ ] Documentation updated
- [ ] Version number updated in pyproject.toml
- [ ] CHANGELOG.md updated
- [ ] Dependencies reviewed and updated
- [ ] Performance benchmarks run
- [ ] Security audit if needed
- [ ] Tested on clean environment
- [ ] Release notes prepared

## Getting Help

### Questions?

- Check existing documentation first
- Search existing issues
- Create new issue with detailed description
- Contact maintainers via project channels

### Issue Template

```markdown
**Description:** Brief description of the issue or suggestion

**Problem:** (for bugs)
Detailed description of the problem

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

**Additional Context:** Any other relevant information, logs, screenshots, etc.
```

## Recognition

Contributors are recognized in:

- CONTRIBUTORS.md file
- Release notes for significant contributions
- Project documentation for major contributions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Assume good intentions
- Resolve conflicts collaboratively

## Additional Resources

- [ARCHITECTURE.md](../ARCHITECTURE.md) - System design
- [COMPONENTS.md](COMPONENTS.md) - Component details
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development guidelines  
- [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) - Known issues
- [README.md](../README.md) - Project overview

Thank you for contributing!