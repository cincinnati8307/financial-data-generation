# Known Issues and Technical Debt

This document tracks known issues, technical debt, and areas needing improvement for the Sensitive Egress Privacy Detection Framework.

## Critical Issues

### 1. Massive Code Duplication

**Severity**: High  
**Impact**: Maintenance burden, inconsistency, development inefficiency

**Description**: 
The financial (`sensitive_egress_poc/`) and health (`health_egress_poc/`) implementations share ~95% of their code but are maintained separately. This creates significant maintenance problems.

**Duplicate Files**:
- `centroid_classifier.py` (100% identical)
- `filters.py` (100% identical) 
- `llm_augmenter.py` (100% identical)
- `cli_centroid.py` (100% identical)
- `cli_augment.py` (100% identical)
- `cli_demo.py` (100% identical)
- `synthetic_generator.py` (95% duplicate, domain-specific configs only)

**Problems**:
- Bug fixes must be applied twice independently
- New features require dual implementation
- Inconsistent updates across domains
- Higher testing overhead
- Risk of divergence over time

**Proposed Solution**: Refactor to unified core framework with domain-specific plugins

**Estimated Effort**: 2-3 weeks for core refactoring

**Related Issues**: #1, #5, #12

---

### 2. Lack of Integration Tests

**Severity**: High  
**Impact**: Reduced confidence in end-to-end functionality

**Description**: 
Current tests focus on unit tests for individual components but lack comprehensive integration tests for the complete data generation → detection → evaluation pipeline.

**Missing Test Coverage**:
- End-to-end workflow from data generation to classification
- Grounding integration pipeline
- Benchmark suite execution with real models
- CLI integration testing
- Error recovery and failure scenarios

**Problems**:
- Integration issues discovered late in development
- Uncertain behavior in complex scenarios
- Difficulty validating system-wide changes
- Limited confidence in production deployment

**Proposed Solution**: Develop comprehensive integration test suite covering all major workflows

**Estimated Effort**: 1-2 weeks

**Related Issues**: #8, #15

---

### 3. Incomplete Grounding Validation

**Severity**: Medium  
**Impact**: Potential quality issues with grounded synthetic data

**Description**: 
Validation of grounding datasets is basic and doesn't catch all quality problems that could affect synthetic data realism or safety.

**Validation Gaps**:
- Insufficient checking for embedding quality
- Limited validation of label consistency
- Missing checks for data leakage scenarios
- No validation of grounding data distribution
- Limited detection of problematic patterns

**Problems**:
- Poor quality grounding data may propagate to synthetic examples
- Hard to detect subtle issues in grounding datasets
- May generate unsafe examples despite grounding

**Proposed Solution**: Enhance grounding validation with comprehensive quality checks

**Estimated Effort**: 1 week

**Related Issues**: #7, #11

## Medium Priority Issues

### 4. Limited Error Recovery and Resilience

**Severity**: Medium  
**Impact**: Poor user experience when things go wrong

**Description**: 
Error handling is primitive and many operations lack graceful degradation or retry mechanisms.

**Specific Issues**:
- No retry logic for transient network errors (LLM API calls)
- No progress preservation in long-running operations
- Poor error messages that don't guide resolution
- No checkpointing for expensive operations
- Limited validation before expensive operations

**User Impact**:
- Lost work when operations fail
- Cryptic error messages
- Need to restart from scratch on failures
- Frustrating debugging experience

**Proposed Solution**: Implement robust error recovery, checkpointing, and clear error messages

**Estimated Effort**: 2-3 weeks

**Related Issues**: #3, #16

---

### 5. Performance Bottlenecks in Large-Scale Processing

**Severity**: Medium  
**Impact**: Limits scalability and usability

**Description**: 
Several operations don't scale well to large datasets (>10,000 examples).

**Performance Issues**:
- Embedding generation is not batched efficiently
- Quality checks are O(n^2) for redundancy detection  
- Centroid training loads all examples into memory
- Benchmark runs don't parallelize across methods
- No early stopping for expensive operations

**Impact**:
- Limitations on dataset size
- Long running times for quality checks
- Memory limitations for large datasets
- Inefficient use of computational resources

**Proposed Solution**: Optimize algorithms and implement better batching/parallelization

**Estimated Effort**: 1-2 weeks

**Related Issues**: #4, #9

---

### 6. Missing Configuration Management

**Severity**: Medium  
**Impact**: Poor reproducibility and usability

**Description**: 
Configuration is scattered across CLI arguments and hardcoded values, making experimentation difficult.

**Problems**:
- No configuration files for reproducible runs
- CLI arguments must be specified manually each time
- Hyperparameters are hardcoded in logic
- No environment-specific configurations
- Difficult to reproduce exact experimental conditions

**User Impact**:
- Hard to reproduce experiments exactly
- Difficult to manage multiple configurations
- Easy to make configuration mistakes
- Poor for systematic experimentation

**Proposed Solution**: Implement configuration file system with validation

**Estimated Effort**: 1 week

**Related Issues**: #6, #10

## Low Priority Issues

### 7. Limited Monitoring and Observability

**Severity**: Low  
**Impact**: Difficult to troubleshoot production issues

**Description**: 
Minimal logging and metrics collection make it hard to understand system behavior in production.

**Missing Features**:
- Structured logging with consistent format
- Performance metrics collection
- Resource usage monitoring
- Error tracking and alerting
- Progress indicators for long operations

**Impact**:
- Hard to troubleshoot issues
- Difficult to identify performance problems
- No visibility into resource usage
- Hard to detect problems early

**Proposed Solution**: Implement comprehensive logging and metrics collection

**Estimated Effort**: 1 week

**Related Issues**: #13, #14

---

### 8. Inconsistent Documentation

**Severity**: Low  
**Impact**: Steeper learning curve for new users

**Description**: 
Documentation exists but is inconsistent in style, depth, and maintenance.

**Documentation Issues**:
- Some functions lack docstrings
- Inconsistent docstring format
- Examples don't always work
- Quick start guide limited
- No troubleshooting guide
- Missing API reference

**User Impact**:
- Harder for new users to get started
- Unclear usage for some features
- Confusion when examples don't work
- Time spent on basic tasks

**Proposed Solution**: Standardize and expand documentation

**Estimated Effort**: 1 week

**Related Issues**: #2

---

### 9. No Validation for Schema Evolution

**Severity**: Low  
**Impact**: Potential data integrity issues over time

**Description**: 
No systematic way to validate that data schema changes don't break existing functionality.

**Problems**:
- No schema versioning
- No migration path for data
- No validation that old data works with new code
- No detection of incompatible changes

**Impact**:
- Risk of breaking existing functionality
- Difficult to maintain backward compatibility
- Risk of data corruption
- Silent failures possible

**Proposed Solution**: Implement schema versioning and validation

**Estimated Effort**: 1 week

**Related Issues**: #17

---

### 10. Limited Extensibility for New Domains

**Severity**: Low  
**Impact**: Higher barrier to adding new privacy domains

**Description**:
Adding new privacy domains currently requires copying entire codebase, which is error-prone and time-consuming.

**Problems**:
- No plugin architecture
- Configuration not externalized
- Template catalogs hardcoded
- No extension points

**Impact**:
- Hard to add new domains
- Duplicate code for each domain
- High barrier to contribution
- Missed opportunities for new research

**Proposed Solution**: Design plugin architecture and domain configuration system

**Estimated Effort**: 2 weeks

**Related Issues**: #18, #19

## Technical Debt Summary

### High Priority (Next 1-2 sprints)

1. **Code duplication** - Address via refactoring to unified core
2. **Integration tests** - Develop comprehensive test suite

### Medium Priority (Next 1-2 months)

3. **Grounding validation** - Enhance quality checks
4. **Error recovery** - Implement robust error handling  
5. **Performance optimization** - Better batching and parallelization
6. **Configuration management** - Add config files

### Low Priority (Ongoing, as time permits)

7. **Monitoring and observability** - Better logging and metrics
8. **Documentation consistency** - Standardize and expand
9. **Schema validation** - Add versioning and migration
10. **Extensibility** - Plugin architecture design

## Technical Debt Metrics

- **Total Estimated Effort**: ~15 weeks
- **Critical Issues**: 2
- **Medium Priority Issues**: 4  
- **Low Priority Issues**: 4
- **Code Duplication**: ~95% between domains
- **Test Coverage**: Estimated 60% (needs improvement)
- **Documentation Completeness**: ~70%

## Dependency Issues

### Outdated Dependencies

- `sentence-transformers` - May need update for latest models
- `transformers` - Version conflicts possible with AI libraries
- `pytest` - May benefit from latest features

### Security Considerations

- OpenAI API key storage in environment (acceptable)
- No secrets in codebase (good)
- Input validation needs review

## Performance Benchmarks

### Current Performance Characteristics

- **Data generation**: ~100 examples/second
- **Embedding creation**: ~50 texts/minute (CPU), ~200/minute (GPU)
- **Quality checks**: O(n^2) complexity, becomes slow >5k examples
- **Benchmark execution**: 30-60 minutes per method (varies by method)
- **Memory usage**: ~2GB for 10k examples

### Performance Goals

- **Doubled throughput** for all operations
- **Near-linear scaling** for quality checks
- **Reduced memory footprint** by 50%
- **Benchmark execution** under 30 minutes

## Architectural Concerns

### Tight Coupling

Many components are tightly coupled, making changes difficult:

- Generator ↔ Grounding store
- Benchmark ↔ Specific methods
- CLI ↔ Core logic

**Solution**: Introduce intermediate interfaces/abstractions

### Global State

Random number generators and configuration use global state:

- Random seeds not consistently applied
- Configuration scattered across modules
- Hidden dependencies

**Solution**: Pass dependencies explicitly, use dependency injection

### Lack of Modularity

Domain boundaries are unclear:

- Financial/health code mixed in some places
- Utility functions in multiple locations
- Unclear separation of concerns

**Solution**: Clear module boundaries, well-defined interfaces

## Future Considerations

### Scale Requirements

The system should support:

- **Dataset sizes**: >100k examples
- **Number of domains**: 5+ privacy domains
- **Benchmark methods**: 10+ detection approaches
- **Concurrent users**: Multiple research teams

### Technology Evolution

Consider preparing for:

- **New embedding models**: Better multilingual support
- **Advanced LLMs**: More capable augmentation
- **Distributed processing**: Large-scale data processing
- **Cloud deployment**: Managed service offering

### Research Integration

Enable easier integration with:

- **New detection methods**: Plugin architecture
- **Novel evaluation metrics**: Extensible metric system
- **Different data sources**: Flexible data ingestion
- **Research prototypes**: Rapid experimentation

## Action Items

### Immediate (Next Sprint)

1. [ ] Begin code duplication refactoring design
2. [ ] Add critical integration tests for main workflows
3. [ ] Enhance grounding validation quality checks

### Short-term (Next Month)

4. [ ] Implement unified core framework
5. [ ] Add comprehensive error recovery
6. [ ] Performance optimization for scaling
7. [ ] Configuration file system

### Medium-term (Next Quarter)

8. [ ] Complete plugin architecture design
9. [ ] Schema versioning and migration
10. [ ] Enhanced monitoring and observability
11. [ ] Documentation standardization

### Long-term (Next Year)

12. [ ] Support for 5+ privacy domains
13. [ ] Cloud deployment preparation
14. [ ] Advanced research integrations
15. [ ] Performance and scalability improvements

## Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) - System design and future direction
- [COMPONENTS.md](COMPONENTS.md) - Detailed component documentation  
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development guidelines
- [README.md](../README.md) - Project overview and usage

This document should be regularly updated as issues are resolved and new technical debt is identified.