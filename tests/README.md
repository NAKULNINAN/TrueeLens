# Media Forensics Test Suite

This directory contains comprehensive test suites for the Media Forensics application, including unit tests, integration tests, performance benchmarks, and sample data generation.

## 📁 Directory Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # pytest configuration and fixtures
├── test_detection.py           # Unit tests for detection modules
├── test_media_processor.py     # Unit tests for media processing utilities
├── test_integration.py         # Integration tests
├── test_performance.py         # Performance benchmarks
├── sample_data_generator.py    # Generate sample test media files
├── test_data/                  # Test data directory
│   ├── images/                 # Sample images for testing
│   ├── videos/                 # Sample videos for testing
│   └── benchmarks/             # Benchmark data and results
└── README.md                   # This file
```

## 🚀 Quick Start

### Install Test Dependencies

```bash
pip install -r requirements_test.txt
```

### Run All Tests

```bash
# From project root directory
python run_tests.py
```

### Run Specific Test Suites

```bash
# Unit tests only
python run_tests.py --unit-only

# Integration tests only
python run_tests.py --integration-only

# Performance benchmarks only
python run_tests.py --performance-only
```

### Generate Test Data

```bash
python run_tests.py --generate-samples
```

### Run with Coverage

```bash
python run_tests.py --with-coverage
```

## 📊 Test Categories

### Unit Tests (`test_detection.py`, `test_media_processor.py`)

Test individual components in isolation:

- **AI Image Detector**: Tests StyleGAN, DALL-E, and Midjourney detectors
- **Media Processor**: Tests image processing, format conversion, metadata extraction
- **Database Operations**: Tests hash storage and retrieval
- **Utility Functions**: Tests error handling, visualization, configuration

Example:
```bash
pytest tests/test_detection.py -v
```

### Integration Tests (`test_integration.py`)

Test complete workflows and component interactions:

- **End-to-end AI Detection**: Complete detection pipeline
- **Database Integration**: Full duplicate detection workflow
- **App Integration**: Main application functionality
- **Cross-module Communication**: Verify modules work together correctly

Example:
```bash
pytest tests/test_integration.py -v
```

### Performance Benchmarks (`test_performance.py`)

Measure and track performance metrics:

- **AI Detection Performance**: Speed and memory usage across image sizes
- **Duplicate Detection Performance**: Hash calculation and search performance
- **Media Processing Performance**: Image processing and format conversion speed
- **Batch Processing**: Throughput measurements

Example:
```bash
pytest tests/test_performance.py -v --tb=short
```

## 🔧 Configuration

### pytest Configuration (`conftest.py`)

Provides shared fixtures and test configuration:

- **Test Data Fixtures**: Automatic test data setup
- **Mock Objects**: Mock Streamlit file uploads
- **Performance Tracking**: Built-in performance measurement
- **Database Cleanup**: Automatic database state management

### Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.slow` - Slow-running tests (can be skipped)
- `@pytest.mark.requires_gpu` - Tests requiring GPU
- `@pytest.mark.requires_models` - Tests requiring model files

### Running Specific Markers

```bash
# Run only unit tests
pytest -m unit

# Run fast tests only (skip slow ones)
pytest -m "not slow"

# Run GPU tests only
pytest -m requires_gpu
```

## 📈 Performance Benchmarking

### Running Benchmarks

```bash
# Run all performance tests
python run_tests.py --performance-only

# Run quick benchmarks (skip slow tests)
python run_tests.py --performance-only --quick

# Run comprehensive benchmarks
python tests/test_performance.py
```

### Benchmark Reports

Benchmark results are saved to:
- `tests/benchmark_results/` - JSON benchmark data
- `test_results/performance_tests.html` - HTML report

### Sample Benchmark Metrics

- **Execution Time**: Processing time per image/video
- **Memory Usage**: Peak memory consumption
- **Throughput**: Images/videos processed per second
- **Scalability**: Performance vs. input size/batch size

## 🎯 Sample Data Generation

### Generate Test Media

```bash
python tests/sample_data_generator.py
```

This creates:
- **Sample Images**: Various sizes and patterns for testing
- **Sample Videos**: Short test videos for video processing tests
- **Synthetic Data**: Controlled test data with known characteristics

### Custom Test Data

To add custom test data:

1. Place images in `tests/test_data/images/`
2. Place videos in `tests/test_data/videos/`
3. Update test fixtures in `conftest.py` if needed

## 📝 Writing New Tests

### Unit Test Example

```python
import pytest
from detection.ai_image_detector import AIImageDetector

def test_ai_detector_initialization():
    """Test AI detector can be initialized"""
    detector = AIImageDetector()
    assert detector is not None

def test_ai_detector_confidence_range():
    """Test confidence scores are in valid range"""
    detector = AIImageDetector()
    # Test implementation...
    assert 0 <= result['confidence'] <= 1
```

### Integration Test Example

```python
import pytest
from app import MediaForensicsApp

class TestWorkflow:
    def test_complete_detection_workflow(self, mock_streamlit_file):
        """Test complete detection from upload to results"""
        app = MediaForensicsApp()
        mock_file = mock_streamlit_file("tests/test_data/images/sample.jpg")
        
        results = app.process_media(mock_file, "Detect AI-Generated Image", 0.5, False)
        
        assert 'confidence' in results
        assert 'is_fake' in results
```

### Performance Test Example

```python
import pytest
from tests.test_performance import PerformanceBenchmark

class TestMyPerformance(PerformanceBenchmark):
    def test_my_function_performance(self):
        """Benchmark my function"""
        benchmark = self.measure_time_and_memory(my_function, args)
        
        assert benchmark['execution_time'] < 5.0  # Should complete in 5s
        assert benchmark['memory_used_mb'] < 100  # Should use less than 100MB
```

## 🔍 Debugging Tests

### Verbose Output

```bash
pytest tests/ -v --tb=long
```

### Run Single Test

```bash
pytest tests/test_detection.py::test_ai_detector_initialization -v
```

### Debug Mode

```bash
pytest tests/ --pdb  # Drop into debugger on failures
```

### Logging

Enable detailed logging during tests:

```bash
pytest tests/ --log-cli-level=DEBUG
```

## 📊 Coverage Reports

### Generate Coverage

```bash
python run_tests.py --with-coverage
```

### View Coverage

- **HTML Report**: `test_results/coverage_html/index.html`
- **JSON Report**: `test_results/coverage.json`
- **Terminal**: Displayed after test run

### Coverage Targets

Aim for:
- **Unit Tests**: >90% line coverage
- **Integration Tests**: >80% feature coverage
- **Critical Paths**: 100% coverage for security-critical code

## 🚀 Continuous Integration

### GitHub Actions Example

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements_test.txt
    - name: Run tests
      run: python run_tests.py --with-coverage
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 📋 Best Practices

### Test Organization

1. **One test file per module** being tested
2. **Group related tests** in classes
3. **Use descriptive test names** that explain what is being tested
4. **Keep tests independent** - no dependencies between tests

### Test Data

1. **Use fixtures** for shared test data
2. **Generate test data programmatically** when possible
3. **Clean up after tests** to avoid state pollution
4. **Use realistic test data** that represents actual usage

### Assertions

1. **Test one thing per test** function
2. **Use specific assertions** (`assertEqual` vs `assertTrue`)
3. **Include helpful assertion messages**
4. **Test both positive and negative cases**

### Performance Tests

1. **Set reasonable timeouts** for performance expectations
2. **Test with different input sizes** to verify scalability
3. **Measure relevant metrics** (time, memory, throughput)
4. **Track performance trends** over time

## 🛠️ Troubleshooting

### Common Issues

1. **Missing dependencies**: Install `requirements_test.txt`
2. **Permission errors**: Run with appropriate permissions
3. **Path issues**: Run from project root directory
4. **Memory issues**: Use `--quick` flag to skip memory-intensive tests

### Getting Help

1. Check test output for detailed error messages
2. Run with `-v` flag for verbose output
3. Use `--tb=long` for detailed tracebacks
4. Check `test_results/` directory for HTML reports

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Plugin](https://pytest-cov.readthedocs.io/)
- [pytest-html Plugin](https://pytest-html.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
