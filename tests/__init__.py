"""
Media Forensics App - Test Suite
================================

Comprehensive test suite for the Media Forensics application including:
- Unit tests for all modules
- Integration tests
- Performance benchmarks
- Sample test data generation
"""

__version__ = "1.0.0"
__author__ = "Media Forensics Team"

# Test categories
UNIT_TESTS = "unit"
INTEGRATION_TESTS = "integration"
PERFORMANCE_TESTS = "performance"
E2E_TESTS = "end_to_end"

# Test data directories
TEST_DATA_DIR = "test_data"
SAMPLE_IMAGES_DIR = f"{TEST_DATA_DIR}/images"
SAMPLE_VIDEOS_DIR = f"{TEST_DATA_DIR}/videos"
BENCHMARK_DATA_DIR = f"{TEST_DATA_DIR}/benchmarks"

# Test configuration
DEFAULT_TEST_CONFIG = {
    "timeout": 30,
    "max_retries": 3,
    "cleanup_after_test": True,
    "save_test_outputs": False,
    "verbose_logging": True
}
