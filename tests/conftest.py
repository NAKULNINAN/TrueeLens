"""
pytest configuration and fixtures for Media Forensics test suite
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
import logging
import json
import time
from typing import Dict, Any, Generator

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from settings import Config
from tests import DEFAULT_TEST_CONFIG, TEST_DATA_DIR


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration fixture"""
    return DEFAULT_TEST_CONFIG.copy()


@pytest.fixture(scope="session")
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp(prefix="media_forensics_test_")
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="session")
def test_data_dir() -> str:
    """Test data directory fixture"""
    data_dir = project_root / "tests" / TEST_DATA_DIR
    data_dir.mkdir(exist_ok=True)
    return str(data_dir)


@pytest.fixture(scope="session")
def sample_images_dir(test_data_dir) -> str:
    """Sample images directory fixture"""
    images_dir = Path(test_data_dir) / "images"
    images_dir.mkdir(exist_ok=True)
    return str(images_dir)


@pytest.fixture(scope="session")
def sample_videos_dir(test_data_dir) -> str:
    """Sample videos directory fixture"""
    videos_dir = Path(test_data_dir) / "videos"
    videos_dir.mkdir(exist_ok=True)
    return str(videos_dir)


@pytest.fixture(scope="function")
def clean_database():
    """Clean test database before each test"""
    # This fixture can be used to clean up database state
    yield
    # Cleanup after test if needed


@pytest.fixture(scope="function")
def test_logger() -> logging.Logger:
    """Test logger fixture"""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


@pytest.fixture(scope="function")
def performance_tracker():
    """Performance tracking fixture"""
    class PerformanceTracker:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.metrics = {}
        
        def start(self):
            self.start_time = time.time()
        
        def end(self):
            self.end_time = time.time()
            return self.end_time - self.start_time if self.start_time else 0
        
        def add_metric(self, name: str, value: Any):
            self.metrics[name] = value
        
        def get_report(self) -> Dict[str, Any]:
            duration = self.end() if self.end_time is None else (self.end_time - self.start_time)
            return {
                "duration": duration,
                "metrics": self.metrics
            }
    
    return PerformanceTracker()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests"""
    # Create required directories
    Config.create_required_directories()
    
    # Setup test logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    yield
    
    # Cleanup after all tests


@pytest.fixture(scope="function")
def mock_streamlit_file():
    """Mock Streamlit uploaded file object"""
    class MockStreamlitFile:
        def __init__(self, file_path: str, name: str = None):
            self.file_path = file_path
            self.name = name or os.path.basename(file_path)
            self.type = self._get_mime_type()
            with open(file_path, 'rb') as f:
                self.content = f.read()
                self.size = len(self.content)
        
        def read(self):
            return self.content
        
        def getvalue(self):
            return self.content
        
        def _get_mime_type(self):
            ext = os.path.splitext(self.file_path)[1].lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.mp4': 'video/mp4',
                '.avi': 'video/avi',
                '.mov': 'video/quicktime'
            }
            return mime_types.get(ext, 'application/octet-stream')
    
    return MockStreamlitFile


# Test markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_models: mark test as requiring model files"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test items during collection"""
    # Add default markers based on test location
    for item in items:
        test_path = str(item.fspath)
        
        if "unit" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "integration" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "performance" in test_path:
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker to tests that typically take longer
        if any(keyword in item.name.lower() for keyword in ['video', 'model', 'benchmark']):
            item.add_marker(pytest.mark.slow)
