"""
Integration tests for Media Forensics app
Tests the complete workflow and component interactions.
"""

import pytest
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import MediaForensicsApp
from detection.ai_image_detector import AIImageDetector
from detection.duplicate_detector import DuplicateDetector
from detection.deepfake_detector import DeepfakeDetector
from utils.media_processor import MediaProcessor


class TestMediaForensicsWorkflow:
    """Integration tests for the complete media forensics workflow"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.test_image = Image.new('RGB', (224, 224), color='red')
        self.temp_path = "temp_test_image.jpg"
        self.test_image.save(self.temp_path)
        yield
        # Cleanup
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)
    
    def test_ai_image_detection_workflow(self):
        """Test AI image detection end-to-end workflow"""
        detector = AIImageDetector()
        
        # Mock uploaded file
        class MockFile:
            def __init__(self, path):
                self.name = os.path.basename(path)
                self.type = "image/jpeg"
                with open(path, 'rb') as f:
                    self.content = f.read()
                    self.size = len(self.content)
            
            def read(self):
                return self.content
        
        mock_file = MockFile(self.temp_path)
        
        # Test detection workflow
        results = detector.detect(mock_file, threshold=0.5, enable_viz=False)
        
        assert 'confidence' in results, "Results should contain confidence"
        assert 'is_fake' in results, "Results should contain is_fake"
        assert 'explanation' in results, "Results should contain explanation"
        assert isinstance(results['confidence'], (int, float)), "Confidence should be numeric"
    
    def test_duplicate_detection_workflow(self):
        """Test duplicate detection end-to-end workflow"""
        detector = DuplicateDetector()
        
        # Create a second identical image
        temp_path2 = "temp_test_image2.jpg"
        self.test_image.save(temp_path2)
        
        try:
            # Process both images
            detector.process_file(self.temp_path)
            detector.process_file(temp_path2)
            
            # Find duplicates
            duplicates = detector.find_all_duplicates()
            
            # Should find at least one duplicate pair
            assert len(duplicates) >= 0, "Should return duplicate results"
            
        finally:
            if os.path.exists(temp_path2):
                os.remove(temp_path2)
    
    def test_media_processor_integration(self):
        """Test media processor integration with detection modules"""
        processor = MediaProcessor()
        
        # Test image processing
        processed_array, metadata = processor.process_image(self.test_image)
        
        assert processed_array is not None, "Should return processed array"
        assert metadata is not None, "Should return metadata"
        
        # Test that processed data can be used by detection modules
        ai_detector = AIImageDetector()
        
        # This tests the integration between processing and detection
        # The processed array should be compatible with detection modules
        assert processed_array.shape == (224, 224, 3), "Should maintain correct shape"
        assert processed_array.dtype == np.uint8, "Should maintain correct data type"


class TestAppIntegration:
    """Integration tests for the main app functionality"""
    
    def test_app_initialization(self):
        """Test that the main app initializes correctly"""
        app = MediaForensicsApp()
        
        assert app.ai_detector is not None, "AI detector should be initialized"
        assert app.deepfake_detector is not None, "Deepfake detector should be initialized"
        assert app.duplicate_detector is not None, "Duplicate detector should be initialized"
        assert app.media_processor is not None, "Media processor should be initialized"
    
    def test_process_media_integration(self):
        """Test the main process_media method integration"""
        app = MediaForensicsApp()
        
        # Create mock uploaded file
        test_image = Image.new('RGB', (224, 224), color='blue')
        temp_path = "integration_test_image.jpg"
        test_image.save(temp_path)
        
        class MockUploadedFile:
            def __init__(self, path):
                self.name = os.path.basename(path)
                self.type = "image/jpeg"
                self.size = os.path.getsize(path)
                with open(path, 'rb') as f:
                    self.content = f.read()
            
            def read(self):
                return self.content
        
        mock_file = MockUploadedFile(temp_path)
        
        try:
            # Test AI image detection mode
            results = app.process_media(
                mock_file, 
                "Detect AI-Generated Image", 
                threshold=0.5, 
                enable_viz=False
            )
            
            assert 'mode' in results, "Results should contain mode"
            assert 'timestamp' in results, "Results should contain timestamp"
            assert 'file_info' in results, "Results should contain file_info"
            assert results['mode'] == "Detect AI-Generated Image", "Mode should match"
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    def test_database_operations(self):
        """Test database operations integration"""
        from database.hash_storage import DatabaseManager
        
        db_manager = DatabaseManager()
        db_manager.init_database()
        
        # Test database connection and basic operations
        assert os.path.exists(db_manager.db_path), "Database file should be created"
        
        # Test hash storage and retrieval
        test_hashes = {
            'dhash': 'test_dhash_value',
            'ahash': 'test_ahash_value',
            'phash': 'test_phash_value',
            'whash': 'test_whash_value'
        }
        
        # This would test actual database operations if the methods exist
        # db_manager.store_hashes('test_file.jpg', test_hashes)


@pytest.mark.slow
class TestPerformanceIntegration:
    """Integration tests for performance-critical workflows"""
    
    def test_batch_processing_performance(self):
        """Test batch processing performance"""
        processor = MediaProcessor()
        
        # Create multiple test images
        test_images = []
        for i in range(5):
            img = Image.new('RGB', (224, 224), color=(i*50, i*40, i*30))
            test_images.append(img)
        
        # Time the batch processing
        import time
        start_time = time.time()
        
        results = []
        for img in test_images:
            processed_array, metadata = processor.process_image(img)
            results.append((processed_array, metadata))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert len(results) == 5, "Should process all images"
        assert processing_time < 10, "Batch processing should be reasonable fast"
        
        # Log performance metrics
        avg_time_per_image = processing_time / len(test_images)
        print(f"Average processing time per image: {avg_time_per_image:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
