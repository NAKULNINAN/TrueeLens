"""
Complete unit tests for all database modules
Tests hash storage, file manager, schema manager, similarity cache, and mock database
"""

import pytest
import os
import tempfile
import sqlite3
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import hashlib

from database.hash_storage import DatabaseManager, HashCalculator
from database.file_manager import FileManager
from database.schema_manager import SchemaManager
from database.similarity_cache import SimilarityCache
from database.mock_database import MockDatabase


class TestDatabaseManagerComplete:
    """Complete tests for Database Manager"""
    
    def setup_method(self):
        """Setup test environment"""
        # Use temporary database for testing
        self.test_db_path = "test_database.db"
        self.db_manager = DatabaseManager(db_path=self.test_db_path)
        self.cleanup_database()
    
    def teardown_method(self):
        """Clean up test environment"""
        self.cleanup_database()
    
    def cleanup_database(self):
        """Clean up test database"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_database_initialization(self):
        """Test database initialization"""
        self.db_manager.init_database()
        
        # Check if database file was created
        assert os.path.exists(self.test_db_path)
        
        # Check if tables were created
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        expected_tables = ['media_hashes', 'duplicate_pairs', 'file_metadata']
        for table in expected_tables:
            assert table in tables
    
    def test_store_and_retrieve_hashes(self):
        """Test storing and retrieving hashes"""
        self.db_manager.init_database()
        
        # Test data
        file_path = "test_image.jpg"
        test_hashes = {
            'dhash': 'a1b2c3d4e5f6g7h8',
            'ahash': 'h8g7f6e5d4c3b2a1',
            'phash': '1234567890abcdef',
            'whash': 'fedcba0987654321'
        }
        
        # Store hashes
        self.db_manager.store_hashes(file_path, test_hashes)
        
        # Retrieve hashes
        retrieved_hashes = self.db_manager.get_hashes(file_path)
        
        assert retrieved_hashes is not None
        for hash_type, hash_value in test_hashes.items():
            assert retrieved_hashes[hash_type] == hash_value
    
    def test_find_similar_hashes(self):
        """Test finding similar hashes"""
        self.db_manager.init_database()
        
        # Store multiple test files with similar hashes
        test_files = [
            ("file1.jpg", {
                'dhash': '1111111111111111',
                'ahash': '2222222222222222',
                'phash': '3333333333333333',
                'whash': '4444444444444444'
            }),
            ("file2.jpg", {
                'dhash': '1111111111111110',  # Very similar to file1
                'ahash': '2222222222222220',
                'phash': '3333333333333330',
                'whash': '4444444444444440'
            }),
            ("file3.jpg", {
                'dhash': 'aaaaaaaaaaaaaaaa',  # Different
                'ahash': 'bbbbbbbbbbbbbbbb',
                'phash': 'cccccccccccccccc',
                'whash': 'dddddddddddddddd'
            })
        ]
        
        # Store all files
        for file_path, hashes in test_files:
            self.db_manager.store_hashes(file_path, hashes)
        
        # Find similar files to file1
        query_hashes = test_files[0][1]
        similar_files = self.db_manager.find_similar_hashes(query_hashes, threshold=5)
        
        # Should find file1 and file2 (similar), but not file3
        similar_paths = [result['file_path'] for result in similar_files]
        assert "file1.jpg" in similar_paths
        assert "file2.jpg" in similar_paths
        assert "file3.jpg" not in similar_paths
    
    def test_hamming_distance_calculation(self):
        """Test Hamming distance calculation"""
        # Test identical hashes
        hash1 = '1111111111111111'
        hash2 = '1111111111111111'
        distance = self.db_manager._hamming_distance(hash1, hash2)
        assert distance == 0
        
        # Test completely different hashes
        hash1 = '1111111111111111'
        hash2 = '0000000000000000'
        distance = self.db_manager._hamming_distance(hash1, hash2)
        assert distance == 16  # All 16 hex digits are different
        
        # Test partially different hashes
        hash1 = '1111111111111111'
        hash2 = '1111111111111110'
        distance = self.db_manager._hamming_distance(hash1, hash2)
        assert distance == 1  # Only last digit is different
    
    def test_database_cleanup(self):
        """Test database cleanup operations"""
        self.db_manager.init_database()
        
        # Add some test data
        test_files = [
            ("old_file.jpg", {'dhash': '1111', 'ahash': '2222', 'phash': '3333', 'whash': '4444'}),
            ("new_file.jpg", {'dhash': '5555', 'ahash': '6666', 'phash': '7777', 'whash': '8888'})
        ]
        
        for file_path, hashes in test_files:
            self.db_manager.store_hashes(file_path, hashes)
        
        # Test clearing all data
        self.db_manager.clear_all_hashes()
        
        # Verify data was cleared
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM media_hashes")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 0
    
    def test_get_statistics(self):
        """Test getting database statistics"""
        self.db_manager.init_database()
        
        # Add some test data
        test_files = [
            ("image1.jpg", {'dhash': '1111', 'ahash': '2222', 'phash': '3333', 'whash': '4444'}),
            ("image2.png", {'dhash': '5555', 'ahash': '6666', 'phash': '7777', 'whash': '8888'}),
            ("video1.mp4", {'dhash': '9999', 'ahash': 'aaaa', 'phash': 'bbbb', 'whash': 'cccc'})
        ]
        
        for file_path, hashes in test_files:
            self.db_manager.store_hashes(file_path, hashes)
        
        # Get statistics
        stats = self.db_manager.get_statistics()
        
        assert 'total_files' in stats
        assert 'images' in stats
        assert 'videos' in stats
        assert stats['total_files'] == 3
        assert stats['images'] == 2  # .jpg and .png
        assert stats['videos'] == 1  # .mp4


class TestHashCalculatorComplete:
    """Complete tests for Hash Calculator"""
    
    def setup_method(self):
        """Setup test environment"""
        self.hash_calculator = HashCalculator()
        self.test_image_path = "test_hash_image.jpg"
        self.create_test_image()
    
    def teardown_method(self):
        """Clean up test environment"""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def create_test_image(self):
        """Create a test image file"""
        from PIL import Image
        import numpy as np
        
        # Create simple test image
        array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        image.save(self.test_image_path)
    
    def test_calculate_image_hashes(self):
        """Test image hash calculation"""
        hashes = self.hash_calculator.calculate_image_hashes(self.test_image_path)
        
        # Check all hash types are present
        expected_hash_types = ['dhash', 'ahash', 'phash', 'whash']
        for hash_type in expected_hash_types:
            assert hash_type in hashes
            assert isinstance(hashes[hash_type], str)
            assert len(hashes[hash_type]) > 0
    
    def test_hash_consistency(self):
        """Test hash calculation consistency"""
        # Calculate hashes twice for the same image
        hashes1 = self.hash_calculator.calculate_image_hashes(self.test_image_path)
        hashes2 = self.hash_calculator.calculate_image_hashes(self.test_image_path)
        
        # Hashes should be identical for the same image
        for hash_type in ['dhash', 'ahash', 'phash', 'whash']:
            assert hashes1[hash_type] == hashes2[hash_type]
    
    def test_hash_different_images(self):
        """Test hashes are different for different images"""
        # Create second test image
        second_image_path = "test_hash_image2.jpg"
        
        try:
            from PIL import Image
            import numpy as np
            
            # Create different test image
            array = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
            image = Image.fromarray(array)
            image.save(second_image_path)
            
            # Calculate hashes for both images
            hashes1 = self.hash_calculator.calculate_image_hashes(self.test_image_path)
            hashes2 = self.hash_calculator.calculate_image_hashes(second_image_path)
            
            # Hashes should be different (at least for some hash types)
            differences = 0
            for hash_type in ['dhash', 'ahash', 'phash', 'whash']:
                if hashes1[hash_type] != hashes2[hash_type]:
                    differences += 1
            
            assert differences > 0, "Hashes should differ for different images"
            
        finally:
            if os.path.exists(second_image_path):
                os.remove(second_image_path)
    
    def test_invalid_file_handling(self):
        """Test handling of invalid files"""
        # Test non-existent file
        with pytest.raises(Exception):
            self.hash_calculator.calculate_image_hashes("nonexistent_file.jpg")
        
        # Test invalid image file
        invalid_path = "invalid_image.jpg"
        with open(invalid_path, 'w') as f:
            f.write("This is not an image")
        
        try:
            with pytest.raises(Exception):
                self.hash_calculator.calculate_image_hashes(invalid_path)
        finally:
            if os.path.exists(invalid_path):
                os.remove(invalid_path)


class TestFileManagerComplete:
    """Complete tests for File Manager"""
    
    def setup_method(self):
        """Setup test environment"""
        self.file_manager = FileManager()
        self.test_dir = "test_file_manager"
        os.makedirs(self.test_dir, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_file_operations(self):
        """Test basic file operations"""
        # Create test file
        test_file = os.path.join(self.test_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        # Test file existence check
        assert self.file_manager.file_exists(test_file) == True
        assert self.file_manager.file_exists("nonexistent.txt") == False
        
        # Test file info retrieval
        file_info = self.file_manager.get_file_info(test_file)
        assert 'size' in file_info
        assert 'modified_time' in file_info
        assert 'created_time' in file_info
        assert file_info['size'] > 0
    
    def test_directory_operations(self):
        """Test directory operations"""
        # Test directory listing
        files = self.file_manager.list_files(self.test_dir, extensions=['.txt', '.jpg'])
        assert isinstance(files, list)
        
        # Create some test files
        test_files = ['test1.txt', 'test2.jpg', 'test3.png']
        for filename in test_files:
            filepath = os.path.join(self.test_dir, filename)
            with open(filepath, 'w') as f:
                f.write("test")
        
        # Test filtered listing
        txt_files = self.file_manager.list_files(self.test_dir, extensions=['.txt'])
        jpg_files = self.file_manager.list_files(self.test_dir, extensions=['.jpg'])
        
        assert len(txt_files) == 1
        assert len(jpg_files) == 1
        assert txt_files[0].endswith('.txt')
        assert jpg_files[0].endswith('.jpg')
    
    def test_file_metadata_extraction(self):
        """Test file metadata extraction"""
        # Create test image with metadata
        from PIL import Image
        import numpy as np
        
        test_image_path = os.path.join(self.test_dir, "test_metadata.jpg")
        array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        image.save(test_image_path)
        
        # Extract metadata
        metadata = self.file_manager.extract_file_metadata(test_image_path)
        
        assert 'file_size' in metadata
        assert 'format' in metadata
        assert 'dimensions' in metadata
        assert metadata['file_size'] > 0
    
    def test_batch_file_processing(self):
        """Test batch file processing"""
        # Create multiple test files
        test_files = []
        for i in range(5):
            filepath = os.path.join(self.test_dir, f"batch_test_{i}.txt")
            with open(filepath, 'w') as f:
                f.write(f"Content {i}")
            test_files.append(filepath)
        
        # Process batch
        results = self.file_manager.process_batch(test_files)
        
        assert len(results) == len(test_files)
        for result in results:
            assert 'file_path' in result
            assert 'success' in result
            assert result['success'] == True
    
    def test_file_validation(self):
        """Test file validation"""
        # Create valid image file
        from PIL import Image
        import numpy as np
        
        valid_image_path = os.path.join(self.test_dir, "valid.jpg")
        array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        image.save(valid_image_path)
        
        # Create invalid file
        invalid_path = os.path.join(self.test_dir, "invalid.jpg")
        with open(invalid_path, 'w') as f:
            f.write("Not an image")
        
        # Test validation
        assert self.file_manager.validate_image_file(valid_image_path) == True
        assert self.file_manager.validate_image_file(invalid_path) == False
        assert self.file_manager.validate_image_file("nonexistent.jpg") == False
    
    def test_file_cleanup(self):
        """Test file cleanup operations"""
        # Create temporary files
        temp_files = []
        for i in range(3):
            filepath = os.path.join(self.test_dir, f"temp_{i}.tmp")
            with open(filepath, 'w') as f:
                f.write("temporary content")
            temp_files.append(filepath)
        
        # Clean up temporary files
        cleaned_count = self.file_manager.cleanup_temp_files(self.test_dir, pattern="temp_*.tmp")
        
        assert cleaned_count == len(temp_files)
        
        # Verify files were deleted
        for filepath in temp_files:
            assert not os.path.exists(filepath)


class TestSchemaManagerComplete:
    """Complete tests for Schema Manager"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_db_path = "test_schema.db"
        self.schema_manager = SchemaManager(db_path=self.test_db_path)
        self.cleanup_database()
    
    def teardown_method(self):
        """Clean up test environment"""
        self.cleanup_database()
    
    def cleanup_database(self):
        """Clean up test database"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_schema_creation(self):
        """Test database schema creation"""
        self.schema_manager.create_schema()
        
        # Verify database file was created
        assert os.path.exists(self.test_db_path)
        
        # Verify tables were created
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Check for expected tables
        expected_tables = ['media_hashes', 'duplicate_pairs', 'file_metadata', 'analysis_results']
        for table in expected_tables:
            assert table in tables
    
    def test_schema_upgrade(self):
        """Test schema upgrade functionality"""
        # Create initial schema
        self.schema_manager.create_schema()
        
        # Test upgrade to newer version
        success = self.schema_manager.upgrade_schema(version=2)
        assert success == True
        
        # Verify schema version
        version = self.schema_manager.get_schema_version()
        assert version >= 2
    
    def test_schema_validation(self):
        """Test schema validation"""
        # Create schema
        self.schema_manager.create_schema()
        
        # Validate schema
        is_valid = self.schema_manager.validate_schema()
        assert is_valid == True
    
    def test_index_creation(self):
        """Test database index creation"""
        self.schema_manager.create_schema()
        
        # Create indexes for performance
        self.schema_manager.create_indexes()
        
        # Verify indexes were created
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Should have some indexes created
        assert len(indexes) > 0
    
    def test_schema_backup_restore(self):
        """Test schema backup and restore"""
        self.schema_manager.create_schema()
        
        # Add some test data
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO media_hashes (file_path, dhash, ahash, phash, whash)
            VALUES (?, ?, ?, ?, ?)
        """, ("test.jpg", "dhash1", "ahash1", "phash1", "whash1"))
        conn.commit()
        conn.close()
        
        # Create backup
        backup_path = "test_schema_backup.db"
        success = self.schema_manager.backup_database(backup_path)
        assert success == True
        assert os.path.exists(backup_path)
        
        # Clean up backup
        if os.path.exists(backup_path):
            os.remove(backup_path)
    
    def test_migration_scripts(self):
        """Test database migration scripts"""
        self.schema_manager.create_schema()
        
        # Test running migration scripts
        migration_results = self.schema_manager.run_migrations()
        
        assert isinstance(migration_results, list)
        # Each migration should have been successful or skipped
        for result in migration_results:
            assert result['status'] in ['success', 'skipped', 'already_applied']


class TestSimilarityCacheComplete:
    """Complete tests for Similarity Cache"""
    
    def setup_method(self):
        """Setup test environment"""
        self.cache = SimilarityCache(max_size=100)
    
    def test_cache_operations(self):
        """Test basic cache operations"""
        # Test storing and retrieving similarity scores
        file1 = "image1.jpg"
        file2 = "image2.jpg"
        similarity_score = 0.85
        
        # Store similarity
        self.cache.store_similarity(file1, file2, similarity_score)
        
        # Retrieve similarity
        retrieved_score = self.cache.get_similarity(file1, file2)
        assert retrieved_score == similarity_score
        
        # Test symmetric retrieval
        retrieved_score_reverse = self.cache.get_similarity(file2, file1)
        assert retrieved_score_reverse == similarity_score
    
    def test_cache_capacity(self):
        """Test cache capacity limits"""
        # Fill cache beyond capacity
        for i in range(150):  # More than max_size of 100
            self.cache.store_similarity(f"file{i}.jpg", f"file{i+1}.jpg", 0.5)
        
        # Cache should not exceed max size
        assert len(self.cache._cache) <= 100
    
    def test_cache_eviction(self):
        """Test cache eviction policy"""
        # Fill cache to capacity
        for i in range(100):
            self.cache.store_similarity(f"file{i}.jpg", f"file{i+1}.jpg", 0.5)
        
        # Access some entries to mark them as recently used
        for i in range(10):
            self.cache.get_similarity(f"file{i}.jpg", f"file{i+1}.jpg")
        
        # Add more entries to trigger eviction
        for i in range(100, 120):
            self.cache.store_similarity(f"file{i}.jpg", f"file{i+1}.jpg", 0.5)
        
        # Recently accessed entries should still be in cache
        for i in range(10):
            score = self.cache.get_similarity(f"file{i}.jpg", f"file{i+1}.jpg")
            assert score == 0.5
    
    def test_cache_statistics(self):
        """Test cache statistics"""
        # Add some entries
        for i in range(10):
            self.cache.store_similarity(f"file{i}.jpg", f"file{i+1}.jpg", 0.5)
        
        # Access some entries
        for i in range(5):
            self.cache.get_similarity(f"file{i}.jpg", f"file{i+1}.jpg")
        
        # Try to access non-existent entry
        self.cache.get_similarity("nonexistent1.jpg", "nonexistent2.jpg")
        
        # Get statistics
        stats = self.cache.get_statistics()
        
        assert 'cache_size' in stats
        assert 'hit_rate' in stats
        assert 'miss_rate' in stats
        assert stats['cache_size'] == 10
        assert stats['hit_rate'] > 0
    
    def test_cache_clear(self):
        """Test cache clearing"""
        # Add entries to cache
        for i in range(10):
            self.cache.store_similarity(f"file{i}.jpg", f"file{i+1}.jpg", 0.5)
        
        assert len(self.cache._cache) == 10
        
        # Clear cache
        self.cache.clear()
        
        assert len(self.cache._cache) == 0
        
        # Verify entries are gone
        score = self.cache.get_similarity("file1.jpg", "file2.jpg")
        assert score is None
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        file1 = "image1.jpg"
        file2 = "image2.jpg"
        
        # Test that keys are generated consistently
        key1 = self.cache._generate_key(file1, file2)
        key2 = self.cache._generate_key(file2, file1)
        
        # Keys should be the same regardless of order
        assert key1 == key2
        
        # Different files should generate different keys
        key3 = self.cache._generate_key("image3.jpg", "image4.jpg")
        assert key1 != key3


class TestMockDatabaseComplete:
    """Complete tests for Mock Database"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mock_db = MockDatabase()
    
    def test_mock_database_initialization(self):
        """Test mock database initialization"""
        assert self.mock_db is not None
        assert hasattr(self.mock_db, 'store_data')
        assert hasattr(self.mock_db, 'retrieve_data')
    
    def test_mock_data_operations(self):
        """Test mock data storage and retrieval"""
        # Test data
        test_key = "test_file.jpg"
        test_data = {
            'hashes': {'dhash': '1234', 'ahash': '5678'},
            'metadata': {'size': 1024, 'format': 'JPEG'}
        }
        
        # Store data
        self.mock_db.store_data(test_key, test_data)
        
        # Retrieve data
        retrieved_data = self.mock_db.retrieve_data(test_key)
        
        assert retrieved_data == test_data
    
    def test_mock_query_operations(self):
        """Test mock query operations"""
        # Add test data
        test_data = [
            ("file1.jpg", {'confidence': 0.8, 'is_fake': False}),
            ("file2.jpg", {'confidence': 0.9, 'is_fake': True}),
            ("file3.jpg", {'confidence': 0.6, 'is_fake': False})
        ]
        
        for key, data in test_data:
            self.mock_db.store_data(key, data)
        
        # Test query by condition
        fake_files = self.mock_db.query_data(lambda data: data.get('is_fake', False))
        
        assert len(fake_files) == 1
        assert fake_files[0][0] == "file2.jpg"
    
    def test_mock_bulk_operations(self):
        """Test mock bulk operations"""
        # Prepare bulk data
        bulk_data = {}
        for i in range(100):
            key = f"bulk_file_{i}.jpg"
            data = {'id': i, 'processed': True}
            bulk_data[key] = data
        
        # Bulk store
        self.mock_db.bulk_store(bulk_data)
        
        # Verify all data was stored
        for key, expected_data in bulk_data.items():
            retrieved_data = self.mock_db.retrieve_data(key)
            assert retrieved_data == expected_data
    
    def test_mock_statistics(self):
        """Test mock database statistics"""
        # Add some test data
        for i in range(10):
            key = f"stats_file_{i}.jpg"
            data = {'id': i, 'category': 'test'}
            self.mock_db.store_data(key, data)
        
        # Get statistics
        stats = self.mock_db.get_statistics()
        
        assert 'total_entries' in stats
        assert stats['total_entries'] == 10
    
    def test_mock_reset(self):
        """Test mock database reset"""
        # Add some data
        for i in range(5):
            key = f"reset_test_{i}.jpg"
            data = {'id': i}
            self.mock_db.store_data(key, data)
        
        # Verify data exists
        assert len(self.mock_db._data) == 5
        
        # Reset database
        self.mock_db.reset()
        
        # Verify data is cleared
        assert len(self.mock_db._data) == 0


# Integration tests for database components
class TestDatabaseIntegration:
    """Integration tests for database components working together"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_db_path = "test_integration.db"
        self.db_manager = DatabaseManager(db_path=self.test_db_path)
        self.schema_manager = SchemaManager(db_path=self.test_db_path)
        self.hash_calculator = HashCalculator()
        self.similarity_cache = SimilarityCache()
        self.file_manager = FileManager()
        
        # Clean up
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def teardown_method(self):
        """Clean up test environment"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_complete_workflow(self):
        """Test complete database workflow"""
        # 1. Create schema
        self.schema_manager.create_schema()
        
        # 2. Initialize database
        self.db_manager.init_database()
        
        # 3. Create test image
        from PIL import Image
        import numpy as np
        
        test_image_path = "integration_test.jpg"
        array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        image.save(test_image_path)
        
        try:
            # 4. Calculate hashes
            hashes = self.hash_calculator.calculate_image_hashes(test_image_path)
            
            # 5. Store hashes in database
            self.db_manager.store_hashes(test_image_path, hashes)
            
            # 6. Retrieve hashes from database
            retrieved_hashes = self.db_manager.get_hashes(test_image_path)
            
            # 7. Verify hashes match
            assert retrieved_hashes == hashes
            
            # 8. Store similarity in cache
            self.similarity_cache.store_similarity(test_image_path, "similar_image.jpg", 0.95)
            
            # 9. Retrieve similarity from cache
            cached_similarity = self.similarity_cache.get_similarity(test_image_path, "similar_image.jpg")
            assert cached_similarity == 0.95
            
            # 10. Get database statistics
            stats = self.db_manager.get_statistics()
            assert stats['total_files'] == 1
            
        finally:
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
    
    def test_batch_processing_integration(self):
        """Test batch processing with all database components"""
        self.schema_manager.create_schema()
        self.db_manager.init_database()
        
        # Create multiple test images
        test_images = []
        for i in range(5):
            from PIL import Image
            import numpy as np
            
            image_path = f"batch_test_{i}.jpg"
            array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            # Add some variation
            array[10:40, 10:40] = [i * 50, i * 30, i * 20]
            image = Image.fromarray(array)
            image.save(image_path)
            test_images.append(image_path)
        
        try:
            # Process all images
            all_hashes = {}
            for image_path in test_images:
                # Calculate and store hashes
                hashes = self.hash_calculator.calculate_image_hashes(image_path)
                self.db_manager.store_hashes(image_path, hashes)
                all_hashes[image_path] = hashes
            
            # Find similar images
            for image_path, hashes in all_hashes.items():
                similar_files = self.db_manager.find_similar_hashes(hashes, threshold=10)
                
                # Should at least find itself
                similar_paths = [result['file_path'] for result in similar_files]
                assert image_path in similar_paths
            
            # Verify all files are in database
            stats = self.db_manager.get_statistics()
            assert stats['total_files'] == len(test_images)
            
        finally:
            # Cleanup
            for image_path in test_images:
                if os.path.exists(image_path):
                    os.remove(image_path)
    
    def test_error_handling_integration(self):
        """Test error handling across database components"""
        # Test schema creation with invalid path
        invalid_schema_manager = SchemaManager(db_path="/invalid/path/test.db")
        
        # Should handle error gracefully
        try:
            invalid_schema_manager.create_schema()
        except Exception as e:
            # Should get permission or path error
            assert isinstance(e, (PermissionError, FileNotFoundError, OSError))
        
        # Test hash calculation with invalid file
        with pytest.raises(Exception):
            self.hash_calculator.calculate_image_hashes("nonexistent_file.jpg")
        
        # Test database operations on non-existent database
        invalid_db_manager = DatabaseManager(db_path="nonexistent_dir/test.db")
        
        try:
            invalid_db_manager.store_hashes("test.jpg", {'dhash': '1234'})
        except Exception:
            # Should handle gracefully
            pass


# Test fixtures for database components
@pytest.fixture
def database_manager():
    """Database manager fixture"""
    db_path = "fixture_test.db"
    db_manager = DatabaseManager(db_path=db_path)
    yield db_manager
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

@pytest.fixture
def hash_calculator():
    """Hash calculator fixture"""
    return HashCalculator()

@pytest.fixture
def similarity_cache():
    """Similarity cache fixture"""
    return SimilarityCache()

@pytest.fixture
def file_manager():
    """File manager fixture"""
    return FileManager()

@pytest.fixture
def test_image_file():
    """Test image file fixture"""
    from PIL import Image
    import numpy as np
    
    image_path = "fixture_test_image.jpg"
    array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    image.save(image_path)
    
    yield image_path
    
    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)
