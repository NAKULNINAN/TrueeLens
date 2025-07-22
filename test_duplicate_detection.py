"""
Test script for the Duplicate Detection System

This script tests the functionality of the duplicate detection system.
"""

import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np
from detection.duplicate_detector import DuplicateDetector, DatabaseManager
import time

def create_test_image(width=100, height=100, color_variation=0):
    """Create a test image with optional color variation"""
    # Create a base image
    base_color = [128, 64, 192]  # Purple-ish base color
    
    # Add variation to create similar but not identical images
    for i in range(3):
        base_color[i] = max(0, min(255, base_color[i] + color_variation))
    
    img_array = np.full((height, width, 3), base_color, dtype=np.uint8)
    
    # Add some pattern for more realistic hashing
    for i in range(0, height, 10):
        for j in range(0, width, 10):
            if (i + j) % 20 == 0:
                img_array[i:i+5, j:j+5] = [255, 255, 255]  # White squares
    
    return Image.fromarray(img_array)

def test_basic_functionality():
    """Test basic duplicate detection functionality"""
    print("Testing basic functionality...")
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test images
        original_image = create_test_image()
        duplicate_image = create_test_image()  # Identical
        similar_image = create_test_image(color_variation=5)  # Slightly different
        different_image = create_test_image(color_variation=100)  # Very different
        
        # Save test images
        original_path = temp_path / "original.jpg"
        duplicate_path = temp_path / "duplicate.jpg"
        similar_path = temp_path / "similar.jpg"
        different_path = temp_path / "different.jpg"
        
        original_image.save(original_path)
        duplicate_image.save(duplicate_path)
        similar_image.save(similar_path)
        different_image.save(different_path)
        
        # Clear database before testing
        detector = DuplicateDetector()
        detector.db_manager.init_database()
        
        # Clear existing data
        import sqlite3
        conn = sqlite3.connect(detector.db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM media_hashes")
        conn.commit()
        conn.close()
        
        print(f"Created test images in {temp_dir}")
        
        # Process files
        print("Processing test files...")
        detector.process_file(str(original_path))
        detector.process_file(str(duplicate_path))
        detector.process_file(str(similar_path))
        detector.process_file(str(different_path))
        
        # Find duplicates
        print("Finding duplicates...")
        duplicates = detector.find_all_duplicates()
        
        print(f"Found {len(duplicates)} duplicate pairs:")
        for i, dup in enumerate(duplicates, 1):
            print(f"  {i}. {Path(dup.original_file).name} <-> {Path(dup.duplicate_file).name}")
            print(f"     Similarity: {dup.similarity_score:.3f} ({dup.hash_type}, {dup.confidence})")
        
        # Test statistics
        stats = detector.get_statistics()
        print(f"\\nStatistics:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Images: {stats['images']}")
        print(f"  Videos: {stats['videos']}")
        
        return len(duplicates)

def test_similarity_thresholds():
    """Test different similarity thresholds"""
    print("\\nTesting similarity thresholds...")
    
    # Test with strict thresholds
    strict_detector = DuplicateDetector({
        'dhash': 2, 'ahash': 2, 'phash': 2, 'whash': 2
    })
    
    # Test with lenient thresholds
    lenient_detector = DuplicateDetector({
        'dhash': 10, 'ahash': 12, 'phash': 15, 'whash': 10
    })
    
    print(f"Strict thresholds: {strict_detector.similarity_threshold}")
    print(f"Lenient thresholds: {lenient_detector.similarity_threshold}")

def test_hash_calculation():
    """Test hash calculation for images"""
    print("\\nTesting hash calculation...")
    
    detector = DuplicateDetector()
    
    # Create two identical images
    image1 = create_test_image()
    image2 = create_test_image()
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp1, \\
         tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp2:
        
        image1.save(tmp1.name)
        image2.save(tmp2.name)
        
        hashes1 = detector.hasher.calculate_image_hashes(tmp1.name)
        hashes2 = detector.hasher.calculate_image_hashes(tmp2.name)
        
        print(f"Image 1 hashes: {hashes1}")
        print(f"Image 2 hashes: {hashes2}")
        
        # Calculate Hamming distances
        for hash_type in ['dhash', 'ahash', 'phash', 'whash']:
            distance = detector.db_manager._hamming_distance(
                hashes1[hash_type], hashes2[hash_type]
            )
            print(f"{hash_type} distance: {distance}")
        
        # Clean up
        os.unlink(tmp1.name)
        os.unlink(tmp2.name)

def test_uploaded_file_simulation():
    """Test the detect() method that simulates uploaded file processing"""
    print("\\nTesting uploaded file simulation...")
    
    # Create a test image
    test_image = create_test_image()
    
    # Save it to get file data
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        test_image.save(tmp_file.name)
        
        # Read file data to simulate uploaded file
        with open(tmp_file.name, 'rb') as f:
            file_data = f.read()
        
        # Create a mock uploaded file object
        class MockUploadedFile:
            def __init__(self, data, name):
                self.data = data
                self.name = name
                self.position = 0
            
            def read(self):
                return self.data
        
        mock_file = MockUploadedFile(file_data, "test_image.jpg")
        
        detector = DuplicateDetector()
        result = detector.detect(mock_file)
        
        print(f"Detection result: {result}")
        
        # Clean up
        os.unlink(tmp_file.name)

def performance_test():
    """Test performance with multiple files"""
    print("\\nPerformance test...")
    
    detector = DuplicateDetector()
    
    # Clear database
    import sqlite3
    conn = sqlite3.connect(detector.db_manager.db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM media_hashes")
    conn.commit()
    conn.close()
    
    # Create multiple test images
    num_images = 50
    print(f"Creating {num_images} test images...")
    
    start_time = time.time()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create images with some duplicates
        for i in range(num_images):
            if i % 10 == 0 and i > 0:
                # Create a duplicate every 10th image
                base_image = create_test_image()
            else:
                # Create a unique image
                base_image = create_test_image(color_variation=i * 2)
            
            image_path = temp_path / f"test_image_{i:03d}.jpg"
            base_image.save(image_path)
            
            # Process the image
            detector.process_file(str(image_path))
        
        processing_time = time.time() - start_time
        print(f"Processing {num_images} images took {processing_time:.2f} seconds")
        
        # Find duplicates
        start_time = time.time()
        duplicates = detector.find_all_duplicates()
        detection_time = time.time() - start_time
        
        print(f"Duplicate detection took {detection_time:.2f} seconds")
        print(f"Found {len(duplicates)} duplicate pairs")
        
        # Show statistics
        stats = detector.get_statistics()
        print(f"Database contains {stats['total_files']} files")

def main():
    """Main test function"""
    print("Duplicate Detection System - Test Suite")
    print("=" * 50)
    
    try:
        # Run basic functionality test
        num_duplicates = test_basic_functionality()
        
        # Run other tests
        test_similarity_thresholds()
        test_hash_calculation()
        test_uploaded_file_simulation()
        performance_test()
        
        print("\\n" + "=" * 50)
        print("All tests completed successfully!")
        print(f"Basic test found {num_duplicates} duplicate pairs")
        
    except Exception as e:
        print(f"\\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
