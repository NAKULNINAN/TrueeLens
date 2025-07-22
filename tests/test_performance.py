"""
Performance benchmarks for Media Forensics components
"""

import pytest
import time
import psutil
import json
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detection.ai_image_detector import AIImageDetector
from detection.duplicate_detector import DuplicateDetector
from utils.media_processor import MediaProcessor


class PerformanceBenchmark:
    """Base class for performance benchmarking"""
    
    def __init__(self):
        self.results = {}
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version,
            "platform": sys.platform
        }
    
    def measure_time_and_memory(self, func, *args, **kwargs):
        """Measure execution time and memory usage"""
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get final memory
        final_memory = process.memory_info().rss / (1024**2)  # MB
        memory_used = final_memory - initial_memory
        
        return {
            "result": result,
            "execution_time": end_time - start_time,
            "memory_used_mb": memory_used,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory
        }
    
    def save_results(self, filename: str):
        """Save benchmark results to JSON file"""
        benchmark_data = {
            "system_info": self.system_info,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"Benchmark results saved to {filename}")


@pytest.mark.performance
class TestAIDetectorPerformance(PerformanceBenchmark):
    """Performance benchmarks for AI detection components"""
    
    def setup_method(self):
        """Setup test data"""
        super().__init__()
        self.detector = AIImageDetector()
        self.test_images = self._generate_test_images()
    
    def _generate_test_images(self, sizes=[(224, 224), (512, 512), (1024, 1024)]) -> List[Image.Image]:
        """Generate test images of different sizes"""
        images = []
        for size in sizes:
            # Create test image with some complexity
            array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            # Add some patterns
            for i in range(0, size[0], 50):
                for j in range(0, size[1], 50):
                    array[i:i+25, j:j+25] = [255, 0, 0]  # Red squares
            images.append(Image.fromarray(array))
        return images
    
    def test_ai_detection_performance_by_image_size(self):
        """Benchmark AI detection performance across different image sizes"""
        results = {}
        
        for i, image in enumerate(self.test_images):
            size = image.size
            print(f"Testing AI detection with image size {size}")
            
            # Create mock file
            class MockFile:
                def __init__(self, img):
                    self.name = f"test_{size[0]}x{size[1]}.jpg"
                    self.type = "image/jpeg"
                    import io
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG')
                    self.content = buffer.getvalue()
                    self.size = len(self.content)
                
                def read(self):
                    return self.content
            
            mock_file = MockFile(image)
            
            # Benchmark detection
            benchmark = self.measure_time_and_memory(
                self.detector.detect, 
                mock_file, 
                threshold=0.5, 
                enable_viz=False
            )
            
            results[f"{size[0]}x{size[1]}"] = {
                "execution_time": benchmark["execution_time"],
                "memory_used_mb": benchmark["memory_used_mb"],
                "pixels": size[0] * size[1],
                "throughput_pixels_per_sec": (size[0] * size[1]) / benchmark["execution_time"]
            }
            
            print(f"  Time: {benchmark['execution_time']:.3f}s, Memory: {benchmark['memory_used_mb']:.2f}MB")
        
        self.results["ai_detection_by_size"] = results
        assert all(r["execution_time"] < 30 for r in results.values()), "All detections should complete within 30s"
    
    def test_batch_ai_detection_performance(self):
        """Benchmark batch processing performance"""
        batch_sizes = [1, 5, 10]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Testing batch AI detection with batch size {batch_size}")
            
            # Use smallest image for batch testing
            test_image = self.test_images[0]
            
            class MockFile:
                def __init__(self, img, index):
                    self.name = f"batch_test_{index}.jpg"
                    self.type = "image/jpeg"
                    import io
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG')
                    self.content = buffer.getvalue()
                    self.size = len(self.content)
                
                def read(self):
                    return self.content
            
            mock_files = [MockFile(test_image, i) for i in range(batch_size)]
            
            def batch_process():
                results = []
                for mock_file in mock_files:
                    result = self.detector.detect(mock_file, threshold=0.5, enable_viz=False)
                    results.append(result)
                return results
            
            benchmark = self.measure_time_and_memory(batch_process)
            
            results[f"batch_{batch_size}"] = {
                "execution_time": benchmark["execution_time"],
                "memory_used_mb": benchmark["memory_used_mb"],
                "avg_time_per_image": benchmark["execution_time"] / batch_size,
                "throughput_images_per_sec": batch_size / benchmark["execution_time"]
            }
            
            print(f"  Total time: {benchmark['execution_time']:.3f}s")
            print(f"  Avg per image: {benchmark['execution_time']/batch_size:.3f}s")
        
        self.results["batch_ai_detection"] = results


@pytest.mark.performance
class TestDuplicateDetectorPerformance(PerformanceBenchmark):
    """Performance benchmarks for duplicate detection"""
    
    def setup_method(self):
        """Setup test environment"""
        super().__init__()
        self.detector = DuplicateDetector()
        
        # Clear database for clean testing
        import sqlite3
        try:
            conn = sqlite3.connect(self.detector.db_manager.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM media_hashes")
            conn.commit()
            conn.close()
        except:
            pass  # Database might not exist yet
    
    def test_hash_calculation_performance(self):
        """Benchmark hash calculation performance"""
        image_sizes = [(224, 224), (512, 512), (1024, 1024)]
        results = {}
        
        for size in image_sizes:
            print(f"Testing hash calculation with image size {size}")
            
            # Create test image
            array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            image = Image.fromarray(array)
            
            # Save to temporary file
            temp_path = f"temp_hash_test_{size[0]}x{size[1]}.jpg"
            image.save(temp_path)
            
            try:
                # Benchmark hash calculation
                benchmark = self.measure_time_and_memory(
                    self.detector.hasher.calculate_image_hashes,
                    temp_path
                )
                
                results[f"{size[0]}x{size[1]}"] = {
                    "execution_time": benchmark["execution_time"],
                    "memory_used_mb": benchmark["memory_used_mb"],
                    "pixels": size[0] * size[1],
                    "hashes_generated": len(benchmark["result"]) if benchmark["result"] else 0
                }
                
                print(f"  Time: {benchmark['execution_time']:.3f}s")
                
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        self.results["hash_calculation"] = results
    
    def test_duplicate_search_performance(self):
        """Benchmark duplicate search performance with different database sizes"""
        database_sizes = [10, 50, 100]
        results = {}
        
        for db_size in database_sizes:
            print(f"Testing duplicate search with database size {db_size}")
            
            # Generate and process multiple images
            temp_files = []
            for i in range(db_size):
                # Create slightly different images to avoid too many duplicates
                array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                array[0:50, 0:50] = [i % 255, (i*2) % 255, (i*3) % 255]  # Unique corner
                image = Image.fromarray(array)
                
                temp_path = f"temp_dup_test_{i}.jpg"
                image.save(temp_path)
                temp_files.append(temp_path)
                
                # Process file
                self.detector.process_file(temp_path)
            
            try:
                # Benchmark duplicate search
                benchmark = self.measure_time_and_memory(
                    self.detector.find_all_duplicates
                )
                
                results[f"db_size_{db_size}"] = {
                    "execution_time": benchmark["execution_time"],
                    "memory_used_mb": benchmark["memory_used_mb"],
                    "database_size": db_size,
                    "duplicates_found": len(benchmark["result"]) if benchmark["result"] else 0,
                    "search_rate_files_per_sec": db_size / benchmark["execution_time"]
                }
                
                print(f"  Time: {benchmark['execution_time']:.3f}s")
                print(f"  Found {len(benchmark['result']) if benchmark['result'] else 0} duplicates")
                
            finally:
                # Cleanup
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
        
        self.results["duplicate_search"] = results


@pytest.mark.performance
class TestMediaProcessorPerformance(PerformanceBenchmark):
    """Performance benchmarks for media processing utilities"""
    
    def setup_method(self):
        """Setup test environment"""
        super().__init__()
        self.processor = MediaProcessor()
    
    def test_image_processing_performance(self):
        """Benchmark image processing performance"""
        image_sizes = [(224, 224), (512, 512), (1024, 1024), (2048, 2048)]
        results = {}
        
        for size in image_sizes:
            print(f"Testing image processing with size {size}")
            
            # Create test image
            array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            image = Image.fromarray(array)
            
            # Benchmark basic processing
            benchmark = self.measure_time_and_memory(
                self.processor.process_image,
                image,
                enhance=True
            )
            
            results[f"{size[0]}x{size[1]}"] = {
                "execution_time": benchmark["execution_time"],
                "memory_used_mb": benchmark["memory_used_mb"],
                "pixels": size[0] * size[1],
                "throughput_pixels_per_sec": (size[0] * size[1]) / benchmark["execution_time"]
            }
            
            print(f"  Time: {benchmark['execution_time']:.3f}s")
        
        self.results["image_processing"] = results
    
    def test_format_conversion_performance(self):
        """Benchmark format conversion performance"""
        formats = ['JPEG', 'PNG', 'WEBP']
        results = {}
        
        # Create test image
        array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        input_path = "temp_conversion_input.jpg"
        image.save(input_path)
        
        try:
            for fmt in formats:
                print(f"Testing conversion to {fmt}")
                
                output_path = f"temp_conversion_output.{fmt.lower()}"
                
                benchmark = self.measure_time_and_memory(
                    self.processor.convert_image_format,
                    input_path,
                    output_path,
                    target_format=fmt,
                    quality=90
                )
                
                # Get file sizes for compression analysis
                input_size = os.path.getsize(input_path) / 1024  # KB
                output_size = os.path.getsize(output_path) / 1024 if os.path.exists(output_path) else 0  # KB
                
                results[fmt] = {
                    "execution_time": benchmark["execution_time"],
                    "memory_used_mb": benchmark["memory_used_mb"],
                    "input_size_kb": input_size,
                    "output_size_kb": output_size,
                    "compression_ratio": input_size / output_size if output_size > 0 else 0
                }
                
                print(f"  Time: {benchmark['execution_time']:.3f}s")
                
                # Cleanup output file
                if os.path.exists(output_path):
                    os.remove(output_path)
        
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
        
        self.results["format_conversion"] = results


@pytest.mark.performance
@pytest.mark.slow
def test_run_all_benchmarks():
    """Run all performance benchmarks and save results"""
    print("Starting comprehensive performance benchmarks...")
    
    # AI Detection Benchmarks
    ai_benchmark = TestAIDetectorPerformance()
    ai_benchmark.setup_method()
    ai_benchmark.test_ai_detection_performance_by_image_size()
    ai_benchmark.test_batch_ai_detection_performance()
    
    # Duplicate Detection Benchmarks
    dup_benchmark = TestDuplicateDetectorPerformance()
    dup_benchmark.setup_method()
    dup_benchmark.test_hash_calculation_performance()
    dup_benchmark.test_duplicate_search_performance()
    
    # Media Processor Benchmarks
    mp_benchmark = TestMediaProcessorPerformance()
    mp_benchmark.setup_method()
    mp_benchmark.test_image_processing_performance()
    mp_benchmark.test_format_conversion_performance()
    
    # Combine all results
    combined_results = {
        "ai_detection": ai_benchmark.results,
        "duplicate_detection": dup_benchmark.results,
        "media_processing": mp_benchmark.results
    }
    
    # Save comprehensive benchmark report
    benchmark_data = {
        "system_info": ai_benchmark.system_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": combined_results
    }
    
    os.makedirs("tests/benchmark_results", exist_ok=True)
    filename = f"tests/benchmark_results/comprehensive_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"Comprehensive benchmark results saved to {filename}")
    
    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*50)
    
    for category, category_results in combined_results.items():
        print(f"\n{category.upper()}:")
        for test_name, test_results in category_results.items():
            print(f"  {test_name}:")
            if isinstance(test_results, dict):
                for key, value in test_results.items():
                    if isinstance(value, dict) and 'execution_time' in value:
                        print(f"    {key}: {value['execution_time']:.3f}s")
                    elif key == 'execution_time':
                        print(f"    {key}: {value:.3f}s")


if __name__ == "__main__":
    test_run_all_benchmarks()
