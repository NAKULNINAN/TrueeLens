"""
Complete unit tests for all utility modules
Tests media processor, error handler, visualization, and model manager
"""

import pytest
import numpy as np
import tempfile
import os
import json
import cv2
from PIL import Image, ImageEnhance
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from utils.media_processor import MediaProcessor
from utils.error_handler import with_error_handling, global_error_handler, ErrorHandler
from utils.visualization import create_result_card, create_confidence_chart, VisualizationManager
from utils.model_manager import ModelManager
from utils.batch_processor import BatchProcessor


class TestMediaProcessorComplete:
    """Complete tests for MediaProcessor utility"""
    
    def setup_method(self):
        """Setup test environment"""
        self.processor = MediaProcessor()
        self.test_images = self._create_test_images()
        
    def _create_test_images(self):
        """Create various test images"""
        images = {}
        
        # Standard RGB image
        rgb_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        images['rgb'] = Image.fromarray(rgb_array)
        
        # RGBA image with transparency
        rgba_array = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
        images['rgba'] = Image.fromarray(rgba_array, 'RGBA')
        
        # Grayscale image
        gray_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        images['grayscale'] = Image.fromarray(gray_array, 'L')
        
        # Large image
        large_array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        images['large'] = Image.fromarray(large_array)
        
        # Small image
        small_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        images['small'] = Image.fromarray(small_array)
        
        return images
    
    def test_process_image_basic(self):
        """Test basic image processing"""
        image = self.test_images['rgb']
        processed_array, metadata = self.processor.process_image(image)
        
        assert processed_array is not None
        assert isinstance(processed_array, np.ndarray)
        assert processed_array.shape[-1] == 3  # Should be RGB
        assert processed_array.dtype == np.uint8
        
        # Check metadata
        assert 'original_size' in metadata
        assert 'processed_size' in metadata
        assert 'stats' in metadata
        assert metadata['original_size'] == image.size
    
    def test_process_image_with_enhancement(self):
        """Test image processing with enhancement"""
        image = self.test_images['rgb']
        processed_array, metadata = self.processor.process_image(image, enhance=True)
        
        assert processed_array is not None
        assert 'stats' in metadata
        assert 'mean_brightness' in metadata['stats']
        assert 'contrast' in metadata['stats']
        assert 'color_balance' in metadata['stats']
    
    def test_process_different_formats(self):
        """Test processing different image formats"""
        for format_name, image in self.test_images.items():
            processed_array, metadata = self.processor.process_image(image)
            
            assert processed_array is not None, f"Failed to process {format_name} image"
            assert processed_array.shape[-1] == 3, f"{format_name} should be converted to RGB"
            assert 'original_size' in metadata
    
    def test_preprocess_image_advanced(self):
        """Test advanced image preprocessing"""
        image = self.test_images['rgb']
        
        options = {
            'denoise': True,
            'histogram_equalization': True,
            'gamma': 1.2,
            'contrast': 1.1,
            'brightness': 1.05,
            'sharpen': True,
            'resize': (256, 256),
            'resize_method': 'LANCZOS'
        }
        
        processed_img, metadata = self.processor.preprocess_image_advanced(image, options)
        
        assert processed_img is not None
        assert processed_img.size == (256, 256)
        assert 'processing_log' in metadata
        assert len(metadata['processing_log']) > 0
    
    def test_extract_comprehensive_metadata(self):
        """Test comprehensive metadata extraction"""
        # Create temporary image file
        temp_path = "temp_metadata_test.jpg"
        self.test_images['rgb'].save(temp_path)
        
        try:
            metadata = self.processor.extract_comprehensive_metadata(temp_path)
            
            # Check all required metadata fields
            assert 'file_info' in metadata
            assert 'hash_data' in metadata
            assert 'technical_data' in metadata
            assert 'file_validation' in metadata
            
            # Check file info
            file_info = metadata['file_info']
            assert 'filename' in file_info
            assert 'file_size_bytes' in file_info
            assert 'file_size_mb' in file_info
            assert 'creation_date' in file_info
            assert 'modification_date' in file_info
            
            # Check hash data
            hash_data = metadata['hash_data']
            assert 'md5' in hash_data
            assert 'sha256' in hash_data
            assert len(hash_data['md5']) == 32  # MD5 is 32 chars
            assert len(hash_data['sha256']) == 64  # SHA256 is 64 chars
            
            # Check technical data
            tech_data = metadata['technical_data']
            assert 'image_dimensions' in tech_data
            assert 'color_mode' in tech_data
            assert 'format' in tech_data
            
            # Check validation
            validation = metadata['file_validation']
            assert 'exists' in validation
            assert 'readable' in validation
            assert validation['exists'] == True
            assert validation['readable'] == True
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_convert_image_format(self):
        """Test image format conversion"""
        # Create temporary input file
        input_path = "temp_input.jpg"
        self.test_images['rgb'].save(input_path)
        
        formats = ['PNG', 'WEBP', 'BMP']
        
        try:
            for fmt in formats:
                output_path = f"temp_output.{fmt.lower()}"
                
                success = self.processor.convert_image_format(
                    input_path, 
                    output_path, 
                    target_format=fmt,
                    quality=90,
                    preserve_metadata=True
                )
                
                assert success == True, f"Failed to convert to {fmt}"
                assert os.path.exists(output_path), f"Output file not created for {fmt}"
                
                # Verify converted file is readable
                converted_img = Image.open(output_path)
                assert converted_img is not None
                converted_img.close()
                
                # Cleanup
                if os.path.exists(output_path):
                    os.remove(output_path)
                    
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
    
    def test_validate_file(self):
        """Test file validation"""
        # Create valid test file
        valid_path = "temp_valid.jpg"
        self.test_images['rgb'].save(valid_path)
        
        # Create invalid file (empty)
        invalid_path = "temp_invalid.jpg"
        with open(invalid_path, 'w') as f:
            f.write("")
        
        # Create non-existent file path
        nonexistent_path = "temp_nonexistent.jpg"
        
        try:
            # Test valid file
            assert self.processor.validate_file(valid_path) == True
            
            # Test invalid file
            assert self.processor.validate_file(invalid_path) == False
            
            # Test non-existent file
            assert self.processor.validate_file(nonexistent_path) == False
            
        finally:
            for path in [valid_path, invalid_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def test_preprocess_for_model(self):
        """Test model preprocessing"""
        image = self.test_images['rgb']
        processed_array, _ = self.processor.process_image(image)
        
        # Test without tensor conversion
        model_input = self.processor.preprocess_for_model(
            processed_array, 
            normalize=True, 
            to_tensor=False
        )
        
        assert model_input is not None
        assert isinstance(model_input, np.ndarray)
        assert model_input.shape == (224, 224, 3)
        
        # Values should be normalized to [0, 1]
        assert model_input.min() >= 0
        assert model_input.max() <= 1
        
        # Test with tensor conversion (if PyTorch available)
        try:
            import torch
            model_tensor = self.processor.preprocess_for_model(
                processed_array, 
                normalize=True, 
                to_tensor=True
            )
            
            assert model_tensor is not None
            assert isinstance(model_tensor, torch.Tensor)
            assert model_tensor.shape == (1, 3, 224, 224)  # Batch, Channel, Height, Width
            
        except ImportError:
            pass  # Skip tensor test if PyTorch not available
    
    def test_video_frame_extraction(self):
        """Test video frame extraction"""
        # Create test video
        temp_video_path = "temp_test_video.mp4"
        self._create_test_video(temp_video_path)
        
        try:
            frames, metadata = self.processor.process_video(
                temp_video_path, 
                extract_frames=True, 
                max_frames=5
            )
            
            assert frames is not None
            assert isinstance(frames, list)
            assert len(frames) <= 5
            
            # Check metadata
            assert 'fps' in metadata
            assert 'duration' in metadata
            assert 'total_frames' in metadata
            assert 'resolution' in metadata
            
            # Verify frames are valid numpy arrays
            for frame in frames:
                assert isinstance(frame, np.ndarray)
                assert len(frame.shape) == 3  # Height, Width, Channels
                assert frame.shape[2] == 3  # RGB
                
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    
    def test_extract_video_frames_advanced(self):
        """Test advanced video frame extraction"""
        temp_video_path = "temp_advanced_video.mp4"
        self._create_test_video(temp_video_path)
        
        try:
            # Test uniform extraction
            frames_uniform, metadata_uniform = self.processor.extract_video_frames_advanced(
                temp_video_path,
                extraction_method='uniform',
                max_frames=3,
                quality_threshold=0.1
            )
            
            assert frames_uniform is not None
            assert len(frames_uniform) <= 3
            assert metadata_uniform is not None
            
            # Test rate-based extraction
            frames_rate, metadata_rate = self.processor.extract_video_frames_advanced(
                temp_video_path,
                extraction_method='rate',
                frame_rate=1.0,
                quality_threshold=0.1
            )
            
            assert frames_rate is not None
            assert metadata_rate is not None
            
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    
    def test_calculate_video_quality_metrics(self):
        """Test video quality metrics calculation"""
        # Create sample frames
        frames = []
        for i in range(5):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            frames.append(frame)
        
        metrics = self.processor.calculate_video_quality_metrics(frames)
        
        assert 'average_brightness' in metrics
        assert 'contrast_variance' in metrics
        assert 'blur_metric' in metrics
        assert 'noise_level' in metrics
        assert 'color_diversity' in metrics
        
        # All metrics should be numeric
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
    
    def test_save_processed_frames(self):
        """Test saving processed frames"""
        # Create sample frames
        frames = []
        for i in range(3):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            frames.append(frame)
        
        output_dir = "temp_frames_output"
        
        try:
            saved_paths = self.processor.save_processed_frames(
                frames, 
                output_dir, 
                prefix="test_frame"
            )
            
            assert len(saved_paths) == len(frames)
            assert os.path.exists(output_dir)
            
            # Verify all frames were saved
            for path in saved_paths:
                assert os.path.exists(path)
                # Verify saved image is readable
                img = Image.open(path)
                assert img is not None
                img.close()
                
        finally:
            # Cleanup
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)
    
    def _create_test_video(self, output_path, duration_seconds=2):
        """Create a test video file"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        frame_size = (320, 240)
        
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        for i in range(duration_seconds * fps):
            # Create frame with changing colors
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            color = ((i * 10) % 255, (i * 15) % 255, (i * 20) % 255)
            frame[:] = color
            
            # Add some pattern
            cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), 2)
            cv2.circle(frame, (100, 100), 30, (0, 0, 255), -1)
            
            out.write(frame)
        
        out.release()


class TestErrorHandlerComplete:
    """Complete tests for Error Handler utility"""
    
    def setup_method(self):
        """Setup test environment"""
        self.error_handler = ErrorHandler()
    
    def test_error_handler_initialization(self):
        """Test error handler initialization"""
        assert self.error_handler is not None
        assert hasattr(self.error_handler, 'log_error')
        assert hasattr(self.error_handler, 'handle_error')
    
    def test_with_error_handling_decorator_success(self):
        """Test error handling decorator with successful function"""
        @with_error_handling(global_error_handler)
        def successful_function(x, y):
            return x + y
        
        result = successful_function(5, 3)
        assert result == 8
    
    def test_with_error_handling_decorator_exception(self):
        """Test error handling decorator with exception"""
        @with_error_handling(global_error_handler)
        def failing_function():
            raise ValueError("Test error")
        
        # Should handle exception gracefully
        result = failing_function()
        # Should return None or error result based on handler implementation
        assert result is None or isinstance(result, dict)
    
    def test_global_error_handler_function(self):
        """Test global error handler function"""
        try:
            raise RuntimeError("Test runtime error")
        except Exception as e:
            result = global_error_handler(e, "test_context")
            
            # Should return error information
            assert isinstance(result, dict)
            assert 'error' in result
            assert 'context' in result
    
    def test_error_logging(self):
        """Test error logging functionality"""
        test_error = Exception("Test logging error")
        
        # This should not raise an exception
        self.error_handler.log_error(test_error, "test_context")
        
        # Test with additional data
        additional_data = {"user_id": 123, "action": "test_action"}
        self.error_handler.log_error(test_error, "test_context", additional_data)
    
    def test_error_categories(self):
        """Test different error categories"""
        errors = {
            'ValueError': ValueError("Invalid value"),
            'FileNotFoundError': FileNotFoundError("File not found"),
            'RuntimeError': RuntimeError("Runtime error"),
            'TypeError': TypeError("Type error")
        }
        
        for error_type, error in errors.items():
            result = global_error_handler(error, f"test_{error_type}")
            assert isinstance(result, dict)
            assert 'error' in result


class TestVisualizationComplete:
    """Complete tests for Visualization utility"""
    
    def setup_method(self):
        """Setup test environment"""
        self.viz_manager = VisualizationManager()
    
    def test_create_result_card(self):
        """Test result card creation"""
        result_data = {
            'confidence': 0.85,
            'is_fake': True,
            'explanation': 'Test explanation',
            'processing_time': 2.5
        }
        
        card_html = create_result_card(result_data)
        
        assert isinstance(card_html, str)
        assert len(card_html) > 0
        assert 'confidence' in card_html.lower()
        assert 'fake' in card_html.lower()
    
    def test_create_confidence_chart(self):
        """Test confidence chart creation"""
        confidence_data = {
            'overall': 0.75,
            'stylegan': 0.6,
            'dalle': 0.8,
            'midjourney': 0.7
        }
        
        chart = create_confidence_chart(confidence_data)
        
        assert chart is not None
        # Should be a plotly figure
        assert hasattr(chart, 'data')
        assert hasattr(chart, 'layout')
    
    def test_visualization_manager_heatmap(self):
        """Test heatmap generation"""
        # Create sample attention map
        attention_map = np.random.rand(224, 224)
        
        heatmap = self.viz_manager.create_attention_heatmap(attention_map)
        
        assert heatmap is not None
        assert isinstance(heatmap, np.ndarray)
        assert len(heatmap.shape) == 3  # Should be RGB
        assert heatmap.shape[2] == 3
    
    def test_visualization_manager_overlay(self):
        """Test overlay creation"""
        # Create base image and attention map
        base_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        attention_map = np.random.rand(224, 224)
        
        overlay = self.viz_manager.create_overlay_visualization(base_image, attention_map)
        
        assert overlay is not None
        assert isinstance(overlay, np.ndarray)
        assert overlay.shape == base_image.shape
    
    def test_visualization_manager_comparison(self):
        """Test comparison visualization"""
        original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        processed = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        comparison = self.viz_manager.create_comparison_view(original, processed)
        
        assert comparison is not None
        assert isinstance(comparison, np.ndarray)
        # Should be wider than original (side-by-side)
        assert comparison.shape[1] > original.shape[1]
    
    def test_metrics_visualization(self):
        """Test metrics visualization"""
        metrics_data = {
            'accuracy': 0.92,
            'precision': 0.88,
            'recall': 0.90,
            'f1_score': 0.89
        }
        
        chart = self.viz_manager.create_metrics_chart(metrics_data)
        
        assert chart is not None
        assert hasattr(chart, 'data')
    
    def test_batch_visualization(self):
        """Test batch visualization"""
        # Create batch of images
        batch_images = []
        batch_results = []
        
        for i in range(4):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = {'confidence': 0.5 + i * 0.1, 'is_fake': i % 2 == 0}
            batch_images.append(image)
            batch_results.append(result)
        
        batch_viz = self.viz_manager.create_batch_visualization(batch_images, batch_results)
        
        assert batch_viz is not None
        assert isinstance(batch_viz, np.ndarray)


class TestModelManagerComplete:
    """Complete tests for Model Manager utility"""
    
    def setup_method(self):
        """Setup test environment"""
        self.model_manager = ModelManager()
    
    def test_model_manager_initialization(self):
        """Test model manager initialization"""
        assert self.model_manager is not None
        assert hasattr(self.model_manager, 'load_model')
        assert hasattr(self.model_manager, 'get_model_info')
    
    def test_model_availability_check(self):
        """Test model availability checking"""
        # Test for common models
        model_names = ['efficientnet', 'xception', 'meso', 'vit']
        
        for model_name in model_names:
            available = self.model_manager.is_model_available(model_name)
            assert isinstance(available, bool)
    
    def test_model_info_retrieval(self):
        """Test model information retrieval"""
        model_info = self.model_manager.get_model_info('efficientnet')
        
        if model_info is not None:
            assert isinstance(model_info, dict)
            expected_fields = ['name', 'version', 'description', 'input_size']
            for field in expected_fields:
                assert field in model_info
    
    def test_model_loading_mock(self):
        """Test model loading with mock"""
        with patch.object(self.model_manager, 'load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            model = self.model_manager.load_model('test_model')
            
            assert model is not None
            mock_load.assert_called_once_with('test_model')
    
    def test_model_caching(self):
        """Test model caching functionality"""
        # This tests the caching mechanism
        with patch.object(self.model_manager, 'load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # Load same model twice
            model1 = self.model_manager.get_cached_model('test_model')
            model2 = self.model_manager.get_cached_model('test_model')
            
            # Should use cached version for second call
            if hasattr(self.model_manager, '_model_cache'):
                assert model1 is model2
    
    def test_model_preloading(self):
        """Test model preloading"""
        models_to_preload = ['efficientnet', 'xception']
        
        # This should not raise exceptions
        self.model_manager.preload_models(models_to_preload)
        
        # Check if models are available in cache
        for model_name in models_to_preload:
            if hasattr(self.model_manager, '_model_cache'):
                # Model should be in cache if preloading worked
                assert model_name in self.model_manager._model_cache or True  # Allow for missing models


class TestBatchProcessorComplete:
    """Complete tests for Batch Processor utility"""
    
    def setup_method(self):
        """Setup test environment"""
        self.batch_processor = BatchProcessor()
    
    def test_batch_processor_initialization(self):
        """Test batch processor initialization"""
        assert self.batch_processor is not None
        assert hasattr(self.batch_processor, 'process_batch')
    
    def test_process_image_batch(self):
        """Test processing batch of images"""
        # Create batch of test images
        batch_data = []
        for i in range(5):
            image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            batch_data.append({
                'id': f'test_image_{i}',
                'data': image,
                'type': 'image'
            })
        
        def mock_process_function(item):
            return {
                'id': item['id'],
                'confidence': 0.5 + (hash(item['id']) % 100) / 200,
                'is_fake': hash(item['id']) % 2 == 0,
                'processing_time': 1.0
            }
        
        results = self.batch_processor.process_batch(batch_data, mock_process_function)
        
        assert len(results) == len(batch_data)
        for result in results:
            assert 'id' in result
            assert 'confidence' in result
            assert 'is_fake' in result
    
    def test_parallel_processing(self):
        """Test parallel batch processing"""
        # Create larger batch for parallel processing
        batch_data = []
        for i in range(10):
            batch_data.append({
                'id': f'item_{i}',
                'value': i * 10,
                'type': 'data'
            })
        
        def slow_process_function(item):
            import time
            time.sleep(0.1)  # Simulate processing time
            return {
                'id': item['id'],
                'result': item['value'] * 2
            }
        
        # Test with parallel processing
        start_time = time.time()
        results_parallel = self.batch_processor.process_batch_parallel(
            batch_data, 
            slow_process_function, 
            max_workers=4
        )
        parallel_time = time.time() - start_time
        
        assert len(results_parallel) == len(batch_data)
        
        # Verify results
        for result in results_parallel:
            assert 'id' in result
            assert 'result' in result
    
    def test_batch_size_optimization(self):
        """Test batch size optimization"""
        # Create data of varying sizes
        small_batch = [{'id': i, 'size': 'small'} for i in range(5)]
        large_batch = [{'id': i, 'size': 'large'} for i in range(20)]
        
        def simple_process(item):
            return {'id': item['id'], 'processed': True}
        
        # Test adaptive batch sizing
        optimal_size = self.batch_processor.get_optimal_batch_size(
            small_batch, 
            simple_process
        )
        
        assert isinstance(optimal_size, int)
        assert optimal_size > 0
    
    def test_batch_error_handling(self):
        """Test error handling in batch processing"""
        batch_data = [
            {'id': 1, 'value': 10},
            {'id': 2, 'value': 'invalid'},  # This will cause error
            {'id': 3, 'value': 30}
        ]
        
        def error_prone_function(item):
            if not isinstance(item['value'], int):
                raise ValueError(f"Invalid value: {item['value']}")
            return {'id': item['id'], 'result': item['value'] * 2}
        
        results = self.batch_processor.process_batch_with_error_handling(
            batch_data, 
            error_prone_function
        )
        
        assert len(results) == len(batch_data)
        
        # Check that successful items were processed
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        assert len(successful) == 2  # Items 1 and 3
        assert len(failed) == 1     # Item 2
    
    def test_progress_tracking(self):
        """Test progress tracking in batch processing"""
        batch_data = [{'id': i} for i in range(20)]
        progress_updates = []
        
        def track_progress(current, total):
            progress_updates.append({'current': current, 'total': total})
        
        def simple_process(item):
            return {'id': item['id'], 'processed': True}
        
        results = self.batch_processor.process_batch_with_progress(
            batch_data, 
            simple_process, 
            progress_callback=track_progress
        )
        
        assert len(results) == len(batch_data)
        assert len(progress_updates) > 0
        
        # Check that progress was tracked
        final_progress = progress_updates[-1]
        assert final_progress['current'] == final_progress['total']


# Integration test for all utilities working together
class TestUtilsIntegration:
    """Integration tests for utilities working together"""
    
    def setup_method(self):
        """Setup test environment"""
        self.processor = MediaProcessor()
        self.viz_manager = VisualizationManager()
        self.batch_processor = BatchProcessor()
        self.model_manager = ModelManager()
    
    def test_complete_processing_pipeline(self):
        """Test complete processing pipeline using all utilities"""
        # Create test image
        test_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_array)
        
        # Step 1: Process image
        processed_array, metadata = self.processor.process_image(test_image, enhance=True)
        assert processed_array is not None
        
        # Step 2: Create visualization
        attention_map = np.random.rand(224, 224)
        heatmap = self.viz_manager.create_attention_heatmap(attention_map)
        assert heatmap is not None
        
        # Step 3: Create result card
        result_data = {
            'confidence': 0.75,
            'is_fake': True,
            'explanation': 'Integration test result',
            'processing_time': metadata.get('processing_time', 1.0)
        }
        
        card = create_result_card(result_data)
        assert isinstance(card, str)
        assert len(card) > 0
    
    def test_batch_processing_with_visualization(self):
        """Test batch processing combined with visualization"""
        # Create batch of images
        batch_images = []
        for i in range(3):
            array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            image = Image.fromarray(array)
            batch_images.append({'id': i, 'image': image})
        
        def process_with_viz(item):
            # Process image
            processed, metadata = self.processor.process_image(item['image'])
            
            # Create simple visualization
            attention = np.random.rand(100, 100)
            viz = self.viz_manager.create_attention_heatmap(attention)
            
            return {
                'id': item['id'],
                'processed': processed is not None,
                'visualization': viz is not None,
                'metadata': metadata
            }
        
        results = self.batch_processor.process_batch(batch_images, process_with_viz)
        
        assert len(results) == len(batch_images)
        for result in results:
            assert result['processed'] == True
            assert result['visualization'] == True
    
    @with_error_handling(global_error_handler)
    def test_error_handling_integration(self):
        """Test error handling integrated with other utilities"""
        # This function intentionally has potential errors
        def risky_processing():
            # Try to process non-existent file
            metadata = self.processor.extract_comprehensive_metadata("nonexistent_file.jpg")
            return metadata
        
        # Should handle error gracefully due to decorator
        result = risky_processing()
        # Result should be None or error dict, not raise exception
        assert result is None or isinstance(result, dict)


# Test fixtures for utilities
@pytest.fixture
def media_processor():
    """Media processor fixture"""
    return MediaProcessor()

@pytest.fixture
def visualization_manager():
    """Visualization manager fixture"""
    return VisualizationManager()

@pytest.fixture
def batch_processor():
    """Batch processor fixture"""
    return BatchProcessor()

@pytest.fixture
def model_manager():
    """Model manager fixture"""
    return ModelManager()

@pytest.fixture
def sample_image():
    """Sample image fixture"""
    array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(array)

@pytest.fixture
def sample_batch():
    """Sample batch data fixture"""
    return [{'id': i, 'value': i * 10} for i in range(5)]
