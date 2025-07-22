#!/usr/bin/env python3
"""
Comprehensive test script for MediaProcessor utilities
Demonstrates all features: video frame extraction, image preprocessing, 
format conversion, metadata extraction, and file validation
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import json

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from media_processor import MediaProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image(filename: str, size: tuple = (800, 600)):
    """Create a test image for demonstration"""
    # Create a test image with some patterns
    img = Image.new('RGB', size, color='white')
    pixels = img.load()
    
    # Add some patterns
    for x in range(size[0]):
        for y in range(size[1]):
            r = int((x / size[0]) * 255)
            g = int((y / size[1]) * 255)
            b = 128
            pixels[x, y] = (r, g, b)
    
    # Add some noise for texture
    import random
    for _ in range(1000):
        x = random.randint(0, size[0]-1)
        y = random.randint(0, size[1]-1)
        pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    img.save(filename)
    logger.info(f"Created test image: {filename}")
    return filename

def test_image_processing():
    """Test image processing functionality"""
    logger.info("=== Testing Image Processing ===")
    
    processor = MediaProcessor()
    
    # Create test image
    test_img_path = "test_image.jpg"
    create_test_image(test_img_path)
    
    try:
        # Test basic image processing
        with Image.open(test_img_path) as img:
            processed_array, metadata = processor.process_image(img, enhance=True)
            
        logger.info(f"Image processed successfully!")
        logger.info(f"Original size: {metadata['original_size']}")
        logger.info(f"Processed size: {metadata['processed_size']}")
        logger.info(f"Stats: brightness={metadata['stats']['mean_brightness']:.2f}, contrast={metadata['stats']['contrast']:.2f}")
        
        # Test advanced preprocessing
        preprocessing_options = {
            'denoise': True,
            'histogram_equalization': True,
            'gamma': 1.2,
            'contrast': 1.1,
            'brightness': 1.05,
            'sharpen': True,
            'resize': (256, 256),
            'resize_method': 'LANCZOS'
        }
        
        processed_advanced, adv_metadata = processor.preprocess_image_advanced(img, preprocessing_options)
        logger.info(f"Advanced preprocessing completed!")
        logger.info(f"Processing log: {adv_metadata['processing_log']}")
        
    except Exception as e:
        logger.error(f"Image processing test failed: {e}")
    
    # Cleanup
    if os.path.exists(test_img_path):
        os.remove(test_img_path)

def test_metadata_extraction():
    """Test comprehensive metadata extraction"""
    logger.info("=== Testing Metadata Extraction ===")
    
    processor = MediaProcessor()
    
    # Create test image with some EXIF-like data
    test_img_path = "test_metadata.jpg"
    create_test_image(test_img_path)
    
    try:
        # Test comprehensive metadata extraction
        metadata = processor.extract_comprehensive_metadata(test_img_path)
        
        logger.info("Comprehensive metadata extracted!")
        logger.info(f"File info: {metadata['file_info']['filename']} ({metadata['file_info']['file_size_mb']} MB)")
        logger.info(f"Hash data: MD5={metadata['hash_data'].get('md5', 'N/A')[:8]}...")
        logger.info(f"Technical data: {metadata['technical_data']}")
        logger.info(f"Validation: exists={metadata['file_validation']['exists']}, readable={metadata['file_validation']['readable']}")
        
        # Save metadata to file for inspection
        with open('metadata_output.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Metadata saved to metadata_output.json")
        
    except Exception as e:
        logger.error(f"Metadata extraction test failed: {e}")
    
    # Cleanup
    if os.path.exists(test_img_path):
        os.remove(test_img_path)

def test_format_conversion():
    """Test format conversion functionality"""
    logger.info("=== Testing Format Conversion ===")
    
    processor = MediaProcessor()
    
    # Create test image
    test_img_path = "test_conversion.png"
    create_test_image(test_img_path)
    
    try:
        # Test format conversion
        output_formats = ['JPEG', 'PNG', 'WEBP', 'BMP']
        
        for fmt in output_formats:
            output_path = f"converted_image.{fmt.lower()}"
            success = processor.convert_image_format(
                test_img_path, 
                output_path, 
                target_format=fmt,
                quality=90,
                preserve_metadata=True
            )
            
            if success:
                logger.info(f"Successfully converted to {fmt}")
                # Validate the converted file
                is_valid = processor.validate_file(output_path)
                logger.info(f"Validation result for {fmt}: {is_valid}")
                
                # Cleanup converted file
                if os.path.exists(output_path):
                    os.remove(output_path)
            else:
                logger.error(f"Failed to convert to {fmt}")
                
    except Exception as e:
        logger.error(f"Format conversion test failed: {e}")
    
    # Cleanup
    if os.path.exists(test_img_path):
        os.remove(test_img_path)

def test_file_validation():
    """Test comprehensive file validation"""
    logger.info("=== Testing File Validation ===")
    
    processor = MediaProcessor()
    
    # Create test files
    test_files = []
    
    # Valid image file
    valid_img = "valid_test.jpg"
    create_test_image(valid_img)
    test_files.append(valid_img)
    
    # Create a corrupted file (empty file with image extension)
    corrupted_img = "corrupted_test.jpg"
    with open(corrupted_img, 'w') as f:
        f.write("")  # Empty file
    test_files.append(corrupted_img)
    
    # Create a text file with wrong extension
    wrong_ext = "text_as_image.jpg"
    with open(wrong_ext, 'w') as f:
        f.write("This is not an image file")
    test_files.append(wrong_ext)
    
    try:
        for test_file in test_files:
            logger.info(f"\nValidating: {test_file}")
            
            # Basic validation
            is_valid = processor.validate_file(test_file)
            logger.info(f"Basic validation: {is_valid}")
            
            # Comprehensive validation
            validation_results = processor._comprehensive_file_validation(test_file)
            logger.info(f"Comprehensive validation results:")
            for key, value in validation_results.items():
                logger.info(f"  {key}: {value}")
                
    except Exception as e:
        logger.error(f"File validation test failed: {e}")
    
    # Cleanup
    for test_file in test_files:
        if os.path.exists(test_file):
            os.remove(test_file)

def test_video_frame_extraction():
    """Test video frame extraction functionality"""
    logger.info("=== Testing Video Frame Extraction ===")
    
    processor = MediaProcessor()
    
    # Create a simple test video using OpenCV (if possible)
    try:
        import cv2
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('test_video.avi', fourcc, 20.0, (640, 480))
        
        # Create frames with different colors
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i in range(100):  # 5 seconds at 20fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            color = colors[i // 20]  # Change color every 20 frames (1 second)
            frame[:, :] = color
            
            # Add frame number text
            cv2.putText(frame, f'Frame {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        logger.info("Test video created: test_video.avi")
        
        # Test basic video processing
        frames, metadata = processor.process_video('test_video.avi', extract_frames=True, max_frames=10)
        logger.info(f"Basic extraction: {len(frames)} frames extracted")
        logger.info(f"Video metadata: FPS={metadata['fps']}, Duration={metadata['duration']:.2f}s")
        
        # Test advanced frame extraction
        extraction_methods = ['uniform', 'keyframes', 'rate']
        
        for method in extraction_methods:
            try:
                if method == 'rate':
                    frames, frame_metadata = processor.extract_video_frames_advanced(
                        'test_video.avi',
                        extraction_method=method,
                        frame_rate=2.0,
                        quality_threshold=0.1
                    )
                else:
                    frames, frame_metadata = processor.extract_video_frames_advanced(
                        'test_video.avi',
                        extraction_method=method,
                        quality_threshold=0.1
                    )
                
                logger.info(f"Advanced extraction ({method}): {len(frames)} frames extracted")
                if frame_metadata:
                    logger.info(f"Sample frame metadata: {frame_metadata[0]}")
                    
            except Exception as e:
                logger.warning(f"Advanced extraction method {method} failed: {e}")
        
        # Test video quality metrics
        if frames:
            quality_metrics = processor.calculate_video_quality_metrics(frames)
            logger.info(f"Video quality metrics: {quality_metrics}")
        
        # Test frame saving
        output_dir = "extracted_frames"
        if frames:
            saved_paths = processor.save_processed_frames(frames, output_dir, prefix="test_frame")
            logger.info(f"Saved {len(saved_paths)} frames to {output_dir}")
            
            # Cleanup saved frames
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)
        
        # Cleanup test video
        if os.path.exists('test_video.avi'):
            os.remove('test_video.avi')
            
    except ImportError:
        logger.warning("OpenCV not available for video testing")
    except Exception as e:
        logger.error(f"Video frame extraction test failed: {e}")

def test_model_preprocessing():
    """Test model preprocessing functionality"""
    logger.info("=== Testing Model Preprocessing ===")
    
    processor = MediaProcessor()
    
    # Create test image
    test_img_path = "test_model_prep.jpg"
    create_test_image(test_img_path)
    
    try:
        with Image.open(test_img_path) as img:
            # Process image first
            processed_array, _ = processor.process_image(img)
            
            # Test model preprocessing without tensor conversion
            model_input = processor.preprocess_for_model(
                processed_array, 
                normalize=True, 
                to_tensor=False
            )
            
            logger.info(f"Model preprocessing completed!")
            logger.info(f"Input shape: {model_input.shape}")
            logger.info(f"Data type: {model_input.dtype}")
            logger.info(f"Value range: [{model_input.min():.3f}, {model_input.max():.3f}]")
            
            # Test with tensor conversion (if PyTorch is available)
            try:
                import torch
                model_tensor = processor.preprocess_for_model(
                    processed_array, 
                    normalize=True, 
                    to_tensor=True
                )
                logger.info(f"Tensor conversion successful!")
                logger.info(f"Tensor shape: {model_tensor.shape}")
                logger.info(f"Tensor type: {type(model_tensor)}")
            except ImportError:
                logger.info("PyTorch not available for tensor conversion test")
                
    except Exception as e:
        logger.error(f"Model preprocessing test failed: {e}")
    
    # Cleanup
    if os.path.exists(test_img_path):
        os.remove(test_img_path)

def main():
    """Run all tests"""
    logger.info("Starting comprehensive MediaProcessor tests...")
    
    # Create output directory for test results
    os.makedirs("test_outputs", exist_ok=True)
    os.chdir("test_outputs")
    
    try:
        # Run all tests
        test_image_processing()
        test_metadata_extraction()
        test_format_conversion()
        test_file_validation()
        test_video_frame_extraction()
        test_model_preprocessing()
        
        logger.info("=== All tests completed! ===")
        logger.info("Check the test_outputs directory for generated files")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
    finally:
        # Cleanup any remaining test files
        for file in os.listdir("."):
            if file.startswith("test_") or file.startswith("converted_"):
                try:
                    os.remove(file)
                except:
                    pass

if __name__ == "__main__":
    main()
