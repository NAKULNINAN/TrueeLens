"""
Unit tests for MediaProcessor utility functions
"""

import pytest
import numpy as np
from PIL import Image
from utils.media_processor import MediaProcessor


def test_process_image():
    """Test basic image processing"""
    processor = MediaProcessor()
    img = Image.new('RGB', (100, 100), color = 'red')
    processed_array, metadata = processor.process_image(img)

    assert processed_array is not None, "Processed array should not be None"
    assert 'original_size' in metadata, "Metadata should contain original_size"
    assert metadata['original_size'] == (100, 100), "Original size should be (100, 100)"


def test_preprocess_image_advanced():
    """Test advanced image preprocessing"""
    processor = MediaProcessor()
    img = Image.new('RGB', (100, 100), color = 'blue')
    options = {
        'resize': (50, 50),
        'enhance_contrast': 1.5
    }
    processed_img, metadata = processor.preprocess_image_advanced(img, options)

    assert processed_img.size == (50, 50), "Image should be resized to (50, 50)"
    assert 'processing_log' in metadata, "Metadata should contain processing_log"


def test_extract_comprehensive_metadata():
    """Test comprehensive metadata extraction"""
    processor = MediaProcessor()
    img = Image.new('RGB', (100, 100), color = 'green')
    img_path = 'test_image.jpg'
    img.save(img_path)

    metadata = processor.extract_comprehensive_metadata(img_path)

    assert 'file_info' in metadata, "Metadata should contain file_info"
    assert metadata['file_info']['filename'] == 'test_image.jpg', "Filename should match"

    import os
    os.remove(img_path)


def test_convert_image_format():
    """Test format conversion"""
    processor = MediaProcessor()
    img = Image.new('RGB', (100, 100), color = 'yellow')
    test_path = 'test_convert.jpg'
    img.save(test_path)

    output_path = 'test_output.png'
    success = processor.convert_image_format(test_path, output_path, target_format='PNG')

    assert success, "Conversion should be successful"
    assert os.path.exists(output_path), "Output path must exist"

    import os
    os.remove(test_path)
    os.remove(output_path)

