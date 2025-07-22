"""
Unit tests for AI Image Detector module
"""

import pytest
import numpy as np
from PIL import Image
from detection.ai_image_detector import AIImageDetector, StyleGANDetector, DALLEDetector, MidjourneyDetector


def test_ai_image_detector_init():
    """Test initialization of AI Image Detector"""
    detector = AIImageDetector()
    assert detector is not None, "AIImageDetector should be initialized"


def test_stylegan_detection():
    """Test StyleGAN artifact detection"""
    detector = StyleGANDetector()
    image = np.zeros((256, 256, 3), np.uint8)  # Simulated image
    results = detector.detect_artifacts(image)
    assert 'stylegan_confidence' in results, "Results should contain stylegan_confidence"
    assert 0 <= results['stylegan_confidence'] <= 1, "Confidence should be between 0 and 1"


def test_dalle_detection():
    """Test DALL-E artifact detection"""
    detector = DALLEDetector()
    image = np.zeros((256, 256, 3), np.uint8)  # Simulated image
    results = detector.detect_artifacts(image)
    assert 'dalle_confidence' in results, "Results should contain dalle_confidence"
    assert 0 <= results['dalle_confidence'] <= 1, "Confidence should be between 0 and 1"


def test_midjourney_detection():
    """Test Midjourney artifact detection"""
    detector = MidjourneyDetector()
    image = np.zeros((256, 256, 3), np.uint8)  # Simulated image
    results = detector.detect_artifacts(image)
    assert 'midjourney_confidence' in results, "Results should contain midjourney_confidence"
    assert 0 <= results['midjourney_confidence'] <= 1, "Confidence should be between 0 and 1"

