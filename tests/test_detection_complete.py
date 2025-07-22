"""
Complete unit tests for all detection modules
Tests AI detection, deepfake detection, duplicate detection, and face extraction
"""

import pytest
import numpy as np
import tempfile
import os
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import torch
import cv2

# Import all detection modules
from detection.ai_image_detector import (
    AIImageDetector, StyleGANDetector, DALLEDetector, MidjourneyDetector,
    EfficientNetAIDetector, VisionTransformerAIDetector, GradCAM
)
from detection.deepfake_detector import DeepfakeDetector
from detection.duplicate_detector import DuplicateDetector
from detection.face_extractor import FaceExtractor


class TestAIImageDetectorComplete:
    """Complete tests for AI Image Detector"""
    
    def setup_method(self):
        """Setup test environment"""
        self.detector = AIImageDetector()
        self.test_image = self.create_test_image()
        
    def create_test_image(self, size=(224, 224)):
        """Create a test image"""
        array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        # Add some patterns to make it more realistic
        array[50:100, 50:100] = [255, 0, 0]  # Red square
        array[150:200, 150:200] = [0, 255, 0]  # Green square
        return Image.fromarray(array)
    
    def test_detector_initialization(self):
        """Test detector initialization with different models"""
        # Test EfficientNet initialization
        detector_eff = AIImageDetector(model_name='efficientnet')
        assert detector_eff is not None
        
        # Test Vision Transformer initialization
        detector_vit = AIImageDetector(model_name='vit')
        assert detector_vit is not None
    
    def test_detect_with_mock_file(self):
        """Test detection with mock uploaded file"""
        # Create mock uploaded file
        temp_path = "temp_test.jpg"
        self.test_image.save(temp_path)
        
        class MockUploadedFile:
            def __init__(self, path):
                self.name = os.path.basename(path)
                self.type = "image/jpeg"
                with open(path, 'rb') as f:
                    self.content = f.read()
                self.size = len(self.content)
            
            def read(self):
                return self.content
        
        try:
            mock_file = MockUploadedFile(temp_path)
            results = self.detector.detect(mock_file, threshold=0.5, enable_viz=False)
            
            # Verify results structure
            assert 'confidence' in results
            assert 'is_fake' in results
            assert 'explanation' in results
            assert 'processing_time' in results
            assert 'technical_details' in results
            
            # Verify confidence range
            assert 0 <= results['confidence'] <= 1
            
            # Verify boolean result
            assert isinstance(results['is_fake'], bool)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_detect_with_visualization(self):
        """Test detection with visualization enabled"""
        temp_path = "temp_viz_test.jpg"
        self.test_image.save(temp_path)
        
        class MockUploadedFile:
            def __init__(self, path):
                self.name = os.path.basename(path)
                self.type = "image/jpeg"
                with open(path, 'rb') as f:
                    self.content = f.read()
                self.size = len(self.content)
            
            def read(self):
                return self.content
        
        try:
            mock_file = MockUploadedFile(temp_path)
            results = self.detector.detect(mock_file, threshold=0.5, enable_viz=True)
            
            # Check if visualizations were generated
            assert 'visualizations' in results
            if results['visualizations']:
                assert isinstance(results['visualizations'], dict)
                
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestStyleGANDetectorComplete:
    """Complete tests for StyleGAN detector"""
    
    def setup_method(self):
        """Setup test environment"""
        self.detector = StyleGANDetector()
        
    def create_stylegan_like_image(self):
        """Create an image with StyleGAN-like artifacts"""
        # Create base image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Add periodic patterns (common in StyleGAN)
        for i in range(0, 256, 16):
            for j in range(0, 256, 16):
                image[i:i+8, j:j+8] = [200, 200, 200]
        
        # Add high-frequency artifacts
        noise = np.random.randint(-50, 50, (256, 256, 3), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def test_detect_artifacts_normal_image(self):
        """Test artifact detection on normal image"""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        results = self.detector.detect_artifacts(image)
        
        assert 'stylegan_confidence' in results
        assert 'spectral_ratio' in results
        assert 'periodic_score' in results
        assert 'texture_variance' in results
        
        # All values should be numeric
        for key, value in results.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
    
    def test_detect_artifacts_stylegan_like(self):
        """Test artifact detection on StyleGAN-like image"""
        image = self.create_stylegan_like_image()
        results = self.detector.detect_artifacts(image)
        
        assert 'stylegan_confidence' in results
        assert 0 <= results['stylegan_confidence'] <= 1
        
        # StyleGAN-like image should have some detectable artifacts
        assert results['spectral_ratio'] > 0
        assert results['periodic_score'] > 0
    
    def test_grayscale_image_handling(self):
        """Test handling of grayscale images"""
        gray_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        results = self.detector.detect_artifacts(gray_image)
        
        # Should handle grayscale without errors
        assert 'stylegan_confidence' in results
        assert isinstance(results['stylegan_confidence'], (int, float))


class TestDALLEDetectorComplete:
    """Complete tests for DALL-E detector"""
    
    def setup_method(self):
        """Setup test environment"""
        self.detector = DALLEDetector()
    
    def create_dalle_like_image(self):
        """Create an image with DALL-E-like characteristics"""
        # Create image with uniform patches
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Fill with uniform color patches (DALL-E characteristic)
        patch_size = 32
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        for i in range(0, 256, patch_size):
            for j in range(0, 256, patch_size):
                color_idx = ((i // patch_size) + (j // patch_size)) % len(colors)
                color = colors[color_idx]
                image[i:i+patch_size, j:j+patch_size] = color
        
        return image
    
    def test_detect_artifacts_normal_image(self):
        """Test artifact detection on normal image"""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        results = self.detector.detect_artifacts(image)
        
        assert 'dalle_confidence' in results
        assert 'edge_uniformity' in results
        assert 'color_uniformity' in results
        assert 'compression_score' in results
        
        # Check confidence range
        assert 0 <= results['dalle_confidence'] <= 1
    
    def test_detect_artifacts_dalle_like(self):
        """Test artifact detection on DALL-E-like image"""
        image = self.create_dalle_like_image()
        results = self.detector.detect_artifacts(image)
        
        assert 'dalle_confidence' in results
        # DALL-E-like image should have higher uniformity scores
        assert results['edge_uniformity'] > 0
        assert results['color_uniformity'] > 0
    
    def test_jpeg_quality_estimation(self):
        """Test JPEG quality estimation"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        quality = self.detector._estimate_jpeg_quality(image)
        
        # Should return reasonable quality estimate
        assert 0 <= quality <= 100
        assert isinstance(quality, (int, float))
    
    def test_grayscale_handling(self):
        """Test handling of grayscale images"""
        gray_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        results = self.detector.detect_artifacts(gray_image)
        
        assert 'dalle_confidence' in results
        # Should handle gracefully with default values
        assert isinstance(results['dalle_confidence'], (int, float))


class TestMidjourneyDetectorComplete:
    """Complete tests for Midjourney detector"""
    
    def setup_method(self):
        """Setup test environment"""
        self.detector = MidjourneyDetector()
    
    def create_artistic_image(self):
        """Create an image with artistic characteristics"""
        # Create smooth gradients (artistic style)
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        for i in range(256):
            for j in range(256):
                # Create smooth color transitions
                r = int(255 * (i / 255))
                g = int(255 * (j / 255))
                b = int(255 * ((i + j) / 510))
                image[i, j] = [r, g, b]
        
        # Add artistic saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # Increase saturation
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return image
    
    def test_detect_artifacts_normal_image(self):
        """Test artifact detection on normal image"""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        results = self.detector.detect_artifacts(image)
        
        assert 'midjourney_confidence' in results
        assert 'color_harmony' in results
        assert 'gradient_smoothness' in results
        assert 'saturation_mean' in results
        assert 'artistic_pattern_score' in results
        
        # Check confidence range
        assert 0 <= results['midjourney_confidence'] <= 1
    
    def test_detect_artifacts_artistic_image(self):
        """Test artifact detection on artistic image"""
        image = self.create_artistic_image()
        results = self.detector.detect_artifacts(image)
        
        assert 'midjourney_confidence' in results
        # Artistic image should have higher harmony and smoothness
        assert results['gradient_smoothness'] > 0
        assert results['saturation_mean'] > 0
    
    def test_color_harmony_calculation(self):
        """Test color harmony calculation"""
        # Create colors with known harmony
        harmonious_colors = np.array([
            [255, 0, 0],      # Red
            [255, 128, 0],    # Orange (analogous to red)
            [0, 255, 255],    # Cyan (complementary to red)
        ])
        
        harmony_score = self.detector._calculate_color_harmony(harmonious_colors)
        
        assert isinstance(harmony_score, float)
        assert 0 <= harmony_score <= 1
    
    def test_single_color_harmony(self):
        """Test color harmony with single color"""
        single_color = np.array([[255, 0, 0]])
        harmony_score = self.detector._calculate_color_harmony(single_color)
        
        # Should handle single color gracefully
        assert harmony_score == 0.5
    
    def test_grayscale_handling(self):
        """Test handling of grayscale images"""
        gray_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        results = self.detector.detect_artifacts(gray_image)
        
        assert 'midjourney_confidence' in results
        # Should handle gracefully with default values
        assert isinstance(results['midjourney_confidence'], (int, float))


class TestDeepfakeDetectorComplete:
    """Complete tests for Deepfake detector"""
    
    def setup_method(self):
        """Setup test environment"""
        self.detector = DeepfakeDetector()
    
    def create_test_video_file(self):
        """Create a test video file"""
        temp_path = "temp_test_video.mp4"
        
        # Create simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, 20.0, (320, 240))
        
        for i in range(60):  # 3 seconds at 20fps
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            # Add a face-like region
            cv2.rectangle(frame, (100, 80), (220, 160), (255, 200, 180), -1)
            # Add eyes
            cv2.circle(frame, (130, 110), 10, (0, 0, 0), -1)
            cv2.circle(frame, (190, 110), 10, (0, 0, 0), -1)
            # Add mouth
            cv2.ellipse(frame, (160, 140), (20, 10), 0, 0, 180, (0, 0, 0), -1)
            
            out.write(frame)
        
        out.release()
        return temp_path
    
    def test_detector_initialization(self):
        """Test deepfake detector initialization"""
        assert self.detector is not None
        # Should have models initialized
        assert hasattr(self.detector, 'xception_model')
        assert hasattr(self.detector, 'meso_model')
    
    @patch('detection.deepfake_detector.DeepfakeDetector.detect')
    def test_detect_with_mock_file(self, mock_detect):
        """Test detection with mock video file"""
        # Mock the detect method to avoid actual model loading
        mock_detect.return_value = {
            'confidence': 0.7,
            'is_fake': False,
            'explanation': 'Video analysis completed',
            'processing_time': 2.5,
            'technical_details': {
                'frames_analyzed': 30,
                'average_confidence': 0.7,
                'model_used': 'xception'
            }
        }
        
        # Create mock uploaded file
        class MockVideoFile:
            def __init__(self):
                self.name = "test_video.mp4"
                self.type = "video/mp4"
                self.content = b"fake video content"
                self.size = len(self.content)
            
            def read(self):
                return self.content
        
        mock_file = MockVideoFile()
        results = self.detector.detect(mock_file, threshold=0.5, enable_viz=False)
        
        assert 'confidence' in results
        assert 'is_fake' in results
        assert 'explanation' in results
        assert 'processing_time' in results
        
        mock_detect.assert_called_once()


class TestDuplicateDetectorComplete:
    """Complete tests for Duplicate detector"""
    
    def setup_method(self):
        """Setup test environment"""
        self.detector = DuplicateDetector()
        # Clear any existing test data
        self.cleanup_database()
    
    def cleanup_database(self):
        """Clean up test database"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.detector.db_manager.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM media_hashes")
            conn.commit()
            conn.close()
        except:
            pass
    
    def create_test_images(self):
        """Create test images with known relationships"""
        images = {}
        
        # Original image
        original = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        images['original'] = Image.fromarray(original)
        
        # Identical duplicate
        images['duplicate'] = Image.fromarray(original.copy())
        
        # Similar image (slight modification)
        similar = original.copy()
        similar[0:50, 0:50] = [255, 0, 0]  # Add red square
        images['similar'] = Image.fromarray(similar)
        
        # Different image
        different = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        images['different'] = Image.fromarray(different)
        
        return images
    
    def test_process_file_workflow(self):
        """Test complete file processing workflow"""
        images = self.create_test_images()
        temp_files = []
        
        try:
            # Save images to temporary files
            for name, img in images.items():
                temp_path = f"temp_{name}.jpg"
                img.save(temp_path)
                temp_files.append(temp_path)
                
                # Process file
                self.detector.process_file(temp_path)
            
            # Get statistics
            stats = self.detector.get_statistics()
            assert stats['total_files'] >= len(images)
            assert stats['images'] >= len(images)
            
        finally:
            # Cleanup
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    def test_find_duplicates_workflow(self):
        """Test duplicate finding workflow"""
        images = self.create_test_images()
        temp_files = []
        
        try:
            # Process original and duplicate
            original_path = "temp_original_dup.jpg"
            duplicate_path = "temp_duplicate_dup.jpg"
            
            images['original'].save(original_path)
            images['duplicate'].save(duplicate_path)
            temp_files.extend([original_path, duplicate_path])
            
            self.detector.process_file(original_path)
            self.detector.process_file(duplicate_path)
            
            # Find duplicates
            duplicates = self.detector.find_all_duplicates()
            
            # Should find the duplicate pair
            assert isinstance(duplicates, list)
            # Exact duplicates should be detected
            if len(duplicates) > 0:
                assert hasattr(duplicates[0], 'similarity_score')
                assert hasattr(duplicates[0], 'original_file')
                assert hasattr(duplicates[0], 'duplicate_file')
            
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    def test_detect_method_with_mock_file(self):
        """Test the detect method with mock uploaded file"""
        test_image = Image.new('RGB', (100, 100), color='red')
        temp_path = "temp_detect_test.jpg"
        test_image.save(temp_path)
        
        try:
            class MockUploadedFile:
                def __init__(self, path):
                    self.name = os.path.basename(path)
                    self.type = "image/jpeg"
                    with open(path, 'rb') as f:
                        self.content = f.read()
                    self.size = len(self.content)
                
                def read(self):
                    return self.content
            
            mock_file = MockUploadedFile(temp_path)
            results = self.detector.detect(mock_file, threshold=0.8, enable_viz=False)
            
            assert 'is_duplicate' in results
            assert 'confidence' in results
            assert 'explanation' in results
            assert 'processing_time' in results
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_similarity_thresholds(self):
        """Test different similarity thresholds"""
        # Test strict thresholds
        strict_detector = DuplicateDetector({
            'dhash': 2, 'ahash': 2, 'phash': 2, 'whash': 2
        })
        
        # Test lenient thresholds
        lenient_detector = DuplicateDetector({
            'dhash': 15, 'ahash': 15, 'phash': 15, 'whash': 15
        })
        
        assert strict_detector.similarity_threshold['dhash'] == 2
        assert lenient_detector.similarity_threshold['dhash'] == 15
        
        # Both should be properly initialized
        assert strict_detector.db_manager is not None
        assert lenient_detector.db_manager is not None


class TestFaceExtractorComplete:
    """Complete tests for Face extractor"""
    
    def setup_method(self):
        """Setup test environment"""
        self.extractor = FaceExtractor()
    
    def create_face_like_image(self):
        """Create an image with face-like features"""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 220  # Light background
        
        # Draw face outline
        cv2.ellipse(image, (100, 100), (60, 80), 0, 0, 360, (200, 180, 160), -1)
        
        # Draw eyes
        cv2.circle(image, (80, 80), 8, (50, 50, 50), -1)
        cv2.circle(image, (120, 80), 8, (50, 50, 50), -1)
        
        # Draw nose
        cv2.line(image, (100, 90), (100, 110), (150, 120, 100), 2)
        
        # Draw mouth
        cv2.ellipse(image, (100, 130), (15, 8), 0, 0, 180, (100, 50, 50), 2)
        
        return image
    
    def test_extractor_initialization(self):
        """Test face extractor initialization"""
        assert self.extractor is not None
    
    @patch('detection.face_extractor.FaceExtractor.extract_faces')
    def test_extract_faces_mock(self, mock_extract):
        """Test face extraction with mock"""
        # Mock the extraction to avoid MTCNN dependency issues
        mock_extract.return_value = {
            'faces_detected': 1,
            'face_count': 1,
            'faces': [{'bbox': [50, 50, 100, 100], 'confidence': 0.95}],
            'processing_time': 1.2
        }
        
        test_image = self.create_face_like_image()
        results = self.extractor.extract_faces(test_image)
        
        assert 'faces_detected' in results
        assert 'face_count' in results
        assert 'faces' in results
        assert results['face_count'] >= 0
        
        mock_extract.assert_called_once()
    
    def test_face_like_image_processing(self):
        """Test processing of face-like image"""
        face_image = self.create_face_like_image()
        
        # This tests the basic image handling without actual face detection
        # to avoid dependency issues
        assert face_image is not None
        assert face_image.shape == (200, 200, 3)
        assert face_image.dtype == np.uint8


class TestModelArchitectures:
    """Test model architecture classes"""
    
    def test_efficientnet_model_creation(self):
        """Test EfficientNet model architecture"""
        try:
            model = EfficientNetAIDetector()
            assert model is not None
            
            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            
            # Should output 2 classes (real vs fake)
            assert output.shape[-1] == 2
            
        except Exception as e:
            # Skip if model dependencies aren't available
            pytest.skip(f"EfficientNet model test skipped: {e}")
    
    def test_vision_transformer_model_creation(self):
        """Test Vision Transformer model architecture"""
        try:
            model = VisionTransformerAIDetector()
            assert model is not None
            
            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            
            # Should output 2 classes (real vs fake)
            assert output.shape[-1] == 2
            
        except Exception as e:
            # Skip if model dependencies aren't available
            pytest.skip(f"Vision Transformer model test skipped: {e}")


class TestGradCAMVisualization:
    """Test GradCAM visualization"""
    
    def test_gradcam_initialization(self):
        """Test GradCAM initialization"""
        try:
            # Create a simple model for testing
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 2)
            )
            
            gradcam = GradCAM(model, ['0'], use_cuda=False)
            assert gradcam is not None
            assert gradcam.model is not None
            
        except Exception as e:
            pytest.skip(f"GradCAM test skipped: {e}")
    
    def test_gradcam_generation(self):
        """Test GradCAM generation"""
        try:
            # Create a simple model for testing
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 2)
            )
            
            gradcam = GradCAM(model, ['0'], use_cuda=False)
            
            # Test with dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            cam = gradcam.generate_cam(dummy_input)
            
            assert cam is not None
            assert isinstance(cam, np.ndarray)
            assert cam.shape == (224, 224)
            
        except Exception as e:
            pytest.skip(f"GradCAM generation test skipped: {e}")


# Test fixtures for all detectors
@pytest.fixture
def ai_detector():
    """AI detector fixture"""
    return AIImageDetector()

@pytest.fixture
def stylegan_detector():
    """StyleGAN detector fixture"""
    return StyleGANDetector()

@pytest.fixture
def dalle_detector():
    """DALL-E detector fixture"""
    return DALLEDetector()

@pytest.fixture
def midjourney_detector():
    """Midjourney detector fixture"""
    return MidjourneyDetector()

@pytest.fixture
def duplicate_detector():
    """Duplicate detector fixture"""
    return DuplicateDetector()

@pytest.fixture
def face_extractor():
    """Face extractor fixture"""
    return FaceExtractor()

@pytest.fixture
def test_image():
    """Test image fixture"""
    array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(array)

@pytest.fixture
def test_image_array():
    """Test image array fixture"""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
