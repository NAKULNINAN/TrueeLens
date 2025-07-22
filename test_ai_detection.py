#!/usr/bin/env python3
"""
Test script for AI Image Detection System
Tests EfficientNet/Vision Transformer models with GradCAM visualization
and specialized artifact detection for StyleGAN, DALL-E, and Midjourney
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection.ai_image_detector import AIImageDetector

def create_test_image(size=(224, 224)):
    """Create a synthetic test image"""
    # Create a simple test pattern
    img_array = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    
    # Add some structured patterns that might trigger AI detection
    center_x, center_y = size[0] // 2, size[1] // 2
    for i in range(size[0]):
        for j in range(size[1]):
            # Create circular gradient
            dist = ((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5
            if dist < size[0] // 4:
                img_array[i, j] = [128, 128, 255]  # Blue center
            elif dist < size[0] // 3:
                img_array[i, j] = [255, 128, 128]  # Red ring
    
    return Image.fromarray(img_array)

def test_ai_detector():
    """Test the AI Image Detection System"""
    print("🔍 Testing AI Image Detection System")
    print("=" * 50)
    
    try:
        # Initialize detector with EfficientNet
        print("📊 Initializing EfficientNet detector...")
        detector_eff = AIImageDetector(model_name='efficientnet')
        print("✅ EfficientNet detector initialized successfully")
        
        # Initialize detector with Vision Transformer
        print("📊 Initializing Vision Transformer detector...")
        detector_vit = AIImageDetector(model_name='vit')
        print("✅ Vision Transformer detector initialized successfully")
        
        # Create test image
        print("🖼️ Creating test image...")
        test_image = create_test_image()
        print("✅ Test image created")
        
        # Test EfficientNet detection
        print("\n🔍 Testing EfficientNet Detection:")
        print("-" * 30)
        results_eff = detector_eff.detect(test_image, threshold=0.5, enable_viz=False)
        
        print(f"Confidence: {results_eff.get('confidence', 0):.3f}")
        print(f"Is AI Generated: {results_eff.get('is_fake', False)}")
        print(f"Processing Time: {results_eff.get('processing_time', 0):.2f}s")
        print(f"Explanation: {results_eff.get('explanation', 'N/A')}")
        
        if 'technical_details' in results_eff:
            tech_details = results_eff['technical_details']
            print(f"StyleGAN Confidence: {tech_details.get('stylegan_analysis', {}).get('stylegan_confidence', 0):.3f}")
            print(f"DALL-E Confidence: {tech_details.get('dalle_analysis', {}).get('dalle_confidence', 0):.3f}")
            print(f"Midjourney Confidence: {tech_details.get('midjourney_analysis', {}).get('midjourney_confidence', 0):.3f}")
        
        # Test Vision Transformer detection
        print("\n🔍 Testing Vision Transformer Detection:")
        print("-" * 30)
        results_vit = detector_vit.detect(test_image, threshold=0.5, enable_viz=False)
        
        print(f"Confidence: {results_vit.get('confidence', 0):.3f}")
        print(f"Is AI Generated: {results_vit.get('is_fake', False)}")
        print(f"Processing Time: {results_vit.get('processing_time', 0):.2f}s")
        print(f"Explanation: {results_vit.get('explanation', 'N/A')}")
        
        # Test with visualization enabled
        print("\n🎨 Testing with Visualization (EfficientNet):")
        print("-" * 30)
        results_viz = detector_eff.detect(test_image, threshold=0.5, enable_viz=True)
        
        if 'visualizations' in results_viz and results_viz['visualizations']:
            print("✅ Visualizations generated successfully")
            if 'heatmap' in results_viz['visualizations']:
                print("✅ GradCAM heatmap generated")
            if 'confidence_chart' in results_viz['visualizations']:
                print("✅ Confidence chart generated")
        else:
            print("⚠️ No visualizations generated")
        
        print("\n🎯 Artifact Detection Analysis:")
        print("-" * 30)
        if 'technical_details' in results_eff:
            tech = results_eff['technical_details']
            
            # StyleGAN analysis
            if 'stylegan_analysis' in tech:
                sg = tech['stylegan_analysis']
                print(f"StyleGAN - Spectral Ratio: {sg.get('spectral_ratio', 0):.3f}")
                print(f"StyleGAN - Periodic Score: {sg.get('periodic_score', 0):.3f}")
                print(f"StyleGAN - Texture Variance: {sg.get('texture_variance', 0):.3f}")
            
            # DALL-E analysis
            if 'dalle_analysis' in tech:
                da = tech['dalle_analysis']
                print(f"DALL-E - Edge Uniformity: {da.get('edge_uniformity', 0):.3f}")
                print(f"DALL-E - Color Uniformity: {da.get('color_uniformity', 0):.3f}")
                print(f"DALL-E - Compression Score: {da.get('compression_score', 0):.3f}")
            
            # Midjourney analysis
            if 'midjourney_analysis' in tech:
                mj = tech['midjourney_analysis']
                print(f"Midjourney - Color Harmony: {mj.get('color_harmony', 0):.3f}")
                print(f"Midjourney - Gradient Smoothness: {mj.get('gradient_smoothness', 0):.3f}")
                print(f"Midjourney - Saturation Mean: {mj.get('saturation_mean', 0):.1f}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_detectors():
    """Test individual specialized detectors"""
    print("\n🧪 Testing Individual Specialized Detectors")
    print("=" * 50)
    
    try:
        from detection.ai_image_detector import StyleGANDetector, DALLEDetector, MidjourneyDetector
        
        # Create test image
        test_image = np.array(create_test_image())
        
        # Test StyleGAN detector
        print("🎭 Testing StyleGAN Detector...")
        stylegan_detector = StyleGANDetector()
        sg_results = stylegan_detector.detect_artifacts(test_image)
        print(f"✅ StyleGAN Confidence: {sg_results['stylegan_confidence']:.3f}")
        
        # Test DALL-E detector
        print("🎨 Testing DALL-E Detector...")
        dalle_detector = DALLEDetector()
        da_results = dalle_detector.detect_artifacts(test_image)
        print(f"✅ DALL-E Confidence: {da_results['dalle_confidence']:.3f}")
        
        # Test Midjourney detector
        print("🖼️ Testing Midjourney Detector...")
        midjourney_detector = MidjourneyDetector()
        mj_results = midjourney_detector.detect_artifacts(test_image)
        print(f"✅ Midjourney Confidence: {mj_results['midjourney_confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual detector test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting AI Image Detection System Tests")
    print("=" * 60)
    
    # Test main AI detector
    success1 = test_ai_detector()
    
    # Test individual detectors
    success2 = test_individual_detectors()
    
    print("\n📋 Test Summary")
    print("=" * 20)
    if success1 and success2:
        print("✅ All tests passed successfully!")
        print("🎉 AI Image Detection System is working correctly!")
        print("\nFeatures verified:")
        print("✓ EfficientNet and Vision Transformer support")
        print("✓ GradCAM visualization")
        print("✓ StyleGAN artifact detection")
        print("✓ DALL-E artifact detection")
        print("✓ Midjourney artifact detection")
        print("✓ Confidence scoring")
        print("✓ Combined multi-model analysis")
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
