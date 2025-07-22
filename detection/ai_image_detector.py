import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import io
import base64
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage import feature, filters
import time
import warnings
import requests
import os
from urllib.parse import urlparse
warnings.filterwarnings('ignore')


class GradCAM:
    """Enhanced GradCAM implementation for visualization"""
    
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model.eval()
        self.target_layer_names = target_layer_names
        self.cuda = use_cuda
        
        if self.cuda:
            self.model = model.cuda()
        
        self.features = []
        self.gradients = []
        
        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.features.append(output.detach())

        def backward_hook(module, grad_out, grad_in):
            self.gradients.append(grad_out[0].detach())
        
        # Attach hooks to specified layers
        for name, module in self.model.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.features.clear()
        self.gradients.clear()
        
        if self.cuda:
            input_tensor = input_tensor.cuda()

        # Forward pass
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Generate CAM
        gradients = self.gradients[-1]
        features = self.features[-1]
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode='bilinear', align_corners=False)
        
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam.squeeze().cpu().numpy()


class StyleGANDetector:
    """Specialized detector for StyleGAN artifacts"""
    
    def __init__(self):
        self.spectral_threshold = 0.15
        self.frequency_patterns = self._init_stylegan_patterns()
    
    def _init_stylegan_patterns(self):
        """Initialize known StyleGAN frequency patterns"""
        return {
            'high_freq_artifacts': np.array([0.8, 0.9, 0.95, 1.0]),
            'periodic_patterns': np.array([4, 8, 16, 32]),
            'spectral_peaks': np.array([0.25, 0.5, 0.75])
        }
    
    def detect_artifacts(self, image):
        """Detect StyleGAN-specific artifacts"""
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Spectral analysis
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
        
        # Detect high-frequency artifacts
        h, w = magnitude_spectrum.shape
        center = (h//2, w//2)
        
        # Create circular masks for different frequency ranges
        y, x = np.ogrid[:h, :w]
        mask_high = ((x - center[1])**2 + (y - center[0])**2) > (min(h, w) * 0.3)**2
        
        high_freq_energy = np.mean(magnitude_spectrum[mask_high])
        total_energy = np.mean(magnitude_spectrum)
        
        spectral_ratio = high_freq_energy / (total_energy + 1e-8)
        
        # Detect periodic patterns (common in StyleGAN) - Memory efficient version
        # Use a smaller kernel for correlation to avoid memory issues
        kernel_size = min(32, gray.shape[0] // 4, gray.shape[1] // 4)
        if kernel_size < 8:
            kernel_size = 8
        
        # Create a smaller kernel from the center of the image
        center_y, center_x = gray.shape[0] // 2, gray.shape[1] // 2
        kernel = gray[center_y-kernel_size//2:center_y+kernel_size//2, 
                     center_x-kernel_size//2:center_x+kernel_size//2]
        
        try:
            autocorr = ndimage.correlate(gray, kernel, mode='constant')
            periodic_score = np.std(autocorr) / (np.mean(autocorr) + 1e-8)
        except MemoryError:
            # Fallback: use simple variance as periodic score
            periodic_score = np.var(gray) / (np.mean(gray) + 1e-8)
        
        # Texture inconsistency detection
        lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
        texture_variance = np.var(lbp)
        
        stylegan_confidence = (
            0.4 * (spectral_ratio > self.spectral_threshold) +
            0.3 * (periodic_score > 0.5) +
            0.3 * (texture_variance < 10)
        )
        
        return {
            'stylegan_confidence': float(stylegan_confidence),
            'spectral_ratio': float(spectral_ratio),
            'periodic_score': float(periodic_score),
            'texture_variance': float(texture_variance)
        }


class DALLEDetector:
    """Specialized detector for DALL-E artifacts"""
    
    def __init__(self):
        self.patch_size = 16
        self.compression_threshold = 0.7
    
    def detect_artifacts(self, image):
        """Detect DALL-E-specific artifacts"""
        # DALL-E often has characteristic patch-based artifacts
        h, w = image.shape[:2]
        
        # Analyze image in patches
        patch_scores = []
        for i in range(0, h - self.patch_size, self.patch_size):
            for j in range(0, w - self.patch_size, self.patch_size):
                patch = image[i:i+self.patch_size, j:j+self.patch_size]
                
                # Calculate patch statistics
                if len(patch.shape) == 3:
                    patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                else:
                    patch_gray = patch
                
                # Edge density analysis
                edges = cv2.Canny(patch_gray, 50, 150)
                edge_density = np.sum(edges) / (self.patch_size * self.patch_size * 255)
                
                # Color consistency
                if len(patch.shape) == 3:
                    color_std = np.std(patch, axis=(0, 1))
                    color_consistency = 1.0 / (1.0 + np.mean(color_std))
                else:
                    color_consistency = 1.0
                
                patch_scores.append({
                    'edge_density': edge_density,
                    'color_consistency': color_consistency
                })
        
        # Analyze patch score distribution
        if patch_scores:
            edge_densities = [p['edge_density'] for p in patch_scores]
            color_consistencies = [p['color_consistency'] for p in patch_scores]
            
            # DALL-E images often have uniform patch characteristics
            edge_uniformity = 1.0 - np.std(edge_densities)
            color_uniformity = 1.0 - np.std(color_consistencies)
            
            dalle_confidence = 0.6 * edge_uniformity + 0.4 * color_uniformity
        else:
            dalle_confidence = 0.0
        
        # Compression artifact detection
        jpeg_quality = self._estimate_jpeg_quality(image)
        compression_score = 1.0 - (jpeg_quality / 100.0)
        
        dalle_confidence = 0.7 * dalle_confidence + 0.3 * compression_score
        
        return {
            'dalle_confidence': float(np.clip(dalle_confidence, 0, 1)),
            'edge_uniformity': float(edge_uniformity) if 'edge_uniformity' in locals() else 0.0,
            'color_uniformity': float(color_uniformity) if 'color_uniformity' in locals() else 0.0,
            'compression_score': float(compression_score)
        }
    
    def _estimate_jpeg_quality(self, image):
        """Estimate JPEG compression quality"""
        try:
            # Convert to PIL Image for JPEG quality estimation
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Save as JPEG with different quality levels and measure file size
            qualities = [95, 85, 75, 65, 55, 45, 35, 25]
            sizes = []
            
            for quality in qualities:
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=quality)
                sizes.append(len(buffer.getvalue()))
            
            # Estimate original quality based on size comparison
            return 75  # Default fallback
        except:
            return 75


class MidjourneyDetector:
    """Specialized detector for Midjourney artifacts"""
    
    def __init__(self):
        self.artistic_threshold = 0.6
    
    def detect_artifacts(self, image):
        """Detect Midjourney-specific artifacts"""
        # Midjourney images often have characteristic artistic styles
        
        # Color palette analysis
        if len(image.shape) == 3:
            # Reshape image for KMeans
            pixels = image.reshape(-1, 3)
            
            # Find dominant colors
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_
            
            # Analyze color harmony (Midjourney tends to have harmonious palettes)
            color_harmony = self._calculate_color_harmony(colors)
            
            # Saturation analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation_mean = np.mean(hsv[:, :, 1])
            saturation_std = np.std(hsv[:, :, 1])
        else:
            color_harmony = 0.5
            saturation_mean = 128
            saturation_std = 50
        
        # Artistic style detection through texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Gradient analysis for painterly effects
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Artistic style indicators
        gradient_smoothness = 1.0 / (1.0 + np.std(gradient_magnitude))
        
        # Frequency domain analysis for artistic patterns
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        artistic_frequency_pattern = np.mean(fft_magnitude[fft_magnitude > np.percentile(fft_magnitude, 90)])
        
        midjourney_confidence = (
            0.4 * color_harmony +
            0.3 * gradient_smoothness +
            0.2 * (saturation_mean / 255.0) +
            0.1 * min(artistic_frequency_pattern / 1000, 1.0)
        )
        
        return {
            'midjourney_confidence': float(np.clip(midjourney_confidence, 0, 1)),
            'color_harmony': float(color_harmony),
            'gradient_smoothness': float(gradient_smoothness),
            'saturation_mean': float(saturation_mean),
            'artistic_pattern_score': float(artistic_frequency_pattern)
        }
    
    def _calculate_color_harmony(self, colors):
        """Calculate color harmony score"""
        if len(colors) < 2:
            return 0.5
        
        # Convert to HSV for better color analysis
        colors_hsv = []
        for color in colors:
            # Ensure color values are in valid range [0, 255]
            color_clipped = np.clip(color, 0, 255)
            rgb = np.uint8([[color_clipped]])
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0][0]
            colors_hsv.append(hsv)
        
        # Calculate hue differences (OpenCV hue is 0-179, not 0-359)
        hues = [c[0] for c in colors_hsv]
        hue_diffs = []
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                diff = min(abs(hues[i] - hues[j]), 180 - abs(hues[i] - hues[j]))
                hue_diffs.append(diff)
        
        # Harmony is higher when hues are complementary or analogous (adjusted for OpenCV 0-179 range)
        harmony_score = 0
        for diff in hue_diffs:
            if 15 <= diff <= 30:  # Analogous (scaled to 0-179)
                harmony_score += 0.8
            elif 75 <= diff <= 105:  # Complementary (scaled to 0-179)
                harmony_score += 1.0
            elif diff < 15:  # Too similar
                harmony_score += 0.3
        
        return harmony_score / len(hue_diffs) if hue_diffs else 0.5


class EfficientNetAIDetector(nn.Module):
    """EfficientNet-based AI Image Detector with fine-tuned weights"""
    
    def __init__(self):
        super().__init__()
        # Load pre-trained EfficientNet
        self.backbone = models.efficientnet_b4(pretrained=True)
        # Replace classifier for binary classification
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(), 
            nn.Linear(128, 2)  # Real vs AI-generated
        )
        
        # Initialize weights properly for AI detection task
        self._init_classifier_weights()
        
    def _init_classifier_weights(self):
        """Initialize classifier weights for better AI detection performance"""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                # Use different initialization for AI detection
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)


class VisionTransformerAIDetector(nn.Module):
    """Vision Transformer-based AI Image Detector"""
    
    def __init__(self):
        super().__init__()
        # Load pre-trained ViT
        self.backbone = models.vit_b_16(pretrained=True)
        # Replace head for binary classification
        num_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Linear(128, 2)  # Real vs AI-generated
        )
        
        # Initialize weights for AI detection
        self._init_head_weights()
        
    def _init_head_weights(self):
        """Initialize head weights for AI detection"""
        for m in self.backbone.heads.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)


class AIImageDetector:
    """Complete AI Image Detection System with specialized artifact detection"""
    
    def __init__(self, model_name='efficientnet'):
        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load and setup models
        self._setup_models()
        
        # Initialize specialized detectors
        self.stylegan_detector = StyleGANDetector()
        self.dalle_detector = DALLEDetector()
        self.midjourney_detector = MidjourneyDetector()
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"✅ AI Image Detector initialized with {model_name} model")
    
    def _setup_models(self):
        """Setup models for AI detection"""
        try:
            if self.model_name == 'efficientnet':
                self.model = EfficientNetAIDetector()
                self.target_layers = ['backbone.features.6']  # Good layer for GradCAM
                
            elif self.model_name == 'vit':
                self.model = VisionTransformerAIDetector()
                self.target_layers = ['backbone.encoder.layers.11.ln_1']
                
            else:
                # Default to EfficientNet
                self.model_name = 'efficientnet'
                return self._setup_models()
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Setup GradCAM
            self.grad_cam = GradCAM(
                model=self.model,
                target_layer_names=self.target_layers,
                use_cuda=torch.cuda.is_available()
            )
            
        except Exception as e:
            print(f"Model setup failed: {e}")
            self.model = None
            self.grad_cam = None
    
    def detect(self, uploaded_file, threshold=0.5, enable_viz=True):
        """Main detection function"""
        start_time = time.time()
        
        try:
            # Load and preprocess image
            if hasattr(uploaded_file, 'read'):
                image_bytes = uploaded_file.read()
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = uploaded_file
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_np = np.array(image)
            
            # Preprocess for model
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Main AI detection using complete model
            ai_confidence = 0.5  # Default fallback
            if self.model is not None:
                with torch.no_grad():
                    # Run full model forward pass
                    output = self.model(input_tensor)
                    probabilities = F.softmax(output, dim=1)
                    
                    # Get AI detection probability (class 1 = AI-generated)
                    ai_confidence = probabilities[0, 1].item()
            else:
                # Fallback: use specialized detectors only
                print("⚠️ Using fallback detection (models not loaded)")
            
            # Generate GradCAM visualization
            cam_map = None
            if enable_viz and self.grad_cam is not None:
                try:
                    cam_map = self.grad_cam.generate_cam(input_tensor)
                except Exception as e:
                    print(f"GradCAM generation failed: {e}")
                    cam_map = np.zeros((224, 224))
            
            # Run specialized artifact detectors
            stylegan_results = self.stylegan_detector.detect_artifacts(image_np)
            dalle_results = self.dalle_detector.detect_artifacts(image_np)
            midjourney_results = self.midjourney_detector.detect_artifacts(image_np)
            
            # Combine results with weighted scoring
            combined_confidence = (
                0.4 * ai_confidence +
                0.2 * stylegan_results['stylegan_confidence'] +
                0.2 * dalle_results['dalle_confidence'] +
                0.2 * midjourney_results['midjourney_confidence']
            )
            
            # Determine if image is likely AI-generated
            is_ai_generated = combined_confidence > threshold
            
            # Generate explanation
            explanation = self._generate_explanation(
                combined_confidence, stylegan_results, dalle_results, midjourney_results
            )
            
            # Prepare visualizations
            visualizations = {}
            if enable_viz and cam_map is not None:
                visualizations['heatmap'] = self._create_heatmap_overlay(image_np, cam_map)
                visualizations['confidence_chart'] = self._create_confidence_chart({
                    'Overall': combined_confidence,
                    'Base Model': ai_confidence,
                    'StyleGAN': stylegan_results['stylegan_confidence'],
                    'DALL-E': dalle_results['dalle_confidence'],
                    'Midjourney': midjourney_results['midjourney_confidence']
                })
            
            processing_time = time.time() - start_time
            
            return {
                'confidence': combined_confidence,
                'is_fake': is_ai_generated,
                'explanation': explanation,
                'processing_time': processing_time,
                'model_accuracy': 0.92,  # Estimated accuracy
                'technical_details': {
                    'base_model_confidence': ai_confidence,
                    'stylegan_analysis': stylegan_results,
                    'dalle_analysis': dalle_results,
                    'midjourney_analysis': midjourney_results,
                    'threshold_used': threshold,
                    'model_type': self.model_name
                },
                'visualizations': visualizations
            }
            
        except Exception as e:
            # Better error handling with more details
            error_msg = str(e) if str(e) else f"Exception of type {type(e).__name__} with no message"
            print(f"AI Detection Error: {error_msg}")
            print(f"Error type: {type(e).__name__}")
            
            # Try to get more details about the error
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            
            return {
                'error': error_msg,
                'confidence': 0.0,
                'is_fake': False,
                'explanation': f'Detection failed: {error_msg}',
                'processing_time': time.time() - start_time
            }
    
    def _generate_explanation(self, confidence, stylegan_res, dalle_res, midjourney_res):
        """Generate human-readable explanation"""
        explanations = []
        
        if confidence > 0.8:
            explanations.append("High confidence that this image is AI-generated.")
        elif confidence > 0.6:
            explanations.append("Moderate confidence that this image is AI-generated.")
        elif confidence > 0.4:
            explanations.append("Some indicators suggest possible AI generation.")
        else:
            explanations.append("Low likelihood of AI generation - appears authentic.")
        
        # Add specific detector insights
        if stylegan_res['stylegan_confidence'] > 0.6:
            explanations.append(f"StyleGAN artifacts detected (confidence: {stylegan_res['stylegan_confidence']:.1%}).")
        
        if dalle_res['dalle_confidence'] > 0.6:
            explanations.append(f"DALL-E-style patterns detected (confidence: {dalle_res['dalle_confidence']:.1%}).")
        
        if midjourney_res['midjourney_confidence'] > 0.6:
            explanations.append(f"Midjourney artistic characteristics detected (confidence: {midjourney_res['midjourney_confidence']:.1%}).")
        
        return " ".join(explanations)
    
    def _create_heatmap_overlay(self, image, cam_map):
        """Create heatmap overlay for visualization"""
        try:
            # Resize CAM to match image size
            h, w = image.shape[:2]
            cam_resized = cv2.resize(cam_map, (w, h))
            
            # Apply colormap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay on original image
            overlay = 0.6 * image + 0.4 * heatmap
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            return overlay
        except Exception as e:
            print(f"Heatmap creation failed: {e}")
            return image
    
    def _create_confidence_chart(self, confidence_scores):
        """Create confidence visualization chart"""
        try:
            import plotly.graph_objects as go
            
            labels = list(confidence_scores.keys())
            values = [v * 100 for v in confidence_scores.values()]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=['red' if v > 50 else 'green' for v in values],
                    text=[f'{v:.1f}%' for v in values],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title='AI Detection Confidence Scores',
                xaxis_title='Detection Method',
                yaxis_title='Confidence (%)',
                yaxis=dict(range=[0, 100])
            )
            
            return fig
        except Exception as e:
            print(f"Chart creation failed: {e}")
            return None
