import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS
import tempfile
import os
import io
import json
import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict, Any
import logging
from settings import Config

try:
    from exifread import process_file
    EXIFREAD_AVAILABLE = True
except ImportError:
    EXIFREAD_AVAILABLE = False
    logging.warning("ExifRead not available. Install with: pip install ExifRead")

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logging.warning("python-magic not available. Install with: pip install python-magic python-magic-bin")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Install with: pip install psutil")

class MediaProcessor:
    """Complete media processing utilities for forensic analysis"""
    
    def __init__(self):
        Config.configure_logging()
        self.logger = logging.getLogger(__name__)
        
        # Processing parameters
        self.target_size = (224, 224)
        self.max_video_frames = 100
        self.frame_sample_rate = 1
        
    def process_image(self, image_data: Union[bytes, np.ndarray, Image.Image], 
                     enhance: bool = True) -> Tuple[np.ndarray, dict]:
        """Process image for analysis with optional enhancement"""
        
        try:
            # Convert input to PIL Image
            if isinstance(image_data, bytes):
                pil_image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
            elif isinstance(image_data, Image.Image):
                pil_image = image_data.copy()
            else:
                raise ValueError("Unsupported image format")
            
            # Store original dimensions
            original_size = pil_image.size
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Optional enhancement for better analysis
            if enhance:
                pil_image = self._enhance_image(pil_image)
            
            # Resize for consistent processing
            processed_image = pil_image.resize(self.target_size, Image.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(processed_image)
            
            # Calculate image statistics
            stats = self._calculate_image_stats(image_array)
            
            metadata = {
                'original_size': original_size,
                'processed_size': self.target_size,
                'color_mode': 'RGB',
                'enhanced': enhance,
                'stats': stats
            }
            
            return image_array, metadata
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise
    
    def extract_metadata(self, file_path: str) -> dict:
        """Extracts metadata from image using ExifRead"""
        
        metadata = {}
        if not EXIFREAD_AVAILABLE:
            self.logger.warning("ExifRead is not available. Returning empty metadata.")
            return metadata

        try:
            with open(file_path, 'rb') as f:
                tags = process_file(f, details=False)
                for tag, value in tags.items():
                    tag_name = TAGS.get(tag, tag)
                    metadata[tag_name] = str(value)
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")

        return metadata

    def validate_file(self, file_path: str) -> bool:
        """Validates file format and existence."""
        
        if not os.path.exists(file_path):
            self.logger.error("File does not exist.")
            return False

        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_file(file_path, mime=True)
                file_type = mimetypes.guess_extension(mime_type)
                if not file_type:
                    self.logger.error("Unsupported file format.")
                    return False
            except Exception as e:
                self.logger.error(f"Error during file validation: {str(e)}")
                return False

        return True

    def convert_format(self, input_path: str, output_path: str, format: str = 'png') -> bool:
        """Converts image format using PIL."""
        
        try:
            with Image.open(input_path) as img:
                img = ImageOps.exif_transpose(img)  # Ensure correct orientation
                img.save(output_path, format.upper())
            return True
        except Exception as e:
            self.logger.error(f"Error converting format: {str(e)}")
            return False

    def process_video(self, video_path: str, 
                     extract_frames: bool = True,
                     max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], dict]:
        """Process video for analysis with frame extraction"""
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            frames = []
            
            if extract_frames:
                frames = self._extract_key_frames(cap, max_frames or self.max_video_frames)
            
            cap.release()
            
            metadata = {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'resolution': (width, height),
                'extracted_frames': len(frames),
                'file_path': video_path
            }
            
            return frames, metadata
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            raise
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply enhancement techniques to improve analysis quality"""
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)  # Slight contrast boost
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)  # Slight sharpness boost
        
        # Apply subtle noise reduction
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return image
    
    def _extract_key_frames(self, cap: cv2.VideoCapture, max_frames: int) -> List[np.ndarray]:
        """Extract key frames from video for analysis"""
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling
        if frame_count <= max_frames:
            frame_indices = list(range(frame_count))
        else:
            # Sample evenly distributed frames
            frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
        
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame for consistent processing
                frame_resized = cv2.resize(frame, self.target_size)
                frames.append(frame_resized)
        
        return frames
    
    def _calculate_image_stats(self, image: np.ndarray) -> dict:
        """Calculate statistical properties of the image"""
        
        # Convert to grayscale for some calculations
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Basic statistics
        stats = {
            'mean_brightness': float(np.mean(gray)),
            'std_brightness': float(np.std(gray)),
            'min_brightness': int(np.min(gray)),
            'max_brightness': int(np.max(gray)),
            'contrast': float(np.std(gray) / np.mean(gray)) if np.mean(gray) > 0 else 0
        }
        
        # Color statistics
        for i, channel in enumerate(['red', 'green', 'blue']):
            channel_data = image[:, :, i]
            stats[f'{channel}_mean'] = float(np.mean(channel_data))
            stats[f'{channel}_std'] = float(np.std(channel_data))
        
        # Edge detection for sharpness estimation
        edges = cv2.Canny(gray, 50, 150)
        stats['edge_density'] = float(np.sum(edges > 0) / edges.size)
        
        # Texture analysis using local binary patterns approximation
        stats['texture_variance'] = float(self._calculate_texture_variance(gray))
        
        return stats
    
    def _calculate_texture_variance(self, gray_image: np.ndarray) -> float:
        """Calculate texture variance as a measure of image complexity"""
        
        # Apply Sobel filters
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Return variance of gradient magnitude
        return float(np.var(gradient_magnitude))
    
    def preprocess_for_model(self, image: np.ndarray, 
                           normalize: bool = True,
                           to_tensor: bool = False) -> Union[np.ndarray, 'torch.Tensor']:
        """Preprocess image for model inference"""
        
        # Ensure image is in correct format
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Normalize to [0, 1]
        if normalize and image.max() > 1.0:
            image = image / 255.0
        
        # Standard ImageNet normalization
        if normalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
        
        # Convert to tensor if requested
        if to_tensor:
            try:
                import torch
                # Convert from HWC to CHW format
                image = np.transpose(image, (2, 0, 1))
                # Convert to tensor and add batch dimension
                image = torch.FloatTensor(image).unsqueeze(0)
            except ImportError:
                self.logger.warning("PyTorch not available, returning numpy array")
        
        return image
    
    def extract_face_regions(self, image: np.ndarray, 
                           face_boxes: List[List[int]]) -> List[np.ndarray]:
        """Extract face regions from image given bounding boxes"""
        
        faces = []
        
        for box in face_boxes:
            x1, y1, x2, y2 = box
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Extract face region
            if x2 > x1 and y2 > y1:
                face = image[y1:y2, x1:x2]
                
                # Resize to target size
                if face.size > 0:
                    face_resized = cv2.resize(face, self.target_size)
                    faces.append(face_resized)
        
        return faces
    
    def calculate_video_quality_metrics(self, frames: List[np.ndarray]) -> dict:
        """Calculate video quality metrics from frames"""
        
        if not frames:
            return {}
        
        metrics = {
            'total_frames': len(frames),
            'avg_brightness': 0.0,
            'avg_contrast': 0.0,
            'avg_sharpness': 0.0,
            'frame_consistency': 0.0
        }
        
        brightness_values = []
        contrast_values = []
        sharpness_values = []
        
        for frame in frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Sharpness using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            brightness_values.append(brightness)
            contrast_values.append(contrast)
            sharpness_values.append(sharpness)
        
        # Calculate averages
        metrics['avg_brightness'] = float(np.mean(brightness_values))
        metrics['avg_contrast'] = float(np.mean(contrast_values))
        metrics['avg_sharpness'] = float(np.mean(sharpness_values))
        
        # Frame consistency (lower values indicate more consistent frames)
        metrics['frame_consistency'] = float(np.std(brightness_values))
        
        return metrics
    
    def save_processed_frames(self, frames: List[np.ndarray], 
                            output_dir: str,
                            prefix: str = "frame") -> List[str]:
        """Save processed frames to directory"""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_paths = []
        
        for i, frame in enumerate(frames):
            filename = f"{prefix}_{i:04d}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_to_save = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_to_save = frame
            
            # Save using PIL for better quality
            pil_image = Image.fromarray(frame_to_save)
            pil_image.save(filepath)
            
            saved_paths.append(filepath)
        
        return saved_paths
    
    def extract_video_frames_advanced(self, video_path: str, 
                                    extraction_method: str = 'uniform',
                                    specific_timestamps: List[float] = None,
                                    frame_rate: float = None,
                                    quality_threshold: float = 0.0) -> Tuple[List[np.ndarray], List[Dict]]:
        """Advanced video frame extraction with multiple methods"""
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            frames = []
            frame_metadata = []
            
            if extraction_method == 'uniform':
                # Extract frames uniformly distributed across the video
                frame_indices = np.linspace(0, total_frames - 1, self.max_video_frames, dtype=int)
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        quality_score = self._calculate_frame_quality(frame)
                        if quality_score >= quality_threshold:
                            frames.append(frame)
                            frame_metadata.append({
                                'frame_index': int(idx),
                                'timestamp': idx / fps,
                                'quality_score': quality_score
                            })
                            
            elif extraction_method == 'keyframes':
                # Extract keyframes using scene change detection
                frames, frame_metadata = self._extract_keyframes_scene_detection(cap, fps, quality_threshold)
                
            elif extraction_method == 'timestamps' and specific_timestamps:
                # Extract frames at specific timestamps
                for timestamp in specific_timestamps:
                    frame_num = int(timestamp * fps)
                    if frame_num < total_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        ret, frame = cap.read()
                        if ret:
                            quality_score = self._calculate_frame_quality(frame)
                            if quality_score >= quality_threshold:
                                frames.append(frame)
                                frame_metadata.append({
                                    'frame_index': frame_num,
                                    'timestamp': timestamp,
                                    'quality_score': quality_score
                                })
                                
            elif extraction_method == 'rate' and frame_rate:
                # Extract frames at specified rate
                frame_interval = int(fps / frame_rate)
                for i in range(0, total_frames, frame_interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        quality_score = self._calculate_frame_quality(frame)
                        if quality_score >= quality_threshold:
                            frames.append(frame)
                            frame_metadata.append({
                                'frame_index': i,
                                'timestamp': i / fps,
                                'quality_score': quality_score
                            })
            
            cap.release()
            return frames, frame_metadata
            
        except Exception as e:
            self.logger.error(f"Error in advanced frame extraction: {str(e)}")
            raise
    
    def _calculate_frame_quality(self, frame: np.ndarray) -> float:
        """Calculate frame quality score based on sharpness and contrast"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance for sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Standard deviation for contrast
        contrast = np.std(gray)
        
        # Combine metrics (normalized)
        quality_score = (sharpness / 1000.0) + (contrast / 255.0)
        
        return min(quality_score, 1.0)
    
    def _extract_keyframes_scene_detection(self, cap: cv2.VideoCapture, fps: float, quality_threshold: float) -> Tuple[List[np.ndarray], List[Dict]]:
        """Extract keyframes using scene change detection"""
        
        frames = []
        frame_metadata = []
        prev_frame = None
        frame_count = 0
        threshold = 30.0  # Scene change threshold
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if prev_frame is not None:
                # Calculate frame difference
                gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                
                diff = cv2.absdiff(gray_curr, gray_prev)
                mean_diff = np.mean(diff)
                
                # If significant change detected, consider as keyframe
                if mean_diff > threshold:
                    quality_score = self._calculate_frame_quality(frame)
                    if quality_score >= quality_threshold:
                        frames.append(frame)
                        frame_metadata.append({
                            'frame_index': frame_count,
                            'timestamp': frame_count / fps,
                            'quality_score': quality_score,
                            'scene_change_score': mean_diff
                        })
            
            prev_frame = frame
            frame_count += 1
            
            # Limit number of keyframes
            if len(frames) >= self.max_video_frames:
                break
        
        return frames, frame_metadata
    
    def preprocess_image_advanced(self, image: Union[np.ndarray, Image.Image], 
                                preprocessing_options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict]:
        """Advanced image preprocessing with multiple options"""
        
        if preprocessing_options is None:
            preprocessing_options = {}
        
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = image.copy()
        
        processing_log = []
        
        # Noise reduction
        if preprocessing_options.get('denoise', False):
            pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))
            processing_log.append('median_filter_applied')
        
        # Histogram equalization
        if preprocessing_options.get('histogram_equalization', False):
            # Convert to numpy for histogram equalization
            np_image = np.array(pil_image)
            if len(np_image.shape) == 3:
                # Apply to each channel
                for i in range(3):
                    np_image[:, :, i] = cv2.equalizeHist(np_image[:, :, i])
            else:
                np_image = cv2.equalizeHist(np_image)
            pil_image = Image.fromarray(np_image)
            processing_log.append('histogram_equalization_applied')
        
        # Gamma correction
        gamma = preprocessing_options.get('gamma', None)
        if gamma and gamma != 1.0:
            np_image = np.array(pil_image)
            np_image = np.power(np_image / 255.0, gamma) * 255.0
            np_image = np.clip(np_image, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(np_image)
            processing_log.append(f'gamma_correction_{gamma}_applied')
        
        # Contrast and brightness adjustment
        contrast_factor = preprocessing_options.get('contrast', 1.0)
        if contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast_factor)
            processing_log.append(f'contrast_adjustment_{contrast_factor}_applied')
        
        brightness_factor = preprocessing_options.get('brightness', 1.0)
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness_factor)
            processing_log.append(f'brightness_adjustment_{brightness_factor}_applied')
        
        # Sharpening
        if preprocessing_options.get('sharpen', False):
            pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            processing_log.append('unsharp_mask_applied')
        
        # Color space conversion
        color_space = preprocessing_options.get('color_space', 'RGB')
        if color_space != 'RGB':
            if color_space == 'GRAY':
                pil_image = pil_image.convert('L')
            elif color_space == 'HSV':
                # Convert to numpy, then HSV, then back to PIL
                np_image = np.array(pil_image)
                hsv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
                pil_image = Image.fromarray(hsv_image)
            processing_log.append(f'color_space_conversion_to_{color_space}')
        
        # Resize
        target_size = preprocessing_options.get('resize', self.target_size)
        if target_size and target_size != pil_image.size:
            resize_method = preprocessing_options.get('resize_method', 'LANCZOS')
            if resize_method == 'LANCZOS':
                pil_image = pil_image.resize(target_size, Image.LANCZOS)
            elif resize_method == 'BICUBIC':
                pil_image = pil_image.resize(target_size, Image.BICUBIC)
            elif resize_method == 'BILINEAR':
                pil_image = pil_image.resize(target_size, Image.BILINEAR)
            processing_log.append(f'resized_to_{target_size}_using_{resize_method}')
        
        # Convert back to numpy array
        processed_array = np.array(pil_image)
        
        metadata = {
            'processing_log': processing_log,
            'final_shape': processed_array.shape,
            'final_dtype': str(processed_array.dtype),
            'processing_options_used': preprocessing_options
        }
        
        return processed_array, metadata
    
    def extract_comprehensive_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from media files"""
        
        metadata = {
            'file_info': {},
            'exif_data': {},
            'technical_data': {},
            'hash_data': {},
            'file_validation': {}
        }
        
        try:
            # Basic file information
            file_path_obj = Path(file_path)
            file_stat = file_path_obj.stat()
            
            metadata['file_info'] = {
                'filename': file_path_obj.name,
                'file_extension': file_path_obj.suffix.lower(),
                'file_size_bytes': file_stat.st_size,
                'file_size_mb': round(file_stat.st_size / (1024 * 1024), 2),
                'creation_time': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'modification_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'access_time': datetime.fromtimestamp(file_stat.st_atime).isoformat()
            }
            
            # File hashes for integrity checking
            metadata['hash_data'] = self._calculate_file_hashes(file_path)
            
            # MIME type detection
            mime_type, _ = mimetypes.guess_type(file_path)
            metadata['file_info']['mime_type'] = mime_type
            
            # Magic number detection if available
            if MAGIC_AVAILABLE:
                try:
                    metadata['file_info']['magic_type'] = magic.from_file(file_path)
                    metadata['file_info']['magic_mime'] = magic.from_file(file_path, mime=True)
                except Exception as e:
                    self.logger.warning(f"Magic detection failed: {e}")
            
            # EXIF data extraction
            if EXIFREAD_AVAILABLE:
                metadata['exif_data'] = self._extract_detailed_exif(file_path)
            
            # PIL-based metadata for images
            try:
                with Image.open(file_path) as img:
                    metadata['technical_data'].update({
                        'image_mode': img.mode,
                        'image_size': img.size,
                        'image_format': img.format,
                        'has_transparency': img.mode in ('RGBA', 'LA', 'P'),
                        'color_profile': 'ICC' if img.info.get('icc_profile') else 'sRGB'
                    })
                    
                    # PIL EXIF data
                    if hasattr(img, '_getexif') and img._getexif():
                        pil_exif = img._getexif()
                        if pil_exif:
                            metadata['exif_data']['pil_exif'] = {
                                TAGS.get(k, k): v for k, v in pil_exif.items()
                            }
            except Exception as e:
                self.logger.debug(f"PIL metadata extraction failed: {e}")
            
            # Video-specific metadata
            if file_path_obj.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
                video_metadata = self._extract_video_metadata(file_path)
                metadata['technical_data'].update(video_metadata)
            
            # File validation
            metadata['file_validation'] = self._comprehensive_file_validation(file_path)
            
        except Exception as e:
            self.logger.error(f"Error extracting comprehensive metadata: {str(e)}")
            metadata['error'] = str(e)
        
        return metadata
    
    def _calculate_file_hashes(self, file_path: str) -> Dict[str, str]:
        """Calculate multiple hash values for file integrity"""
        
        hashes = {}
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                
                # MD5
                hashes['md5'] = hashlib.md5(content).hexdigest()
                
                # SHA-1
                hashes['sha1'] = hashlib.sha1(content).hexdigest()
                
                # SHA-256
                hashes['sha256'] = hashlib.sha256(content).hexdigest()
                
                # CRC32
                import zlib
                hashes['crc32'] = format(zlib.crc32(content) & 0xffffffff, '08x')
                
        except Exception as e:
            self.logger.error(f"Error calculating file hashes: {str(e)}")
        
        return hashes
    
    def _extract_detailed_exif(self, file_path: str) -> Dict[str, Any]:
        """Extract detailed EXIF data using ExifRead"""
        
        exif_data = {}
        
        try:
            with open(file_path, 'rb') as f:
                tags = process_file(f, details=True)
                
                for tag, value in tags.items():
                    # Convert tag to string and clean up
                    tag_str = str(tag)
                    value_str = str(value)
                    
                    # Organize by category
                    if tag_str.startswith('EXIF'):
                        category = 'camera_settings'
                    elif tag_str.startswith('GPS'):
                        category = 'gps_data'
                    elif tag_str.startswith('Image'):
                        category = 'image_data'
                    elif tag_str.startswith('Thumbnail'):
                        category = 'thumbnail'
                    else:
                        category = 'other'
                    
                    if category not in exif_data:
                        exif_data[category] = {}
                    
                    exif_data[category][tag_str] = value_str
                    
        except Exception as e:
            self.logger.error(f"Error extracting detailed EXIF: {str(e)}")
        
        return exif_data
    
    def _extract_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract detailed video metadata"""
        
        video_metadata = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if cap.isOpened():
                video_metadata.update({
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
                    'backend': cap.getBackendName()
                })
                
                # Calculate duration
                fps = video_metadata['fps']
                frame_count = video_metadata['frame_count']
                if fps > 0:
                    video_metadata['duration_seconds'] = frame_count / fps
                    video_metadata['duration_formatted'] = f"{int(video_metadata['duration_seconds'] // 60)}:{int(video_metadata['duration_seconds'] % 60):02d}"
                
                cap.release()
                
        except Exception as e:
            self.logger.error(f"Error extracting video metadata: {str(e)}")
        
        return video_metadata
    
    def _comprehensive_file_validation(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive file validation"""
        
        validation_results = {
            'exists': False,
            'readable': False,
            'valid_format': False,
            'format_matches_extension': False,
            'file_size_valid': False,
            'corruption_check': 'unknown'
        }
        
        try:
            # File existence
            validation_results['exists'] = os.path.exists(file_path)
            
            if validation_results['exists']:
                # File readability
                try:
                    with open(file_path, 'rb') as f:
                        f.read(1024)  # Try to read first 1KB
                    validation_results['readable'] = True
                except:
                    validation_results['readable'] = False
                
                # File size validation
                file_size = os.path.getsize(file_path)
                validation_results['file_size_valid'] = file_size > 0
                validation_results['file_size_bytes'] = file_size
                
                # Format validation
                file_ext = Path(file_path).suffix.lower()
                
                # Magic number validation
                if MAGIC_AVAILABLE:
                    try:
                        detected_type = magic.from_file(file_path, mime=True)
                        validation_results['detected_mime_type'] = detected_type
                        
                        # Check if extension matches detected type
                        expected_ext = mimetypes.guess_extension(detected_type)
                        validation_results['format_matches_extension'] = (expected_ext == file_ext)
                        validation_results['valid_format'] = True
                        
                    except Exception as e:
                        validation_results['magic_error'] = str(e)
                
                # Corruption check for images
                if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                    try:
                        with Image.open(file_path) as img:
                            img.verify()  # Verify image integrity
                        validation_results['corruption_check'] = 'passed'
                    except Exception as e:
                        validation_results['corruption_check'] = f'failed: {str(e)}'
                
                # Corruption check for videos
                elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                    try:
                        cap = cv2.VideoCapture(file_path)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret:
                                validation_results['corruption_check'] = 'passed'
                            else:
                                validation_results['corruption_check'] = 'failed: cannot read frames'
                        else:
                            validation_results['corruption_check'] = 'failed: cannot open video'
                        cap.release()
                    except Exception as e:
                        validation_results['corruption_check'] = f'failed: {str(e)}'
                
        except Exception as e:
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    def convert_image_format(self, input_path: str, output_path: str, 
                           target_format: str = 'PNG', 
                           quality: int = 95,
                           preserve_metadata: bool = True) -> bool:
        """Advanced image format conversion with options"""
        
        try:
            with Image.open(input_path) as img:
                # Handle EXIF orientation
                img = ImageOps.exif_transpose(img)
                
                # Convert color mode if necessary
                if target_format.upper() == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for JPEG
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif target_format.upper() == 'PNG' and img.mode not in ('RGB', 'RGBA', 'L', 'LA', 'P'):
                    img = img.convert('RGBA')
                
                # Prepare save arguments
                save_kwargs = {'format': target_format.upper()}
                
                if target_format.upper() == 'JPEG':
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif target_format.upper() == 'PNG':
                    save_kwargs['optimize'] = True
                elif target_format.upper() == 'WEBP':
                    save_kwargs['quality'] = quality
                    save_kwargs['method'] = 6
                
                # Preserve metadata if requested
                if preserve_metadata and hasattr(img, 'info'):
                    save_kwargs['exif'] = img.info.get('exif', b'')
                    save_kwargs['icc_profile'] = img.info.get('icc_profile')
                
                img.save(output_path, **save_kwargs)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error converting image format: {str(e)}")
            return False
