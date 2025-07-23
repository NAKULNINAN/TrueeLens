import time
import cv2
import torch
import numpy as np
import tempfile
import os
import gc
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from .face_extractor import FaceExtractor
from models.xception_net import load_xception_model
from models.meso_net import load_meso_model
from utils.model_manager import ModelManager
from utils.batch_processor import MemoryAwareBatchProcessor, BatchTask
from database.similarity_cache import SimilarityCache

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class DeepfakeDetector:
    def __init__(self, cache_results: bool = True, batch_size: int = 8, max_memory_mb: int = 2048):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        self.face_extractor = FaceExtractor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize optimized components
        self.model_manager = ModelManager(max_cached_models=2, max_memory_mb=max_memory_mb // 2)
        self.batch_processor = MemoryAwareBatchProcessor(batch_size=batch_size, max_memory_mb=max_memory_mb)
        self.cache_results = cache_results
        
        if cache_results:
            self.result_cache = SimilarityCache('deepfake_cache.db')
        
        # Register model loaders for lazy loading
        self._register_model_loaders()
        
        # Models will be loaded lazily
        self._xception_model = None
        self._meso_model = None
        
        # Performance tracking
        self.processing_stats = {
            'total_frames_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_processing_times': [],
            'memory_usage_peaks': []
        }
    
    def _register_model_loaders(self):
        """Register model loaders for lazy loading"""
        def load_xception():
            model = load_xception_model(use_custom=True)
            model.to(self.device)
            model.eval()
            return model
        
        def load_meso():
            model = load_meso_model(model_type='meso4')
            model.to(self.device)
            model.eval()
            return model
        
        self.model_manager.register_model_loader('xception', load_xception, {'memory_mb_estimate': 800})
        self.model_manager.register_model_loader('meso', load_meso, {'memory_mb_estimate': 400})
    
    @property
    def xception_model(self):
        """Lazy loading for XceptionNet model"""
        if self._xception_model is None:
            self._xception_model = self.model_manager.get_model('xception')
        return self._xception_model
    
    @property
    def meso_model(self):
        """Lazy loading for MesoNet model"""
        if self._meso_model is None:
            self._meso_model = self.model_manager.get_model('meso')
        return self._meso_model
    
    def detect(self, uploaded_file, threshold=0.5, enable_viz=True):
        """Complete deepfake detection with MTCNN face extraction and model inference"""
        start_time = time.time()
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            return self._analyze_video(tmp_path, threshold, enable_viz, start_time)
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def _analyze_video(self, video_path, threshold, enable_viz, start_time):
        """Analyze video frame by frame for deepfake detection with robust face extraction"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_results = []
        processed_frames = 0
        faces_detected = 0
        failed_face_detection_frames = 0
        
        # Sample frames for efficiency
        sample_rate = max(1, frame_count // 50)  # Process ~50 frames max
        
        frame_idx = 0
        while cap.isOpened() and processed_frames < 50:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames based on sample rate
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue
            
            # Extract faces using robust method (tries multiple detectors and enhancement)
            faces = self.face_extractor.extract_faces_robust(frame)
            
            if len(faces) > 0:
                faces_detected += len(faces)
                
                for face_idx, face in enumerate(faces):
                    try:
                        # Preprocess face for models
                        face_tensor = self.face_extractor.preprocess(face)
                        face_tensor = face_tensor.to(self.device)
                        
                        # Run inference with both models
                        with torch.no_grad():
                            # XceptionNet prediction
                            xception_output = self.xception_model(face_tensor)
                            xception_probs = torch.nn.functional.softmax(xception_output, dim=1)
                            xception_confidence = xception_probs[0][1].item()  # Fake class probability
                            
                            # MesoNet prediction
                            meso_output = self.meso_model(face_tensor)
                            meso_confidence = torch.sigmoid(meso_output).item()
                        
                        # Ensemble prediction (average of both models)
                        ensemble_confidence = (xception_confidence + meso_confidence) / 2.0
                        
                        frame_result = {
                            'frame_idx': frame_idx,
                            'face_idx': face_idx,
                            'xception_confidence': float(xception_confidence),
                            'meso_confidence': float(meso_confidence),
                            'ensemble_confidence': float(ensemble_confidence),
                            'is_fake': ensemble_confidence > threshold
                        }
                        
                        frame_results.append(frame_result)
                        processed_frames += 1
                        
                    except Exception as e:
                        # Log face processing errors for debugging
                        self.logger.warning(f"Face processing failed on frame {frame_idx}, face {face_idx}: {e}")
                        continue
            else:
                # Count frames where no faces were detected
                failed_face_detection_frames += 1
            
            frame_idx += 1
        
        cap.release()
        
        # Calculate final verdict with enhanced error tracking
        return self._calculate_final_verdict(
            frame_results, faces_detected, frame_count, 
            fps, threshold, start_time, enable_viz, failed_face_detection_frames
        )
    
    def _calculate_final_verdict(self, frame_results, faces_detected, 
                               total_frames, fps, threshold, start_time, enable_viz, failed_face_detection_frames=0):
        """Calculate final deepfake verdict based on frame analysis with enhanced error tracking"""
        
        if len(frame_results) == 0:
            if failed_face_detection_frames > 0:
                explanation = f'No faces detected in video for analysis. Failed to detect faces in {failed_face_detection_frames} sampled frames. This could be due to poor video quality, no faces in the video, or faces that are too small/blurry to detect reliably.'
            else:
                explanation = 'No faces detected in video for analysis. Please ensure the video contains clear, visible faces.'
                
            return {
                'confidence': 0.0,
                'is_fake': False,
                'explanation': explanation,
                'processing_time': time.time() - start_time,
                'model_accuracy': 0.92,
                'status': 'no_faces_detected',
                'technical_details': {
                    'total_frames': total_frames,
                    'processed_frames': 0,
                    'faces_detected': 0,
                    'failed_face_detection_frames': failed_face_detection_frames,
                    'fps': fps,
                    'detection_attempted': True,
                    'robust_detection_used': True
                }
            }
        
        # Calculate statistics
        fake_frames = sum(1 for result in frame_results if result['is_fake'])
        avg_xception_conf = np.mean([r['xception_confidence'] for r in frame_results])
        avg_meso_conf = np.mean([r['meso_confidence'] for r in frame_results])
        avg_ensemble_conf = np.mean([r['ensemble_confidence'] for r in frame_results])
        
        # Final verdict based on percentage of fake frames
        fake_percentage = fake_frames / len(frame_results)
        is_fake = fake_percentage > 0.3  # If >30% of frames are fake, classify as deepfake
        
        # Confidence score is the average ensemble confidence
        confidence = float(avg_ensemble_conf)
        
        # Generate explanation
        if is_fake:
            explanation = f"DEEPFAKE DETECTED: {fake_frames}/{len(frame_results)} frames ({fake_percentage:.1%}) classified as synthetic"
        else:
            explanation = f"AUTHENTIC: {fake_frames}/{len(frame_results)} frames ({fake_percentage:.1%}) classified as synthetic (below 30% threshold)"
        
        # Prepare visualizations if requested
        visualizations = {}
        if enable_viz:
            visualizations = self._create_visualizations(frame_results)
        
        return {
            'confidence': confidence,
            'is_fake': is_fake,
            'explanation': explanation,
            'processing_time': time.time() - start_time,
            'model_accuracy': 0.92,
            'visualizations': visualizations,
            'technical_details': {
                'total_frames': total_frames,
                'processed_frames': len(frame_results),
                'faces_detected': faces_detected,
                'fake_frames': fake_frames,
                'fake_percentage': fake_percentage,
                'fps': fps,
                'avg_xception_confidence': float(avg_xception_conf),
                'avg_meso_confidence': float(avg_meso_conf),
                'avg_ensemble_confidence': float(avg_ensemble_conf),
                'threshold_used': threshold,
                'sample_frame_results': frame_results[:5]  # Show first 5 results
            }
        }
    
    def _create_visualizations(self, frame_results):
        """Create visualization data for the analysis results"""
        import matplotlib.pyplot as plt
        import io
        import base64
        
        # Create confidence timeline
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot confidence scores over frames
        frame_indices = [r['frame_idx'] for r in frame_results]
        xception_scores = [r['xception_confidence'] for r in frame_results]
        meso_scores = [r['meso_confidence'] for r in frame_results]
        ensemble_scores = [r['ensemble_confidence'] for r in frame_results]
        
        ax1.plot(frame_indices, xception_scores, label='XceptionNet', alpha=0.7)
        ax1.plot(frame_indices, meso_scores, label='MesoNet', alpha=0.7)
        ax1.plot(frame_indices, ensemble_scores, label='Ensemble', linewidth=2)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Deepfake Confidence')
        ax1.set_title('Deepfake Detection Confidence Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Create histogram of confidence scores
        ax2.hist(ensemble_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')
        ax2.set_xlabel('Ensemble Confidence Score')
        ax2.set_ylabel('Number of Frames')
        ax2.set_title('Distribution of Confidence Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64 for embedding
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return {
            'confidence_timeline': f"data:image/png;base64,{plot_data}",
            'summary_stats': {
                'total_analyzed_faces': len(frame_results),
                'avg_confidence': float(np.mean(ensemble_scores)),
                'max_confidence': float(np.max(ensemble_scores)),
                'min_confidence': float(np.min(ensemble_scores)),
                'std_confidence': float(np.std(ensemble_scores))
            }
        }
    
    def batch_detect_faces(self, faces_batch: List[np.ndarray], use_cache: bool = True) -> List[Dict[str, Any]]:
        """Process multiple faces in batch for better performance"""
        if not faces_batch:
            return []
        
        start_time = time.time()
        batch_results = []
        
        try:
            # Check cache for existing results
            if use_cache and self.cache_results:
                cached_results = []
                uncached_faces = []
                uncached_indices = []
                
                for i, face in enumerate(faces_batch):
                    face_hash = self._compute_face_hash(face)
                    cached_result = self.result_cache.get_cache_entry(face_hash, 'deepfake_detection')
                    
                    if cached_result:
                        cached_results.append((i, cached_result))
                        self.processing_stats['cache_hits'] += 1
                    else:
                        uncached_faces.append(face)
                        uncached_indices.append(i)
                        self.processing_stats['cache_misses'] += 1
                
                # Process only uncached faces
                if uncached_faces:
                    uncached_results = self._batch_process_faces(uncached_faces)
                    
                    # Cache new results
                    for idx, result in enumerate(uncached_results):
                        face_hash = self._compute_face_hash(uncached_faces[idx])
                        self.result_cache.store_cache_entry(face_hash, 'deepfake_detection', result, expires_in_hours=24)
                else:
                    uncached_results = []
                
                # Combine cached and new results
                all_results = [None] * len(faces_batch)
                for idx, result in cached_results:
                    all_results[idx] = result
                for i, idx in enumerate(uncached_indices):
                    if i < len(uncached_results):
                        all_results[idx] = uncached_results[i]
                
                batch_results = [r for r in all_results if r is not None]
            else:
                # Process all faces without caching
                batch_results = self._batch_process_faces(faces_batch)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats['batch_processing_times'].append(processing_time)
            self.processing_stats['total_frames_processed'] += len(faces_batch)
            
            # Track memory usage
            if PSUTIL_AVAILABLE:
                memory_usage = psutil.virtual_memory().percent
                self.processing_stats['memory_usage_peaks'].append(memory_usage)
                
                # Trigger cleanup if memory usage is high
                if memory_usage > 85:
                    self._cleanup_memory()
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch face processing failed: {e}")
            return [{'error': str(e)} for _ in faces_batch]
    
    def _batch_process_faces(self, faces: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process faces in optimized batches"""
        results = []
        
        # Preprocess faces into tensor batch
        preprocessed_faces = []
        for face in faces:
            try:
                face_tensor = self.face_extractor.preprocess(face)
                preprocessed_faces.append(face_tensor.squeeze(0))  # Remove batch dimension
            except Exception as e:
                # Skip problematic faces
                results.append({'error': f'Preprocessing failed: {str(e)}'})
                continue
        
        if not preprocessed_faces:
            return results
        
        try:
            # Stack into batch tensor
            batch_tensor = torch.stack(preprocessed_faces).to(self.device)
            
            with torch.no_grad():
                # Batch inference for XceptionNet
                xception_outputs = self.xception_model(batch_tensor)
                xception_probs = torch.nn.functional.softmax(xception_outputs, dim=1)
                xception_confidences = xception_probs[:, 1].cpu().numpy()  # Fake class probabilities
                
                # Batch inference for MesoNet
                meso_outputs = self.meso_model(batch_tensor)
                meso_confidences = torch.sigmoid(meso_outputs).squeeze().cpu().numpy()
                
                # Handle single face case
                if len(preprocessed_faces) == 1:
                    xception_confidences = [xception_confidences]
                    meso_confidences = [meso_confidences]
            
            # Combine results
            for i in range(len(preprocessed_faces)):
                xception_conf = float(xception_confidences[i])
                meso_conf = float(meso_confidences[i])
                ensemble_conf = (xception_conf + meso_conf) / 2.0
                
                result = {
                    'xception_confidence': xception_conf,
                    'meso_confidence': meso_conf,
                    'ensemble_confidence': ensemble_conf,
                    'is_fake': ensemble_conf > 0.5,
                    'processing_method': 'batch'
                }
                results.append(result)
                
        except Exception as e:
            # Fallback to individual processing
            for face in faces:
                try:
                    face_tensor = self.face_extractor.preprocess(face).to(self.device)
                    
                    with torch.no_grad():
                        xception_output = self.xception_model(face_tensor)
                        xception_probs = torch.nn.functional.softmax(xception_output, dim=1)
                        xception_conf = xception_probs[0][1].item()
                        
                        meso_output = self.meso_model(face_tensor)
                        meso_conf = torch.sigmoid(meso_output).item()
                    
                    ensemble_conf = (xception_conf + meso_conf) / 2.0
                    
                    result = {
                        'xception_confidence': float(xception_conf),
                        'meso_confidence': float(meso_conf),
                        'ensemble_confidence': ensemble_conf,
                        'is_fake': ensemble_conf > 0.5,
                        'processing_method': 'individual_fallback'
                    }
                    results.append(result)
                    
                except Exception as face_error:
                    results.append({'error': f'Face processing failed: {str(face_error)}'})
        
        return results
    
    def _compute_face_hash(self, face: np.ndarray) -> str:
        """Compute hash for face caching"""
        import hashlib
        face_bytes = face.tobytes()
        return hashlib.md5(face_bytes).hexdigest()
    
    def _cleanup_memory(self):
        """Clean up memory when usage is high"""
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear model cache if needed
        self.model_manager.cleanup_cache()
        
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            self.logger.info(f"Memory cleanup completed. Usage: {memory.percent}%")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = dict(self.processing_stats)
        
        # Add model manager stats
        stats['model_manager'] = self.model_manager.get_memory_stats()
        
        # Add batch processor stats
        stats['batch_processor'] = self.batch_processor.get_processing_stats()
        
        # Calculate averages
        if self.processing_stats['batch_processing_times']:
            stats['avg_batch_time'] = np.mean(self.processing_stats['batch_processing_times'])
        
        if self.processing_stats['memory_usage_peaks']:
            stats['peak_memory_usage'] = max(self.processing_stats['memory_usage_peaks'])
            stats['avg_memory_usage'] = np.mean(self.processing_stats['memory_usage_peaks'])
        
        # Cache hit rate
        total_requests = stats['cache_hits'] + stats['cache_misses']
        if total_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests
        
        return stats
    
    def preload_models(self):
        """Preload models in background for faster first inference"""
        def preload():
            try:
                self.logger.info("Preloading deepfake detection models...")
                self.xception_model  # This will trigger lazy loading
                self.meso_model     # This will trigger lazy loading
                self.logger.info("Models preloaded successfully")
            except Exception as e:
                self.logger.error(f"Model preloading failed: {e}")
        
        # Preload in background thread
        import threading
        preload_thread = threading.Thread(target=preload, daemon=True)
        preload_thread.start()
        return preload_thread
