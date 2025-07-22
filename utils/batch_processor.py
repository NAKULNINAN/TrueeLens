import asyncio
import concurrent.futures
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
import queue
import time
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import gc

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

@dataclass
class BatchTask:
    """Represents a single batch processing task"""
    task_id: str
    data: Any
    task_type: str
    priority: int = 0  # Higher numbers = higher priority
    metadata: Dict[str, Any] = None

@dataclass
class BatchResult:
    """Represents the result of a batch processing task"""
    task_id: str
    result: Any
    success: bool
    error: str = None
    processing_time: float = 0.0

class MemoryAwareBatchProcessor:
    """Batch processor with memory management and optimization"""
    
    def __init__(self, max_workers: int = None, max_memory_mb: int = 2048, 
                 batch_size: int = 8, use_gpu: bool = True):
        self.max_workers = max_workers or min(mp.cpu_count(), 4)
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        
        self.logger = logging.getLogger(__name__)
        
        # Task queue with priority support
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        
        # Processing statistics
        self.stats = {
            'tasks_processed': 0,
            'total_processing_time': 0.0,
            'average_batch_time': 0.0,
            'memory_peaks': [],
            'gpu_utilization': []
        }
        
        # GPU settings
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        if self.use_gpu and torch.cuda.is_available():
            # Set memory fraction for GPU
            torch.cuda.set_per_process_memory_fraction(0.7)
    
    def add_task(self, task: BatchTask):
        """Add a task to the processing queue"""
        # Use negative priority for max-heap behavior (higher priority first)
        self.task_queue.put((-task.priority, time.time(), task))
    
    def add_tasks(self, tasks: List[BatchTask]):
        """Add multiple tasks to the queue"""
        for task in tasks:
            self.add_task(task)
    
    def process_batch_images(self, image_batch: List[np.ndarray], 
                           processor_func: Callable, **kwargs) -> List[Any]:
        """Process a batch of images efficiently"""
        if not image_batch:
            return []
        
        start_time = time.time()
        results = []
        
        try:
            # Check memory before processing
            if PSUTIL_AVAILABLE:
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > 80:
                    self.logger.warning(f"High memory usage: {memory_usage}%")
                    gc.collect()
            
            # Convert to tensor batch if using GPU
            if self.use_gpu and torch.cuda.is_available():
                # Stack images into batch tensor
                try:
                    batch_tensor = torch.stack([
                        torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0 
                        for img in image_batch
                    ]).to(self.device)
                    
                    # Process batch on GPU
                    with torch.no_grad():
                        batch_results = processor_func(batch_tensor, **kwargs)
                    
                    # Convert results back to CPU if necessary
                    if isinstance(batch_results, torch.Tensor):
                        batch_results = batch_results.cpu().numpy()
                    
                    results = [batch_results[i] for i in range(len(image_batch))]
                    
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    self.logger.warning(f"GPU processing failed: {e}. Falling back to CPU.")
                    # Fall back to CPU processing
                    results = [processor_func(img, **kwargs) for img in image_batch]
            else:
                # CPU processing
                results = [processor_func(img, **kwargs) for img in image_batch]
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            self.stats['tasks_processed'] += len(image_batch)
            
            if PSUTIL_AVAILABLE:
                memory_usage = psutil.virtual_memory().percent
                self.stats['memory_peaks'].append(memory_usage)
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            results = [None] * len(image_batch)
        
        return results
    
    def process_video_frames_batch(self, video_paths: List[str], 
                                 frame_extractor: Callable, 
                                 max_frames_per_video: int = 30) -> List[List[np.ndarray]]:
        """Extract frames from multiple videos in batch"""
        all_frames = []
        
        # Use multiprocessing for video frame extraction
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_video = {
                executor.submit(self._extract_video_frames, video_path, frame_extractor, max_frames_per_video): video_path
                for video_path in video_paths
            }
            
            for future in concurrent.futures.as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    frames = future.result()
                    all_frames.append(frames)
                except Exception as e:
                    self.logger.error(f"Frame extraction failed for {video_path}: {e}")
                    all_frames.append([])
        
        return all_frames
    
    def _extract_video_frames(self, video_path: str, frame_extractor: Callable, 
                            max_frames: int) -> List[np.ndarray]:
        """Extract frames from a single video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            
            # Sample frames evenly
            if frame_count <= max_frames:
                frame_indices = list(range(frame_count))
            else:
                frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    processed_frame = frame_extractor(frame)
                    if processed_frame is not None:
                        frames.append(processed_frame)
            
            cap.release()
            return frames
            
        except Exception as e:
            self.logger.error(f"Error extracting frames from {video_path}: {e}")
            return []
    
    def process_with_threading(self, tasks: List[BatchTask], 
                             processor_func: Callable) -> List[BatchResult]:
        """Process tasks using ThreadPoolExecutor"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._process_single_task, task, processor_func): task
                for task in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    error_result = BatchResult(
                        task_id=task.task_id,
                        result=None,
                        success=False,
                        error=str(e)
                    )
                    results.append(error_result)
        
        return results
    
    def _process_single_task(self, task: BatchTask, processor_func: Callable) -> BatchResult:
        """Process a single task"""
        start_time = time.time()
        
        try:
            result = processor_func(task.data, **(task.metadata or {}))
            processing_time = time.time() - start_time
            
            return BatchResult(
                task_id=task.task_id,
                result=result,
                success=True,
                processing_time=processing_time
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return BatchResult(
                task_id=task.task_id,
                result=None,
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def adaptive_batch_processing(self, tasks: List[BatchTask], 
                                processor_func: Callable) -> List[BatchResult]:
        """Adaptively process tasks based on system resources"""
        if not tasks:
            return []
        
        # Check system resources
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Adapt batch size based on resources
            if memory_percent > 75:
                effective_batch_size = max(1, self.batch_size // 2)
            elif cpu_percent > 80:
                effective_batch_size = max(1, self.batch_size // 2)
            else:
                effective_batch_size = self.batch_size
        else:
            effective_batch_size = self.batch_size
        
        # Process tasks in adaptive batches
        results = []
        for i in range(0, len(tasks), effective_batch_size):
            batch = tasks[i:i + effective_batch_size]
            batch_results = self.process_with_threading(batch, processor_func)
            results.extend(batch_results)
            
            # Optional: brief pause between batches for system recovery
            if len(batch) == effective_batch_size:
                time.sleep(0.1)
        
        return results
    
    def parallel_hash_computation(self, file_paths: List[str], 
                                hash_func: Callable) -> List[Dict[str, str]]:
        """Compute hashes for multiple files in parallel"""
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            hash_results = list(executor.map(hash_func, file_paths))
        
        return hash_results
    
    def batch_similarity_comparison(self, hash_list1: List[str], 
                                  hash_list2: List[str],
                                  similarity_func: Callable,
                                  threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """Compare hashes in batches for similarity detection"""
        similarities = []
        
        # Create comparison tasks
        comparison_tasks = []
        task_id = 0
        
        for i, hash1 in enumerate(hash_list1):
            for j, hash2 in enumerate(hash_list2):
                if i != j:  # Don't compare with self
                    task = BatchTask(
                        task_id=f"compare_{task_id}",
                        data=(hash1, hash2, i, j),
                        task_type="similarity_comparison"
                    )
                    comparison_tasks.append(task)
                    task_id += 1
        
        # Process comparisons
        def compare_hashes(data):
            hash1, hash2, i, j = data
            similarity = similarity_func(hash1, hash2)
            return (i, j, similarity) if similarity >= threshold else None
        
        results = self.adaptive_batch_processing(comparison_tasks, compare_hashes)
        
        # Filter successful comparisons above threshold
        for result in results:
            if result.success and result.result is not None:
                similarities.append(result.result)
        
        return similarities
    
    def cleanup_memory(self):
        """Clean up memory and GPU cache"""
        gc.collect()
        
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Log memory status
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            self.logger.info(f"Memory cleanup - Usage: {memory.percent}%, Available: {memory.available / (1024**3):.2f}GB")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = dict(self.stats)
        
        if self.stats['tasks_processed'] > 0:
            stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['tasks_processed']
            )
        
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            stats['current_memory_usage'] = {
                'percent': memory.percent,
                'available_gb': memory.available / (1024**3),
                'used_gb': (memory.total - memory.available) / (1024**3)
            }
        
        if self.use_gpu and torch.cuda.is_available():
            stats['gpu_memory'] = {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'cached_gb': torch.cuda.memory_reserved() / (1024**3)
            }
        
        return stats

class ImageBatchProcessor:
    """Specialized batch processor for images"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.batch_processor = MemoryAwareBatchProcessor()
    
    def preprocess_images_batch(self, images: List[Union[np.ndarray, str]], 
                              normalize: bool = True) -> np.ndarray:
        """Preprocess a batch of images efficiently"""
        processed_images = []
        
        for img in images:
            if isinstance(img, str):
                # Load from file path
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img_resized = cv2.resize(img, self.target_size)
            
            # Normalize
            if normalize:
                img_resized = img_resized.astype(np.float32) / 255.0
                # ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_resized = (img_resized - mean) / std
            
            processed_images.append(img_resized)
        
        return np.array(processed_images)
    
    def extract_features_batch(self, images: List[np.ndarray], 
                             feature_extractor: Callable) -> np.ndarray:
        """Extract features from a batch of images"""
        # Convert to tensor batch
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        preprocessed = self.preprocess_images_batch(images)
        
        # Convert to torch tensor
        batch_tensor = torch.from_numpy(preprocessed).permute(0, 3, 1, 2).to(device)
        
        with torch.no_grad():
            features = feature_extractor(batch_tensor)
        
        return features.cpu().numpy()
