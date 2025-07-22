import os
import requests
import torch
import hashlib
import logging
import gc
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from functools import lru_cache, wraps
from collections import OrderedDict
from contextlib import contextmanager

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class LRUModelCache:
    """LRU Cache for loaded models with memory management"""
    
    def __init__(self, max_models: int = 3, max_memory_mb: int = 4096):
        self.max_models = max_models
        self.max_memory_mb = max_memory_mb
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.model_memory = {}
        
    def get(self, key: str):
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                model = self.cache.pop(key)
                self.cache[key] = model
                return model
            return None
    
    def put(self, key: str, model, memory_mb: float = 0):
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                self.cache.pop(key)
            
            # Check memory constraints
            current_memory = sum(self.model_memory.values())
            while (len(self.cache) >= self.max_models or 
                   current_memory + memory_mb > self.max_memory_mb) and self.cache:
                oldest_key = next(iter(self.cache))
                self._evict(oldest_key)
                current_memory = sum(self.model_memory.values())
            
            self.cache[key] = model
            self.model_memory[key] = memory_mb
    
    def _evict(self, key: str):
        """Evict a model from cache and free memory"""
        if key in self.cache:
            model = self.cache.pop(key)
            memory = self.model_memory.pop(key, 0)
            
            # Clear CUDA cache if model was on GPU
            if hasattr(model, 'device') and model.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            del model
            gc.collect()
    
    def clear(self):
        with self.lock:
            keys = list(self.cache.keys())
            for key in keys:
                self._evict(key)
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'cached_models': len(self.cache),
                'max_models': self.max_models,
                'total_memory_mb': sum(self.model_memory.values()),
                'max_memory_mb': self.max_memory_mb,
                'model_keys': list(self.cache.keys())
            }

class ModelManager:
    def __init__(self, cache_dir='model_cache', log_level=logging.INFO, 
                 max_cached_models=3, max_memory_mb=4096):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
        
        # Model cache for lazy loading
        self.model_cache = LRUModelCache(max_cached_models, max_memory_mb)
        self.model_loaders = {}
        self.model_configs = {}
        
        # Performance tracking
        self.load_times = {}
        self.access_counts = {}
        
        # Memory monitoring
        self.memory_threshold = 0.85  # 85% memory usage threshold

    def download_model(self, url, model_name, version='latest', checksum=None):
        model_dir = self.cache_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'model.pth'
        
        # If model is already cached, return the path
        if model_path.exists():
            self.logger.info(f"Model {model_name} version {version} is cached.")
            return model_path

        self.logger.info(f"Downloading {model_name}...")

        try:
            response = requests.get(url, timeout=10)  # Timeout after 10 seconds
            response.raise_for_status()  # Raise error for bad responses

            # Save model
            with open(model_path, 'wb') as f:
                f.write(response.content)

            self.logger.info(f"Downloaded {model_name} version {version}.")

            # Verify checksum
            if checksum:
                if not self.verify_checksum(model_path, checksum):
                    self.logger.error("Checksum verification failed.")
                    raise ValueError("Checksum verification failed.")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download model: {e}")
            raise

        return model_path

    def verify_checksum(self, file_path, checksum):
        """Verify file integrity using a checksum (e.g., SHA256)"""
        self.logger.debug(f"Verifying checksum for {file_path}...")
        file_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)

        calculated_checksum = file_hash.hexdigest()
        is_valid = calculated_checksum == checksum
        self.logger.debug(f"Calculated checksum: {calculated_checksum}, Expected checksum: {checksum}")
        return is_valid

    def lazy_load_model(self, model_key: str, model_loader: Callable[[], torch.nn.Module], 
                     memory_mb_estimate: float = 500) -> torch.nn.Module:
        """Load models with LRU cache to manage memory"""
        start_time = time.time()
        
        # Check cache first
        cached_model = self.model_cache.get(model_key)
        if cached_model is not None:
            self.access_counts[model_key] = self.access_counts.get(model_key, 0) + 1
            self.logger.debug(f"Model {model_key} loaded from cache.")
            return cached_model
        
        # Check memory before loading
        if PSUTIL_AVAILABLE:
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > self.memory_threshold:
                self.logger.warning(f"High memory usage ({memory_percent:.1%}). Clearing cache.")
                self.model_cache.clear()
                gc.collect()
        
        # Load model
        self.logger.info(f"Loading model {model_key}...")
        model = model_loader()
        
        # Track performance
        load_time = time.time() - start_time
        self.load_times[model_key] = load_time
        self.access_counts[model_key] = self.access_counts.get(model_key, 0) + 1
        
        # Cache the model
        self.model_cache.put(model_key, model, memory_mb_estimate)
        self.logger.info(f"Model {model_key} loaded and cached in {load_time:.2f}s.")
        return model
    
    def register_model_loader(self, model_key: str, loader_func: Callable, config: Dict[str, Any] = None):
        """Register a model loader for lazy loading"""
        self.model_loaders[model_key] = loader_func
        self.model_configs[model_key] = config or {}
        self.logger.debug(f"Registered loader for model {model_key}")
    
    def get_model(self, model_key: str, **kwargs) -> torch.nn.Module:
        """Get a model using registered loader"""
        if model_key not in self.model_loaders:
            raise ValueError(f"No loader registered for model {model_key}")
        
        # Merge config with runtime kwargs
        config = {**self.model_configs[model_key], **kwargs}
        
        # Create loader with config
        loader = lambda: self.model_loaders[model_key](**config)
        
        # Estimate memory usage
        memory_estimate = config.get('memory_mb_estimate', 500)
        
        return self.lazy_load_model(model_key, loader, memory_estimate)
    
    @contextmanager
    def temporary_model(self, model_key: str, **kwargs):
        """Context manager for temporary model usage"""
        model = self.get_model(model_key, **kwargs)
        try:
            yield model
        finally:
            # Optional: remove from cache if it's a temporary model
            if kwargs.get('temporary', False):
                self.model_cache._evict(model_key)
    
    def preload_model(self, model_key: str, **kwargs):
        """Preload a model in background"""
        def _preload():
            try:
                self.get_model(model_key, **kwargs)
                self.logger.info(f"Preloaded model {model_key}")
            except Exception as e:
                self.logger.error(f"Failed to preload model {model_key}: {e}")
        
        thread = threading.Thread(target=_preload, daemon=True)
        thread.start()
        return thread
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory and performance statistics"""
        stats = {
            'model_cache': self.model_cache.get_stats(),
            'load_times': dict(self.load_times),
            'access_counts': dict(self.access_counts),
            'registered_models': list(self.model_loaders.keys())
        }
        
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            stats['system_memory'] = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_percent': memory.percent,
                'free_gb': memory.free / (1024**3)
            }
        
        return stats
    
    def cleanup_cache(self, force: bool = False):
        """Cleanup model cache based on memory usage"""
        if PSUTIL_AVAILABLE:
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > self.memory_threshold or force:
                self.logger.info(f"Cleaning up model cache (memory usage: {memory_percent:.1%})")
                self.model_cache.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def load_model(self, model_loader, model_name, version='latest', offline_mode=False):
        """Legacy method for backward compatibility"""
        model_path = self.cache_dir / model_name / version / 'model.pth'
        
        if not model_path.exists():
            if offline_mode:
                self.logger.warning(f"Model {model_name} version {version} not found and offline mode is enabled.")
                raise FileNotFoundError("Model not available locally, and offline mode is enabled.")
            else:
                self.logger.info(f"Model {model_name} version {version} not found. Attempting download...")
                raise FileNotFoundError("Model not available locally.")
        
        # Use lazy loading for legacy calls
        model_key = f"{model_name}_{version}"
        loader = lambda: model_loader(model_path=model_path)
        return self.lazy_load_model(model_key, loader)

