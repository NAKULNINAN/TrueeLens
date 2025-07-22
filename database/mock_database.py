"""
Mock database for simulating media forensics operations.
Provides functionality for storing known media hashes, similarity comparisons, and result caching.
"""

import sqlite3
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import random
import time

from .schema_manager import SchemaManager
from .hash_storage import HashStorage
from .similarity_cache import SimilarityCache
from .file_manager import FileManager


class MockMediaDatabase:
    """
    Comprehensive mock database for a media forensics application.
    Provides high-level operations for hash storage, similarity comparisons, result caching, and file management.
    """
    
    def __init__(self, db_path: str = "mock_media_forensics.db"):
        """Initialize mock media database components."""
        self.db_path = db_path
        self.schema_manager = SchemaManager(db_path)
        self.hash_storage = HashStorage(db_path)
        self.similarity_cache = SimilarityCache(db_path)
        self.file_manager = FileManager(db_path)
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize database schema and prepare components."""
        print("Initializing mock media forensics database...")
        success = self.schema_manager.initialize_database()
        if success:
            self._initialized = True
            print("Database initialized successfully.")
            return True
        else:
            print("Database initialization failed.")
            return False
    
    def is_initialized(self) -> bool:
        """Check if database is initialized."""
        return self._initialized
    
    # High-level operations
    
    def add_media_file(self, file_path: str, compute_hashes: bool = True, 
                      hash_types: List[str] = None) -> Optional[int]:
        """Add a media file with optional hash computation."""
        if not self._initialized:
            print("Database not initialized. Call initialize() first.")
            return None
        
        # Register file
        file_id = self.file_manager.register_file(file_path)
        if not file_id:
            return None
        
        # Compute and store hashes if requested
        if compute_hashes:
            self.hash_storage.compute_and_store_hashes(file_id, file_path, hash_types)
        
        return file_id
    
    def find_duplicates(self, file_id: int, hash_type: str = 'sha256', 
                       similarity_threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Find duplicate or similar files."""
        if not self._initialized:
            return []
        
        file_hash = self.hash_storage.get_hash(file_id, hash_type)
        if not file_hash:
            return []
        
        if hash_type in ['perceptual', 'dhash', 'phash', 'ahash', 'whash']:
            # Use similarity search for perceptual hashes
            threshold = int((1.0 - similarity_threshold) * 64)  # Convert to Hamming distance
            similar_files = self.hash_storage.find_similar_hashes(file_hash, hash_type, threshold)
            
            duplicates = []
            for similar_file_id, similar_hash, distance in similar_files:
                similarity_score = 1.0 - (distance / 64.0)
                file_info = self.file_manager.get_file_info(similar_file_id)
                if file_info:
                    duplicates.append({
                        'file_id': similar_file_id,
                        'file_path': file_info['file_path'],
                        'similarity_score': similarity_score,
                        'hash_distance': distance
                    })
            return duplicates
        else:
            # Exact match for cryptographic hashes
            matching_files = self.hash_storage.find_by_hash(file_hash, hash_type)
            duplicates = []
            
            for match_id in matching_files:
                if match_id != file_id:  # Exclude the original file
                    file_info = self.file_manager.get_file_info(match_id)
                    if file_info:
                        duplicates.append({
                            'file_id': match_id,
                            'file_path': file_info['file_path'],
                            'similarity_score': 1.0,
                            'hash_distance': 0
                        })
            return duplicates
    
    def compare_files(self, file1_id: int, file2_id: int, 
                     comparison_type: str = 'perceptual', 
                     algorithm: str = 'phash') -> Dict[str, Any]:
        """Compare two files and cache the result."""
        if not self._initialized:
            return {'error': 'Database not initialized'}
        
        # Check if comparison already cached
        cached_result = self.similarity_cache.get_similarity(
            file1_id, file2_id, comparison_type, algorithm
        )
        
        if cached_result:
            cached_result['cached'] = True
            return cached_result
        
        # Simulate comparison (in real implementation, this would call actual comparison algorithms)
        start_time = time.time()
        similarity_score = self._simulate_comparison(file1_id, file2_id, comparison_type)
        processing_time = time.time() - start_time
        
        # Store result in cache
        metadata = {
            'processing_time': processing_time,
            'algorithm_parameters': {'hash_size': 8, 'highfreq_factor': 4}
        }
        
        self.similarity_cache.store_similarity(
            file1_id, file2_id, comparison_type, similarity_score, 
            algorithm, '1.0', metadata
        )
        
        return {
            'similarity_score': similarity_score,
            'algorithm': algorithm,
            'algorithm_version': '1.0',
            'metadata': metadata,
            'cached': False,
            'processing_time': processing_time
        }
    
    def detect_ai_generated(self, file_id: int, model_name: str = 'mock_detector') -> Dict[str, Any]:
        """Simulate AI-generated content detection."""
        if not self._initialized:
            return {'error': 'Database not initialized'}
        
        # Check cache first
        cached_result = self.similarity_cache.get_detection_result(file_id, 'ai_generated')
        if cached_result:
            cached_result['cached'] = True
            return cached_result
        
        # Simulate detection
        start_time = time.time()
        confidence_score = random.uniform(0.1, 0.9)  # Random confidence for simulation
        processing_time = time.time() - start_time
        
        result_data = {
            'predicted_class': 'ai_generated' if confidence_score > 0.5 else 'real',
            'processing_time': processing_time,
            'features_analyzed': ['texture', 'artifacts', 'frequency_domain']
        }
        
        # Store result
        self.similarity_cache.store_detection_result(
            file_id, 'ai_generated', confidence_score, result_data,
            model_name, '1.0', processing_time
        )
        
        return {
            'confidence_score': confidence_score,
            'result_data': result_data,
            'model_name': model_name,
            'model_version': '1.0',
            'processing_time': processing_time,
            'cached': False
        }
    
    def detect_deepfake(self, file_id: int, model_name: str = 'mock_deepfake_detector') -> Dict[str, Any]:
        """Simulate deepfake detection."""
        if not self._initialized:
            return {'error': 'Database not initialized'}
        
        # Check cache first
        cached_result = self.similarity_cache.get_detection_result(file_id, 'deepfake')
        if cached_result:
            cached_result['cached'] = True
            return cached_result
        
        # Simulate detection
        start_time = time.time()
        confidence_score = random.uniform(0.05, 0.95)  # Random confidence for simulation
        processing_time = time.time() - start_time + 0.5  # Deepfake detection takes longer
        
        result_data = {
            'predicted_class': 'deepfake' if confidence_score > 0.5 else 'real',
            'processing_time': processing_time,
            'face_regions_analyzed': random.randint(1, 5),
            'consistency_score': random.uniform(0.3, 0.9)
        }
        
        # Store result
        self.similarity_cache.store_detection_result(
            file_id, 'deepfake', confidence_score, result_data,
            model_name, '1.0', processing_time
        )
        
        return {
            'confidence_score': confidence_score,
            'result_data': result_data,
            'model_name': model_name,
            'model_version': '1.0',
            'processing_time': processing_time,
            'cached': False
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        if not self._initialized:
            return {'error': 'Database not initialized'}
        
        stats = {
            'database_info': {
                'schema_version': self.schema_manager.get_schema_version(),
                'database_path': self.db_path,
                'tables': self.schema_manager.get_table_stats()
            },
            'files': self.file_manager.get_file_statistics(),
            'hashes': self.hash_storage.get_hash_statistics(),
            'cache': self.similarity_cache.get_cache_statistics()
        }
        
        return stats
    
    def cleanup_database(self) -> Dict[str, int]:
        """Perform database cleanup operations."""
        if not self._initialized:
            return {'error': 'Database not initialized'}
        
        cleanup_stats = {
            'missing_files_cleaned': self.file_manager.cleanup_missing_files(),
            'orphaned_hashes_removed': self.hash_storage.cleanup_orphaned_hashes(),
            'expired_cache_entries_removed': self.similarity_cache.cleanup_expired_cache()
        }
        
        return cleanup_stats
    
    def scan_directory(self, directory_path: str, recursive: bool = True, 
                      compute_hashes: bool = True) -> Dict[str, Any]:
        """Scan directory and add all media files."""
        if not self._initialized:
            return {'error': 'Database not initialized'}
        
        # Scan and register files
        scan_results = self.file_manager.scan_directory(directory_path, recursive)
        
        if compute_hashes and scan_results.get('success', 0) > 0:
            # Get recently added files and compute hashes
            recent_files = self.file_manager.list_files(limit=scan_results['success'])
            
            hash_results = self.hash_storage.batch_compute_hashes(
                recent_files, ['sha256', 'perceptual']
            )
            
            scan_results['hashes_computed'] = len(hash_results)
        
        return scan_results
    
    def create_processing_job(self, job_type: str, file_id: int = None, 
                             parameters: Dict[str, Any] = None) -> Optional[int]:
        """Create a processing job (for batch operations)."""
        if not self._initialized:
            return None
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """INSERT INTO processing_jobs (job_type, file_id, parameters)
                       VALUES (?, ?, ?)""",
                    (job_type, file_id, json.dumps(parameters) if parameters else None)
                )
                return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error creating processing job: {e}")
            return None
    
    def _simulate_comparison(self, file1_id: int, file2_id: int, comparison_type: str) -> float:
        """Simulate file comparison with mock similarity score."""
        # In a real implementation, this would perform actual comparison
        # For simulation, generate a reasonable similarity score
        
        if comparison_type == 'identical':
            return 1.0 if file1_id == file2_id else 0.0
        
        # Generate pseudo-random but consistent similarity based on file IDs
        seed = abs(file1_id - file2_id)
        random.seed(seed)
        
        if seed == 0:  # Same file
            return 1.0
        elif seed <= 2:  # Very similar files
            return random.uniform(0.8, 0.95)
        elif seed <= 5:  # Somewhat similar files
            return random.uniform(0.4, 0.7)
        else:  # Different files
            return random.uniform(0.0, 0.3)
    
    def export_data(self, format_type: str = 'json') -> str:
        """Export database data for backup or analysis."""
        if not self._initialized:
            return json.dumps({'error': 'Database not initialized'})
        
        if format_type == 'json':
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'statistics': self.get_system_statistics(),
                'schema_version': self.schema_manager.get_schema_version()
            }
            return json.dumps(export_data, indent=2)
        else:
            return self.schema_manager.backup_schema()
    
    def validate_integrity(self) -> Dict[str, Any]:
        """Validate database integrity."""
        if not self._initialized:
            return {'error': 'Database not initialized'}
        
        return self.schema_manager.validate_schema()

