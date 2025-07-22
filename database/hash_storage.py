"""
Hash storage component for managing media file hashes.
Handles storage and retrieval of various hash types including perceptual hashes.
"""

import sqlite3
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import imagehash
from PIL import Image
import os


class HashStorage:
    """Manages storage and retrieval of media file hashes."""
    
    SUPPORTED_HASH_TYPES = [
        'md5', 'sha1', 'sha256', 'sha512',
        'perceptual', 'dhash', 'phash', 'ahash', 'whash'
    ]
    
    def __init__(self, db_path: str):
        """Initialize hash storage with database path."""
        self.db_path = db_path
    
    def store_hash(self, file_id: int, hash_type: str, hash_value: str, 
                   algorithm_version: str = None) -> bool:
        """Store a hash for a file."""
        if hash_type not in self.SUPPORTED_HASH_TYPES:
            raise ValueError(f"Unsupported hash type: {hash_type}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO media_hashes 
                       (file_id, hash_type, hash_value, algorithm_version) 
                       VALUES (?, ?, ?, ?)""",
                    (file_id, hash_type, hash_value, algorithm_version)
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error storing hash: {e}")
            return False
    
    def get_hash(self, file_id: int, hash_type: str) -> Optional[str]:
        """Get a specific hash for a file."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT hash_value FROM media_hashes WHERE file_id = ? AND hash_type = ?",
                    (file_id, hash_type)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except sqlite3.Error as e:
            print(f"Error retrieving hash: {e}")
            return None
    
    def get_all_hashes(self, file_id: int) -> Dict[str, str]:
        """Get all hashes for a file."""
        hashes = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT hash_type, hash_value FROM media_hashes WHERE file_id = ?",
                    (file_id,)
                )
                for hash_type, hash_value in cursor:
                    hashes[hash_type] = hash_value
        except sqlite3.Error as e:
            print(f"Error retrieving hashes: {e}")
        
        return hashes
    
    def find_by_hash(self, hash_value: str, hash_type: str = None) -> List[int]:
        """Find files with matching hash value."""
        file_ids = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                if hash_type:
                    cursor = conn.execute(
                        "SELECT file_id FROM media_hashes WHERE hash_value = ? AND hash_type = ?",
                        (hash_value, hash_type)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT file_id FROM media_hashes WHERE hash_value = ?",
                        (hash_value,)
                    )
                file_ids = [row[0] for row in cursor]
        except sqlite3.Error as e:
            print(f"Error finding files by hash: {e}")
        
        return file_ids
    
    def compute_file_hash(self, file_path: str, hash_type: str) -> Optional[str]:
        """Compute hash for a file."""
        if not os.path.exists(file_path):
            return None
        
        try:
            if hash_type in ['md5', 'sha1', 'sha256', 'sha512']:
                return self._compute_cryptographic_hash(file_path, hash_type)
            elif hash_type in ['perceptual', 'dhash', 'phash', 'ahash', 'whash']:
                return self._compute_perceptual_hash(file_path, hash_type)
            else:
                raise ValueError(f"Unsupported hash type: {hash_type}")
        except Exception as e:
            print(f"Error computing hash for {file_path}: {e}")
            return None
    
    def _compute_cryptographic_hash(self, file_path: str, hash_type: str) -> str:
        """Compute cryptographic hash (MD5, SHA1, SHA256, etc.)."""
        hash_func = getattr(hashlib, hash_type)()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def _compute_perceptual_hash(self, file_path: str, hash_type: str) -> str:
        """Compute perceptual hash for images."""
        try:
            with Image.open(file_path) as img:
                if hash_type == 'perceptual' or hash_type == 'phash':
                    return str(imagehash.phash(img))
                elif hash_type == 'dhash':
                    return str(imagehash.dhash(img))
                elif hash_type == 'ahash':
                    return str(imagehash.average_hash(img))
                elif hash_type == 'whash':
                    return str(imagehash.whash(img))
                else:
                    raise ValueError(f"Unknown perceptual hash type: {hash_type}")
        except Exception as e:
            print(f"Error computing perceptual hash: {e}")
            return None
    
    def compute_and_store_hashes(self, file_id: int, file_path: str, 
                                hash_types: List[str] = None) -> Dict[str, str]:
        """Compute and store multiple hash types for a file."""
        if hash_types is None:
            # Default hash types based on file type
            if self._is_image_file(file_path):
                hash_types = ['sha256', 'perceptual', 'dhash']
            else:
                hash_types = ['sha256']
        
        computed_hashes = {}
        
        for hash_type in hash_types:
            hash_value = self.compute_file_hash(file_path, hash_type)
            if hash_value:
                if self.store_hash(file_id, hash_type, hash_value):
                    computed_hashes[hash_type] = hash_value
        
        return computed_hashes
    
    def _is_image_file(self, file_path: str) -> bool:
        """Check if file is an image based on extension."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
        return os.path.splitext(file_path)[1].lower() in image_extensions
    
    def find_similar_hashes(self, hash_value: str, hash_type: str, 
                           threshold: int = 5) -> List[Tuple[int, str, int]]:
        """Find similar perceptual hashes within a threshold."""
        if hash_type not in ['perceptual', 'dhash', 'phash', 'ahash', 'whash']:
            raise ValueError("Similarity search only supported for perceptual hashes")
        
        similar_files = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT file_id, hash_value FROM media_hashes WHERE hash_type = ?",
                    (hash_type,)
                )
                
                target_hash = imagehash.hex_to_hash(hash_value)
                
                for file_id, stored_hash in cursor:
                    if stored_hash == hash_value:
                        continue  # Skip exact match
                    
                    try:
                        stored_hash_obj = imagehash.hex_to_hash(stored_hash)
                        distance = target_hash - stored_hash_obj
                        
                        if distance <= threshold:
                            similar_files.append((file_id, stored_hash, distance))
                    except Exception as e:
                        print(f"Error comparing hashes: {e}")
                        continue
        
        except sqlite3.Error as e:
            print(f"Error finding similar hashes: {e}")
        
        # Sort by distance (most similar first)
        similar_files.sort(key=lambda x: x[2])
        return similar_files
    
    def get_hash_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored hashes."""
        stats = {
            'total_hashes': 0,
            'hash_types': {},
            'files_with_hashes': 0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total hash count
                cursor = conn.execute("SELECT COUNT(*) FROM media_hashes")
                stats['total_hashes'] = cursor.fetchone()[0]
                
                # Count by hash type
                cursor = conn.execute(
                    "SELECT hash_type, COUNT(*) FROM media_hashes GROUP BY hash_type"
                )
                for hash_type, count in cursor:
                    stats['hash_types'][hash_type] = count
                
                # Unique files with hashes
                cursor = conn.execute("SELECT COUNT(DISTINCT file_id) FROM media_hashes")
                stats['files_with_hashes'] = cursor.fetchone()[0]
        
        except sqlite3.Error as e:
            print(f"Error getting hash statistics: {e}")
        
        return stats
    
    def cleanup_orphaned_hashes(self) -> int:
        """Remove hashes for files that no longer exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """DELETE FROM media_hashes 
                       WHERE file_id NOT IN (SELECT id FROM media_files)"""
                )
                return cursor.rowcount
        except sqlite3.Error as e:
            print(f"Error cleaning up orphaned hashes: {e}")
            return 0
    
    def batch_compute_hashes(self, file_records: List[Dict[str, Any]], 
                            hash_types: List[str] = None) -> Dict[int, Dict[str, str]]:
        """Compute hashes for multiple files in batch."""
        if hash_types is None:
            hash_types = ['sha256', 'perceptual']
        
        results = {}
        
        for record in file_records:
            file_id = record['id']
            file_path = record['file_path']
            
            computed_hashes = self.compute_and_store_hashes(
                file_id, file_path, hash_types
            )
            
            if computed_hashes:
                results[file_id] = computed_hashes
        
        return results
