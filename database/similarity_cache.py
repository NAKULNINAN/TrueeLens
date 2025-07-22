"""
Similarity cache for storing and retrieving media comparison results.
Handles caching of similarity scores and comparison metadata to avoid repeated computations.
"""

import sqlite3
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import hashlib


class SimilarityCache:
    """Manages caching of similarity comparison results."""
    
    def __init__(self, db_path: str):
        """Initialize similarity cache with database path."""
        self.db_path = db_path
    
    def store_similarity(self, file1_id: int, file2_id: int, comparison_type: str,
                        similarity_score: float, algorithm: str, 
                        algorithm_version: str = None, 
                        metadata: Dict[str, Any] = None) -> bool:
        """Store a similarity comparison result."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO similarity_comparisons 
                       (file1_id, file2_id, comparison_type, similarity_score, 
                        algorithm, algorithm_version, comparison_metadata) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (file1_id, file2_id, comparison_type, similarity_score,
                     algorithm, algorithm_version, json.dumps(metadata) if metadata else None)
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error storing similarity result: {e}")
            return False
    
    def get_similarity(self, file1_id: int, file2_id: int, comparison_type: str,
                      algorithm: str) -> Optional[Dict[str, Any]]:
        """Get cached similarity result between two files."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Try both file ID orders since similarity is symmetric
                for id1, id2 in [(file1_id, file2_id), (file2_id, file1_id)]:
                    cursor = conn.execute(
                        """SELECT similarity_score, algorithm_version, comparison_metadata, created_at
                           FROM similarity_comparisons 
                           WHERE file1_id = ? AND file2_id = ? AND comparison_type = ? AND algorithm = ?""",
                        (id1, id2, comparison_type, algorithm)
                    )
                    result = cursor.fetchone()
                    if result:
                        score, version, metadata_json, created_at = result
                        return {
                            'similarity_score': score,
                            'algorithm_version': version,
                            'metadata': json.loads(metadata_json) if metadata_json else None,
                            'created_at': created_at
                        }
        except sqlite3.Error as e:
            print(f"Error retrieving similarity result: {e}")
        
        return None
    
    def find_similar_files(self, file_id: int, comparison_type: str = None, 
                          min_similarity: float = 0.5, 
                          algorithm: str = None) -> List[Dict[str, Any]]:
        """Find files similar to the given file."""
        similar_files = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        CASE WHEN file1_id = ? THEN file2_id ELSE file1_id END as other_file_id,
                        similarity_score, comparison_type, algorithm, algorithm_version,
                        comparison_metadata, created_at
                    FROM similarity_comparisons 
                    WHERE (file1_id = ? OR file2_id = ?) 
                    AND similarity_score >= ?
                """
                params = [file_id, file_id, file_id, min_similarity]
                
                if comparison_type:
                    query += " AND comparison_type = ?"
                    params.append(comparison_type)
                
                if algorithm:
                    query += " AND algorithm = ?"
                    params.append(algorithm)
                
                query += " ORDER BY similarity_score DESC"
                
                cursor = conn.execute(query, params)
                
                for row in cursor:
                    other_file_id, score, comp_type, alg, alg_version, metadata_json, created_at = row
                    similar_files.append({
                        'file_id': other_file_id,
                        'similarity_score': score,
                        'comparison_type': comp_type,
                        'algorithm': alg,
                        'algorithm_version': alg_version,
                        'metadata': json.loads(metadata_json) if metadata_json else None,
                        'created_at': created_at
                    })
                    
        except sqlite3.Error as e:
            print(f"Error finding similar files: {e}")
        
        return similar_files
    
    def store_detection_result(self, file_id: int, detection_type: str,
                              confidence_score: float, result_data: Dict[str, Any] = None,
                              model_name: str = None, model_version: str = None,
                              processing_time: float = None) -> bool:
        """Store a detection result (AI-generated, deepfake, etc.)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO detection_results 
                       (file_id, detection_type, confidence_score, result_data,
                        model_name, model_version, processing_time) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (file_id, detection_type, confidence_score, 
                     json.dumps(result_data) if result_data else None,
                     model_name, model_version, processing_time)
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error storing detection result: {e}")
            return False
    
    def get_detection_result(self, file_id: int, detection_type: str) -> Optional[Dict[str, Any]]:
        """Get cached detection result for a file."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """SELECT confidence_score, result_data, model_name, model_version,
                              processing_time, created_at
                       FROM detection_results 
                       WHERE file_id = ? AND detection_type = ?
                       ORDER BY created_at DESC LIMIT 1""",
                    (file_id, detection_type)
                )
                result = cursor.fetchone()
                if result:
                    confidence, data_json, model_name, model_version, proc_time, created_at = result
                    return {
                        'confidence_score': confidence,
                        'result_data': json.loads(data_json) if data_json else None,
                        'model_name': model_name,
                        'model_version': model_version,
                        'processing_time': proc_time,
                        'created_at': created_at
                    }
        except sqlite3.Error as e:
            print(f"Error retrieving detection result: {e}")
        
        return None
    
    def store_cache_entry(self, cache_key: str, cache_type: str, data: Dict[str, Any],
                         expires_in_hours: int = 24) -> bool:
        """Store a generic cache entry."""
        expires_at = datetime.now() + timedelta(hours=expires_in_hours)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO cache_entries 
                       (cache_key, cache_type, data, expires_at) 
                       VALUES (?, ?, ?, ?)""",
                    (cache_key, cache_type, json.dumps(data), expires_at.isoformat())
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error storing cache entry: {e}")
            return False
    
    def get_cache_entry(self, cache_key: str, cache_type: str = None) -> Optional[Dict[str, Any]]:
        """Get a cached entry by key."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT data, expires_at FROM cache_entries WHERE cache_key = ?"
                params = [cache_key]
                
                if cache_type:
                    query += " AND cache_type = ?"
                    params.append(cache_type)
                
                cursor = conn.execute(query, params)
                result = cursor.fetchone()
                
                if result:
                    data_json, expires_at = result
                    
                    # Check if expired
                    if expires_at:
                        expires_datetime = datetime.fromisoformat(expires_at)
                        if datetime.now() > expires_datetime:
                            # Entry is expired, remove it
                            self._remove_cache_entry(cache_key, cache_type)
                            return None
                    
                    # Update access statistics
                    conn.execute(
                        """UPDATE cache_entries 
                           SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                           WHERE cache_key = ?""",
                        (cache_key,)
                    )
                    conn.commit()
                    
                    return json.loads(data_json)
                    
        except sqlite3.Error as e:
            print(f"Error retrieving cache entry: {e}")
        
        return None
    
    def _remove_cache_entry(self, cache_key: str, cache_type: str = None) -> bool:
        """Remove a cache entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if cache_type:
                    conn.execute(
                        "DELETE FROM cache_entries WHERE cache_key = ? AND cache_type = ?",
                        (cache_key, cache_type)
                    )
                else:
                    conn.execute(
                        "DELETE FROM cache_entries WHERE cache_key = ?",
                        (cache_key,)
                    )
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error removing cache entry: {e}")
            return False
    
    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE expires_at < ?",
                    (datetime.now().isoformat(),)
                )
                return cursor.rowcount
        except sqlite3.Error as e:
            print(f"Error cleaning up expired cache: {e}")
            return 0
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        stats = {
            'total_entries': 0,
            'entries_by_type': {},
            'expired_entries': 0,
            'most_accessed': [],
            'cache_hit_potential': 0.0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total entries
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                stats['total_entries'] = cursor.fetchone()[0]
                
                # Entries by type
                cursor = conn.execute(
                    "SELECT cache_type, COUNT(*) FROM cache_entries GROUP BY cache_type"
                )
                for cache_type, count in cursor:
                    stats['entries_by_type'][cache_type] = count
                
                # Expired entries
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM cache_entries WHERE expires_at < ?",
                    (datetime.now().isoformat(),)
                )
                stats['expired_entries'] = cursor.fetchone()[0]
                
                # Most accessed entries
                cursor = conn.execute(
                    """SELECT cache_key, cache_type, access_count 
                       FROM cache_entries 
                       ORDER BY access_count DESC LIMIT 10"""
                )
                stats['most_accessed'] = [
                    {'key': key, 'type': cache_type, 'access_count': count}
                    for key, cache_type, count in cursor
                ]
                
                # Calculate cache hit potential (entries accessed more than once)
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries WHERE access_count > 1")
                multi_access = cursor.fetchone()[0]
                if stats['total_entries'] > 0:
                    stats['cache_hit_potential'] = multi_access / stats['total_entries']
                    
        except sqlite3.Error as e:
            print(f"Error getting cache statistics: {e}")
        
        return stats
    
    def generate_cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def bulk_store_similarities(self, similarities: List[Dict[str, Any]]) -> int:
        """Store multiple similarity results in batch."""
        stored_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for sim in similarities:
                    conn.execute(
                        """INSERT OR REPLACE INTO similarity_comparisons 
                           (file1_id, file2_id, comparison_type, similarity_score, 
                            algorithm, algorithm_version, comparison_metadata) 
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            sim['file1_id'], sim['file2_id'], sim['comparison_type'],
                            sim['similarity_score'], sim['algorithm'],
                            sim.get('algorithm_version'), 
                            json.dumps(sim.get('metadata')) if sim.get('metadata') else None
                        )
                    )
                    stored_count += 1
                
                conn.commit()
                
        except sqlite3.Error as e:
            print(f"Error bulk storing similarities: {e}")
        
        return stored_count
