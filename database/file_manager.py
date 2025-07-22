"""
File manager for media file registration and metadata management.
Handles file registration, metadata storage, and file system operations.
"""

import sqlite3
import json
import os
import mimetypes
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path


class FileManager:
    """Manages file registration and metadata in the database."""
    
    SUPPORTED_MEDIA_TYPES = {
        'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'],
        'video': ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'],
        'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
    }
    
    def __init__(self, db_path: str):
        """Initialize file manager with database path."""
        self.db_path = db_path
    
    def register_file(self, file_path: str, metadata: Dict[str, Any] = None) -> Optional[int]:
        """Register a file in the database."""
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
        
        file_info = self._get_file_info(file_path)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """INSERT OR REPLACE INTO media_files 
                       (file_path, filename, file_size, file_type, mime_type, 
                        metadata, checksum, status) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        file_path,
                        file_info['filename'],
                        file_info['file_size'],
                        file_info['file_type'],
                        file_info['mime_type'],
                        json.dumps(metadata) if metadata else None,
                        None,  # Will be computed separately if needed
                        'active'
                    )
                )
                return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error registering file: {e}")
            return None
    
    def get_file_info(self, file_id: int) -> Optional[Dict[str, Any]]:
        """Get file information by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """SELECT id, file_path, filename, file_size, file_type, 
                              mime_type, created_at, modified_at, last_accessed,
                              metadata, checksum, status
                       FROM media_files WHERE id = ?""",
                    (file_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    file_info = {
                        'id': result[0],
                        'file_path': result[1],
                        'filename': result[2],
                        'file_size': result[3],
                        'file_type': result[4],
                        'mime_type': result[5],
                        'created_at': result[6],
                        'modified_at': result[7],
                        'last_accessed': result[8],
                        'metadata': json.loads(result[9]) if result[9] else None,
                        'checksum': result[10],
                        'status': result[11]
                    }
                    
                    # Update last accessed time
                    conn.execute(
                        "UPDATE media_files SET last_accessed = CURRENT_TIMESTAMP WHERE id = ?",
                        (file_id,)
                    )
                    conn.commit()
                    
                    return file_info
        except sqlite3.Error as e:
            print(f"Error retrieving file info: {e}")
        
        return None
    
    def find_file_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Find file by path."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT id FROM media_files WHERE file_path = ?",
                    (file_path,)
                )
                result = cursor.fetchone()
                
                if result:
                    return self.get_file_info(result[0])
        except sqlite3.Error as e:
            print(f"Error finding file by path: {e}")
        
        return None
    
    def update_file_metadata(self, file_id: int, metadata: Dict[str, Any]) -> bool:
        """Update file metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE media_files SET metadata = ?, modified_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (json.dumps(metadata), file_id)
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error updating file metadata: {e}")
            return False
    
    def update_file_checksum(self, file_id: int, checksum: str) -> bool:
        """Update file checksum."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE media_files SET checksum = ? WHERE id = ?",
                    (checksum, file_id)
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error updating file checksum: {e}")
            return False
    
    def mark_file_deleted(self, file_id: int) -> bool:
        """Mark file as deleted (soft delete)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE media_files SET status = 'deleted', modified_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (file_id,)
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error marking file as deleted: {e}")
            return False
    
    def list_files(self, file_type: str = None, status: str = 'active', 
                   limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List files with optional filters."""
        files = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """SELECT id, file_path, filename, file_size, file_type, 
                                 mime_type, created_at, status
                          FROM media_files WHERE 1=1"""
                params = []
                
                if file_type:
                    query += " AND file_type = ?"
                    params.append(file_type)
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor = conn.execute(query, params)
                
                for row in cursor:
                    files.append({
                        'id': row[0],
                        'file_path': row[1],
                        'filename': row[2],
                        'file_size': row[3],
                        'file_type': row[4],
                        'mime_type': row[5],
                        'created_at': row[6],
                        'status': row[7]
                    })
                    
        except sqlite3.Error as e:
            print(f"Error listing files: {e}")
        
        return files
    
    def search_files(self, search_term: str, search_in: List[str] = None) -> List[Dict[str, Any]]:
        """Search files by filename or metadata."""
        if search_in is None:
            search_in = ['filename', 'file_path']
        
        files = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conditions = []
                params = []
                
                if 'filename' in search_in:
                    conditions.append("filename LIKE ?")
                    params.append(f"%{search_term}%")
                
                if 'file_path' in search_in:
                    conditions.append("file_path LIKE ?")
                    params.append(f"%{search_term}%")
                
                if 'metadata' in search_in:
                    conditions.append("metadata LIKE ?")
                    params.append(f"%{search_term}%")
                
                if not conditions:
                    return files
                
                query = f"""
                    SELECT id, file_path, filename, file_size, file_type, 
                           mime_type, created_at, status
                    FROM media_files 
                    WHERE ({' OR '.join(conditions)}) AND status = 'active'
                    ORDER BY created_at DESC
                    LIMIT 50
                """
                
                cursor = conn.execute(query, params)
                
                for row in cursor:
                    files.append({
                        'id': row[0],
                        'file_path': row[1],
                        'filename': row[2],
                        'file_size': row[3],
                        'file_type': row[4],
                        'mime_type': row[5],
                        'created_at': row[6],
                        'status': row[7]
                    })
                    
        except sqlite3.Error as e:
            print(f"Error searching files: {e}")
        
        return files
    
    def get_file_statistics(self) -> Dict[str, Any]:
        """Get file storage statistics."""
        stats = {
            'total_files': 0,
            'files_by_type': {},
            'files_by_status': {},
            'total_size': 0,
            'average_size': 0.0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total files
                cursor = conn.execute("SELECT COUNT(*) FROM media_files")
                stats['total_files'] = cursor.fetchone()[0]
                
                # Files by type
                cursor = conn.execute(
                    "SELECT file_type, COUNT(*) FROM media_files GROUP BY file_type"
                )
                for file_type, count in cursor:
                    stats['files_by_type'][file_type] = count
                
                # Files by status
                cursor = conn.execute(
                    "SELECT status, COUNT(*) FROM media_files GROUP BY status"
                )
                for status, count in cursor:
                    stats['files_by_status'][status] = count
                
                # Total size
                cursor = conn.execute("SELECT SUM(file_size) FROM media_files WHERE status = 'active'")
                result = cursor.fetchone()[0]
                stats['total_size'] = result if result else 0
                
                # Average size
                if stats['total_files'] > 0:
                    stats['average_size'] = stats['total_size'] / stats['total_files']
                    
        except sqlite3.Error as e:
            print(f"Error getting file statistics: {e}")
        
        return stats
    
    def cleanup_missing_files(self) -> int:
        """Mark files as deleted if they no longer exist on disk."""
        cleaned_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT id, file_path FROM media_files WHERE status = 'active'"
                )
                
                for file_id, file_path in cursor:
                    if not os.path.exists(file_path):
                        conn.execute(
                            "UPDATE media_files SET status = 'deleted', modified_at = CURRENT_TIMESTAMP WHERE id = ?",
                            (file_id,)
                        )
                        cleaned_count += 1
                
                conn.commit()
                
        except sqlite3.Error as e:
            print(f"Error cleaning up missing files: {e}")
        
        return cleaned_count
    
    def batch_register_files(self, file_paths: List[str]) -> Dict[str, int]:
        """Register multiple files in batch."""
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                results['skipped'] += 1
                continue
            
            # Check if already registered
            existing = self.find_file_by_path(file_path)
            if existing:
                results['skipped'] += 1
                continue
            
            file_id = self.register_file(file_path)
            if file_id:
                results['success'] += 1
            else:
                results['failed'] += 1
        
        return results
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic file information from filesystem."""
        path_obj = Path(file_path)
        stat = path_obj.stat()
        
        file_type = self._determine_file_type(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        
        return {
            'filename': path_obj.name,
            'file_size': stat.st_size,
            'file_type': file_type,
            'mime_type': mime_type,
            'created_at': datetime.fromtimestamp(stat.st_ctime),
            'modified_at': datetime.fromtimestamp(stat.st_mtime)
        }
    
    def _determine_file_type(self, file_path: str) -> str:
        """Determine file type category from extension."""
        extension = os.path.splitext(file_path)[1].lower()
        
        for file_type, extensions in self.SUPPORTED_MEDIA_TYPES.items():
            if extension in extensions:
                return file_type
        
        return 'unknown'
    
    def scan_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """Scan directory and register all media files."""
        if not os.path.isdir(directory_path):
            return {'error': 'Directory does not exist'}
        
        found_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if self._is_media_file(file_path):
                        found_files.append(file_path)
        else:
            for item in os.listdir(directory_path):
                file_path = os.path.join(directory_path, item)
                if os.path.isfile(file_path) and self._is_media_file(file_path):
                    found_files.append(file_path)
        
        results = self.batch_register_files(found_files)
        results['total_found'] = len(found_files)
        
        return results
    
    def _is_media_file(self, file_path: str) -> bool:
        """Check if file is a supported media type."""
        extension = os.path.splitext(file_path)[1].lower()
        
        for extensions in self.SUPPORTED_MEDIA_TYPES.values():
            if extension in extensions:
                return True
        
        return False
