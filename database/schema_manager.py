"""
Schema manager for the mock media forensics database.
Handles table creation, migrations, and schema validation.
"""

import sqlite3
from typing import List, Dict, Any
import json
from datetime import datetime


class SchemaManager:
    """Manages database schema and migrations."""
    
    SCHEMA_VERSION = "1.0.0"
    
    TABLES = {
        'media_files': '''
            CREATE TABLE IF NOT EXISTS media_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_type TEXT NOT NULL,
                mime_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,  -- JSON metadata
                checksum TEXT,
                status TEXT DEFAULT 'active'  -- active, deleted, moved
            )
        ''',
        
        'media_hashes': '''
            CREATE TABLE IF NOT EXISTS media_hashes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                hash_type TEXT NOT NULL,  -- md5, sha256, perceptual, etc.
                hash_value TEXT NOT NULL,
                algorithm_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES media_files (id) ON DELETE CASCADE,
                UNIQUE(file_id, hash_type)
            )
        ''',
        
        'similarity_comparisons': '''
            CREATE TABLE IF NOT EXISTS similarity_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file1_id INTEGER NOT NULL,
                file2_id INTEGER NOT NULL,
                comparison_type TEXT NOT NULL,  -- perceptual, structural, semantic
                similarity_score REAL NOT NULL,  -- 0.0 to 1.0
                algorithm TEXT NOT NULL,
                algorithm_version TEXT,
                comparison_metadata TEXT,  -- JSON with detailed results
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file1_id) REFERENCES media_files (id) ON DELETE CASCADE,
                FOREIGN KEY (file2_id) REFERENCES media_files (id) ON DELETE CASCADE,
                UNIQUE(file1_id, file2_id, comparison_type, algorithm)
            )
        ''',
        
        'detection_results': '''
            CREATE TABLE IF NOT EXISTS detection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                detection_type TEXT NOT NULL,  -- ai_generated, deepfake, duplicate
                confidence_score REAL NOT NULL,  -- 0.0 to 1.0
                result_data TEXT,  -- JSON with detailed results
                model_name TEXT,
                model_version TEXT,
                processing_time REAL,  -- seconds
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES media_files (id) ON DELETE CASCADE
            )
        ''',
        
        'cache_entries': '''
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT UNIQUE NOT NULL,
                cache_type TEXT NOT NULL,  -- hash, similarity, detection
                data TEXT NOT NULL,  -- JSON cached data
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1
            )
        ''',
        
        'processing_jobs': '''
            CREATE TABLE IF NOT EXISTS processing_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,  -- hash_generation, similarity_check, detection
                file_id INTEGER,
                status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
                progress REAL DEFAULT 0.0,  -- 0.0 to 1.0
                parameters TEXT,  -- JSON job parameters
                result TEXT,  -- JSON job result
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES media_files (id) ON DELETE CASCADE
            )
        ''',
        
        'schema_version': '''
            CREATE TABLE IF NOT EXISTS schema_version (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        '''
    }
    
    INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_media_files_path ON media_files(file_path)",
        "CREATE INDEX IF NOT EXISTS idx_media_files_type ON media_files(file_type)",
        "CREATE INDEX IF NOT EXISTS idx_media_files_checksum ON media_files(checksum)",
        "CREATE INDEX IF NOT EXISTS idx_media_hashes_hash ON media_hashes(hash_value)",
        "CREATE INDEX IF NOT EXISTS idx_media_hashes_type ON media_hashes(hash_type)",
        "CREATE INDEX IF NOT EXISTS idx_similarity_score ON similarity_comparisons(similarity_score)",
        "CREATE INDEX IF NOT EXISTS idx_similarity_type ON similarity_comparisons(comparison_type)",
        "CREATE INDEX IF NOT EXISTS idx_detection_type ON detection_results(detection_type)",
        "CREATE INDEX IF NOT EXISTS idx_detection_confidence ON detection_results(confidence_score)",
        "CREATE INDEX IF NOT EXISTS idx_cache_key ON cache_entries(cache_key)",
        "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_type ON processing_jobs(job_type)",
    ]
    
    def __init__(self, db_path: str):
        """Initialize schema manager with database path."""
        self.db_path = db_path
    
    def initialize_database(self) -> bool:
        """Initialize database with all required tables and indexes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Create all tables
                for table_name, sql in self.TABLES.items():
                    conn.execute(sql)
                
                # Create indexes
                for index_sql in self.INDEXES:
                    conn.execute(index_sql)
                
                # Record schema version
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version, description) VALUES (?, ?)",
                    (self.SCHEMA_VERSION, "Initial schema creation")
                )
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
            return False
    
    def get_schema_version(self) -> str:
        """Get current schema version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
                )
                result = cursor.fetchone()
                return result[0] if result else "0.0.0"
        except sqlite3.Error:
            return "0.0.0"
    
    def validate_schema(self) -> Dict[str, Any]:
        """Validate database schema integrity."""
        validation_result = {
            "valid": True,
            "tables": {},
            "indexes": {},
            "foreign_keys": True,
            "issues": []
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check tables
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row[0] for row in cursor.fetchall()}
                
                for table_name in self.TABLES.keys():
                    validation_result["tables"][table_name] = table_name in existing_tables
                    if table_name not in existing_tables:
                        validation_result["valid"] = False
                        validation_result["issues"].append(f"Missing table: {table_name}")
                
                # Check indexes
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
                existing_indexes = {row[0] for row in cursor.fetchall() if row[0]}
                
                for index_sql in self.INDEXES:
                    # Extract index name from SQL
                    index_name = index_sql.split()[5]  # "CREATE INDEX IF NOT EXISTS idx_name"
                    validation_result["indexes"][index_name] = index_name in existing_indexes
                    if index_name not in existing_indexes:
                        validation_result["valid"] = False
                        validation_result["issues"].append(f"Missing index: {index_name}")
                
                # Check foreign key support
                cursor = conn.execute("PRAGMA foreign_keys")
                fk_enabled = cursor.fetchone()[0]
                validation_result["foreign_keys"] = bool(fk_enabled)
                if not fk_enabled:
                    validation_result["issues"].append("Foreign keys are not enabled")
                
        except sqlite3.Error as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Database error: {e}")
        
        return validation_result
    
    def get_table_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tables."""
        stats = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for table_name in self.TABLES.keys():
                    if table_name == 'schema_version':
                        continue
                    
                    # Get row count
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    
                    # Get table info
                    cursor = conn.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    
                    stats[table_name] = {
                        "row_count": row_count,
                        "column_count": len(columns),
                        "columns": [col[1] for col in columns]  # Column names
                    }
                    
        except sqlite3.Error as e:
            print(f"Error getting table stats: {e}")
        
        return stats
    
    def backup_schema(self) -> str:
        """Generate SQL dump of schema only."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get schema dump
                schema_dump = []
                cursor = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type IN ('table', 'index') AND sql IS NOT NULL"
                )
                for row in cursor:
                    schema_dump.append(row[0] + ";")
                
                return "\n".join(schema_dump)
                
        except sqlite3.Error as e:
            return f"-- Error generating schema dump: {e}"
