"""
Database package for media forensics application.
Provides mock database functionality for storing media hashes, similarity comparisons,
result caching, and file management utilities.
"""

from .mock_database import MockMediaDatabase
from .schema_manager import SchemaManager
from .hash_storage import HashStorage
from .similarity_cache import SimilarityCache
from .file_manager import FileManager

__all__ = [
    'MockMediaDatabase',
    'SchemaManager',
    'HashStorage',
    'SimilarityCache',
    'FileManager'
]
