"""
Command-line interface for Duplicate Detection System

This script provides a CLI for testing and managing the duplicate detection functionality.
"""

import argparse
import sys
from pathlib import Path
from detection.duplicate_detector import DuplicateDetector, DatabaseManager
import json
import time

class DuplicateCLI:
    """Command-line interface for duplicate detection"""
    
    def __init__(self):
        self.detector = DuplicateDetector()
        self.db_manager = DatabaseManager()
    
    def scan_directory(self, directory: str, recursive: bool = True):
        """Scan a directory for media files"""
        print(f"Scanning directory: {directory}")
        print(f"Recursive: {recursive}")
        
        start_time = time.time()
        
        try:
            self.detector.scan_directory(directory, recursive)
            elapsed_time = time.time() - start_time
            print(f"Scan completed in {elapsed_time:.2f} seconds")
            
            # Display statistics
            self.show_statistics()
            
        except Exception as e:
            print(f"Error scanning directory: {e}")
    
    def find_duplicates(self, output_file: str = None):
        """Find all duplicates in the database"""
        print("Finding duplicates...")
        
        start_time = time.time()
        duplicates = self.detector.find_all_duplicates()
        elapsed_time = time.time() - start_time
        
        print(f"Found {len(duplicates)} duplicate pairs in {elapsed_time:.2f} seconds")
        
        if not duplicates:
            print("No duplicates found.")
            return
        
        # Group by confidence level
        high_conf = [d for d in duplicates if d.confidence == 'high']
        medium_conf = [d for d in duplicates if d.confidence == 'medium']
        low_conf = [d for d in duplicates if d.confidence == 'low']
        
        print(f"\\nConfidence breakdown:")
        print(f"  High confidence: {len(high_conf)}")
        print(f"  Medium confidence: {len(medium_conf)}")
        print(f"  Low confidence: {len(low_conf)}")
        
        # Display duplicates
        print(f"\\nDuplicate pairs:")
        for i, duplicate in enumerate(duplicates, 1):
            print(f"{i:3d}. {duplicate.original_file}")
            print(f"     <-> {duplicate.duplicate_file}")
            print(f"     Similarity: {duplicate.similarity_score:.3f} "
                  f"({duplicate.hash_type}, {duplicate.confidence})")
            print()
        
        # Save to file if requested
        if output_file:
            self.save_duplicates_to_file(duplicates, output_file)
    
    def save_duplicates_to_file(self, duplicates, filename: str):
        """Save duplicates to a JSON file"""
        duplicate_data = []
        for dup in duplicates:
            duplicate_data.append({
                'original_file': dup.original_file,
                'duplicate_file': dup.duplicate_file,
                'similarity_score': dup.similarity_score,
                'hash_type': dup.hash_type,
                'confidence': dup.confidence
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.time(),
                'total_duplicates': len(duplicates),
                'duplicates': duplicate_data
            }, f, indent=2)
        
        print(f"Duplicates saved to {filename}")
    
    def show_statistics(self):
        """Display database statistics"""
        stats = self.detector.get_statistics()
        
        print("\\n=== Database Statistics ===")
        print(f"Total files: {stats['total_files']}")
        print(f"Images: {stats['images']}")
        print(f"Videos: {stats['videos']}")
        print(f"Total size: {stats['total_size']:,} bytes ({stats['total_size']/1024/1024:.1f} MB)")
        
    def process_single_file(self, file_path: str):
        """Process a single file and add it to the database"""
        print(f"Processing file: {file_path}")
        
        if not Path(file_path).exists():
            print(f"Error: File {file_path} does not exist")
            return
        
        try:
            self.detector.process_file(file_path)
            print(f"File processed successfully")
        except Exception as e:
            print(f"Error processing file: {e}")
    
    def clear_database(self):
        """Clear the entire database"""
        import sqlite3
        
        confirm = input("Are you sure you want to clear the entire database? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Operation cancelled")
            return
        
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM media_hashes")
        conn.commit()
        conn.close()
        
        print("Database cleared successfully")
    
    def configure_thresholds(self, dhash: int = None, ahash: int = None, 
                           phash: int = None, whash: int = None):
        """Configure similarity thresholds"""
        current_thresholds = self.detector.similarity_threshold.copy()
        
        if dhash is not None:
            current_thresholds['dhash'] = dhash
        if ahash is not None:
            current_thresholds['ahash'] = ahash
        if phash is not None:
            current_thresholds['phash'] = phash
        if whash is not None:
            current_thresholds['whash'] = whash
        
        self.detector.similarity_threshold = current_thresholds
        
        print("Updated similarity thresholds:")
        for hash_type, threshold in current_thresholds.items():
            print(f"  {hash_type}: {threshold}")
    
    def export_database(self, output_file: str):
        """Export database to JSON file"""
        all_hashes = self.db_manager.get_all_hashes()
        
        export_data = {
            'timestamp': time.time(),
            'total_files': len(all_hashes),
            'hashes': []
        }
        
        for media_hash in all_hashes:
            export_data['hashes'].append({
                'file_path': media_hash.file_path,
                'file_type': media_hash.file_type,
                'dhash': media_hash.dhash,
                'ahash': media_hash.ahash,
                'phash': media_hash.phash,
                'whash': media_hash.whash,
                'file_size': media_hash.file_size,
                'timestamp': media_hash.timestamp
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Database exported to {output_file}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Duplicate Detection System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scan --directory ./images --recursive
  %(prog)s find --output duplicates.json
  %(prog)s stats
  %(prog)s process --file image.jpg
  %(prog)s configure --dhash 3 --phash 8
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan directory for media files')
    scan_parser.add_argument('--directory', '-d', required=True, 
                           help='Directory to scan')
    scan_parser.add_argument('--recursive', '-r', action='store_true', default=True,
                           help='Scan recursively (default: True)')
    scan_parser.add_argument('--no-recursive', action='store_false', dest='recursive',
                           help='Disable recursive scanning')
    
    # Find duplicates command
    find_parser = subparsers.add_parser('find', help='Find duplicate files')
    find_parser.add_argument('--output', '-o', help='Output file for duplicates (JSON)')
    
    # Statistics command
    subparsers.add_parser('stats', help='Show database statistics')
    
    # Process single file command
    process_parser = subparsers.add_parser('process', help='Process a single file')
    process_parser.add_argument('--file', '-f', required=True, help='File to process')
    
    # Clear database command
    subparsers.add_parser('clear', help='Clear the database')
    
    # Configure thresholds command
    config_parser = subparsers.add_parser('configure', help='Configure similarity thresholds')
    config_parser.add_argument('--dhash', type=int, help='dHash threshold')
    config_parser.add_argument('--ahash', type=int, help='aHash threshold')
    config_parser.add_argument('--phash', type=int, help='pHash threshold')
    config_parser.add_argument('--whash', type=int, help='wHash threshold')
    
    # Export database command
    export_parser = subparsers.add_parser('export', help='Export database to JSON')
    export_parser.add_argument('--output', '-o', required=True, 
                              help='Output file for database export')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = DuplicateCLI()
    
    try:
        if args.command == 'scan':
            cli.scan_directory(args.directory, args.recursive)
        elif args.command == 'find':
            cli.find_duplicates(args.output)
        elif args.command == 'stats':
            cli.show_statistics()
        elif args.command == 'process':
            cli.process_single_file(args.file)
        elif args.command == 'clear':
            cli.clear_database()
        elif args.command == 'configure':
            cli.configure_thresholds(args.dhash, args.ahash, args.phash, args.whash)
        elif args.command == 'export':
            cli.export_database(args.output)
            
    except KeyboardInterrupt:
        print("\\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
