"""
Sample Data Generator for Duplicate Detection System

This script creates mock sample entries in the database for testing
the duplicate detection functionality.
"""

import os
import random
import string
from pathlib import Path
import time
from detection.duplicate_detector import DuplicateDetector, MediaHash, DatabaseManager
from PIL import Image
import numpy as np

class SampleDataGenerator:
    """Generates sample data for testing duplicate detection"""
    
    def __init__(self):
        self.detector = DuplicateDetector()
        self.db_manager = DatabaseManager()
        
    def generate_mock_hash(self) -> str:
        """Generate a random hash-like string"""
        return ''.join(random.choices(string.hexdigits.lower(), k=16))
    
    def create_sample_images(self, count: int = 20):
        """Create sample image files and their database entries"""
        print(f"Creating {count} sample image entries...")
        
        sample_dir = Path("sample_images")
        sample_dir.mkdir(exist_ok=True)
        
        sample_files = []
        
        for i in range(count):
            # Create a simple test image
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            image = Image.fromarray(img_array)
            
            filename = f"sample_image_{i:03d}.jpg"
            filepath = sample_dir / filename
            image.save(filepath)
            
            # Generate similar hashes for some images to simulate duplicates
            base_hash = self.generate_mock_hash()
            
            # Create variations of the hash for potential duplicates
            if i % 5 == 0 and i > 0:  # Every 5th image is similar to a previous one
                # Make it similar to the previous group
                reference_idx = i - (i % 5)
                base_hash = self.modify_hash(base_hash, changes=random.randint(1, 3))
            
            # Create slightly different hashes for each hash type
            dhash = base_hash
            ahash = self.modify_hash(base_hash, changes=1)
            phash = self.modify_hash(base_hash, changes=2)
            whash = self.modify_hash(base_hash, changes=1)
            
            media_hash = MediaHash(
                file_path=str(filepath.absolute()),
                file_type='image',
                dhash=dhash,
                ahash=ahash,
                phash=phash,
                whash=whash,
                file_size=os.path.getsize(filepath),
                timestamp=str(time.time())
            )
            
            self.db_manager.insert_media_hash(media_hash)
            sample_files.append(str(filepath.absolute()))
        
        print(f"Created {count} sample image entries in database")
        return sample_files
    
    def create_sample_videos(self, count: int = 10):
        """Create mock video entries (without actual video files)"""
        print(f"Creating {count} sample video entries...")
        
        sample_files = []
        
        for i in range(count):
            # Mock video file path
            filename = f"sample_video_{i:03d}.mp4"
            filepath = Path("sample_videos") / filename
            
            # Generate similar hashes for some videos to simulate duplicates
            base_hash = self.generate_mock_hash()
            
            # Create variations for potential duplicates
            if i % 4 == 0 and i > 0:  # Every 4th video is similar to a previous one
                base_hash = self.modify_hash(base_hash, changes=random.randint(1, 4))
            
            dhash = base_hash
            ahash = self.modify_hash(base_hash, changes=2)
            phash = self.modify_hash(base_hash, changes=3)
            whash = self.modify_hash(base_hash, changes=1)
            
            media_hash = MediaHash(
                file_path=str(filepath.absolute()),
                file_type='video',
                dhash=dhash,
                ahash=ahash,
                phash=phash,
                whash=whash,
                file_size=random.randint(1000000, 50000000),  # Mock file size
                timestamp=str(time.time())
            )
            
            self.db_manager.insert_media_hash(media_hash)
            sample_files.append(str(filepath.absolute()))
        
        print(f"Created {count} sample video entries in database")
        return sample_files
    
    def modify_hash(self, original_hash: str, changes: int = 1) -> str:
        """Modify a hash by changing a few characters"""
        hash_chars = list(original_hash)
        positions = random.sample(range(len(hash_chars)), min(changes, len(hash_chars)))
        
        for pos in positions:
            # Change to a different hex character
            current_char = hash_chars[pos]
            new_char = random.choice([c for c in string.hexdigits.lower() if c != current_char])
            hash_chars[pos] = new_char
        
        return ''.join(hash_chars)
    
    def create_exact_duplicates(self, count: int = 5):
        """Create exact duplicate entries for testing"""
        print(f"Creating {count} exact duplicate pairs...")
        
        for i in range(count):
            base_hash = self.generate_mock_hash()
            base_path = f"original_file_{i:03d}.jpg"
            duplicate_path = f"duplicate_file_{i:03d}.jpg"
            
            # Create original
            original_hash = MediaHash(
                file_path=base_path,
                file_type='image',
                dhash=base_hash,
                ahash=base_hash,
                phash=base_hash,
                whash=base_hash,
                file_size=random.randint(100000, 1000000),
                timestamp=str(time.time())
            )
            
            # Create exact duplicate
            duplicate_hash = MediaHash(
                file_path=duplicate_path,
                file_type='image',
                dhash=base_hash,  # Same hash = exact duplicate
                ahash=base_hash,
                phash=base_hash,
                whash=base_hash,
                file_size=original_hash.file_size,  # Same size
                timestamp=str(time.time() + 1)  # Slightly different timestamp
            )
            
            self.db_manager.insert_media_hash(original_hash)
            self.db_manager.insert_media_hash(duplicate_hash)
        
        print(f"Created {count} exact duplicate pairs")
    
    def create_near_duplicates(self, count: int = 8):
        """Create near-duplicate entries for testing"""
        print(f"Creating {count} near-duplicate pairs...")
        
        for i in range(count):
            base_hash = self.generate_mock_hash()
            original_path = f"near_original_{i:03d}.jpg"
            near_duplicate_path = f"near_duplicate_{i:03d}.jpg"
            
            # Create original
            original_hash = MediaHash(
                file_path=original_path,
                file_type='image',
                dhash=base_hash,
                ahash=base_hash,
                phash=base_hash,
                whash=base_hash,
                file_size=random.randint(100000, 1000000),
                timestamp=str(time.time())
            )
            
            # Create near duplicate (modify 1-3 characters in hashes)
            near_duplicate_hash = MediaHash(
                file_path=near_duplicate_path,
                file_type='image',
                dhash=self.modify_hash(base_hash, changes=random.randint(1, 2)),
                ahash=self.modify_hash(base_hash, changes=random.randint(1, 3)),
                phash=self.modify_hash(base_hash, changes=random.randint(1, 2)),
                whash=self.modify_hash(base_hash, changes=random.randint(1, 2)),
                file_size=original_hash.file_size + random.randint(-1000, 1000),
                timestamp=str(time.time() + 1)
            )
            
            self.db_manager.insert_media_hash(original_hash)
            self.db_manager.insert_media_hash(near_duplicate_hash)
        
        print(f"Created {count} near-duplicate pairs")
    
    def display_statistics(self):
        """Display current database statistics"""
        stats = self.detector.get_statistics()
        print("\n=== Database Statistics ===")
        print(f"Total files: {stats['total_files']}")
        print(f"Images: {stats['images']}")
        print(f"Videos: {stats['videos']}")
        print(f"Total size: {stats['total_size']:,} bytes")
        
        # Find and display duplicates
        duplicates = self.detector.find_all_duplicates()
        print(f"\n=== Duplicate Detection Results ===")
        print(f"Found {len(duplicates)} duplicate pairs")
        
        confidence_counts = {'high': 0, 'medium': 0, 'low': 0}
        hash_type_counts = {'dhash': 0, 'ahash': 0, 'phash': 0, 'whash': 0}
        
        for duplicate in duplicates[:10]:  # Show first 10
            confidence_counts[duplicate.confidence] += 1
            hash_type_counts[duplicate.hash_type] += 1
            print(f"  {duplicate.original_file} <-> {duplicate.duplicate_file}")
            print(f"    Similarity: {duplicate.similarity_score:.3f}, "
                  f"Hash: {duplicate.hash_type}, "
                  f"Confidence: {duplicate.confidence}")
        
        print(f"\nConfidence distribution: {confidence_counts}")
        print(f"Hash type distribution: {hash_type_counts}")
    
    def clear_database(self):
        """Clear all entries from the database"""
        import sqlite3
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM media_hashes")
        conn.commit()
        conn.close()
        print("Database cleared")
    
    def generate_all_samples(self):
        """Generate a complete set of sample data"""
        print("Generating complete sample dataset...")
        
        # Clear existing data
        self.clear_database()
        
        # Generate different types of sample data
        self.create_sample_images(15)
        self.create_sample_videos(8)
        self.create_exact_duplicates(5)
        self.create_near_duplicates(6)
        
        # Display results
        self.display_statistics()
        
        print("\nSample data generation complete!")


def main():
    """Main function to run the sample data generator"""
    generator = SampleDataGenerator()
    
    print("Media Forensics - Sample Data Generator")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Generate all sample data")
        print("2. Generate sample images only")
        print("3. Generate sample videos only")
        print("4. Generate exact duplicates")
        print("5. Generate near duplicates")
        print("6. Display current statistics")
        print("7. Clear database")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            generator.generate_all_samples()
        elif choice == '2':
            count = int(input("Enter number of images to generate (default 15): ") or "15")
            generator.create_sample_images(count)
        elif choice == '3':
            count = int(input("Enter number of videos to generate (default 8): ") or "8")
            generator.create_sample_videos(count)
        elif choice == '4':
            count = int(input("Enter number of exact duplicate pairs (default 5): ") or "5")
            generator.create_exact_duplicates(count)
        elif choice == '5':
            count = int(input("Enter number of near duplicate pairs (default 6): ") or "6")
            generator.create_near_duplicates(count)
        elif choice == '6':
            generator.display_statistics()
        elif choice == '7':
            confirm = input("Are you sure you want to clear the database? (y/N): ")
            if confirm.lower() == 'y':
                generator.clear_database()
        elif choice == '8':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
