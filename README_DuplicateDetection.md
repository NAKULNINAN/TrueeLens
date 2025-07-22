# Duplicate Detection System

A comprehensive media duplicate detection system using perceptual hashing for images and video frame comparison.

## Features

- **Perceptual Hashing**: Uses multiple hash types (dHash, aHash, pHash, wHash) for robust duplicate detection
- **Multi-media Support**: Handles both images and videos
- **Similarity Thresholds**: Configurable similarity matching with confidence levels
- **SQLite Database**: Persistent storage of file hashes for fast comparison
- **Video Frame Analysis**: Samples video frames at intervals for duplicate detection
- **Batch Processing**: Can scan entire directories recursively
- **Performance Optimized**: Efficient Hamming distance calculation with database indexing

## Hash Types Explained

1. **dHash (Difference Hash)**: Sensitive to structural changes, good for detecting cropped or resized images
2. **aHash (Average Hash)**: Less sensitive to brightness changes, good for color-adjusted duplicates
3. **pHash (Perceptual Hash)**: Good for detecting rotations and scaling changes
4. **wHash (Wavelet Hash)**: Effective at handling noise and compression artifacts

## Installation

The required dependencies are already in your `requirements.txt`:

```
imagehash
opencv-python
pillow
numpy
```

## Usage

### 1. Basic Integration (Streamlit App)

The duplicate detector is already integrated into your Streamlit app. When a file is uploaded, it will:

```python
from detection.duplicate_detector import DuplicateDetector

detector = DuplicateDetector()
result = detector.detect(uploaded_file)

# Result contains:
# - confidence: similarity score (0-1)
# - is_duplicate: boolean
# - explanation: detailed explanation
# - processing_time: time taken
# - matches: list of similar files found (if any)
```

### 2. Command Line Interface

Use the CLI for batch processing and management:

```bash
# Scan a directory for media files
python duplicate_cli.py scan --directory ./media_files --recursive

# Find all duplicates in database
python duplicate_cli.py find --output duplicates.json

# Show database statistics
python duplicate_cli.py stats

# Process a single file
python duplicate_cli.py process --file image.jpg

# Configure similarity thresholds
python duplicate_cli.py configure --dhash 3 --ahash 5 --phash 8 --whash 4

# Export database to JSON
python duplicate_cli.py export --output database_backup.json

# Clear database
python duplicate_cli.py clear
```

### 3. Python API

```python
from detection.duplicate_detector import DuplicateDetector

# Initialize with custom thresholds
detector = DuplicateDetector({
    'dhash': 5,   # Lower = more strict
    'ahash': 8,
    'phash': 10,
    'whash': 8
})

# Scan directory
detector.scan_directory('/path/to/media', recursive=True)

# Find duplicates
duplicates = detector.find_all_duplicates()

for duplicate in duplicates:
    print(f"Original: {duplicate.original_file}")
    print(f"Duplicate: {duplicate.duplicate_file}")
    print(f"Similarity: {duplicate.similarity_score:.3f}")
    print(f"Confidence: {duplicate.confidence}")
    print(f"Hash type: {duplicate.hash_type}")
    print()

# Get statistics
stats = detector.get_statistics()
print(f"Total files: {stats['total_files']}")
print(f"Images: {stats['images']}")
print(f"Videos: {stats['videos']}")
```

### 4. Sample Data Generation

Generate test data for development and testing:

```bash
# Run interactive sample data generator
python sample_data_generator.py

# Options available:
# 1. Generate all sample data (recommended for testing)
# 2. Generate sample images only
# 3. Generate sample videos only
# 4. Generate exact duplicates
# 5. Generate near duplicates
# 6. Display current statistics
# 7. Clear database
```

## Configuration

### Similarity Thresholds

Lower values = more strict matching, higher values = more lenient matching:

- **dhash: 5** - Good for structural similarity
- **ahash: 8** - Handles brightness variations
- **phash: 10** - Handles rotations/scaling
- **whash: 8** - Handles noise and compression

### Confidence Levels

- **High**: Hamming distance ≤ 2 (very similar)
- **Medium**: Hamming distance ≤ 5 (similar)
- **Low**: Hamming distance > 5 (possibly similar)

## Supported File Formats

### Images
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

### Videos
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)

## Database Schema

The system uses SQLite with the following schema:

```sql
CREATE TABLE media_hashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_type TEXT NOT NULL,  -- 'image' or 'video'
    dhash TEXT NOT NULL,
    ahash TEXT NOT NULL,
    phash TEXT NOT NULL,
    whash TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    timestamp TEXT NOT NULL
);
```

Indexes are created on all hash columns for fast similarity searching.

## Testing

Run the test suite to verify functionality:

```bash
python test_duplicate_detection.py
```

This will run comprehensive tests including:
- Basic functionality test
- Similarity threshold testing
- Hash calculation verification
- Uploaded file simulation
- Performance testing with multiple files

## Performance

The system is optimized for performance:

- **Hash Calculation**: ~0.1-0.5 seconds per image
- **Database Operations**: Indexed for fast similarity searches
- **Memory Usage**: Minimal - only stores hash strings, not image data
- **Scalability**: Can handle thousands of files efficiently

## Algorithm Details

### Image Processing
1. Load image using PIL
2. Calculate 4 different perceptual hashes
3. Store hashes as hex strings in database
4. Compare using Hamming distance

### Video Processing
1. Extract frames at configurable intervals (default: every 30th frame)
2. Calculate perceptual hashes for each sampled frame
3. Use most common hash for each hash type across all frames
4. Store combined representative hashes

### Duplicate Detection
1. Compare all hash types for each file pair
2. Use the hash type with the lowest Hamming distance
3. Apply configurable thresholds for each hash type
4. Calculate similarity score: 1.0 - (distance / max_possible_distance)
5. Assign confidence level based on distance

## Integration with Main App

The duplicate detection system integrates seamlessly with your existing Streamlit app:

```python
# In your main app.py, the detector is already imported and used:
from detection.duplicate_detector import DuplicateDetector

# Initialize once (in your app initialization)
duplicate_detector = DuplicateDetector()

# Use in your file upload handler
if uploaded_file is not None:
    # Your existing deepfake detection...
    
    # Add duplicate detection
    duplicate_result = duplicate_detector.detect(uploaded_file)
    
    if duplicate_result['is_duplicate']:
        st.warning(f"⚠️ Potential duplicate detected!")
        st.info(duplicate_result['explanation'])
```

## Troubleshooting

### Common Issues

1. **"No module named 'detection'"**: Make sure you're running from the correct directory
2. **SQLite database locked**: Close other connections to the database
3. **Memory issues with large videos**: Adjust frame_interval parameter to sample fewer frames
4. **False positives**: Lower the similarity thresholds for stricter matching
5. **Missing duplicates**: Increase the similarity thresholds for more lenient matching

### Performance Optimization

- Use SSD storage for the database for faster I/O
- Increase frame_interval for videos to reduce processing time
- Consider using a separate database file for large datasets
- Regular database maintenance (VACUUM) for optimal performance

## Future Enhancements

Potential improvements for the system:
- GPU acceleration for hash calculation
- Advanced video analysis (scene detection)
- Machine learning-based similarity scoring
- Distributed processing for large datasets
- Web interface for database management
- Integration with cloud storage services
