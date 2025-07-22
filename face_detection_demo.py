#!/usr/bin/env python3
"""
Face Detection Pipeline Demo
Demonstrates MTCNN/MediaPipe face detection with alignment, quality assessment, and batch processing
"""

import cv2
import numpy as np
import os
import sys
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from detection.face_extractor import FaceExtractor

def demo_single_image(image_path: str, detector_type: str = "mtcnn"):
    """Demo face detection on a single image"""
    print(f"Processing single image: {image_path} with {detector_type}")
    
    extractor = FaceExtractor(detector_type=detector_type)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Process the image
    results = extractor.process_frame(image, apply_alignment=True, quality_threshold=True)
    
    print(f"Found {len(results)} high-quality faces")
    
    # Display results
    for i, result in enumerate(results):
        print(f"\nFace {i+1}:")
        print(f"  Quality metrics: {result['quality_metrics']}")
        print(f"  Bounding box: {result['bbox']}")
    
    # Save processed faces
    if results:
        output_dir = f"demo_output_{detector_type}"
        extractor.save_faces(results, output_dir)
        print(f"Saved faces to {output_dir}")
    
    return results

def demo_video_batch(video_path: str, detector_type: str = "mtcnn", batch_size: int = 16):
    """Demo batch processing of video frames"""
    print(f"Processing video: {video_path} with {detector_type} (batch size: {batch_size})")
    
    extractor = FaceExtractor(detector_type=detector_type)
    
    # Process video in batches
    results = extractor.process_video_batch(
        video_path, 
        batch_size=batch_size, 
        frame_skip=30  # Process every 30th frame (1 frame per second at 30fps)
    )
    
    print(f"Processed video and found {len(results)} faces across all frames")
    
    # Get statistics
    stats = extractor.get_statistics(results)
    print("\nVideo Processing Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Save sample faces
    if results:
        output_dir = f"video_output_{detector_type}"
        sample_results = results[:10]  # Save first 10 faces
        extractor.save_faces(sample_results, output_dir)
        print(f"Saved sample faces to {output_dir}")
    
    return results, stats

def demo_image_batch(image_dir: str, detector_type: str = "mtcnn"):
    """Demo batch processing of multiple images"""
    print(f"Processing image directory: {image_dir} with {detector_type}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = [
        str(p) for p in Path(image_dir).glob('*') 
        if p.suffix.lower() in image_extensions
    ]
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return [], {}
    
    print(f"Found {len(image_paths)} images")
    
    extractor = FaceExtractor(detector_type=detector_type)
    
    # Process batch of images
    results = extractor.process_image_batch(image_paths)
    
    print(f"Processed {len(image_paths)} images and found {len(results)} faces")
    
    # Get statistics
    stats = extractor.get_statistics(results)
    print("\nImage Batch Processing Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Save all detected faces
    if results:
        output_dir = f"batch_output_{detector_type}"
        extractor.save_faces(results, output_dir)
        print(f"Saved all faces to {output_dir}")
    
    return results, stats

def compare_detectors(image_path: str):
    """Compare MTCNN vs MediaPipe performance"""
    print(f"Comparing detectors on: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    results_comparison = {}
    
    for detector_type in ["mtcnn", "mediapipe"]:
        print(f"\nTesting {detector_type}...")
        extractor = FaceExtractor(detector_type=detector_type)
        
        import time
        start_time = time.time()
        results = extractor.process_frame(image, apply_alignment=True, quality_threshold=False)
        processing_time = time.time() - start_time
        
        stats = extractor.get_statistics(results)
        results_comparison[detector_type] = {
            'face_count': len(results),
            'processing_time': processing_time,
            'stats': stats
        }
        
        print(f"  Found {len(results)} faces in {processing_time:.3f}s")
        if stats:
            print(f"  Average sharpness: {stats.get('avg_sharpness', 0):.2f}")
            print(f"  High quality ratio: {stats.get('quality_ratio', 0):.2f}")
    
    return results_comparison

def create_quality_report(results: list, output_path: str = "quality_report.json"):
    """Create detailed quality assessment report"""
    if not results:
        print("No results to analyze")
        return
    
    report = {
        'total_faces': len(results),
        'quality_distribution': {
            'high_quality': 0,
            'low_quality': 0
        },
        'metrics': {
            'sharpness': [],
            'brightness': [],
            'contrast': [],
            'size': []
        },
        'detailed_results': []
    }
    
    for i, result in enumerate(results):
        metrics = result['quality_metrics']
        
        # Update distribution
        if metrics['is_high_quality']:
            report['quality_distribution']['high_quality'] += 1
        else:
            report['quality_distribution']['low_quality'] += 1
        
        # Collect metrics
        report['metrics']['sharpness'].append(metrics['sharpness'])
        report['metrics']['brightness'].append(metrics['brightness'])
        report['metrics']['contrast'].append(metrics['contrast'])
        report['metrics']['size'].append(metrics['size'])
        
        # Add detailed result
        detailed_result = {
            'face_id': i,
            'bbox': result['bbox'],
            'quality_metrics': metrics
        }
        
        if 'frame_index' in result:
            detailed_result['frame_index'] = result['frame_index']
            detailed_result['timestamp'] = result['timestamp']
        
        if 'image_path' in result:
            detailed_result['image_path'] = result['image_path']
        
        report['detailed_results'].append(detailed_result)
    
    # Calculate summary statistics
    for metric in report['metrics']:
        values = report['metrics'][metric]
        report['metrics'][metric] = {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Quality report saved to {output_path}")
    return report

def visualize_quality_metrics(results: list, output_dir: str = "visualizations"):
    """Create visualizations of quality metrics"""
    if not results:
        print("No results to visualize")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    metrics_data = {
        'sharpness': [r['quality_metrics']['sharpness'] for r in results],
        'brightness': [r['quality_metrics']['brightness'] for r in results],
        'contrast': [r['quality_metrics']['contrast'] for r in results],
        'size': [r['quality_metrics']['size'] for r in results]
    }
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Face Quality Metrics Distribution', fontsize=16)
    
    axes = axes.flatten()
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        axes[i].hist(values, bins=20, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{metric.capitalize()} Distribution')
        axes[i].set_xlabel(metric.capitalize())
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Quality vs metrics scatter plots
    high_quality = [r['quality_metrics']['is_high_quality'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Quality Assessment vs Metrics', fontsize=16)
    
    axes = axes.flatten()
    colors = ['red' if not hq else 'green' for hq in high_quality]
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        axes[i].scatter(range(len(values)), values, c=colors, alpha=0.6)
        axes[i].set_title(f'{metric.capitalize()} by Quality')
        axes[i].set_xlabel('Face Index')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='High Quality'),
        Patch(facecolor='red', label='Low Quality')
    ]
    fig.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_vs_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Face Detection Pipeline Demo')
    parser.add_argument('--mode', choices=['image', 'video', 'batch', 'compare'], 
                      required=True, help='Demo mode')
    parser.add_argument('--input', required=True, help='Input file or directory path')
    parser.add_argument('--detector', choices=['mtcnn', 'mediapipe'], 
                      default='mtcnn', help='Face detector to use')
    parser.add_argument('--batch-size', type=int, default=16, 
                      help='Batch size for video processing')
    parser.add_argument('--create-report', action='store_true', 
                      help='Create detailed quality report')
    parser.add_argument('--visualize', action='store_true', 
                      help='Create quality metric visualizations')
    
    args = parser.parse_args()
    
    results = []
    
    try:
        if args.mode == 'image':
            results = demo_single_image(args.input, args.detector)
        
        elif args.mode == 'video':
            results, stats = demo_video_batch(args.input, args.detector, args.batch_size)
        
        elif args.mode == 'batch':
            results, stats = demo_image_batch(args.input, args.detector)
        
        elif args.mode == 'compare':
            comparison = compare_detectors(args.input)
            print("\nComparison Results:")
            for detector, data in comparison.items():
                print(f"{detector}: {data['face_count']} faces, {data['processing_time']:.3f}s")
            return
        
        # Generate reports if requested
        if args.create_report and results:
            report_path = f"{args.mode}_{args.detector}_quality_report.json"
            create_quality_report(results, report_path)
        
        if args.visualize and results:
            viz_dir = f"{args.mode}_{args.detector}_visualizations"
            visualize_quality_metrics(results, viz_dir)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
