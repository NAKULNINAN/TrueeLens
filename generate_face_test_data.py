#!/usr/bin/env python3
"""
Generate realistic test data with faces for deepfake detection testing
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
import requests
from urllib.parse import urlparse

def create_synthetic_face_image(width=512, height=512, save_path=None):
    """Create a synthetic face-like image for testing"""
    
    # Create base image
    img = np.ones((height, width, 3), dtype=np.uint8) * 220  # Light background
    
    # Face oval
    center_x, center_y = width // 2, height // 2
    face_width, face_height = width // 3, height // 2.5
    
    # Draw face shape
    cv2.ellipse(img, (center_x, center_y), (int(face_width), int(face_height)), 
                0, 0, 360, (200, 180, 160), -1)
    
    # Eyes
    eye_y = int(center_y - face_height // 4)
    left_eye_x = int(center_x - face_width // 3)
    right_eye_x = int(center_x + face_width // 3)
    
    # Eye whites
    cv2.ellipse(img, (left_eye_x, eye_y), (25, 15), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (right_eye_x, eye_y), (25, 15), 0, 0, 360, (255, 255, 255), -1)
    
    # Pupils
    cv2.circle(img, (left_eye_x, eye_y), 8, (50, 50, 50), -1)
    cv2.circle(img, (right_eye_x, eye_y), 8, (50, 50, 50), -1)
    
    # Nose
    nose_points = np.array([
        [center_x - 8, center_y - 10],
        [center_x + 8, center_y - 10],
        [center_x, center_y + 20]
    ], np.int32)
    cv2.fillPoly(img, [nose_points], (180, 160, 140))
    
    # Mouth
    mouth_y = int(center_y + face_height // 3)
    cv2.ellipse(img, (center_x, mouth_y), (30, 12), 0, 0, 180, (150, 100, 100), 3)
    
    # Add some texture/noise to make it more realistic
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    if save_path:
        cv2.imwrite(save_path, img)
        print(f"Created synthetic face image: {save_path}")
    
    return img

def download_sample_face_image(url, save_path):
    """Download a sample face image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded face image: {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def create_face_test_dataset():
    """Create a comprehensive test dataset with faces"""
    
    # Create directories
    test_dir = "tests/test_data"
    images_dir = os.path.join(test_dir, "images_with_faces")
    videos_dir = os.path.join(test_dir, "videos_with_faces")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    print("Creating face test dataset...")
    
    # Create synthetic face images
    for i in range(5):
        img_path = os.path.join(images_dir, f"synthetic_face_{i}.jpg")
        create_synthetic_face_image(save_path=img_path)
    
    # Create different sized face images
    sizes = [(256, 256), (512, 512), (720, 480), (1024, 768)]
    for i, (w, h) in enumerate(sizes):
        img_path = os.path.join(images_dir, f"face_size_{w}x{h}.jpg")
        create_synthetic_face_image(width=w, height=h, save_path=img_path)
    
    # Create a simple test video with a face
    create_test_video_with_face(os.path.join(videos_dir, "test_face_video.mp4"))
    
    print(f"Test dataset created in {test_dir}")
    print(f"Images with faces: {images_dir}")
    print(f"Videos with faces: {videos_dir}")

def create_test_video_with_face(output_path, duration_seconds=5, fps=10):
    """Create a test video with a moving synthetic face"""
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    for frame_num in range(total_frames):
        # Create background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 100
        
        # Calculate face position (moving across screen)
        progress = frame_num / total_frames
        face_x = int(100 + (width - 200) * progress)
        face_y = height // 2
        
        # Draw moving face
        face_size = 80
        
        # Face
        cv2.circle(frame, (face_x, face_y), face_size, (200, 180, 160), -1)
        
        # Eyes
        eye_offset = face_size // 3
        cv2.circle(frame, (face_x - eye_offset, face_y - 20), 12, (255, 255, 255), -1)
        cv2.circle(frame, (face_x + eye_offset, face_y - 20), 12, (255, 255, 255), -1)
        cv2.circle(frame, (face_x - eye_offset, face_y - 20), 6, (50, 50, 50), -1)
        cv2.circle(frame, (face_x + eye_offset, face_y - 20), 6, (50, 50, 50), -1)
        
        # Nose
        cv2.circle(frame, (face_x, face_y), 8, (180, 160, 140), -1)
        
        # Mouth
        cv2.ellipse(frame, (face_x, face_y + 25), (20, 10), 0, 0, 180, (150, 100, 100), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video with face: {output_path}")

if __name__ == "__main__":
    create_face_test_dataset()
