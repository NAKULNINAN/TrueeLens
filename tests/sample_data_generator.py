"""
Sample Data Generator for Media Forensics Tests
Generate sample media files for testing.
"""

import os
from PIL import Image
import numpy as np
import cv2


def generate_sample_images(output_dir, num_images=10):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_images} sample images...")
    for i in range(num_images):
        # Create a sample RGB image with random patterns
        array = np.random.randint(256, size=(256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(array)
        img.save(os.path.join(output_dir, f"sample_image_{i}.jpg"))
        print(f"Generated sample_image_{i}.jpg")


def generate_sample_videos(output_dir, num_videos=5):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_videos} sample videos...")
    for i in range(num_videos):
        video_path = os.path.join(output_dir, f"sample_video_{i}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (320, 240))

        for frame_num in range(100):  # 5 seconds at 20 fps
            frame = np.random.randint(256, size=(240, 320, 3), dtype=np.uint8)
            video_writer.write(frame)
        video_writer.release()
        print(f"Generated sample_video_{i}.avi")


def main():
    sample_images_dir = os.path.join('tests', 'test_data', 'images')
    sample_videos_dir = os.path.join('tests', 'test_data', 'videos')

    generate_sample_images(sample_images_dir)
    generate_sample_videos(sample_videos_dir)


if __name__ == '__main__':
    main()

