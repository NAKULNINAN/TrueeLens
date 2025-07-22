"""Comprehensive Sample Data Generator for Media Forensics Tests
Generate diverse sample media files including images, videos, and specialized test cases.
"""

import os
import json
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import cv2
from datetime import datetime
import random
import hashlib


class ComprehensiveSampleGenerator:
    """Generate comprehensive sample data for testing media forensics"""
    
    def __init__(self):
        self.generated_files = []
        self.metadata = {
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0',
            'files': []
        }
    
    def generate_realistic_images(self, output_dir, count=10):
        """Generate realistic test images with various characteristics"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating {count} realistic sample images...")
        
        image_types = [
            self._create_natural_scene,
            self._create_portrait_like,
            self._create_geometric_pattern,
            self._create_textured_image,
            self._create_high_contrast_image
        ]
        
        for i in range(count):
            # Vary image characteristics
            size = random.choice([(224, 224), (512, 512), (1024, 768), (800, 600)])
            image_type = random.choice(image_types)
            
            # Create base image
            img = image_type(size)
            
            # Add random modifications
            img = self._apply_random_effects(img)
            
            # Save with metadata
            filename = f"realistic_image_{i:03d}.jpg"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath, quality=random.randint(85, 98))
            
            # Record metadata
            self._record_file_metadata(filepath, 'image', {
                'type': image_type.__name__,
                'size': size,
                'modifications_applied': True
            })
            
            print(f"Generated {filename}")
    
    def generate_ai_like_artifacts(self, output_dir, count=5):
        """Generate images with AI-generation-like artifacts"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating {count} AI-artifact images...")
        
        for i in range(count):
            size = (512, 512)
            
            # Create base image
            img = self._create_natural_scene(size)
            
            # Add AI-like artifacts
            if i % 3 == 0:
                img = self._add_stylegan_artifacts(img)
                artifact_type = 'stylegan'
            elif i % 3 == 1:
                img = self._add_dalle_artifacts(img)
                artifact_type = 'dalle'
            else:
                img = self._add_midjourney_artifacts(img)
                artifact_type = 'midjourney'
            
            filename = f"ai_artifact_{artifact_type}_{i:03d}.jpg"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath, quality=random.randint(80, 95))
            
            self._record_file_metadata(filepath, 'ai_artifact', {
                'artifact_type': artifact_type,
                'intended_for_detection': True
            })
            
            print(f"Generated {filename}")
    
    def generate_duplicate_sets(self, output_dir, sets=3):
        """Generate sets of duplicate and near-duplicate images"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating {sets} duplicate sets...")
        
        for set_id in range(sets):
            # Create original image
            original = self._create_natural_scene((400, 400))
            
            # Save original
            original_filename = f"duplicate_set_{set_id:02d}_original.jpg"
            original_path = os.path.join(output_dir, original_filename)
            original.save(original_path, quality=95)
            
            # Create exact duplicate
            exact_filename = f"duplicate_set_{set_id:02d}_exact.jpg"
            exact_path = os.path.join(output_dir, exact_filename)
            original.save(exact_path, quality=95)
            
            # Create near duplicates with small modifications
            modifications = [
                ('resized', lambda img: img.resize((380, 380))),
                ('compressed', lambda img: self._compress_image(img, 70)),
                ('brightened', lambda img: ImageEnhance.Brightness(img).enhance(1.1)),
                ('slightly_rotated', lambda img: img.rotate(1, expand=False)),
                ('cropped', lambda img: img.crop((10, 10, 390, 390)))
            ]
            
            for mod_name, mod_func in modifications:
                modified = mod_func(original.copy())
                mod_filename = f"duplicate_set_{set_id:02d}_{mod_name}.jpg"
                mod_path = os.path.join(output_dir, mod_filename)
                modified.save(mod_path, quality=random.randint(85, 95))
                
                self._record_file_metadata(mod_path, 'duplicate', {
                    'set_id': set_id,
                    'modification': mod_name,
                    'original': original_filename
                })
            
            self._record_file_metadata(original_path, 'duplicate_original', {'set_id': set_id})
            self._record_file_metadata(exact_path, 'duplicate_exact', {'set_id': set_id})
            
            print(f"Generated duplicate set {set_id}")
    
    def generate_test_videos(self, output_dir, count=5):
        """Generate diverse test videos with various characteristics"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating {count} test videos...")
        
        video_types = [
            ('static_scene', self._create_static_video),
            ('moving_objects', self._create_moving_objects_video),
            ('color_changing', self._create_color_changing_video),
            ('face_like', self._create_face_like_video),
            ('complex_motion', self._create_complex_motion_video)
        ]
        
        for i in range(count):
            video_type, create_func = video_types[i % len(video_types)]
            
            filename = f"test_video_{video_type}_{i:03d}.mp4"
            filepath = os.path.join(output_dir, filename)
            
            # Video parameters
            duration = random.randint(3, 8)  # 3-8 seconds
            fps = random.choice([15, 20, 24, 30])
            resolution = random.choice([(320, 240), (640, 480), (720, 576)])
            
            create_func(filepath, duration, fps, resolution)
            
            self._record_file_metadata(filepath, 'video', {
                'type': video_type,
                'duration': duration,
                'fps': fps,
                'resolution': resolution
            })
            
            print(f"Generated {filename}")
    
    def generate_edge_cases(self, output_dir):
        """Generate edge cases for robust testing"""
        os.makedirs(output_dir, exist_ok=True)
        print("Generating edge case files...")
        
        edge_cases = [
            ('tiny_image', lambda: self._create_natural_scene((32, 32))),
            ('large_image', lambda: self._create_natural_scene((2048, 1536))),
            ('square_image', lambda: self._create_natural_scene((512, 512))),
            ('wide_image', lambda: self._create_natural_scene((1920, 480))),
            ('tall_image', lambda: self._create_natural_scene((480, 1920))),
            ('grayscale', lambda: self._create_natural_scene((400, 400)).convert('L')),
            ('high_noise', lambda: self._add_noise(self._create_natural_scene((400, 400)), 0.3)),
            ('low_contrast', lambda: ImageEnhance.Contrast(self._create_natural_scene((400, 400))).enhance(0.3)),
            ('high_saturation', lambda: ImageEnhance.Color(self._create_natural_scene((400, 400))).enhance(2.0))
        ]
        
        for case_name, create_func in edge_cases:
            img = create_func()
            filename = f"edge_case_{case_name}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            if img.mode == 'L':  # Grayscale
                img.save(filepath, quality=90)
            else:
                img.save(filepath, quality=90)
            
            self._record_file_metadata(filepath, 'edge_case', {'case_type': case_name})
            print(f"Generated edge case: {filename}")
    
    def _create_natural_scene(self, size):
        """Create a natural-looking scene"""
        img = Image.new('RGB', size, color=(135, 206, 235))  # Sky blue background
        draw = ImageDraw.Draw(img)
        
        # Add ground
        ground_y = size[1] // 2
        draw.rectangle([(0, ground_y), size], fill=(34, 139, 34))  # Forest green
        
        # Add some random elements
        for _ in range(random.randint(3, 8)):
            # Random shapes
            x1, y1 = random.randint(0, size[0]//2), random.randint(0, size[1]//2)
            x2, y2 = x1 + random.randint(20, 100), y1 + random.randint(20, 100)
            color = tuple(random.randint(0, 255) for _ in range(3))
            
            if random.choice([True, False]):
                draw.ellipse([(x1, y1), (x2, y2)], fill=color)
            else:
                draw.rectangle([(x1, y1), (x2, y2)], fill=color)
        
        return img
    
    def _create_portrait_like(self, size):
        """Create a portrait-like image"""
        img = Image.new('RGB', size, color=(245, 245, 220))  # Beige background
        draw = ImageDraw.Draw(img)
        
        # Face-like oval
        center_x, center_y = size[0] // 2, size[1] // 2
        face_w, face_h = size[0] // 3, size[1] // 2
        draw.ellipse([
            (center_x - face_w//2, center_y - face_h//2),
            (center_x + face_w//2, center_y + face_h//2)
        ], fill=(255, 220, 177))  # Skin tone
        
        # Eyes
        eye_y = center_y - face_h//4
        eye_offset = face_w//4
        for eye_x in [center_x - eye_offset, center_x + eye_offset]:
            draw.ellipse([(eye_x-8, eye_y-8), (eye_x+8, eye_y+8)], fill=(0, 0, 0))
        
        # Mouth
        mouth_y = center_y + face_h//6
        draw.arc([(center_x-20, mouth_y-10), (center_x+20, mouth_y+10)], 0, 180, fill=(255, 0, 0))
        
        return img
    
    def _create_geometric_pattern(self, size):
        """Create geometric patterns"""
        img = Image.new('RGB', size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Create grid pattern
        grid_size = 20
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for x in range(0, size[0], grid_size):
            for y in range(0, size[1], grid_size):
                color = colors[((x//grid_size) + (y//grid_size)) % len(colors)]
                draw.rectangle([(x, y), (x+grid_size-1, y+grid_size-1)], fill=color)
        
        return img
    
    def _create_textured_image(self, size):
        """Create textured image"""
        # Start with noise
        array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        img = Image.fromarray(array)
        
        # Apply blur to create texture
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Enhance contrast
        img = ImageEnhance.Contrast(img).enhance(1.5)
        
        return img
    
    def _create_high_contrast_image(self, size):
        """Create high contrast image"""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Alternating black and white stripes
        stripe_width = size[0] // 10
        for i in range(0, size[0], stripe_width * 2):
            color = (255, 255, 255) if (i // stripe_width) % 2 == 0 else (0, 0, 0)
            draw.rectangle([(i, 0), (i + stripe_width, size[1])], fill=color)
        
        return img
    
    def _apply_random_effects(self, img):
        """Apply random effects to image"""
        effects = [
            lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.8, 1.2)),
            lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.8, 1.2)),
            lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.8, 1.2)),
            lambda x: x.filter(ImageFilter.BLUR) if random.random() < 0.3 else x,
            lambda x: x.filter(ImageFilter.SHARPEN) if random.random() < 0.3 else x
        ]
        
        # Apply 1-3 random effects
        num_effects = random.randint(1, 3)
        selected_effects = random.sample(effects, num_effects)
        
        for effect in selected_effects:
            img = effect(img)
        
        return img
    
    def _add_stylegan_artifacts(self, img):
        """Add StyleGAN-like artifacts"""
        # Add periodic patterns
        array = np.array(img)
        height, width = array.shape[:2]
        
        # Add high-frequency artifacts
        for i in range(0, height, 16):
            for j in range(0, width, 16):
                if random.random() < 0.3:
                    array[i:i+8, j:j+8] += np.random.randint(-20, 20, (8, 8, 3))
        
        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)
    
    def _add_dalle_artifacts(self, img):
        """Add DALL-E-like artifacts"""
        # Add patch-based uniformity
        array = np.array(img)
        height, width = array.shape[:2]
        patch_size = 32
        
        for i in range(0, height-patch_size, patch_size):
            for j in range(0, width-patch_size, patch_size):
                if random.random() < 0.4:
                    patch = array[i:i+patch_size, j:j+patch_size]
                    avg_color = np.mean(patch, axis=(0, 1))
                    array[i:i+patch_size, j:j+patch_size] = avg_color
        
        return Image.fromarray(array.astype(np.uint8))
    
    def _add_midjourney_artifacts(self, img):
        """Add Midjourney-like artifacts"""
        # Enhance artistic effects
        img = ImageEnhance.Color(img).enhance(1.3)  # High saturation
        img = img.filter(ImageFilter.SMOOTH_MORE)   # Smooth gradients
        
        # Add slight gaussian blur for painterly effect
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img
    
    def _add_noise(self, img, intensity=0.1):
        """Add noise to image"""
        array = np.array(img)
        noise = np.random.normal(0, intensity * 255, array.shape).astype(np.int16)
        array = array.astype(np.int16) + noise
        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)
    
    def _compress_image(self, img, quality):
        """Compress image with specified quality"""
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
    
    def _create_static_video(self, output_path, duration, fps, resolution):
        """Create video with static scene"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        
        # Create single frame
        frame = np.random.randint(0, 255, (*resolution[::-1], 3), dtype=np.uint8)
        
        # Write same frame multiple times
        for _ in range(duration * fps):
            out.write(frame)
        
        out.release()
    
    def _create_moving_objects_video(self, output_path, duration, fps, resolution):
        """Create video with moving objects"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        
        total_frames = duration * fps
        
        for frame_num in range(total_frames):
            # Create background
            frame = np.full((*resolution[::-1], 3), (50, 100, 150), dtype=np.uint8)
            
            # Add moving circle
            x = int((frame_num / total_frames) * resolution[0])
            y = resolution[1] // 2
            cv2.circle(frame, (x, y), 20, (255, 255, 0), -1)
            
            # Add moving rectangle
            rect_x = int(((total_frames - frame_num) / total_frames) * resolution[0])
            rect_y = resolution[1] // 4
            cv2.rectangle(frame, (rect_x-15, rect_y-15), (rect_x+15, rect_y+15), (255, 0, 255), -1)
            
            out.write(frame)
        
        out.release()
    
    def _create_color_changing_video(self, output_path, duration, fps, resolution):
        """Create video with changing colors"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        
        total_frames = duration * fps
        
        for frame_num in range(total_frames):
            # Create gradient background that changes over time
            t = frame_num / total_frames
            
            r = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * t)))
            g = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * t + 2 * np.pi / 3)))
            b = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * t + 4 * np.pi / 3)))
            
            frame = np.full((*resolution[::-1], 3), (b, g, r), dtype=np.uint8)
            out.write(frame)
        
        out.release()
    
    def _create_face_like_video(self, output_path, duration, fps, resolution):
        """Create video with face-like content"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        
        total_frames = duration * fps
        
        for frame_num in range(total_frames):
            # Background
            frame = np.full((*resolution[::-1], 3), (200, 200, 200), dtype=np.uint8)
            
            # Face outline
            center_x, center_y = resolution[0] // 2, resolution[1] // 2
            face_size = min(resolution) // 3
            
            cv2.ellipse(frame, (center_x, center_y), (face_size, int(face_size * 1.2)), 
                       0, 0, 360, (255, 220, 177), -1)
            
            # Eyes (blinking animation)
            eye_open = (frame_num % 60) < 50  # Blink every 3 seconds
            eye_y = center_y - face_size // 3
            
            if eye_open:
                cv2.circle(frame, (center_x - face_size//3, eye_y), 8, (0, 0, 0), -1)
                cv2.circle(frame, (center_x + face_size//3, eye_y), 8, (0, 0, 0), -1)
            else:
                cv2.line(frame, (center_x - face_size//3 - 8, eye_y), 
                        (center_x - face_size//3 + 8, eye_y), (0, 0, 0), 2)
                cv2.line(frame, (center_x + face_size//3 - 8, eye_y), 
                        (center_x + face_size//3 + 8, eye_y), (0, 0, 0), 2)
            
            # Mouth
            mouth_y = center_y + face_size // 3
            cv2.ellipse(frame, (center_x, mouth_y), (20, 10), 0, 0, 180, (100, 50, 50), 2)
            
            out.write(frame)
        
        out.release()
    
    def _create_complex_motion_video(self, output_path, duration, fps, resolution):
        """Create video with complex motion patterns"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        
        total_frames = duration * fps
        
        for frame_num in range(total_frames):
            # Dynamic background
            frame = np.zeros((*resolution[::-1], 3), dtype=np.uint8)
            
            t = frame_num / total_frames
            
            # Multiple moving objects with different patterns
            for i in range(5):
                # Circular motion
                angle = 2 * np.pi * t + i * np.pi / 2.5
                radius = 50 + i * 20
                center_x = resolution[0] // 2 + int(radius * np.cos(angle))
                center_y = resolution[1] // 2 + int(radius * np.sin(angle))
                
                color = (
                    int(255 * (0.5 + 0.5 * np.sin(angle + i))),
                    int(255 * (0.5 + 0.5 * np.cos(angle + i))),
                    int(255 * (0.5 + 0.5 * np.sin(angle * 2 + i)))
                )
                
                if 0 <= center_x < resolution[0] and 0 <= center_y < resolution[1]:
                    cv2.circle(frame, (center_x, center_y), 10 + i * 2, color, -1)
            
            out.write(frame)
        
        out.release()
    
    def _record_file_metadata(self, filepath, file_type, extra_info):
        """Record metadata about generated file"""
        file_info = {
            'path': filepath,
            'filename': os.path.basename(filepath),
            'type': file_type,
            'size_bytes': os.path.getsize(filepath),
            'generated_at': datetime.now().isoformat(),
            **extra_info
        }
        
        # Add file hash for verification
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
            file_info['md5_hash'] = file_hash
        
        self.metadata['files'].append(file_info)
        self.generated_files.append(filepath)
    
    def save_metadata(self, output_dir):
        """Save metadata about all generated files"""
        metadata_path = os.path.join(output_dir, 'generated_files_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")
    
    def generate_all(self, base_output_dir='tests/test_data'):
        """Generate all types of sample data"""
        print("Starting comprehensive sample data generation...")
        
        # Create directory structure
        images_dir = os.path.join(base_output_dir, 'images')
        videos_dir = os.path.join(base_output_dir, 'videos')
        ai_artifacts_dir = os.path.join(base_output_dir, 'ai_artifacts')
        duplicates_dir = os.path.join(base_output_dir, 'duplicates')
        edge_cases_dir = os.path.join(base_output_dir, 'edge_cases')
        
        # Generate all sample types
        self.generate_realistic_images(images_dir, 15)
        self.generate_ai_like_artifacts(ai_artifacts_dir, 10)
        self.generate_duplicate_sets(duplicates_dir, 5)
        self.generate_edge_cases(edge_cases_dir)
        self.generate_test_videos(videos_dir, 8)
        
        # Save metadata
        self.save_metadata(base_output_dir)
        
        print(f"\nGeneration complete!")
        print(f"Total files generated: {len(self.generated_files)}")
        print(f"Output directory: {base_output_dir}")
        
        return self.generated_files


def main():
    """Main function to generate comprehensive sample data"""
    generator = ComprehensiveSampleGenerator()
    generated_files = generator.generate_all()
    
    print("\n" + "=" * 50)
    print("SAMPLE DATA GENERATION COMPLETE")
    print("=" * 50)
    print(f"Generated {len(generated_files)} files")
    print("\nGenerated categories:")
    print("✓ Realistic images (15 files)")
    print("✓ AI artifact images (10 files)")
    print("✓ Duplicate image sets (5 sets, ~30 files)")
    print("✓ Edge case images (9 files)")
    print("✓ Test videos (8 files)")
    print("✓ Comprehensive metadata file")
    print("\nFiles ready for comprehensive testing!")


if __name__ == '__main__':
    main()
