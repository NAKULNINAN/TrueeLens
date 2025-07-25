import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import dlib
import torchvision.transforms as transforms
import mediapipe as mp
import os
import urllib.request
from typing import List, Tuple, Optional

class FaceExtractor:
    def __init__(self, detector_type="mtcnn"):
        self.detector_type = detector_type
        
        # Always initialize both detectors for robust extraction with more sensitive settings
        self.mtcnn = MTCNN(
            keep_all=True, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            min_face_size=20,  # Detect smaller faces
            thresholds=[0.6, 0.7, 0.7],  # More lenient thresholds
            factor=0.709  # Better scale factor
        )
        
        # Initialize OpenCV face detector as additional fallback
        self.opencv_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize MediaPipe components
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        
        # Initialize dlib face predictor
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        self._download_predictor_if_needed()
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.detector = dlib.get_frontal_face_detector()
    
    def _download_predictor_if_needed(self):
        if not os.path.exists(self.predictor_path):
            print("Downloading dlib face landmark predictor...")
            url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
            urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
            # Extract bz2 file
            import bz2
            with bz2.open("shape_predictor_68_face_landmarks.dat.bz2", 'rb') as f_in:
                with open(self.predictor_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove("shape_predictor_68_face_landmarks.dat.bz2")

    def extract_faces_mtcnn(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = self.mtcnn.detect(pil_image)
        
        faces = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    faces.append((face, (x1, y1, x2, y2)))
        return faces
    
    def extract_faces_mediapipe(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    faces.append((face, (x1, y1, x2, y2)))
        return faces
    
    def extract_faces_opencv(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract faces using OpenCV Haar cascade detector"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with more lenient parameters
        faces_rects = self.opencv_face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,  # More lenient
            minSize=(30, 30),  # Smaller minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        for (x, y, w, h) in faces_rects:
            # Extract face region
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                faces.append((face, (x, y, x+w, y+h)))
        
        return faces

    def extract_faces(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        if self.detector_type == "mtcnn":
            return self.extract_faces_mtcnn(frame)
        else:
            return self.extract_faces_mediapipe(frame)
    
    def extract_faces_robust(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract faces with robust fallback mechanism and image enhancement.
        
        This method tries multiple detection strategies in order:
        1. MTCNN on original frame
        2. MediaPipe on original frame
        3. OpenCV on original frame
        4. MTCNN on enhanced frame (histogram equalization)
        5. MediaPipe on enhanced frame
        6. OpenCV on enhanced frame
        
        Returns the results from the first successful detection.
        """
        # Try MTCNN first
        faces = self.extract_faces_mtcnn(frame)
        if len(faces) > 0:
            return faces
        
        # Try MediaPipe if MTCNN fails
        faces = self.extract_faces_mediapipe(frame)
        if len(faces) > 0:
            return faces
        
        # Try OpenCV if MediaPipe fails
        faces = self.extract_faces_opencv(frame)
        if len(faces) > 0:
            return faces
        
        # If no faces detected, try with enhanced frame
        enhanced_frame = self._enhance_frame(frame)
        
        # Try MTCNN on enhanced frame
        faces = self.extract_faces_mtcnn(enhanced_frame)
        if len(faces) > 0:
            # Map coordinates back to original frame
            return [(self._map_face_to_original(face, enhanced_frame, frame), bbox) for face, bbox in faces]
        
        # Try MediaPipe on enhanced frame
        faces = self.extract_faces_mediapipe(enhanced_frame)
        if len(faces) > 0:
            # Map coordinates back to original frame
            return [(self._map_face_to_original(face, enhanced_frame, frame), bbox) for face, bbox in faces]
        
        # Try OpenCV on enhanced frame
        faces = self.extract_faces_opencv(enhanced_frame)
        if len(faces) > 0:
            # Map coordinates back to original frame
            return [(self._map_face_to_original(face, enhanced_frame, frame), bbox) for face, bbox in faces]
        
        # Return empty list if no faces found
        return []
    
    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame quality to improve face detection.
        
        Applies histogram equalization to improve contrast and brightness.
        """
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel_enhanced = clahe.apply(l_channel)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_channel_enhanced, a_channel, b_channel])
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_frame
    
    def _map_face_to_original(self, face: np.ndarray, enhanced_frame: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Map face region from enhanced frame back to original frame.
        
        Since enhancement doesn't change coordinates, we can extract the same region from original frame.
        """
        # For simplicity, we assume the face coordinates are the same
        # In practice, we could do more sophisticated mapping if needed
        return face

    def align_face(self, face: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        
        if len(faces) == 0:
            return face
        
        shape = self.predictor(gray, faces[0])
        points = np.array([[p.x, p.y] for p in shape.parts()])
        
        # Get eye coordinates for alignment
        left_eye = np.mean(points[36:42], axis=0).astype(int)
        right_eye = np.mean(points[42:48], axis=0).astype(int)
        
        # Calculate angle for alignment
        angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        angle_degrees = np.degrees(angle)
        
        # Calculate center for rotation
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        
        # Create rotation matrix and apply transformation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1)
        aligned_face = cv2.warpAffine(face, rotation_matrix, (face.shape[1], face.shape[0]))
        
        return aligned_face

    def assess_quality(self, face: np.ndarray) -> dict:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Sharpness assessment using Laplacian variance
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness assessment
        brightness = np.mean(gray)
        
        # Contrast assessment
        contrast = gray.std()
        
        # Size assessment
        height, width = face.shape[:2]
        size_score = min(height, width)
        
        # Quality scores - More lenient thresholds
        quality_metrics = {
            'sharpness': sharpness,
            'brightness': brightness,
            'contrast': contrast,
            'size': size_score,
            'is_high_quality': (
                sharpness > 10 and  # Much more lenient
                20 < brightness < 250 and  # Wider range
                contrast > 5 and  # Lower threshold
                size_score >= 32  # Smaller minimum size
            )
        }
        
        return quality_metrics

    def process_frame(self, frame: np.ndarray, model=None, apply_alignment: bool = True, quality_threshold: bool = True) -> List[dict]:
        faces_data = self.extract_faces(frame)
        results = []
        
        for face, bbox in faces_data:
            result = {
                'original_face': face,
                'bbox': bbox,
                'quality_metrics': self.assess_quality(face)
            }
            
            # Skip low quality faces if threshold is enabled
            if quality_threshold and not result['quality_metrics']['is_high_quality']:
                continue
                
            # Apply face alignment if requested
            if apply_alignment:
                aligned_face = self.align_face(face)
                result['aligned_face'] = aligned_face
            else:
                result['aligned_face'] = face
            
            # Preprocess for model if model is provided
            if model is not None:
                preprocessed_face = self.preprocess(result['aligned_face'])
                result['preprocessed'] = preprocessed_face
                
                # Run model inference
                with torch.no_grad():
                    output = model(preprocessed_face)
                    result['model_output'] = output
            
            results.append(result)
        
        return results

    def process_video_batch(self, video_path: str, model=None, batch_size: int = 32, frame_skip: int = 1) -> List[dict]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        all_results = []
        frame_count = 0
        batch_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                batch_frames.append((frame, frame_count, frame_count / fps))
                
                if len(batch_frames) >= batch_size:
                    batch_results = self._process_frame_batch(batch_frames, model)
                    all_results.extend(batch_results)
                    batch_frames = []
            
            frame_count += 1
        
        # Process remaining frames
        if batch_frames:
            batch_results = self._process_frame_batch(batch_frames, model)
            all_results.extend(batch_results)
        
        cap.release()
        return all_results
    
    def _process_frame_batch(self, batch_frames: List[Tuple[np.ndarray, int, float]], model=None) -> List[dict]:
        batch_results = []
        
        for frame, frame_idx, timestamp in batch_frames:
            frame_results = self.process_frame(frame, model)
            
            for result in frame_results:
                result['frame_index'] = frame_idx
                result['timestamp'] = timestamp
                batch_results.append(result)
        
        return batch_results

    def process_image_batch(self, image_paths: List[str], model=None) -> List[dict]:
        all_results = []
        
        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            image_results = self.process_frame(image, model)
            
            for result in image_results:
                result['image_path'] = image_path
                result['image_index'] = i
                all_results.append(result)
        
        return all_results

    def preprocess(self, face: np.ndarray) -> torch.Tensor:
        # Resize face to standard size
        face_resized = cv2.resize(face, (224, 224))
        
        # Convert to PIL Image and apply transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        pil_face = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
        face_tensor = transform(pil_face).unsqueeze(0)
        
        return face_tensor
    
    def save_faces(self, results: List[dict], output_dir: str, save_aligned: bool = True) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            # Save original face
            original_path = os.path.join(output_dir, f"face_{i}_original.jpg")
            cv2.imwrite(original_path, result['original_face'])
            
            # Save aligned face if available
            if save_aligned and 'aligned_face' in result:
                aligned_path = os.path.join(output_dir, f"face_{i}_aligned.jpg")
                cv2.imwrite(aligned_path, result['aligned_face'])
    
    def get_statistics(self, results: List[dict]) -> dict:
        if not results:
            return {}
        
        total_faces = len(results)
        high_quality_faces = sum(1 for r in results if r['quality_metrics']['is_high_quality'])
        
        avg_sharpness = np.mean([r['quality_metrics']['sharpness'] for r in results])
        avg_brightness = np.mean([r['quality_metrics']['brightness'] for r in results])
        avg_contrast = np.mean([r['quality_metrics']['contrast'] for r in results])
        
        return {
            'total_faces': total_faces,
            'high_quality_faces': high_quality_faces,
            'quality_ratio': high_quality_faces / total_faces if total_faces > 0 else 0,
            'avg_sharpness': avg_sharpness,
            'avg_brightness': avg_brightness,
            'avg_contrast': avg_contrast
        }

