"""
Face Recognition Service Core Logic - Using OpenCV
Lightweight face recognition without heavy ML dependencies.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

# Cache directory for storing face data
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_FILE = os.path.join(CACHE_DIR, 'face_embeddings.json')

# Load OpenCV face cascade for detection
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


class FaceRecognitionService:
    """
    Service for face recognition operations.
    Uses OpenCV for face detection and histogram-based features for recognition.
    """
    
    def __init__(self):
        self.embeddings_cache: Dict[int, np.ndarray] = {}
        self._load_cache()
        logger.info("FaceRecognitionService initialized with OpenCV")
    
    def _load_cache(self):
        """Load cached face embeddings from disk."""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    for employee_id, embedding_list in data.items():
                        self.embeddings_cache[int(employee_id)] = np.array(embedding_list)
                logger.info(f"Loaded {len(self.embeddings_cache)} face embeddings from cache")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                self.embeddings_cache = {}
        else:
            logger.info("No existing cache found, starting fresh")
    
    def _save_cache(self):
        """Save face embeddings to disk."""
        try:
            data = {
                str(employee_id): embedding.tolist()
                for employee_id, embedding in self.embeddings_cache.items()
            }
            with open(CACHE_FILE, 'w') as f:
                json.dump(data, f)
            logger.info(f"Saved {len(data)} face embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in image and return face region.
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            
        Returns:
            Tuple (x, y, w, h) of face region or None
        """
        # Convert RGB to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) == 0:
            return None
        
        # Return the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest_face)
    
    def _local_binary_pattern(self, image: np.ndarray) -> np.ndarray:
        """Compute Local Binary Pattern for texture analysis."""
        height, width = image.shape
        lbp = np.zeros((height-2, width-2), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        
        return lbp
    
    def get_face_encoding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face feature vector using histogram-based approach.
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            
        Returns:
            Feature vector as numpy array or None if no face detected
        """
        face_region = self.detect_face(image)
        
        if face_region is None:
            logger.warning("No face detected in image")
            return None
        
        x, y, w, h = face_region
        
        # Extract face region
        face_img = image[y:y+h, x:x+w]
        
        # Convert to grayscale
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        
        # Resize to standard size
        face_resized = cv2.resize(face_gray, (100, 100))
        
        # Extract histogram features
        # Use multiple regions for better accuracy
        h, w = face_resized.shape
        features = []
        
        # Divide face into 4x4 grid and compute histogram for each region
        grid_size = 4
        cell_h, cell_w = h // grid_size, w // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell = face_resized[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                hist = cv2.calcHist([cell], [0], None, [16], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                features.extend(hist)
        
        # Add LBP histogram for texture features
        lbp = self._local_binary_pattern(face_resized)
        lbp_hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [32], [0, 32])
        lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
        features.extend(lbp_hist)
        
        return np.array(features, dtype=np.float32)
    
    def store_embedding(self, employee_id: int, embedding: np.ndarray):
        """Store face embedding for an employee."""
        self.embeddings_cache[employee_id] = embedding
        self._save_cache()
        logger.info(f"Stored embedding for employee {employee_id}")
    
    def get_embedding(self, employee_id: int) -> Optional[np.ndarray]:
        """Retrieve stored embedding for an employee."""
        return self.embeddings_cache.get(employee_id)
    
    def remove_embedding(self, employee_id: int):
        """Remove stored embedding for an employee."""
        if employee_id in self.embeddings_cache:
            del self.embeddings_cache[employee_id]
            self._save_cache()
            logger.info(f"Removed embedding for employee {employee_id}")
    
    def get_all_enrolled_employees(self) -> List[int]:
        """Get list of all enrolled employee IDs."""
        return list(self.embeddings_cache.keys())
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two face embeddings using cosine similarity."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        confidence = (cosine_sim + 1) / 2
        
        return float(confidence)
    
    def compare_with_employee(self, employee_id: int, captured_embedding: np.ndarray, threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        """Compare captured face against a specific employee."""
        stored_embedding = self.get_embedding(employee_id)
        
        if stored_embedding is None:
            logger.warning(f"No stored embedding found for employee {employee_id}")
            return None
        
        confidence = self.calculate_similarity(stored_embedding, captured_embedding)
        
        if confidence >= threshold:
            return {'employee_id': employee_id, 'confidence': confidence, 'matched': True}
        
        return None
    
    def compare_against_all(self, captured_embedding: np.ndarray, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Compare captured face against all enrolled employees."""
        matches = []
        
        for employee_id, stored_embedding in self.embeddings_cache.items():
            confidence = self.calculate_similarity(stored_embedding, captured_embedding)
            
            if confidence >= threshold:
                matches.append({'employee_id': employee_id, 'confidence': confidence, 'matched': True})
        
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches
    
    def is_enrolled(self, employee_id: int) -> bool:
        """Check if an employee has face enrollment."""
        return employee_id in self.embeddings_cache
