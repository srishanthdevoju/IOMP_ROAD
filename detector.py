"""
YOLO-based Object Detector for Road Safety System
Detects: cars, pedestrians, trucks, buses, motorcycles, bicycles
"""

from ultralytics import YOLO
import numpy as np
from config import YOLO_MODEL, TARGET_CLASSES, CONFIDENCE_THRESHOLD


class Detector:
    def __init__(self, model_path: str = YOLO_MODEL):
        """Initialize YOLO detector."""
        self.model = YOLO(model_path)
        self.target_classes = TARGET_CLASSES
        self.confidence_threshold = CONFIDENCE_THRESHOLD
    
    def detect(self, frame: np.ndarray) -> list:
        """
        Detect objects in a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of detections, each containing:
            {
                'bbox': [x1, y1, x2, y2],
                'center': (cx, cy),
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'area': int
            }
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Filter by target classes and confidence
            if class_id not in self.target_classes:
                continue
            if confidence < self.confidence_threshold:
                continue
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'center': (cx, cy),
                'class_id': class_id,
                'class_name': self.target_classes[class_id],
                'confidence': confidence,
                'area': area
            })
        
        return detections
