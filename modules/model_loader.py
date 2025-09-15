# File: modules/model_loader.py

import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


from ultralytics import YOLO

def load_yolo_model(model_path='yolov8n.pt'):
    model = YOLO(model_path)
    original_call = model.__call__

    def call_with_person_only(*args, **kwargs):
        # Run normal detection (with or without classes filter)
        results = original_call(*args, **kwargs)
        
        # Filter detections: keep only class 0 (person)
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                cls = boxes.cls.cpu() if hasattr(boxes, 'cls') else None
                if cls is not None:
                    mask = cls == 0
                    boxes.data = boxes.data[mask]
                    boxes.cls = boxes.cls[mask]
                    boxes.conf = boxes.conf[mask]
        return results

    model.__call__ = call_with_person_only

    return model



def init_tracker():
    """Initialize DeepSORT tracker."""
    tracker = DeepSort(max_age=30, n_init=1)
    return tracker
