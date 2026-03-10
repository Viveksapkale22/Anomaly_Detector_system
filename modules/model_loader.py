# File: modules/model_loader.py

import os
import sys
import contextlib
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


from ultralytics import YOLO

def load_yolo_model(model_path='yolov8n.pt'):
    model = YOLO(model_path)
    original_call = model.__call__

    def call_with_person_only(*args, **kwargs):
        # Force person-only detection and suppress YOLO console output.
        # This prevents printing "3 persons, 1 bowl, 2 bananas" etc.
        kwargs['classes'] = [0]
        kwargs['verbose'] = False
        kwargs['show'] = False
        kwargs['hide_labels'] = True
        kwargs['hide_conf'] = True
        kwargs['save'] = False

        # Some versions still print to stdout; suppress it entirely.
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            results = original_call(*args, **kwargs)

        # Filter detections: keep only class 0 (person) as a safety net
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
