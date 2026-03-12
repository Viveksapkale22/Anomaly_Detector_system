# File: modules/model_loader.py

import os
import sys
import contextlib

import numpy as np
# Fix for numpy>=1.24: deep_sort_realtime still uses np.float (deprecated)
if not hasattr(np, "float"):
    np.float = float

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


from ultralytics import YOLO

def load_yolo_model(model_path='best.pt'):
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
    """Initialize DeepSORT tracker.

    Tuned for more stable IDs and fewer short-lived/duplicate tracks.
    """
    # Some deep_sort_realtime versions do not expose max_iou_distance as an init
    # argument. Use introspection to pick a supported signature.
    from inspect import signature

    sig = signature(DeepSort.__init__)
    kwargs = {"max_age": 60, "n_init": 3}
    if "max_iou_distance" in sig.parameters:
        kwargs["max_iou_distance"] = 0.8

    tracker = DeepSort(**kwargs)
    return tracker
