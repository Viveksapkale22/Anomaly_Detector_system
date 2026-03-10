# File: modules/detection.py

import os

import cv2
import time
import threading
from ultralytics import YOLO

from modules.utils import count_persons, boxes_intersect, play_alert
from modules.utils import send_alert_email
from modules.face_analysis import build_model, analyze_gender
from config import Config

# ✅ Load gender model once (fast, single inference model)
# This is the model used to label (male/female) above the person box.
gender_model = build_model(r"static/best_gender_model.pth")

# Optional face detection model (YOLO). Disable for max speed.
USE_FACE_DETECTION = False
FACE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "test", "model.pt")
face_model = YOLO(FACE_MODEL_PATH) if USE_FACE_DETECTION else None

# Per-person gender inference cooldown (seconds)
GENDER_INFERENCE_COOLDOWN = 1.5


def analyze_gender_wrapper(face_crop, global_state, person_id):
    """Thread-safe wrapper to run gender inference and update state."""
    try:
        gender_result = analyze_gender(face_crop, gender_model)
        global_state['gender_labels'][person_id] = gender_result
    except Exception as e:
        print(f"[GenderModel] Wrapper Error: {e}")
        global_state['gender_labels'][person_id] = "Unknown"
    finally:
        global_state['processing'][person_id] = False  # mark as done


def generate_frames(video_source, model, tracker, global_state):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error opening video source.")
        return

    frame_skip = 3
    frame_count = 0
    resize_factor = 0.5

    while not global_state['stop_video_flag'].is_set():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # ✅ Person counting
        person_count = count_persons(model, frame) if global_state['counting_enabled'] else 0
        cv2.putText(frame, f"Persons: {person_count}", (20, frame.shape[0] - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        small_frame = cv2.resize(frame, (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor)))

        # Always run detection and gender inference (no motion gating)
        # Run person detection (only class 0 is kept by model_loader filters)
        results = model(small_frame)
        global_state['detected_persons'].clear()

        # Optionally run face detection once per frame (help gender classifier focus on the face)
        faces = []
        if USE_FACE_DETECTION and face_model is not None:
            face_results = face_model(small_frame)
            for fr in face_results:
                if fr.boxes is None:
                    continue
                for fbox in fr.boxes.xyxy:
                    fx1, fy1, fx2, fy2 = map(int, fbox.cpu().numpy())
                    # Scale to original frame coordinates
                    fx1, fy1, fx2, fy2 = int(fx1 / resize_factor), int(fy1 / resize_factor), int(fx2 / resize_factor), int(fy2 / resize_factor)
                    faces.append((fx1, fy1, fx2, fy2))

        # Process people and (optionally) match detected faces
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0].item())
                conf = box.conf[0].item()

                # Scale back to original frame size
                x1, y1, x2, y2 = int(x1 / resize_factor), int(y1 / resize_factor), int(x2 / resize_factor), int(y2 / resize_factor)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{result.names[cls]} {conf:.2f}"
                # Draw label above the bounding box
                cv2.putText(frame, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                if cls != 0:
                    continue

                person_id = track_person(global_state, (x1, y1, x2, y2))
                global_state['detected_persons'].append((person_id, x1, y1, x2, y2))

                # Assign face to this person (closest overlapping face)
                best_face = None
                best_area = 0
                if faces:
                    for fx1, fy1, fx2, fy2 in faces:
                        # compute intersection area
                        ix1 = max(x1, fx1)
                        iy1 = max(y1, fy1)
                        ix2 = min(x2, fx2)
                        iy2 = min(y2, fy2)
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        area = (ix2 - ix1) * (iy2 - iy1)
                        if area > best_area:
                            best_area = area
                            best_face = (fx1, fy1, fx2, fy2)

                # ✅ Check restricted area & trigger alert
                if global_state['restricted_area'] and boxes_intersect((x1, y1, x2, y2), global_state['restricted_area']):
                    gender = global_state['gender_labels'].get(person_id, "unknown").lower()
                    if gender == "unknown" or (global_state['selected_gender'] != "both" and gender != global_state['selected_gender'].lower()):
                        now = time.time()
                        if person_id not in global_state['last_alert_time'] or now - global_state['last_alert_time'][person_id] > 20:
                            play_alert()
                            cv2.putText(frame, "ALERT!", (50, frame.shape[0] - 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            alert_frame = frame.copy()
                            email = global_state.get("email", "viveksapkale022@gmail.com")
                            threading.Thread(target=send_alert_email,
                                             args=(email, alert_frame, person_id, person_count)).start()
                            if Config.DEBUG_ALERTS:
                                print(f"alert triggered: person_id={person_id}, person_count={person_count}")
                            global_state['last_alert_time'][person_id] = now

                # ✅ Gender analysis (threaded, one per person at a time)
                now = time.time()
                last_gender_time = global_state.get('last_gender_time', {}).get(person_id, 0)
                if (person_id not in global_state['processing'] or not global_state['processing'][person_id]) and \
                   (now - last_gender_time) > GENDER_INFERENCE_COOLDOWN:

                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue

                    # Prefer face crop (if found) otherwise use the full person crop
                    if best_face:
                        fx1, fy1, fx2, fy2 = best_face
                        face_crop = frame[fy1:fy2, fx1:fx2]
                        # (face box removed for speed / UI clarity)
                    else:
                        face_crop = person_crop

                    if face_crop.size == 0:
                        continue

                    global_state['processing'][person_id] = True
                    global_state['last_gender_time'][person_id] = now
                    threading.Thread(target=analyze_gender_wrapper,
                                     args=(face_crop, global_state, person_id),
                                     daemon=True).start()

        # ✅ Draw restricted area overlay if set
        if global_state['restricted_area']:
            x1, y1, x2, y2 = global_state['restricted_area']
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ✅ Draw gender labels on each detected person
        for pid, x1, y1, x2, y2 in global_state['detected_persons']:
            gender_text = global_state['gender_labels'].get(pid, "Unknown")
            cv2.putText(frame, f"ID {pid}: {gender_text}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
 
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)

    cap.release()


def track_person(state, bbox):
    for pid, prev_bbox in state['person_tracks'].items():
        if boxes_intersect(bbox, prev_bbox):
            state['person_tracks'][pid] = bbox
            return pid
    state['person_counter'] += 1
    state['person_tracks'][state['person_counter']] = bbox
    return state['person_counter']
