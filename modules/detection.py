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


# --- ADD THIS TO TOP OF detection.py ---
try:
    fire_model = YOLO(r'train_fire_weapon.pt')
except Exception as e:
    print(f"[FireModel] Failed to load fire/weapon model: {e}")
    fire_model = None

# Optional face detection model (YOLO). Disable for max speed.
USE_FACE_DETECTION = False
FACE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "test", "model.pt")
face_model = None
if USE_FACE_DETECTION:
    if not os.path.exists(FACE_MODEL_PATH):
        print(f"[GenderModel] Face model not found at {FACE_MODEL_PATH}; disabling face-crop mode.")
        USE_FACE_DETECTION = False
    else:
        try:
            face_model = YOLO(FACE_MODEL_PATH)
        except Exception as e:
            print(f"[GenderModel] Failed to load face detection model: {e}")
            face_model = None
            USE_FACE_DETECTION = False

# Per-person gender inference cooldown (seconds)
# (We're doing gender inference less often to save compute and keep IDs stable)
GENDER_INFERENCE_COOLDOWN = 10.0

# How often we run face detection (in frames)
FACE_DETECTION_INTERVAL = 5  # run face detector every N frames (only if USE_FACE_DETECTION=True)

# How often we run full person detection (in frames)
DETECTION_FRAME_SKIP = 2  # run person detection every N frames (lower = smoother but slower)


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

    frame_skip = DETECTION_FRAME_SKIP
    frame_count = 0
    resize_factor = 0.5
    face_frame_counter = 0
    current_faces = []

    while not global_state['stop_video_flag'].is_set():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Run person detection every N frames (configured by DETECTION_FRAME_SKIP)
        if frame_count % frame_skip != 0:
            # Still update UI elements, but skip detection + tracking this frame
            # to save compute.
            # Draw previous tracked boxes + labels if we have them.
            for pid, x1, y1, x2, y2 in global_state['detected_persons']:
                gender_text = global_state['gender_labels'].get(pid, "Unknown")
                cv2.putText(frame, f"ID {pid}: {gender_text}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
            cv2.putText(frame,
                        f"Persons: {len(global_state['detected_persons'])}",
                        (20, frame.shape[0] - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            yield_frame = frame.copy()
            success, buffer = cv2.imencode('.jpg', yield_frame)
            if not success:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.01)
            continue

        # ✅ Person counting
        # We'll update person_count after tracking so it reflects stable IDs.
        person_count = 0

        small_frame = cv2.resize(frame, (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor)))

        # Always run detection and gender inference (no motion gating)
        # Run person detection (only class 0 is kept by model_loader filters)
        results = model(small_frame)
        global_state['detected_persons'].clear()

        # Optionally run face detection every few frames (helps performance)
        if USE_FACE_DETECTION and face_model is not None:
            face_frame_counter += 1
            if face_frame_counter % FACE_DETECTION_INTERVAL == 0:
                current_faces = []
                face_results = face_model(small_frame)
                for fr in face_results:
                    if fr.boxes is None:
                        continue
                    for fbox in fr.boxes.xyxy:
                        fx1, fy1, fx2, fy2 = map(int, fbox.cpu().numpy())
                        # Scale to original frame coordinates
                        fx1, fy1, fx2, fy2 = int(fx1 / resize_factor), int(fy1 / resize_factor), int(fx2 / resize_factor), int(fy2 / resize_factor)
                        current_faces.append((fx1, fy1, fx2, fy2))
        faces = current_faces

        # Build person detections for tracker (small-frame coords)
        tracker_detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                if cls != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0].item())
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                tracker_detections.append(([x1, y1, w, h], conf, "person"))

        # Update tracker and get stable IDs
        tracked_people = []  # list of (person_id, x1, y1, x2, y2) in small-frame coords
        if tracker is not None:
            if tracker_detections:
                tracks = tracker.update_tracks(tracker_detections, frame=small_frame)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    tlbr = track.to_ltrb()
                    if tlbr is None:
                        continue
                    tx1, ty1, tx2, ty2 = map(int, tlbr)
                    # Filter out tiny/invalid tracks (can appear when tracker drifts)
                    if tx2 - tx1 < 20 or ty2 - ty1 < 20:
                        continue
                    tracked_people.append((track.track_id, tx1, ty1, tx2, ty2))
            else:
                # No detections; keep previous tracked people (or none)
                tracks = []
        else:
            # Fallback to simple intersection tracking
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0].item())
                    if cls != 0:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_id = track_person(global_state, (x1, y1, x2, y2))
                    tracked_people.append((person_id, x1, y1, x2, y2))

        # Process each tracked person (convert to original frame scale, run face/gender)
        global_state['detected_persons'].clear()
        person_count = len(tracked_people)
        for person_id, tx1, ty1, tx2, ty2 in tracked_people:
            # Convert track box back to full frame coords
            x1, y1, x2, y2 = int(tx1 / resize_factor), int(ty1 / resize_factor), int(tx2 / resize_factor), int(ty2 / resize_factor)

            # Draw person bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {person_id}", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            global_state['detected_persons'].append((person_id, x1, y1, x2, y2))

            # Ensure we have a placeholder label to avoid blank text on first frame
            if person_id not in global_state['gender_labels']:
                global_state['gender_labels'][person_id] = "Calculating"

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
                    if person_id not in global_state['last_alert_time'] or now - global_state['last_alert_time'][person_id] > Config.ALERT_INTERVAL:
                        play_alert()
                        cv2.putText(frame, "ALERT!", (50, frame.shape[0] - 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        alert_frame = frame.copy()
                        email = global_state.get("email", "viveksapkale022@gmail.com")
                        subject = "Security Alert: Restricted Area Breach"
                        body = f"Person ID {person_id} entered the restricted area. Total persons: {person_count}."
                        threading.Thread(
                            target=send_alert_email,
                            args=(email, alert_frame),
                            kwargs={"subject": subject, "body": body},
                        ).start()
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

                # Prefer face crop (if found) otherwise use the upper portion of the person box
                if best_face:
                    fx1, fy1, fx2, fy2 = best_face
                    face_crop = frame[fy1:fy2, fx1:fx2]
                    # (face box removed for speed / UI clarity)
                else:
                    h = max(1, y2 - y1)
                    face_crop = frame[y1:y1 + int(h * 0.4), x1:x2]

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

        # Show active tracking IDs to help diagnose duplicate/flickering tracks
        active_ids = ",".join(str(pid) for pid, *_ in global_state['detected_persons'])
        if active_ids:
            cv2.putText(frame, f"IDs: {active_ids}",
                        (20, frame.shape[0] - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # ✅ Overlay person + gender counts (helps verify all IDs are being processed)
        person_count = len(global_state['detected_persons'])
        male_count = female_count = unknown_count = 0
        for pid, *_ in global_state['detected_persons']:
            gender_text = global_state['gender_labels'].get(pid, "Unknown").lower()
            if gender_text == "male":
                male_count += 1
            elif gender_text == "female":
                female_count += 1
            else:
                unknown_count += 1

        cv2.putText(frame,
                    f"Persons: {person_count} (M:{male_count} F:{female_count} U:{unknown_count})",
                    (20, frame.shape[0] - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)

    cap.release()




def generate_fire_frames(video_source, fire_state):
    """Isolated video generation pipeline for Fire & Weapon detection."""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("[FireModel] Error opening fire video source.")
        return

    ALERT_COOLDOWN = 60

    while True:
        if fire_state.get("video_terminated", False):
            break
            
        success, frame = cap.read()
        if not success:
            break

        if fire_model is not None:
            # We lowered the threshold to 0.3 just to make sure the model is actually seeing things
            results = fire_model(frame, conf=0.40, verbose=False) 
            threat_detected = None

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Instead of hardcoding 0 and 1, we pull the exact name from your model!
                    class_name = fire_model.names[cls].upper()
                    
                    # Check if the model detected fire or a weapon/gun
                    if "FIRE" in class_name or "WEAPON" in class_name or "GUN" in class_name:
                        threat_detected = class_name
                        color = (0, 0, 255) # Red
                    else:
                        # If it detects something else, draw it in Blue so you know the model is working
                        threat_detected = class_name
                        color = (255, 165, 0) # Orange
                    
                    # Draw the bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Print to your server console so you can debug!
                    print(f"[FireModel Debug] Found {class_name} with {conf:.2f} confidence!")

            current_time = time.time()
            
            # Send Email if a threat is found (respecting cooldown)
            if threat_detected and ("FIRE" in threat_detected or "WEAPON" in threat_detected or "GUN" in threat_detected):
                if (current_time - fire_state.get("last_alert_time", 0) > ALERT_COOLDOWN):
                    fire_state["last_alert_time"] = current_time
                    email = "viveksapkale022@gmail.com"
                    subject = f"URGENT: {threat_detected} Detected!"
                    body = f"A {threat_detected} was just detected on your camera feed. Please check immediately."
                    
                    print(f"[FireModel] Triggering email alert for {threat_detected}!")
                    threading.Thread(target=send_alert_email, args=(email, frame), 
                                     kwargs={"subject": subject, "body": body}).start()

                cv2.putText(frame, "THREAT DETECTED - ALERT SENT", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
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
