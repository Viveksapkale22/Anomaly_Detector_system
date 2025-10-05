# File: modules/detection.py

import cv2
import time
import threading
from modules.utils import count_persons, boxes_intersect, play_alert, detect_motion
from modules.utils import send_alert_email
from modules.face_analysis import build_model, analyze_gender

# ✅ Load gender model once
gender_model = build_model(r"static\gender_mobilenetv2.pth")

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

    frame_skip = 1
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

        if detect_motion(small_frame):
            results = model(small_frame)
            global_state['detected_persons'].clear()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()

                    # Scale back to original frame size
                    x1, y1, x2, y2 = int(x1 / resize_factor), int(y1 / resize_factor), int(x2 / resize_factor), int(y2 / resize_factor)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"{result.names[cls]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                    if cls == 0:  # person class
                        person_id = track_person(global_state, (x1, y1, x2, y2))
                        global_state['detected_persons'].append((person_id, x1, y1, x2, y2))

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
                                    global_state['last_alert_time'][person_id] = now

                        # ✅ Gender analysis (threaded, one per person at a time)
                        if person_id not in global_state['processing'] or not global_state['processing'][person_id]:
                            face_crop = frame[y1:y2, x1:x2]
                            if face_crop.size == 0:
                                continue
                            global_state['processing'][person_id] = True
                            threading.Thread(target=analyze_gender_wrapper,
                                             args=(face_crop, global_state, person_id),
                                             daemon=True).start()

        else:
            cv2.putText(frame, "standby mode (no motion)", (10, frame.shape[0] - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
