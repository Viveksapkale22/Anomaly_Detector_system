import cv2
# UPDATED IMPORTS: Now importing individual models
from modules.model_loader import fire_model, weapon_model, person_model
from modules.utils import handle_threat_alert, draw_smooth_boxes

def generate_security_frames(video_source, system_state, alerts_collection=None):
    """
    Core AI video loop using updated individual V2/V3 models.
    """
    cap = cv2.VideoCapture(video_source)
    
    frame_count = 0
    skip_frames = 2 
    
    memory_boxes = {"FIRE": [], "WEAPON": [], "PERSON": []}
    
    username = system_state.get("username", "Unknown")
    user_email = system_state.get("email", "viveksapkale0022@gmail.com")

    while True:
        if system_state.get("video_terminated", False):
            break
            
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # ==========================================
        # 1. AI DETECTION (Individual Models)
        # ==========================================
        if frame_count % (skip_frames + 1) == 0:
            memory_boxes = {"FIRE": [], "WEAPON": [], "PERSON": []} 

            # --- A. Fire Detection (V2) ---
            if system_state.get("FIRE", True) and fire_model:
                results = fire_model(frame, conf=0.45, verbose=False)
                for r in results:
                    for box in r.boxes:
                        memory_boxes["FIRE"].append(box.xyxy[0].cpu().numpy())

            # --- B. Weapon Detection (V3) ---
            if system_state.get("WEAPON", True) and weapon_model:
                results = weapon_model(frame, conf=0.45, verbose=False)
                for r in results:
                    for box in r.boxes:
                        memory_boxes["WEAPON"].append(box.xyxy[0].cpu().numpy())

            # --- C. Person / Crowd Detection (v2d) ---
            if system_state.get("PERSON", True) and person_model:
                results = person_model(frame, conf=0.50, verbose=False)
                for r in results:
                    for box in r.boxes:
                        memory_boxes["PERSON"].append(box.xyxy[0].cpu().numpy())

        # ==========================================
        # 2. DRAWING & RENDERING 
        # ==========================================
        frame = draw_smooth_boxes(frame, memory_boxes["FIRE"], (0, 165, 255), "FIRE")      
        frame = draw_smooth_boxes(frame, memory_boxes["WEAPON"], (0, 0, 255), "WEAPON")    
        frame = draw_smooth_boxes(frame, memory_boxes["PERSON"], (175, 214, 11), "PERSON") 
        
        person_count = len(memory_boxes["PERSON"])
        if system_state.get("PERSON", True):
            cv2.putText(frame, f"Persons: {person_count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (175, 214, 11), 3)

        # ==========================================
        # 3. ASYNCHRONOUS ALERTS LOGIC 
        # ==========================================
        if len(memory_boxes["FIRE"]) > 0:
            handle_threat_alert("FIRE", frame, username, user_email, alerts_collection)
            
        if len(memory_boxes["WEAPON"]) > 0:
            handle_threat_alert("WEAPON", frame, username, user_email, alerts_collection)
            
        if person_count > 0: # <-- Keep at 0 for testing, change to > 5 for production
            handle_threat_alert("PERSON", frame, username, user_email, alerts_collection, count=person_count)

        # ==========================================
        # 4. ENCODE & YIELD
        # ==========================================
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()