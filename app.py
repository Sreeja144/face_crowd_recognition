import streamlit as st
import cv2
import numpy as np
import os
import pickle
import smtplib
import psycopg2
import time
from datetime import datetime
from email.message import EmailMessage
from insightface.app import FaceAnalysis
from ultralytics import YOLO
import mediapipe as mp
import pygame

# === AUDIO INITIALIZATION WITH FALLBACK FOR DOCKER/RENDER ===
IS_RENDER = os.environ.get("RENDER", "false").lower() == "true"

if not IS_RENDER:
    try:
        pygame.mixer.init()
        # Ensure these paths are correct for your audio files
        crowd_alert_sound = pygame.mixer.Sound("crowd_alert.mp3")
        restricted_alert_sound = pygame.mixer.Sound("restricted_alert.mp3")
        theft_alert_sound = pygame.mixer.Sound("theft.mp3")
    except Exception as e:
        crowd_alert_sound = None
        restricted_alert_sound = None
        theft_alert_sound = None
        print(f"[WARNING] Audio disabled: {e}")
else:
    crowd_alert_sound = None
    restricted_alert_sound = None
    theft_alert_sound = None


# === CONFIGURATION ===
VIDEO_PATH = "video/theft.mp4" # Ensure this path is correct for your video
EMAIL_SENDER = "smadala4@gitam.in"
EMAIL_PASSWORD = "kljn nztp qqot juwe" # REMINDER: This is an app password, not your main Gmail password.
# Never expose your main password directly in code.
DB_URL = "postgresql://faceuser:gruqofbpAImi7EY6tyrGQjVsmMgMPiG6@dpg-d1oiqqadbo4c73b4fca0-a.frankfurt-postgres.render.com/face_db_7r21"

os.makedirs("unknown_faces", exist_ok=True)

# Load registered and blacklist faces
try:
    with open("registered_faces.pkl", "rb") as f:
        registered_faces = pickle.load(f)
except FileNotFoundError:
    st.error("Error: registered_faces.pkl not found. Please ensure it exists.")
    registered_faces = {} # Initialize as empty to prevent errors
try:
    with open("blacklist_faces.pkl", "rb") as f:
        blacklist_faces = pickle.load(f)
except FileNotFoundError:
    st.error("Error: blacklist_faces.pkl not found. Please ensure it exists.")
    blacklist_faces = {} # Initialize as empty to prevent errors


# === ALERT SOUND FILES ===
# Make sure these MP3 files exist in your project directory
CROWD_ALERT = "crowd_alert.mp3"
RESTRICTED_ALERT = "restricted_alert.mp3" 
THIEF_ALERT = "theft.mp3"

# === STREAMLIT UI ===
st.set_page_config(page_title="Crowd & Face Monitoring", layout="wide")
st.title("📽️ Crowd & Face Monitoring System with Alerts")

frame_display = st.empty()
email_icon = st.empty()
start_btn = st.button("▶️ Start Monitoring")

# Initialize session state variables
if "email_sent" not in st.session_state:
    st.session_state.email_sent = False
if "final_email" not in st.session_state:
    st.session_state.final_email = ""
if "unknown_faces" not in st.session_state:
    st.session_state.unknown_faces =[]
if "thief_alert_triggered" not in st.session_state:
    st.session_state.thief_alert_triggered = False
if "standing_count" not in st.session_state: # Added for overall standing count
    st.session_state.standing_count = 0
if "bending_count" not in st.session_state: # Added for overall bending count
    st.session_state.bending_count = 0
if "alerts_sidebar" not in st.session_state: # To store alerts for sidebar display
    st.session_state.alerts_sidebar = []
if "current_stats" not in st.session_state: # To store current stats for sidebar display
    st.session_state.current_stats = {"crowd": 0, "standing": 0, "bending": 0}


# === FACE MATCHING HELPERS ===
seen_unknown_embeddings =[]

def is_duplicate(embedding, seen_list, threshold=0.6):
    for emb in seen_list:
        sim = np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb))
        if sim > threshold:
            return True
    return False

def find_match(embedding, face_db, threshold=0.45):
    best_name, best_score = None, -1
    for name, db_emb in face_db.items():
        sim = np.dot(embedding, db_emb) / (np.linalg.norm(embedding) * np.linalg.norm(db_emb))
        if sim > best_score and sim > threshold:
            return name, sim
    return best_name, best_score

def insert_to_db(path, score):
    try:
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO unknown_faces (image_path, similarity_score) VALUES (%s, %s)", (path, score))
                conn.commit()
    except Exception as e:
        st.error(f"❌ DB Error (Unknown Face): {e}")

def log_alert_to_db(alert_type, frame_path="none", description=""):
    try:
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO alerts_triggered (timestamp, alert_type, frame_reference, description) VALUES (%s, %s, %s, %s)",
                    (datetime.now(), alert_type, frame_path, description)
                )
                conn.commit()
    except Exception as e:
        print(f" {e}")

def send_email_with_images(image_paths, to_email, standing, bending, crowd):
    if not to_email:
        return

    msg = EmailMessage()
    msg["Subject"] = "🔍 Face Detection Report"
    msg["From"] = EMAIL_SENDER
    msg["To"] = to_email
    msg.set_content(f"""

Hi,

Here's the summary of today's monitoring:

- Unknown faces detected: {len(image_paths)}
- Standing People Count: {standing}
- Bending/Fall Count: {bending}
- Max Crowd Count: {crowd}

Images are attached. They're also stored in the 'unknown_faces' folder and logged to DB. Thanks,
Crowd Monitoring System

""")

    for path in image_paths[:5]: # Attach up to 5 images
        try:
            with open(path, 'rb') as img:
                msg.add_attachment(img.read(), maintype='image', subtype='jpeg', filename=os.path.basename(path))
        except Exception as e:
            print(f"Error attaching image {path}: {e}")

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        st.toast("✅ Email sent successfully!", icon="📨")
        st.success(f"📧 Sent to {to_email}")

        # Log email
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO email_logs (sent_to, unknown_face_count, standing_count, bending_count, crowd_count, restricted_triggered)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (to_email, len(image_paths), standing, bending, crowd, False)) # restricted_triggered set to False
            conn.commit()
    except Exception as e:
        st.error("❌ Email Failed")
        st.code(str(e))

def send_theft_alert(to_email, description="N/A"):
    if not to_email:
        return
    msg = EmailMessage()
    msg["Subject"] = "🚨 Potential Theft Detected!"
    msg["From"] = EMAIL_SENDER
    msg["To"] = to_email
    msg.set_content(f"""

Hi,

A potential theft situation has been detected: {description}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please review immediately.

Thanks,
Crowd Monitoring System

""")
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f" {e}")


# Function to calculate angle between three points (useful for pose)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Use 2D coordinates for vector calculation
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))

    return angle

# === NEW STANDING/BENDING/THEFT HELPER FUNCTIONS ===

def detect_standing(pose_landmarks):
    """
    Detects if a person is standing based on the angle between shoulder, hip, and knee.
    A relatively straight body (angle > 160) is considered standing.
    """
    try:
        left_shoulder = [pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_hip = [pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                    pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                     pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
        
        angle = calculate_angle(left_shoulder, left_hip, left_knee)
        return angle > 160
    except (IndexError, AttributeError, ValueError):
        return False

def detect_bending(pose_landmarks):
    """
    Detects if a person is bending based on the angle between shoulder, hip, and knee.
    An angled body (angle < 150) is considered bending.
    """
    try:
        left_shoulder = [pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_hip = [pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                    pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                     pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]

        angle = calculate_angle(left_shoulder, left_hip, left_knee)
        return angle < 150
    except (IndexError, AttributeError, ValueError):
        return False

def detect_theft(pose_landmarks):
    """
    Detects potential theft if a person is bending.
    """
    return detect_bending(pose_landmarks)

# === MONITORING LOGIC ===
if start_btn:
    face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    yolo_model = YOLO("yolov8n.pt") # Yolov8n for person detection. Consider yolov8x.pt for higher accuracy if performance allows.

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    from mediapipe.framework.formats import landmark_pb2

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        st.error(f"❌ Could not open video at: {VIDEO_PATH}. Please check the path and file.")
        st.stop()

    unknown_faces, seen_unknown_embeddings =[],[]
    max_crowd_count = 0
    standing_total_overall = 0
    bending_total_overall = 0
    frame_count = 0
    frame_skip = 10 # Process every 10th frame for better performance balance

    # Clear sidebar alerts and stats at start
    st.session_state.alerts_sidebar = []
    st.session_state.current_stats = {"crowd": 0, "standing": 0, "bending": 0}


    while True:
        for _ in range(frame_skip):
            cap.read()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = yolo_model.track(rgb_frame, persist=True, classes=0, verbose=False, tracker="bytetrack.yaml")
        crowd_count = 0
        
        current_frame_standing_count = 0
        current_frame_bending_count = 0
        is_theft_in_current_frame = False


        if results and results[0].boxes:
            for r in results[0].boxes:
                cls = int(r.cls)
                conf = float(r.conf)
                track_id = int(r.id) if r.id is not None else None

                if cls == 0 and conf > 0.5 and track_id is not None:
                    crowd_count += 1 # Increment crowd count for each detected person

                    bbox = r.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Draw YOLO bounding box and ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)


                    # --- Pose Estimation ---
                    x1_p, y1_p, x2_p, y2_p = bbox
                    pad_y = int((y2_p - y1_p) * 0.25)
                    pad_x = int((x2_p - x1_p) * 0.25)

                    y1_crop = max(0, y1_p - pad_y)
                    y2_crop = min(frame.shape[0], y2_p + pad_y)
                    x1_crop = max(0, x1_p - pad_x)
                    x2_crop = min(frame.shape[1], x2_p + pad_x)

                    person_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]

                    if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                        rgb_person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                        person_holistic_results = holistic.process(rgb_person_crop)

                        if person_holistic_results.pose_landmarks:
                            scaled_pose_landmarks = landmark_pb2.NormalizedLandmarkList()
                            
                            h_crop, w_crop, _ = person_crop.shape
                            h_frame, w_frame, _ = frame.shape
                            
                            for landmark in person_holistic_results.pose_landmarks.landmark:
                                new_landmark = scaled_pose_landmarks.landmark.add()
                                new_landmark.x = (landmark.x * w_crop + x1_crop) / w_frame
                                new_landmark.y = (landmark.y * h_crop + y1_crop) / h_frame
                                new_landmark.z = landmark.z
                                new_landmark.visibility = landmark.visibility

                            mp_drawing.draw_landmarks(
                                frame,
                                scaled_pose_landmarks,
                                mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=4, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=4, circle_radius=4)  
                            )

                            pose_status = ""
                            if detect_standing(person_holistic_results.pose_landmarks.landmark):
                                current_frame_standing_count += 1
                                standing_total_overall += 1
                                pose_status = "Standing"
                            elif detect_bending(person_holistic_results.pose_landmarks.landmark):
                                current_frame_bending_count += 1
                                bending_total_overall += 1
                                pose_status = "Bending"
                            
                            if detect_theft(person_holistic_results.pose_landmarks.landmark):
                                is_theft_in_current_frame = True
                                pose_status = "THIEF!" # Override status if theft detected

                            # Draw pose status inside the bounding box, just above the person's head if possible
                            if pose_status:
                                text_size = cv2.getTextSize(pose_status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                text_x = x1 + (x2 - x1 - text_size[0]) // 2 # Center text horizontally
                                text_y = y1 + text_size[1] + 5 # Just below top of bbox
                                
                                # Use different color for THIEF!
                                text_color_status = (0, 255, 255) # Yellow for bending/standing
                                if pose_status == "THIEF!":
                                    text_color_status = (0, 0, 255) # Red for thief

                                cv2.putText(frame, pose_status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color_status, 2)

        max_crowd_count = max(max_crowd_count, crowd_count)

        # Update current_stats for sidebar display
        st.session_state.current_stats["crowd"] = crowd_count
        st.session_state.current_stats["standing"] = current_frame_standing_count
        st.session_state.current_stats["bending"] = current_frame_bending_count

        if is_theft_in_current_frame and not st.session_state.thief_alert_triggered:
            if theft_alert_sound:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                theft_alert_sound.play()
            if st.session_state.final_email:
                send_theft_alert(st.session_state.final_email, "Person bending detected")
            log_alert_to_db("Theft Alert (Behavioral)", description="Person bending detected")
            st.session_state.thief_alert_triggered = True
            st.session_state.alerts_sidebar.append(("error", "🚨🚨 THIEF ALERT!!"))
        elif not is_theft_in_current_frame and st.session_state.thief_alert_triggered:
            st.session_state.thief_alert_triggered = False


        # --- Face Detection and Recognition ---
        faces = face_app.get(rgb_frame)
        for face in faces:
            bbox_face = face.bbox.astype(int)
            x1_f, y1_f, x2_f, y2_f = bbox_face
            face_embedding = face.embedding

            name = "Unknown"
            text_color = (0, 0, 255) # Red for unknown

            matched_name, score = find_match(face_embedding, registered_faces)
            if matched_name:
                # Format name for display (e.g., "john.doe@example.com" -> "John Doe")
                name_display = matched_name.split('@')[0].replace('.', ' ').title()
                name = name_display
                text_color = (0, 255, 0)

            blacklisted_name, blacklist_score = find_match(face_embedding, blacklist_faces)
            if blacklisted_name:
                name = "BLACKLISTED" # Simply show "BLACKLISTED" on frame
                text_color = (0, 0, 0) # Black for blacklisted
                cv2.rectangle(frame, (x1_f, y1_f), (x2_f, y2_f), (0, 0, 0), 3) # Black box for blacklist
                
                if restricted_alert_sound:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                    restricted_alert_sound.play() # Using RESTRICTED_ALERT sound for blacklist
                
                if st.session_state.final_email:
                    msg_subject = "🚨 Blacklisted Person Detected!"
                    msg_content = f"""
Hi,

A blacklisted person ({blacklisted_name}) has been detected.
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please take necessary action.

Thanks,
Crowd Monitoring System
"""
                    msg = EmailMessage()
                    msg["Subject"] = msg_subject
                    msg["From"] = EMAIL_SENDER
                    msg["To"] = st.session_state.final_email
                    msg.set_content(msg_content)
                    try:
                        with smtplib.SMTP("smtp.gmail.com", 587) as server:
                            server.starttls()
                            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                            server.send_message(msg)
                    except Exception as e:
                        print(f"[Email Alert Error for Blacklist] {e}")

                log_alert_to_db("Blacklist Face Detected", description=f"Blacklisted: {blacklisted_name}")
                st.session_state.alerts_sidebar.append(("error", f"🚫 Blacklisted Face: {blacklisted_name}"))

            # Draw face name/status on the frame
            face_text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            face_text_x = x1_f + (x2_f - x1_f - face_text_size[0]) // 2 # Center text horizontally
            face_text_y = y1_f + face_text_size[1] + 5 # Just below top of face bbox
            
            cv2.rectangle(frame, (x1_f, y1_f), (x2_f, y2_f), text_color, 2)
            cv2.putText(frame, name, (face_text_x, face_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            if name == "Unknown":
                if not is_duplicate(face_embedding, seen_unknown_embeddings):
                    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join("unknown_faces", f"unknown_face_{current_time_str}_{len(st.session_state.unknown_faces)}.jpg")
                    face_crop = frame[y1_f:y2_f, x1_f:x2_f]
                    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                        cv2.imwrite(filename, face_crop)
                        st.session_state.unknown_faces.append(filename)
                        seen_unknown_embeddings.append(face_embedding)
                        insert_to_db(filename, score if score!= -1 else 0.0)
                        log_alert_to_db("Unknown Face", filename, "Unknown face detected")
                        st.session_state.alerts_sidebar.append(("warning", "❓ Unknown Face Detected"))
                    else:
                        print(f"Warning: Invalid face crop for saving: {bbox_face}")

        # Removed Crowd Count display on the video frame


        # Update Streamlit display
        frame_display.image(frame, channels="BGR", use_column_width=True)

        if crowd_count > 2:
            if crowd_alert_sound:
                if not pygame.mixer.music.get_busy():
                    crowd_alert_sound.play()
            log_alert_to_db("Overcrowding")
            st.session_state.alerts_sidebar.append(("error", "🎉 Overcrowding Detected!"))
        else:
            if ("error", "🎉 Overcrowding Detected!") in st.session_state.alerts_sidebar:
                pass

        # Display current stats and alerts in the sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Current Statistics")
        st.sidebar.info(f"👥 **Crowd Count:** {st.session_state.current_stats['crowd']}")
        st.sidebar.info(f"🧍 **Standing People:** {st.session_state.current_stats['standing']}")
        st.sidebar.info(f"🤸 **Bending People:** {st.session_state.current_stats['bending']}")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Active Alerts")
        if not st.session_state.alerts_sidebar:
            st.sidebar.success("✅ No active alerts.")
        else:
            displayed_alerts = set()
            for alert_type, message in reversed(st.session_state.alerts_sidebar):
                if message not in displayed_alerts:
                    if alert_type == "error":
                        st.sidebar.error(message)
                    elif alert_type == "warning":
                        st.sidebar.warning(message)
                    elif alert_type == "success":
                        st.sidebar.success(message)
                    displayed_alerts.add(message)
            st.session_state.alerts_sidebar = st.session_state.alerts_sidebar[-10:]


        time.sleep(0.0005)

    cap.release()
    st.session_state.unknown_faces = unknown_faces
    st.session_state.standing_count = standing_total_overall
    st.session_state.bending_count = bending_total_overall
    st.session_state.max_crowd = max_crowd_count
# === EMAIL SECTION ===
with st.form("final_email_form"):
    st.markdown("#### 📧 Enter email to receive the report")
    st.session_state.final_email = st.text_input("Email")
    submit = st.form_submit_button("Send Report")
    if submit and not st.session_state.email_sent:
        send_email_with_images(st.session_state.unknown_faces, st.session_state.final_email, st.session_state.standing_count, st.session_state.bending_count, st.session_state.max_crowd)
        st.session_state.email_sent = True
