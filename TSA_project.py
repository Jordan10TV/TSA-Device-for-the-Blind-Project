import cv2
import time
import threading
import subprocess
import json
import os
import pickle
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from collections import Counter

def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"

def get_time_string():
    now = datetime.now()
    hour = now.strftime("%I").lstrip("0")
    minute = now.strftime("%M")
    ampm = now.strftime("%p")
    if minute == "00":
        return f"The time is {hour} {ampm}"
    else:
        return f"The time is {hour} {minute} {ampm}"

def show_startup_screen():
    screen = np.zeros((480, 640, 3), dtype=np.uint8)

    logo_path = r"C:\Users\jarbeeny\Downloads\tsa_logo.webp"
    if os.path.exists(logo_path):
        logo = cv2.imread(logo_path)
        if logo is not None:
            logo = cv2.resize(logo, (180, 180))
            x_offset = (640 - 180) // 2
            y_offset = 20
            screen[y_offset:y_offset+180, x_offset:x_offset+180] = logo

    cv2.putText(screen, "Device for the Blind", (150, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.line(screen, (50, 255), (590, 255), (0, 255, 0), 1)
    cv2.putText(screen, "* Object Detection & Distance Warning", (150, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(screen, "* Left / Right / Center Positioning", (150, 315),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(screen, "* AI Face Recognition & Memory", (150, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(screen, "* Real Time Voice Alerts", (150, 365),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.line(screen, (50, 385), (590, 385), (0, 255, 0), 1)
    cv2.putText(screen, "By: Jordan Arbeeny and Braden Harris", (120, 415),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)
    cv2.putText(screen, "TSA Project 2026", (230, 445),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    for i in range(4, 0, -1):
        screen_copy = screen.copy()
        cv2.putText(screen_copy, f"Starting in {i}...", (240, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.imshow("Blind Assist Goggles", screen_copy)
        cv2.waitKey(1000)

def draw_mute_symbol(frame):
    cx, cy = 440, 25
    cv2.circle(frame, (cx, cy), 18, (0, 0, 255), -1)
    cv2.rectangle(frame, (cx-8, cy-5), (cx-2, cy+5), (255, 255, 255), -1)
    pts = np.array([[cx-2, cy-5], [cx+6, cy-10], [cx+6, cy+10], [cx-2, cy+5]], np.int32)
    cv2.fillPoly(frame, [pts], (255, 255, 255))
    cv2.line(frame, (cx-5, cy-8), (cx+8, cy+8), (0, 0, 255), 2)
    cv2.line(frame, (cx+8, cy-8), (cx-5, cy+8), (0, 0, 255), 2)
    cv2.putText(frame, "MUTED", (415, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def draw_voice_indicator(frame, voice_index):
    voices = ["Default", "David", "Zira"]
    colors = [(255, 255, 255), (100, 200, 255), (255, 100, 200)]
    name = voices[voice_index]
    color = colors[voice_index]
    cv2.putText(frame, f"Voice: {name}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

# Load AI model
model = YOLO("yolov8n.pt")

# Show startup screen
show_startup_screen()

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Face recognition setup
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Face database files
FACE_DB_FILE = "face_database.json"
FACE_SAMPLES_FILE = "face_samples.pkl"
FACE_MODEL_FILE = "face_model.yml"

face_names = {}
face_samples = {}
next_face_id = 0
is_trained = False
is_muted = False
current_voice = 0

# Separate buffer per face location
face_buffers = {}
BUFFER_SIZE = 10

VOICE_NAMES = [
    "Default",
    "Microsoft David Desktop",
    "Microsoft Zira Desktop"
]

def load_face_database():
    global face_names, next_face_id, face_samples, is_trained
    if os.path.exists(FACE_DB_FILE):
        with open(FACE_DB_FILE, 'r') as f:
            data = json.load(f)
            face_names = {int(k): v for k, v in data.get("names", {}).items()}
            next_face_id = data.get("next_id", 0)
        print(f"Loaded {len(face_names)} known faces!")
    else:
        print("No face database found, starting fresh!")
    if os.path.exists(FACE_SAMPLES_FILE):
        with open(FACE_SAMPLES_FILE, 'rb') as f:
            face_samples = pickle.load(f)
    if os.path.exists(FACE_MODEL_FILE) and face_names:
        face_recognizer.read(FACE_MODEL_FILE)
        is_trained = True
        print("Face model loaded!")

def save_face_database():
    with open(FACE_DB_FILE, 'w') as f:
        json.dump({"names": face_names, "next_id": next_face_id}, f)
    with open(FACE_SAMPLES_FILE, 'wb') as f:
        pickle.dump(face_samples, f)

def train_recognizer():
    global is_trained
    if not face_samples:
        is_trained = False
        return
    faces = []
    labels = []
    for face_id, samples in face_samples.items():
        for sample in samples:
            faces.append(sample)
            labels.append(face_id)
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.save(FACE_MODEL_FILE)
    is_trained = True
    print("Face recognizer trained and saved!")

def add_new_face(name, samples):
    global next_face_id
    new_id = next_face_id
    next_face_id += 1
    face_names[new_id] = name
    face_samples[new_id] = samples
    save_face_database()
    train_recognizer()
    print(f"Saved face as: {name}")
    t = threading.Thread(target=speak, args=(f"Got it! I will remember {name}",))
    t.daemon = True
    t.start()

def get_stable_name(face_id, new_prediction):
    # Each face slot gets its own buffer
    if face_id not in face_buffers:
        face_buffers[face_id] = []

    buf = face_buffers[face_id]
    buf.append(new_prediction)

    if len(buf) > BUFFER_SIZE:
        buf.pop(0)

    if len(buf) < 5:
        return "Unknown"

    counts = Counter(buf)
    most_common_name, most_common_count = counts.most_common(1)[0]
    if most_common_count / len(buf) >= 0.6:
        return most_common_name
    return "Unknown"

# Speech queue so voices never overlap
speech_queue = []
speech_lock = threading.Lock()
speech_thread_running = True

def speech_worker():
    while speech_thread_running:
        text = None
        voice = None
        with speech_lock:
            if speech_queue:
                text, voice = speech_queue.pop(0)
        if text and not is_muted:
            if voice == 0:
                subprocess.call([
                    'powershell', '-Command',
                    f'Add-Type -AssemblyName System.Speech; '
                    f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                    f'$s.Speak("{text}")'
                ])
            else:
                voice_name = VOICE_NAMES[voice]
                subprocess.call([
                    'powershell', '-Command',
                    f'Add-Type -AssemblyName System.Speech; '
                    f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                    f'$s.SelectVoice("{voice_name}"); '
                    f'$s.Speak("{text}")'
                ])
        else:
            time.sleep(0.05)

# Start single speech worker thread
worker = threading.Thread(target=speech_worker)
worker.daemon = True
worker.start()

def speak(text):
    if is_muted:
        return
    with speech_lock:
        # Clear queue so new alerts don't pile up
        speech_queue.clear()
        speech_queue.append((text, current_voice))

# Variables
last_spoken = {}
last_boxes = []
frame_count = 0
last_clear_time = 0
nothing_detected_count = 0
collecting_face = None
face_collect_samples = []
SAMPLES_NEEDED = 75
last_time_announcement = 0

def get_distance(x1, y1, x2, y2, frame_width, frame_height):
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    ratio = box_area / frame_area
    if ratio > 0.4:
        return "danger"
    elif ratio > 0.15:
        return "caution"
    else:
        return "safe"

def get_position(x1, x2, frame_width):
    center_x = (x1 + x2) / 2
    if center_x < frame_width * 0.33:
        return "right"
    elif center_x > frame_width * 0.66:
        return "left"
    else:
        return "center"

def build_message(label, distance, position):
    if distance == "danger":
        if position == "center":
            return f"Warning! {label} directly in front of you"
        else:
            return f"Warning! {label} very close on your {position}"
    elif distance == "caution":
        if position == "center":
            return f"Warning! {label} ahead"
        else:
            return f"Warning! {label} on your {position}"
    else:
        if position == "center":
            return f"Warning! {label} detected ahead"
        elif position == "left":
            return f"Warning! {label} on your left"
        else:
            return f"Warning! {label} on your right"

load_face_database()

greeting = get_greeting()
time_str = get_time_string()
speak(f"{greeting}. {time_str}")
last_time_announcement = time.time()

print("Camera started!")
print("Press P to mute/unmute")
print("Press V to cycle voices")
print("Press N to name a face")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        print("Camera not found!")
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    current_time = time.time()
    if current_time - last_time_announcement >= 600:
        if not is_muted:
            last_time_announcement = current_time
            speak(get_time_string())

    faces_detected = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    # Clear buffers for faces no longer in frame
    active_slots = set(range(len(faces_detected)))
    for slot in list(face_buffers.keys()):
        if slot not in active_slots:
            del face_buffers[slot]

    for i, (fx, fy, fw, fh) in enumerate(faces_detected):
        face_img = gray[fy:fy+fh, fx:fx+fw]
        face_img = cv2.resize(face_img, (100, 100))

        if collecting_face is not None:
            face_collect_samples.append(face_img)
            cv2.putText(frame, f"Collecting... {len(face_collect_samples)}/{SAMPLES_NEEDED} - Move head slowly!",
                        (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)
            if len(face_collect_samples) >= SAMPLES_NEEDED:
                add_new_face(collecting_face, face_collect_samples.copy())
                collecting_face = None
                face_collect_samples = []
            continue

        raw_name = "Unknown"
        if is_trained:
            try:
                label_id, confidence = face_recognizer.predict(face_img)
                if confidence < 100:
                    raw_name = face_names.get(label_id, "Unknown")
            except:
                raw_name = "Unknown"

        # Use slot index as face ID so each face has its own buffer
        name = get_stable_name(i, raw_name)

        color = (255, 165, 0) if name == "Unknown" else (0, 255, 255)
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)
        cv2.putText(frame, name, (fx, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        current_time = time.time()
        face_key = f"face_{name}"
        if face_key not in last_spoken or current_time - last_spoken[face_key] > 4:
            last_spoken[face_key] = current_time
            position = get_position(fx, fx+fw, frame_width)
            if name == "Unknown":
                msg = f"Warning! Unknown person on your {position}"
            else:
                msg = f"Warning! {name} on your {position}"
            speak(msg)

    if frame_count % 5 == 0:
        results = model(frame, verbose=False)
        last_boxes = []
        detected_labels = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])

                if label == "person":
                    continue

                if confidence > 0.4:
                    distance = get_distance(x1, y1, x2, y2, frame_width, frame_height)
                    position = get_position(x1, x2, frame_width)

                    if distance == "danger":
                        color = (0, 0, 255)
                    elif distance == "caution":
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)

                    last_boxes.append((x1, y1, x2, y2, label, color, distance, position))
                    detected_labels.append((label, distance, position))

        if len(detected_labels) == 0 and len(faces_detected) == 0:
            nothing_detected_count += 1
            if nothing_detected_count >= 10:
                current_time = time.time()
                if current_time - last_clear_time > 8:
                    if not is_muted:
                        last_clear_time = current_time
                        nothing_detected_count = 0
                        speak("Clear path ahead")
        else:
            nothing_detected_count = 0

        for (label, distance, position) in detected_labels:
            current_time = time.time()
            key = f"{label}_{position}"
            if key not in last_spoken or current_time - last_spoken[key] > 4:
                last_spoken[key] = current_time
                speak(build_message(label, distance, position))

    for (x1, y1, x2, y2, label, color, distance, position) in last_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} - {distance}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    draw_voice_indicator(frame, current_voice)

    if is_muted:
        draw_mute_symbol(frame)

    cv2.imshow("Blind Assist Goggles", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        is_muted = not is_muted
        print("Muted" if is_muted else "Unmuted")
    elif key == ord('v'):
        current_voice = (current_voice + 1) % 3
        print(f"Switched to: {VOICE_NAMES[current_voice]}")
        speak(f"Voice changed to {VOICE_NAMES[current_voice]}")
    elif key == ord('n'):
        name_input = input("Enter name for this face: ")
        if name_input:
            collecting_face = name_input
            face_collect_samples = []
            print(f"Slowly move your head while looking at the camera!")

speech_thread_running = False
cap.release()
cv2.destroyAllWindows()