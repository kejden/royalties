import torch
from ultralytics.nn import tasks
import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition
from sklearn.cluster import DBSCAN
from PIL import Image
import collections
from tqdm import tqdm

# --- CONFIGURATION ---
VIDEO_NAME = "video_5"

# --- PATHS ---
INPUT_VIDEO_PATH = f"../movies/{VIDEO_NAME}.mp4"
OUTPUT_GIF_PATH = f"../gifs/{VIDEO_NAME}.gif"
OUTPUT_SUMMARY_PATH = f"../gifs/{VIDEO_NAME}_summary.txt"

# --- PROCESSING SETTINGS ---
PROCESS_EVERY_N_FRAMES = 2
# 🚀 PERFORMANCE FIX: Increased resolution for better detection accuracy.
RESIZE_FACTOR = 0.75

# --- CLUSTERING SETTINGS ---
DBSCAN_EPS = 0.38
MIN_SAMPLES = 2

# --- INITIALIZATION ---
torch.serialization.add_safe_globals([tasks.DetectionModel])

print("Loading YOLO model...")
# 🚀 PERFORMANCE FIX: Using the powerful 'large' model for maximum person detection accuracy.
person_detector = YOLO("yolov8l.pt") # Upgraded from 'm' to 'l'
PERSON_CLASS_ID = 0

all_faces_data = []

# --- PASS 1: DATA EXTRACTION ---
print(f"Starting Pass 1: Extracting face data from '{INPUT_VIDEO_PATH}'...")
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

for frame_num in tqdm(range(total_frames), desc="Pass 1: Extracting face data"):
    ret, frame = cap.read()
    if not ret:
        break

    if frame_num % PROCESS_EVERY_N_FRAMES == 0:
        height, width, _ = frame.shape
        small_frame = cv2.resize(frame, (int(width * RESIZE_FACTOR), int(height * RESIZE_FACTOR)))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        person_results = person_detector(small_frame, classes=[PERSON_CLASS_ID], verbose=False, device=0, conf=0.15)
        person_boxes = person_results[0].boxes.xyxy.cpu().numpy()

        face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_box = [left, top, right, bottom]
            containing_person_box = None
            for p_box in person_boxes:
                px1, py1, px2, py2 = p_box
                face_center_x = (left + right) / 2
                face_center_y = (top + bottom) / 2
                if px1 < face_center_x < px2 and py1 < face_center_y < py2:
                    containing_person_box = p_box
                    break
            
            if containing_person_box is not None:
                all_faces_data.append({
                    "frame_num": frame_num,
                    "person_box": [int(x / RESIZE_FACTOR) for x in containing_person_box],
                    "face_box": [int(x / RESIZE_FACTOR) for x in face_box],
                    "embedding": face_encodings[i]
                })

cap.release()
print(f"Pass 1 complete. Found a total of {len(all_faces_data)} face instances.")

# --- PASS 2 is unchanged ---
if not all_faces_data:
    print("No faces were detected in the video. Exiting.")
    exit()

print("\nStarting Pass 2: Clustering faces...")
embeddings = np.array([data["embedding"] for data in all_faces_data])
db = DBSCAN(eps=DBSCAN_EPS, min_samples=MIN_SAMPLES, metric='euclidean').fit(embeddings)
cluster_labels = db.labels_
person_ids = {}
person_counter = 0
for i, label in enumerate(cluster_labels):
    if label != -1:
        if label not in person_ids:
            person_ids[label] = f"Person {person_counter}"
            person_counter += 1
        all_faces_data[i]["person_id"] = person_ids[label]
    else:
        all_faces_data[i]["person_id"] = "Unknown"
print(f"Clustering complete. Found {person_counter} unique persons.")

person_frame_counts = collections.defaultdict(set)
for data in all_faces_data:
    if data["person_id"] != "Unknown":
        person_frame_counts[data["person_id"]].add(data["frame_num"])

with open(OUTPUT_SUMMARY_PATH, "w") as f:
    f.write(f"On-Screen Time Summary for: {VIDEO_NAME}.mp4\n")
    f.write("-" * 40 + "\n")
    for person_id, frames in sorted(person_frame_counts.items()):
        duration_seconds = len(frames) * PROCESS_EVERY_N_FRAMES / fps
        f.write(f"{person_id}: {duration_seconds:.2f} seconds\n")
print(f"On-screen time summary saved to '{OUTPUT_SUMMARY_PATH}'")

print("\nRendering GIF...")
frames_with_faces = collections.defaultdict(list)
for data in all_faces_data:
    frames_with_faces[data["frame_num"]].append(data)

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
gif_frames = []
for frame_num in tqdm(sorted(frames_with_faces.keys()), desc="Pass 2: Rendering GIF frames"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue
    for data in frames_with_faces[frame_num]:
        p_box = data["person_box"]
        f_box = data["face_box"]
        person_id = data["person_id"]
        cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (255, 0, 0), 2)
        cv2.rectangle(frame, (f_box[0], f_box[1]), (f_box[2], f_box[3]), (0, 255, 0), 2)
        label_pos = (f_box[0], f_box[1] - 10)
        cv2.putText(frame, person_id, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    rgb_frame_for_gif = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame_for_gif)
    gif_frames.append(pil_image)

cap.release()

if gif_frames:
    print(f"\nSaving GIF to '{OUTPUT_GIF_PATH}'...")
    gif_frames[0].save(
        OUTPUT_GIF_PATH,
        save_all=True,
        append_images=gif_frames[1:],
        optimize=False,
        duration=100,
        loop=0
    )
    print("✅ Done!")
else:
    print("Could not generate GIF as no frames with faces were processed.")
