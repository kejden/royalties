import torch
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import defaultdict
from sklearn.cluster import DBSCAN
import os
import warnings
from tqdm import tqdm

# --- Konfiguracja ---
VIDEO_PATH = '../movies/video_4.mp4'  # Ścieżka do pliku wideo
MIN_DETECTION_CONFIDENCE = 0.6  # Minimalna pewność detekcji YOLO (0.0 - 1.0)
DBSCAN_EPS = 0.5  # Parametr DBSCAN: maksymalna odległość między próbkami w jednym klastrze
DBSCAN_MIN_SAMPLES = 5  # Parametr DBSCAN: minimalna liczba próbek w sąsiedztwie, aby punkt był uznany za centralny
FRAME_PROCESSING_INTERVAL = 5 # Co która klatka będzie przetwarzana (1 = każda)

# --- Inicjalizacja modeli ---
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używane urządzenie: {device}")

# Inicjalizacja YOLOv8
print("Ładowanie modelu YOLOv8...")
yolo = YOLO('yolov8n.pt')
print("Model YOLOv8 załadowany.")

# Inicjalizacja DeepSORT
print("Ładowanie modelu DeepSORT...")
cfg_deep = get_config()
cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                    max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                    min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                    nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                    use_cuda=True if device.type == 'cuda' else False)
print("Model DeepSORT załadowany.")


# --- Przetwarzanie wideo ---
track_frames = defaultdict(list)
track_features = defaultdict(list)

print(f"\nRozpoczynanie przetwarzania wideo: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Błąd: Nie można otworzyć pliku wideo: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

progress_bar = tqdm(total=total_frames, desc="Analiza klatek", unit="klatka")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    progress_bar.update(1)

    if frame_count % FRAME_PROCESSING_INTERVAL != 0:
        continue

    # Detekcja obiektów (klasa 0 to 'person')
    results = yolo(frame, classes=[0], conf=MIN_DETECTION_CONFIDENCE, verbose=False)

    detections = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
        conf = result.conf[0].cpu().numpy()
        detections.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], conf, 0))

    # Aktualizacja śledzenia
    tracks = deepsort.update(detections, frame)

    # Zbieranie cech i klatek dla każdej ścieżki
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        feature = track.get_feature()
        track_frames[track_id].append(frame_count)
        track_features[track_id].append(feature)

progress_bar.close()
cap.release()
print("Przetwarzanie wideo zakończone.")

# --- Klasteryzacja i obliczanie czasu ---
print("\nRozpoczynanie klasteryzacji...")

all_features = []
track_id_map = []
for track_id, features in track_features.items():
    if len(features) > 0:
        # Uśrednienie wektorów cech dla danego śladu zwiększa stabilność klasteryzacji
        avg_feature = np.mean(features, axis=0)
        all_features.append(avg_feature)
        track_id_map.append(track_id)

if not all_features:
    print("Nie wykryto żadnych osób w filmie.")
    exit()

all_features = np.array(all_features)

# Uruchomienie algorytmu DBSCAN
clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='euclidean').fit(all_features)
labels = clustering.labels_
num_unique_persons = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Klasteryzacja zakończona. Znaleziono {num_unique_persons} unikalnych osób.")

# Mapowanie tymczasowych ID śladów na finalne ID osób
person_map = {track_id: f"Osoba_{label}" for track_id, label in zip(track_id_map, labels) if label != -1}

# Obliczanie czasu na ekranie
screen_time = defaultdict(float)
frame_duration = 1.0 / fps

for track_id, person_label in person_map.items():
    num_processed_frames = len(track_frames[track_id])
    estimated_total_frames = num_processed_frames * FRAME_PROCESSING_INTERVAL
    time_in_track = estimated_total_frames * frame_duration
    screen_time[person_label] += time_in_track

# --- Wyświetlanie wyników ---
print("\n--- Wyniki Analizy ---")
if not screen_time:
    print("Nie zidentyfikowano żadnych unikalnych osób.")
    print("Wskazówka: Spróbuj dostosować parametry DBSCAN (np. DBSCAN_MIN_SAMPLES, DBSCAN_EPS).")
else:
    sorted_screen_time = sorted(screen_time.items(), key=lambda item: item[1], reverse=True)
    for person, time in sorted_screen_time:
        print(f"{person}: {time:.2f} sekund")
