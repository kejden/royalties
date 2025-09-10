

import cv2
import face_recognition
from sklearn.cluster import DBSCAN
from collections import defaultdict
import numpy as np
import os
import torch

# Biblioteki do interfejsu w konsoli i YOLO
from rich.console import Console
from rich.table import Table
from rich.progress import track
from ultralytics import YOLO

# --- KONFIGURACJA ---
VIDEO_PATH = "../movies/video_1.mp4"
OUTPUT_DIR = "output_portraits"
FRAME_SKIP = 2  # Możemy analizować więcej klatek, bo GPU jest szybsze
DBSCAN_EPS = 0.45

# --- INICJALIZACJA ---
console = Console()
os.makedirs(OUTPUT_DIR, exist_ok=True)
face_data = []

# Sprawdź, czy GPU jest dostępne dla PyTorch/YOLO
device = 'cuda' if torch.cuda.is_available() else 'cpu'
console.print(f"Używane urządzenie: [bold {'green' if device == 'cuda' else 'red'}]{device}[/]")
#
# Załaduj model YOLOv8 wytrenowany do wykrywania twarzy.
# Model zostanie automatycznie pobrany przy pierwszym uruchomieniu.
console.print("[cyan]Ładowanie modelu detekcji twarzy YOLOv8...[/cyan]")
yolo_model = YOLO('yolov8n-face.pt')

# --- FAZA 1: ZBIERANIE DANYCH Z WIDEO (DETEKCJA NA GPU) ---
video_capture = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video_capture.get(cv2.CAP_PROP_FPS)

console.print(f"[bold cyan]Faza 1: Analizowanie {total_frames} klatek wideo z użyciem YOLOv8...[/bold cyan]")

for frame_num in track(range(total_frames), description="Przetwarzanie..."):
    ret, frame = video_capture.read()
    if not ret:
        break

    if frame_num % FRAME_SKIP == 0:
        # 1. DETEKCJA TWARZY ZA POMOCĄ YOLO (szybkie, na GPU)
        results = yolo_model.predict(frame, device=device, verbose=False)
        # Pobierz współrzędne ramek (bounding boxes)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        
        # Konwertuj format ramki z (x1, y1, x2, y2) na format dlib (top, right, bottom, left)
        yolo_locations = [(box[1], box[2], box[3], box[0]) for box in boxes]

        # 2. TWORZENIE "ODCISKÓW" TWARZY (szybkie, bo tylko dla znalezionych twarzy)
        face_encodings = face_recognition.face_encodings(frame, yolo_locations)

        for i, encoding in enumerate(face_encodings):
            face_data.append({
                "encoding": encoding,
                "frame": frame,
                "location": yolo_locations[i]
            })

video_capture.release()
console.print(f"[bold green]✔ Zakończono. Znaleziono {len(face_data)} wystąpień twarzy.[/bold green]")


# --- FAZA 2, 3 i 4: KLASTERYZACJA, ZAPIS I WYNIKI (bez zmian) ---

if not face_data:
    console.print("[bold red]Nie znaleziono żadnych twarzy w wideo.[/bold red]")
    exit()

console.print("[bold cyan]Faza 2: Grupowanie podobnych twarzy...[/bold cyan]")
all_encodings = [d["encoding"] for d in face_data]
clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=DBSCAN_EPS)
clt.fit(all_encodings)

unique_labels = set(clt.labels_)
num_unique_people = len(unique_labels) - (1 if -1 in unique_labels else 0)
console.print(f"[bold green]✔ Odkryto {num_unique_people} unikalnych osób.[/bold green]")

console.print("[bold cyan]Faza 3: Zapisywanie portretów i obliczanie czasu...[/bold cyan]")
saved_portraits = {}
label_counts = defaultdict(int)

for i, label in enumerate(clt.labels_):
    if label == -1: continue
    label_counts[label] += 1
    if label not in saved_portraits:
        top, right, bottom, left = face_data[i]["location"]
        frame = face_data[i]["frame"]
        padding = 20
        face_image = frame[max(0, top-padding):bottom+padding, max(0, left-padding):right+padding]
        portrait_filename = f"Osoba_{label + 1}.jpg"
        portrait_path = os.path.join(OUTPUT_DIR, portrait_filename)
        cv2.imwrite(portrait_path, face_image)
        saved_portraits[label] = portrait_filename

table = Table(title="Wyniki Analizy Czasu na Ekranie (GPU + YOLOv8)")
table.add_column("ID Osoby", justify="center", style="cyan")
table.add_column("Identyfikator Zdjęcia", justify="left", style="magenta")
table.add_column("Szacowany Czas na Ekranie", justify="center", style="green")

sorted_labels = sorted(label_counts.keys(), key=lambda l: label_counts[l], reverse=True)

for label in sorted_labels:
    count = label_counts[label]
    portrait_file = saved_portraits.get(label, "Brak zdjęcia")
    total_frames = count * FRAME_SKIP
    total_seconds = total_frames / fps
    minutes, seconds = divmod(total_seconds, 60)
    time_str = f"{int(minutes):02d}:{int(seconds):02d}"
    table.add_row(f"Osoba_{label + 1}", portrait_file, time_str)

console.print(table)
console.print(f"\n[bold yellow]Zdjęcia identyfikacyjne zostały zapisane w folderze '{OUTPUT_DIR}'.[/bold yellow]")
