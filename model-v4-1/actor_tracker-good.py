#!/usr/bin/env python3
"""
Actor Detection and Tracking System for Movies
Implements state-of-the-art multi-modal tracking with face recognition and person ReID
"""

import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import time

from ultralytics import YOLO
from tqdm import tqdm
import insightface
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN

# Configure torch for optimal performance
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()

@dataclass
class Detection:
    """Single detection result"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    frame_id: int
    
@dataclass
class Track:
    """Single track state"""
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    hits: int = 0
    time_since_update: int = 0
    face_embeddings: List[np.ndarray] = None
    body_features: List[np.ndarray] = None
    
    def __post_init__(self):
        if self.face_embeddings is None:
            self.face_embeddings = []
        if self.body_features is None:
            self.body_features = []

class ByteTracker:
    """ByteTrack-inspired multi-object tracker"""
    
    def __init__(self, max_disappeared: int = 30, high_thresh: float = 0.6, low_thresh: float = 0.1):
        self.max_disappeared = max_disappeared
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.next_id = 0
        self.tracks = {}
        self.kalman_filters = {}
        
    def _create_kalman_filter(self, bbox: Tuple[float, float, float, float]) -> KalmanFilter:
        """Create Kalman filter for motion prediction"""
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State: [x, y, w, h, vx, vy, vw, vh]
        kf.x = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], 0, 0, 0, 0])
        
        # State transition matrix
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Covariance matrices
        kf.R *= 0.01  # Measurement noise
        kf.P[4:, 4:] *= 1000  # Initial velocity uncertainty
        kf.Q[-1, -1] *= 0.01  # Process noise
        kf.Q[4:, 4:] *= 0.01
        
        return kf
    
    def _iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
            
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks with new detections"""
        # Predict all tracks
        for track_id, kf in self.kalman_filters.items():
            kf.predict()
            pred_bbox = kf.x[:4].copy()
            pred_bbox[2] += pred_bbox[0]  # Convert w,h to x2,y2
            pred_bbox[3] += pred_bbox[1]
            self.tracks[track_id].bbox = tuple(pred_bbox)
            self.tracks[track_id].time_since_update += 1
        
        # Separate high and low confidence detections
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]
        
        # Associate high confidence detections
        self._associate_detections(high_dets, high_confidence=True)
        
        # Associate remaining low confidence detections with unmatched tracks
        unmatched_tracks = [t for t in self.tracks.values() if t.time_since_update > 0]
        if unmatched_tracks and low_dets:
            self._associate_detections(low_dets, high_confidence=False, tracks_subset=unmatched_tracks)
        
        # Remove lost tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            del self.kalman_filters[track_id]
        
        return list(self.tracks.values())
    
    def _associate_detections(self, detections: List[Detection], high_confidence: bool, tracks_subset: List[Track] = None):
        """Associate detections with tracks using Hungarian algorithm (simplified)"""
        tracks = tracks_subset if tracks_subset else list(self.tracks.values())
        
        if not tracks or not detections:
            # Create new tracks for unmatched detections
            for det in detections:
                if high_confidence or len(tracks) == 0:
                    self._create_new_track(det)
            return
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, det.bbox)
        
        # Simple greedy assignment (for production, use Hungarian algorithm)
        matched_tracks = set()
        matched_detections = set()
        
        # Find best matches above threshold
        iou_threshold = 0.3
        while True:
            max_iou = np.max(iou_matrix)
            if max_iou < iou_threshold:
                break
                
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            track = tracks[i]
            det = detections[j]
            
            # Update track
            self._update_track(track, det)
            matched_tracks.add(i)
            matched_detections.add(j)
            
            # Remove from consideration
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
        
        # Create new tracks for unmatched high-confidence detections
        if high_confidence:
            for j, det in enumerate(detections):
                if j not in matched_detections:
                    self._create_new_track(det)
    
    def _create_new_track(self, detection: Detection):
        """Create new track from detection"""
        track = Track(
            track_id=self.next_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            hits=1,
            time_since_update=0
        )
        
        self.tracks[self.next_id] = track
        self.kalman_filters[self.next_id] = self._create_kalman_filter(detection.bbox)
        self.next_id += 1
    
    def _update_track(self, track: Track, detection: Detection):
        """Update existing track with new detection"""
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.hits += 1
        track.time_since_update = 0
        
        # Update Kalman filter
        kf = self.kalman_filters[track.track_id]
        measurement = np.array([detection.bbox[0], detection.bbox[1], 
                              detection.bbox[2] - detection.bbox[0],
                              detection.bbox[3] - detection.bbox[1]])
        kf.update(measurement)

class FaceRecognizer:
    """Face recognition module using InsightFace"""
    
    def __init__(self, model_name: str = 'buffalo_l'):
        self.app = None
        self.model_name = model_name
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize InsightFace model"""
        try:
            self.app = insightface.app.FaceAnalysis(name=self.model_name)
            ctx_id = 0 if torch.cuda.is_available() else -1
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            print(f"✓ Face recognition model '{self.model_name}' loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load face recognition model: {e}")
            print("Face recognition will be disabled")
    
    def extract_faces(self, frame: np.ndarray, bbox: Tuple[float, float, float, float] = None) -> List[Dict]:
        """Extract face embeddings from frame"""
        if self.app is None:
            return []
            
        try:
            # Crop to person bbox if provided
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                roi = frame[y1:y2, x1:x2]
            else:
                roi = frame
                
            if roi.size == 0:
                return []
                
            faces = self.app.get(roi)
            
            results = []
            for face in faces:
                # Adjust bbox coordinates if we cropped
                if bbox:
                    face_bbox = face.bbox
                    face_bbox[0] += x1  # Adjust x coordinates
                    face_bbox[2] += x1
                    face_bbox[1] += y1  # Adjust y coordinates  
                    face_bbox[3] += y1
                
                results.append({
                    'bbox': face.bbox,
                    'embedding': face.embedding,
                    'confidence': getattr(face, 'det_score', 1.0)
                })
            
            return results
        except Exception as e:
            print(f"Error in face extraction: {e}")
            return []

class ActorTracker:
    """Main actor tracking system"""
    
    def __init__(self, 
                 model_path: str = 'yolov8n.pt',
                 conf_threshold: float = 0.5,
                 device: str = 'auto'):
        
        # Setup device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"🚀 Using device: {self.device}")
        
        # Initialize models
        self.person_detector = YOLO(model_path)
        self.person_detector.to(self.device)
        
        self.face_recognizer = FaceRecognizer()
        self.tracker = ByteTracker()
        
        # Configuration
        self.conf_threshold = conf_threshold
        self.person_class_id = 0  # COCO person class
        
        # Results storage
        self.all_tracks = defaultdict(list)  # track_id -> list of track states
        self.face_embeddings = defaultdict(list)  # track_id -> list of embeddings
        self.frame_results = []
        
    def process_video(self, 
                     video_path: str, 
                     output_path: str = None,
                     skip_frames: int = 1,
                     max_frames: int = None) -> Dict:
        """Process entire video and return results"""
        
        print(f"🎬 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            
        print(f"📊 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if output_path is None:
            output_path = f"{Path(video_path).stem}_tracked.mp4"
            
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames with progress bar
        frame_count = 0
        processed_count = 0
        
        pbar = tqdm(total=total_frames//skip_frames, desc="Processing frames", unit="frame")
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_count >= max_frames):
                    break
                
                # Skip frames for efficiency
                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue
                    
                # Process frame
                results = self.process_frame(frame, frame_count)
                
                # Draw results on frame
                annotated_frame = self.draw_results(frame, results)
                
                # Write frame
                out.write(annotated_frame)
                
                # Store results
                self.frame_results.append({
                    'frame_id': frame_count,
                    'tracks': results
                })
                
                # Update progress
                processed_count += 1
                pbar.update(1)
                
                # Update progress bar description with speed info
                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_processing = processed_count / elapsed
                    pbar.set_description(f"Processing frames ({fps_processing:.1f} FPS)")
                
                frame_count += 1
                
        finally:
            pbar.close()
            cap.release()
            out.release()
            
        print(f"✅ Video processing completed: {output_path}")
        
        # Generate final results
        results = self.generate_results()
        
        # Save results to JSON
        results_path = f"{Path(video_path).stem}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
            
        print(f"📋 Results saved: {results_path}")
        
        return results
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> List[Track]:
        """Process single frame"""
        # Person detection
        results = self.person_detector(frame, conf=self.conf_threshold, verbose=False)
        
        # Extract person detections
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    if int(boxes.cls[i]) == self.person_class_id:  # Person class
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i])
                        
                        detections.append(Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_id=self.person_class_id,
                            frame_id=frame_id
                        ))
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Extract face embeddings for each track
        for track in tracks:
            # Extract faces from person bounding box
            faces = self.face_recognizer.extract_faces(frame, track.bbox)
            
            # Store face embeddings
            if faces:
                for face in faces:
                    self.face_embeddings[track.track_id].append({
                        'frame_id': frame_id,
                        'embedding': face['embedding'],
                        'confidence': face['confidence']
                    })
            
            # Store track history
            self.all_tracks[track.track_id].append({
                'frame_id': frame_id,
                'bbox': track.bbox,
                'confidence': track.confidence,
                'faces_count': len(faces)
            })
        
        return tracks
    
    def draw_results(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw tracking results on frame"""
        annotated_frame = frame.copy()
        
        # Color palette for different tracks
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (128, 128, 0), (0, 128, 128), (128, 0, 0), (0, 128, 0)
        ]
        
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            color = colors[track.track_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Actor_{track.track_id:03d}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence
            conf_text = f"{track.confidence:.2f}"
            cv2.putText(annotated_frame, conf_text, (x2 - 50, y1 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw frame info
        frame_info = f"Actors: {len(tracks)}"
        cv2.putText(annotated_frame, frame_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def generate_results(self) -> Dict:
        """Generate final analysis results"""
        results = {
            'summary': {
                'total_actors': len(self.all_tracks),
                'total_frames_processed': len(self.frame_results),
                'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'actors': {}
        }
        
        # Process each actor track
        for track_id, track_history in self.all_tracks.items():
            actor_name = f"Actor_{track_id:03d}"
            
            # Calculate screen time (number of frames present)
            screen_frames = len(track_history)
            
            # Get frame appearances
            appearances = []
            for entry in track_history:
                appearances.append({
                    'frame_id': entry['frame_id'],
                    'bbox': entry['bbox'],
                    'confidence': entry['confidence']
                })
            
            # Face recognition summary
            face_data = self.face_embeddings.get(track_id, [])
            face_confidence = np.mean([f['confidence'] for f in face_data]) if face_data else 0.0
            
            results['actors'][actor_name] = {
                'track_id': track_id,
                'screen_frames': screen_frames,
                'first_appearance': min(track_history, key=lambda x: x['frame_id'])['frame_id'],
                'last_appearance': max(track_history, key=lambda x: x['frame_id'])['frame_id'],
                'avg_confidence': float(np.mean([t['confidence'] for t in track_history])),
                'face_detections': len(face_data),
                'face_confidence': float(face_confidence),
                'appearances': appearances
            }
        
        return results
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Actor Detection and Tracking in Movies")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("-m", "--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("-c", "--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("-s", "--skip", type=int, default=1, help="Skip frames for faster processing")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cuda/cpu)")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ActorTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device
    )
    
    # Process video
    results = tracker.process_video(
        video_path=args.input_video,
        output_path=args.output,
        skip_frames=args.skip,
        max_frames=args.max_frames
    )
    
    # Print summary
    print("\n📊 PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total actors detected: {results['summary']['total_actors']}")
    print(f"Total frames processed: {results['summary']['total_frames_processed']}")
    
    # Print top actors by screen time
    actors = results['actors']
    sorted_actors = sorted(actors.items(), key=lambda x: x[1]['screen_frames'], reverse=True)
    
    print("\n🎭 TOP ACTORS BY SCREEN TIME:")
    for i, (actor_name, data) in enumerate(sorted_actors[:10]):
        print(f"{i+1:2d}. {actor_name}: {data['screen_frames']} frames "
              f"(conf: {data['avg_confidence']:.2f})")

if __name__ == "__main__":
    main()