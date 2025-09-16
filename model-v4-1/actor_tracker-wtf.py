#!/usr/bin/env python3
"""
Enhanced Actor Detection and Tracking System for Movies
Integrates face recognition directly into tracking for better identity association
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import time
from ultralytics import YOLO
from tqdm import tqdm
import insightface
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import cosine, euclidean
from scipy.optimize import linear_sum_assignment
import hdbscan
from sklearn.preprocessing import normalize
import torchvision.transforms as T
from torchvision.models import resnet50

# Configure torch for optimal performance
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()

@dataclass
class Detection:
    """Enhanced detection with appearance features"""
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    frame_id: int
    face_embedding: Optional[np.ndarray] = None
    body_feature: Optional[np.ndarray] = None
    
@dataclass
class Track:
    """Enhanced track with multi-modal features"""
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    hits: int = 0
    time_since_update: int = 0
    face_embeddings: List[np.ndarray] = field(default_factory=list)
    body_features: List[np.ndarray] = field(default_factory=list)
    kalman_filter: Optional[KalmanFilter] = None
    avg_face_embedding: Optional[np.ndarray] = None
    avg_body_feature: Optional[np.ndarray] = None
    scene_id: int = 0
    
    def update_averages(self):
        """Update average embeddings for robust matching"""
        if self.face_embeddings:
            # Weighted average with recent embeddings having more weight
            weights = np.exp(np.linspace(-1, 0, len(self.face_embeddings[-10:])))
            weights /= weights.sum()
            self.avg_face_embedding = np.average(
                self.face_embeddings[-10:], axis=0, weights=weights
            )
        
        if self.body_features:
            weights = np.exp(np.linspace(-1, 0, len(self.body_features[-10:])))
            weights /= weights.sum()
            self.avg_body_feature = np.average(
                self.body_features[-10:], axis=0, weights=weights
            )

class SceneDetector:
    """Detect scene changes to handle tracking discontinuities"""
    
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold
        self.prev_frame = None
        self.scene_id = 0
        
    def detect_scene_change(self, frame: np.ndarray) -> Tuple[bool, int]:
        """Detect if there's a scene change between frames"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))  # Reduce for faster computation
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, self.scene_id
            
        # Calculate histogram difference
        hist_curr = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_prev = cv2.calcHist([self.prev_frame], [0], None, [256], [0, 256])
        
        hist_diff = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_CHISQR)
        
        # Also check mean absolute difference
        frame_diff = np.mean(np.abs(gray.astype(float) - self.prev_frame.astype(float)))
        
        is_scene_change = hist_diff > self.threshold or frame_diff > 50
        
        if is_scene_change:
            self.scene_id += 1
            
        self.prev_frame = gray
        return is_scene_change, self.scene_id

class BodyFeatureExtractor:
    """Extract body features for ReID when faces are not visible"""
    
    def __init__(self):
        self.model = resnet50(pretrained=True)
        self.model.eval()
        # Remove final classification layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Extract body features from person bbox"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            person_crop = frame[y1:y2, x1:x2]
            
            # Transform and extract features
            input_tensor = self.transform(person_crop).unsqueeze(0)
            
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.squeeze().cpu().numpy()
                features = normalize(features.reshape(1, -1))[0]  # L2 normalize
                
            return features
            
        except Exception as e:
            return None

class HybridTracker:
    """Enhanced tracker with integrated face recognition and body ReID"""
    
    def __init__(self, 
                 max_disappeared: int = 30,
                 face_weight: float = 0.7,
                 body_weight: float = 0.2,
                 motion_weight: float = 0.1,
                 face_threshold: float = 0.65,
                 body_threshold: float = 0.5):
        
        self.max_disappeared = max_disappeared
        self.face_weight = face_weight
        self.body_weight = body_weight
        self.motion_weight = motion_weight
        self.face_threshold = face_threshold
        self.body_threshold = body_threshold
        
        self.next_id = 0
        self.tracks = {}
        self.lost_tracks = {}  # Buffer for recently lost tracks
        self.lost_buffer_size = 100  # Keep lost tracks for potential re-identification
        
    def create_kalman_filter(self, bbox: Tuple[float, float, float, float]) -> KalmanFilter:
        """Create Kalman filter for motion prediction"""
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State: [x_center, y_center, width, height, vx, vy, vw, vh]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0])
        
        # State transition matrix
        kf.F = np.eye(8)
        for i in range(4):
            kf.F[i, i+4] = 1
        
        # Measurement matrix
        kf.H = np.zeros((4, 8))
        for i in range(4):
            kf.H[i, i] = 1
        
        # Covariance matrices
        kf.R *= 0.01
        kf.P[4:, 4:] *= 1000
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        
        return kf
    
    def calculate_cost_matrix(self, tracks: List[Track], detections: List[Detection], 
                            scene_changed: bool) -> np.ndarray:
        """Calculate hybrid cost matrix combining motion, face, and body features"""
        n_tracks = len(tracks)
        n_dets = len(detections)
        
        if n_tracks == 0 or n_dets == 0:
            return np.empty((0, 0))
        
        cost_matrix = np.ones((n_tracks, n_dets)) * 1e6  # High cost for non-matches
        
        for i, track in enumerate(tracks):
            # Predict track position
            if track.kalman_filter and not scene_changed:
                track.kalman_filter.predict()
                predicted_state = track.kalman_filter.x[:4].copy()
                predicted_bbox = (
                    predicted_state[0] - predicted_state[2]/2,
                    predicted_state[1] - predicted_state[3]/2,
                    predicted_state[0] + predicted_state[2]/2,
                    predicted_state[1] + predicted_state[3]/2
                )
            else:
                predicted_bbox = track.bbox
            
            for j, det in enumerate(detections):
                costs = []
                weights = []
                
                # Motion cost (IoU-based) - only if not scene change
                if not scene_changed:
                    iou = self._calculate_iou(predicted_bbox, det.bbox)
                    motion_cost = 1 - iou
                    costs.append(motion_cost)
                    weights.append(self.motion_weight)
                
                # Face similarity cost
                if det.face_embedding is not None and track.avg_face_embedding is not None:
                    face_sim = 1 - cosine(det.face_embedding, track.avg_face_embedding)
                    face_cost = 1 - face_sim
                    costs.append(face_cost)
                    weights.append(self.face_weight)
                
                # Body feature cost
                if det.body_feature is not None and track.avg_body_feature is not None:
                    body_sim = 1 - cosine(det.body_feature, track.avg_body_feature)
                    body_cost = 1 - body_sim
                    costs.append(body_cost)
                    weights.append(self.body_weight)
                
                # Calculate weighted cost
                if costs:
                    weights = np.array(weights) / np.sum(weights)  # Normalize weights
                    cost_matrix[i, j] = np.dot(costs, weights)
                    
        return cost_matrix
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IoU between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections: List[Detection], scene_changed: bool = False, 
              scene_id: int = 0) -> List[Track]:
        """Update tracks with new detections"""
        
        # If scene changed, move active tracks to lost buffer
        if scene_changed:
            for track_id, track in self.tracks.items():
                self.lost_tracks[track_id] = track
            # Keep only recent lost tracks
            if len(self.lost_tracks) > self.lost_buffer_size:
                oldest_ids = sorted(self.lost_tracks.keys())[:len(self.lost_tracks) - self.lost_buffer_size]
                for tid in oldest_ids:
                    del self.lost_tracks[tid]
        
        # Update time since update for all tracks
        for track in self.tracks.values():
            track.time_since_update += 1
        
        # Try to associate with active tracks first
        active_tracks = list(self.tracks.values())
        if active_tracks and detections:
            cost_matrix = self.calculate_cost_matrix(active_tracks, detections, scene_changed)
            
            # Hungarian algorithm for optimal assignment
            if cost_matrix.size > 0:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                matched_tracks = set()
                matched_detections = set()
                
                for row, col in zip(row_indices, col_indices):
                    # Check if cost is below threshold
                    if cost_matrix[row, col] < 0.5:  # Reasonable match threshold
                        track = active_tracks[row]
                        det = detections[col]
                        self._update_track(track, det, scene_id)
                        matched_tracks.add(row)
                        matched_detections.add(col)
                
                # Handle unmatched detections
                unmatched_det_indices = set(range(len(detections))) - matched_detections
            else:
                unmatched_det_indices = set(range(len(detections)))
        else:
            unmatched_det_indices = set(range(len(detections)))
        
        # Try to re-identify from lost tracks
        for det_idx in list(unmatched_det_indices):
            det = detections[det_idx]
            
            if det.face_embedding is not None and self.lost_tracks:
                best_match_id = None
                best_similarity = 0.0
                
                for track_id, track in self.lost_tracks.items():
                    if track.avg_face_embedding is not None:
                        similarity = 1 - cosine(det.face_embedding, track.avg_face_embedding)
                        if similarity > self.face_threshold and similarity > best_similarity:
                            best_similarity = similarity
                            best_match_id = track_id
                
                if best_match_id is not None:
                    # Revive track
                    track = self.lost_tracks[best_match_id]
                    del self.lost_tracks[best_match_id]
                    self.tracks[best_match_id] = track
                    self._update_track(track, det, scene_id)
                    unmatched_det_indices.remove(det_idx)
        
        # Create new tracks for remaining unmatched detections
        for det_idx in unmatched_det_indices:
            det = detections[det_idx]
            self._create_new_track(det, scene_id)
        
        # Remove lost tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_disappeared:
                tracks_to_remove.append(track_id)
                self.lost_tracks[track_id] = track  # Move to lost buffer
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return list(self.tracks.values())
    
    def _create_new_track(self, detection: Detection, scene_id: int):
        """Create new track from detection"""
        track = Track(
            track_id=self.next_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            hits=1,
            time_since_update=0,
            scene_id=scene_id
        )
        
        track.kalman_filter = self.create_kalman_filter(detection.bbox)
        
        if detection.face_embedding is not None:
            track.face_embeddings.append(detection.face_embedding)
            track.avg_face_embedding = detection.face_embedding
            
        if detection.body_feature is not None:
            track.body_features.append(detection.body_feature)
            track.avg_body_feature = detection.body_feature
        
        self.tracks[self.next_id] = track
        self.next_id += 1
    
    def _update_track(self, track: Track, detection: Detection, scene_id: int):
        """Update existing track with new detection"""
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.hits += 1
        track.time_since_update = 0
        track.scene_id = scene_id
        
        # Update Kalman filter
        if track.kalman_filter:
            cx = (detection.bbox[0] + detection.bbox[2]) / 2
            cy = (detection.bbox[1] + detection.bbox[3]) / 2
            w = detection.bbox[2] - detection.bbox[0]
            h = detection.bbox[3] - detection.bbox[1]
            measurement = np.array([cx, cy, w, h])
            track.kalman_filter.update(measurement)
        
        # Update features
        if detection.face_embedding is not None:
            track.face_embeddings.append(detection.face_embedding)
            if len(track.face_embeddings) > 20:  # Keep last 20
                track.face_embeddings = track.face_embeddings[-20:]
                
        if detection.body_feature is not None:
            track.body_features.append(detection.body_feature)
            if len(track.body_features) > 20:
                track.body_features = track.body_features[-20:]
        
        # Update averages
        track.update_averages()

class TrackConsolidator:
    """Post-processing consolidation using HDBSCAN clustering"""
    
    def __init__(self, min_cluster_size: int = 8, min_samples: int = 5):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        
    def consolidate(self, all_tracks: Dict[int, List], 
                   face_embeddings: Dict[int, List]) -> Dict[int, int]:
        """Consolidate tracks using HDBSCAN on face embeddings"""
        
        print("🔄 Running HDBSCAN consolidation...")
        
        # Extract high-quality face embeddings
        track_embeddings = {}
        track_ids = []
        embedding_matrix = []
        
        for track_id, embeddings in face_embeddings.items():
            if len(embeddings) >= 3:  # Minimum embeddings required
                # Use high-confidence embeddings
                high_conf = [e for e in embeddings if e['confidence'] > 0.7]
                
                if high_conf:
                    # Calculate robust average
                    emb_vectors = [e['embedding'] for e in high_conf[:10]]
                    avg_embedding = np.mean(emb_vectors, axis=0)
                    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                    
                    track_embeddings[track_id] = avg_embedding
                    track_ids.append(track_id)
                    embedding_matrix.append(avg_embedding)
        
        if len(embedding_matrix) < 2:
            print("   Not enough tracks with face embeddings for consolidation")
            return {}
        
        embedding_matrix = np.array(embedding_matrix)
        
        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=0.3
        )
        
        labels = clusterer.fit_predict(embedding_matrix)
        
        # Create mapping
        track_mapping = {}
        cluster_representatives = {}
        
        for i, (track_id, label) in enumerate(zip(track_ids, labels)):
            if label == -1:  # Noise point - keep original
                track_mapping[track_id] = track_id
            else:
                if label not in cluster_representatives:
                    cluster_representatives[label] = track_id
                track_mapping[track_id] = cluster_representatives[label]
        
        # Add unmapped tracks
        for track_id in all_tracks.keys():
            if track_id not in track_mapping:
                track_mapping[track_id] = track_id
        
        n_consolidated = len(track_mapping) - len(set(track_mapping.values()))
        print(f"   Consolidated {n_consolidated} duplicate tracks")
        print(f"   Final unique actors: {len(set(track_mapping.values()))}")
        
        return track_mapping

class EnhancedActorTracker:
    """Main enhanced actor tracking system"""
    
    def __init__(self, 
                 model_path: str = 'yolov8x.pt',  # Use larger model
                 conf_threshold: float = 0.4,
                 device: str = 'auto',
                 enable_consolidation: bool = True):
        
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
        self.body_extractor = BodyFeatureExtractor()
        self.scene_detector = SceneDetector()
        
        # Use hybrid tracker instead of ByteTracker
        self.tracker = HybridTracker()
        
        # Consolidation
        self.consolidator = TrackConsolidator()
        self.enable_consolidation = enable_consolidation
        
        # Configuration
        self.conf_threshold = conf_threshold
        self.person_class_id = 0
        
        # Results storage
        self.all_tracks = defaultdict(list)
        self.face_embeddings = defaultdict(list)
        self.frame_results = []
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[List[Track], bool, int]:
        """Process single frame with scene detection"""
        
        # Detect scene change
        scene_changed, scene_id = self.scene_detector.detect_scene_change(frame)
        
        # Person detection
        results = self.person_detector(frame, conf=self.conf_threshold, verbose=False)
        
        # Extract detections with features
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    if int(boxes.cls[i]) == self.person_class_id:
                        bbox = tuple(boxes.xyxy[i].cpu().numpy())
                        conf = float(boxes.conf[i])
                        
                        det = Detection(
                            bbox=bbox,
                            confidence=conf,
                            class_id=self.person_class_id,
                            frame_id=frame_id
                        )
                        
                        # Extract face embedding
                        faces = self.face_recognizer.extract_faces(frame, bbox)
                        if faces:
                            best_face = max(faces, key=lambda f: f['confidence'])
                            det.face_embedding = best_face['embedding']
                            det.face_embedding = det.face_embedding / np.linalg.norm(det.face_embedding)
                        
                        # Extract body features
                        body_feat = self.body_extractor.extract(frame, bbox)
                        if body_feat is not None:
                            det.body_feature = body_feat
                        
                        detections.append(det)
        
        # Update tracker with scene awareness
        tracks = self.tracker.update(detections, scene_changed, scene_id)
        
        # Store results
        for track in tracks:
            self.all_tracks[track.track_id].append({
                'frame_id': frame_id,
                'bbox': track.bbox,
                'confidence': track.confidence,
                'scene_id': scene_id
            })
            
            # Store face embeddings
            if track.face_embeddings:
                for emb in track.face_embeddings[-1:]:  # Latest only
                    self.face_embeddings[track.track_id].append({
                        'frame_id': frame_id,
                        'embedding': emb,
                        'confidence': track.confidence
                    })
        
        return tracks, scene_changed, scene_id
    
    def process_video(self, 
                     video_path: str, 
                     output_path: str = None,
                     skip_frames: int = 1,
                     max_frames: int = None) -> Dict:
        """Process entire video"""
        
        print(f"🎬 Processing video: {video_path}")
        
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
        
        # Setup output
        if output_path is None:
            output_path = f"{Path(video_path).stem}_tracked.mp4"
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        pbar = tqdm(total=total_frames//skip_frames, desc="Processing frames")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_count >= max_frames):
                    break
                
                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue
                    
                # Process frame
                tracks, scene_changed, scene_id = self.process_frame(frame, frame_count)
                
                # Draw results
                annotated_frame = self.draw_results(frame, tracks, scene_changed, scene_id)
                out.write(annotated_frame)
                
                # Store results
                self.frame_results.append({
                    'frame_id': frame_count,
                    'tracks': tracks,
                    'scene_changed': scene_changed,
                    'scene_id': scene_id
                })
                
                pbar.update(1)
                frame_count += 1
                
        finally:
            pbar.close()
            cap.release()
            out.release()
            
        print(f"✅ Video processing completed: {output_path}")
        
        # Apply consolidation
        if self.enable_consolidation:
            track_mapping = self.consolidator.consolidate(
                self.all_tracks, 
                self.face_embeddings
            )
            self.apply_track_consolidation(track_mapping)
        
        # Generate results
        results = self.generate_results()
        
        # Save results
        results_path = f"{Path(video_path).stem}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
            
        print(f"📋 Results saved: {results_path}")
        
        return results
    
    
    def draw_results(self, frame: np.ndarray, tracks: List[Track], 
                scene_changed: bool, scene_id: int) -> np.ndarray:
        """Draw tracking results on frame"""
        annotated_frame = frame.copy()
    
        # Get frame dimensions
        height, width = frame.shape[:2]
    
        # Color palette
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
    
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            color = colors[track.track_id % len(colors)]
        
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
            # Draw label
            label = f"Actor_{track.track_id:03d}"
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
        # Draw scene info
        if scene_changed:
            cv2.putText(annotated_frame, "SCENE CHANGE", (width // 2 - 100, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
        scene_text = f"Scene: {scene_id} | Actors: {len(tracks)}"
        cv2.putText(annotated_frame, scene_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
        return annotated_frame


    def apply_track_consolidation(self, track_mapping: Dict[int, int]):
        """Apply consolidation mapping"""
        if not track_mapping:
            return
            
        new_all_tracks = defaultdict(list)
        for old_id, history in self.all_tracks.items():
            new_id = track_mapping.get(old_id, old_id)
            new_all_tracks[new_id].extend(history)
        
        for track_id in new_all_tracks:
            new_all_tracks[track_id].sort(key=lambda x: x['frame_id'])
        
        self.all_tracks = new_all_tracks
    
    def generate_results(self) -> Dict:
        """Generate final analysis results"""
        results = {
            'summary': {
                'total_actors': len(self.all_tracks),
                'total_frames': len(self.frame_results),
                'total_scenes': max([f['scene_id'] for f in self.frame_results]) + 1
            },
            'actors': {}
        }
        
        for track_id, history in self.all_tracks.items():
            actor_name = f"Actor_{track_id:03d}"
            
            # Calculate metrics
            screen_frames = len(history)
            scenes_appeared = len(set([h['scene_id'] for h in history]))
            
            results['actors'][actor_name] = {
                'track_id': track_id,
                'screen_frames': screen_frames,
                'scenes_appeared': scenes_appeared,
                'first_frame': min(h['frame_id'] for h in history),
                'last_frame': max(h['frame_id'] for h in history),
                'avg_confidence': float(np.mean([h['confidence'] for h in history]))
            }
        
        return results
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class FaceRecognizer:
    """Face recognition using InsightFace"""
    
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
            print(f"✓ Face recognition model loaded")
        except Exception as e:
            print(f"⚠ Warning: Could not load face model: {e}")
    
    def extract_faces(self, frame: np.ndarray, bbox: Tuple = None) -> List[Dict]:
        """Extract face embeddings from frame"""
        if self.app is None:
            return []
            
        try:
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
                if bbox:
                    face.bbox[0] += x1
                    face.bbox[2] += x1
                    face.bbox[1] += y1
                    face.bbox[3] += y1
                
                results.append({
                    'bbox': face.bbox,
                    'embedding': face.embedding,
                    'confidence': getattr(face, 'det_score', 1.0)
                })
            
            return results
        except Exception:
            return []

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Actor Tracking")
    parser.add_argument("input_video", help="Input video path")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("-m", "--model", default="yolov8x.pt", help="YOLO model")
    parser.add_argument("-c", "--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("-s", "--skip", type=int, default=1, help="Skip frames")
    parser.add_argument("--max-frames", type=int, help="Max frames to process")
    parser.add_argument("--no-consolidation", action="store_true", help="Disable consolidation")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = EnhancedActorTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        enable_consolidation=not args.no_consolidation
    )
    
    # Process video
    results = tracker.process_video(
        video_path=args.input_video,
        output_path=args.output,
        skip_frames=args.skip,
        max_frames=args.max_frames
    )
    
    # Print summary
    print("\n📊 FINAL RESULTS")
    print("=" * 50)
    print(f"Total actors: {results['summary']['total_actors']}")
    print(f"Total scenes: {results['summary']['total_scenes']}")
    print(f"Total frames: {results['summary']['total_frames']}")
    
    # Show top actors
    actors = results['actors']
    sorted_actors = sorted(actors.items(), key=lambda x: x[1]['screen_frames'], reverse=True)
    
    print("\n🎭 TOP ACTORS BY SCREEN TIME:")
    for i, (name, data) in enumerate(sorted_actors[:10]):
        pct = (data['screen_frames'] / results['summary']['total_frames']) * 100
        print(f"{i+1}. {name}: {data['screen_frames']} frames ({pct:.1f}%)")
        print(f"   Appeared in {data['scenes_appeared']} scenes")

if __name__ == "__main__":
    main()
