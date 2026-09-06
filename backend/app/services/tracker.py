import time
from typing import List, Tuple
from collections import OrderedDict
from dataclasses import dataclass

from app.schemas.detection import DetectedObject, BoundingBox

@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    first_seen: float
    last_seen: float
    last_detection_time: float
    missed_updates: int = 0
    velocity_x: float = 0.0
    velocity_y: float = 0.0

def compute_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    xA = max(boxA.x, boxB.x)
    yA = max(boxA.y, boxB.y)
    xB = min(boxA.x + boxA.width, boxB.x + boxB.width)
    yB = min(boxA.y + boxA.height, boxB.y + boxB.height)

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea == 0.0:
        return 0.0

    boxAArea = boxA.width * boxA.height
    boxBArea = boxB.width * boxB.height
    return interArea / float(boxAArea + boxBArea - interArea)

class SimpleTracker:
    def __init__(self, max_missed_detections: int = 5, iou_threshold: float = 0.3, max_prediction_age_ms: float = 2000.0):
        self.tracks: dict[int, Track] = OrderedDict()
        self.next_id = 1
        self.max_missed_detections = max_missed_detections
        self.iou_threshold = iou_threshold
        self.max_prediction_age_ms = max_prediction_age_ms

    def update(self, detections: List[DetectedObject], timestamp: float) -> None:
        """Update tracker with fresh YOLO detections."""
        unmatched_tracks = set(self.tracks.keys())
        
        for det in detections:
            best_iou = self.iou_threshold
            best_track_id = None
            
            for track_id in unmatched_tracks:
                track = self.tracks[track_id]
                if track.class_id != det.class_id:
                    continue
                    
                iou = compute_iou(det.bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
                    
            if best_track_id is not None:
                track = self.tracks[best_track_id]
                
                # Compute velocity
                dt = timestamp - track.last_detection_time
                if dt > 0:
                    dx = det.bbox.x - track.bbox.x
                    dy = det.bbox.y - track.bbox.y
                    # EMA for velocity
                    alpha = 0.7
                    track.velocity_x = alpha * (dx / dt) + (1 - alpha) * track.velocity_x
                    track.velocity_y = alpha * (dy / dt) + (1 - alpha) * track.velocity_y

                track.bbox = det.bbox
                track.confidence = det.confidence
                track.last_seen = timestamp
                track.last_detection_time = timestamp
                track.missed_updates = 0
                
                unmatched_tracks.remove(best_track_id)
            else:
                self.tracks[self.next_id] = Track(
                    track_id=self.next_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    last_detection_time=timestamp
                )
                self.next_id += 1
                
        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            track.missed_updates += 1
            if track.missed_updates >= self.max_missed_detections:
                del self.tracks[track_id]
                
    def predict(self, current_time: float) -> List[DetectedObject]:
        """Generate predicted detections at current_time."""
        results = []
        stale_tracks = []
        
        for track_id, track in self.tracks.items():
            dt = current_time - track.last_detection_time
            prediction_age_ms = dt * 1000.0
            
            if prediction_age_ms > self.max_prediction_age_ms:
                stale_tracks.append(track_id)
                continue
                
            predicted_x = track.bbox.x + (track.velocity_x * dt)
            predicted_y = track.bbox.y + (track.velocity_y * dt)
            
            # Constrain to plausible values (simplistic)
            predicted_x = max(0.0, predicted_x)
            predicted_y = max(0.0, predicted_y)
            
            pred_bbox = BoundingBox(
                x=predicted_x,
                y=predicted_y,
                width=track.bbox.width,
                height=track.bbox.height
            )
            
            obj = DetectedObject(
                id=track.class_id,  # Map id to class_id or a unique int to satisfy schema
                class_id=track.class_id,
                class_name=track.class_name,
                confidence=track.confidence,
                bbox=pred_bbox,
                track_id=track.track_id,
                source="prediction" if dt > 0.05 else "detection", # Very rough heuristic, better set by manager
                first_seen=track.first_seen,
                last_seen=current_time,
                age=(current_time - track.first_seen) * 1000.0,
                detection_age_ms=prediction_age_ms,
                prediction_age_ms=prediction_age_ms,
                velocity_x=track.velocity_x,
                velocity_y=track.velocity_y
            )
            results.append(obj)
            
        for track_id in stale_tracks:
            del self.tracks[track_id]
            
        return results
