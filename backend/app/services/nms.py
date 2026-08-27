"""Non-Maximum Suppression (NMS) utilities."""
import numpy as np
from app.domain.detection import RawDetection

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """Compute IoU between a box and an array of boxes.
    
    Boxes are in format [x1, y1, x2, y2].
    """
    x1 = np.maximum(box1[0], box2[:, 0])
    y1 = np.maximum(box1[1], box2[:, 1])
    x2 = np.minimum(box1[2], box2[:, 2])
    y2 = np.minimum(box1[3], box2[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)
    
    return iou

def class_aware_nms(detections: list[RawDetection], iou_threshold: float = 0.45) -> list[RawDetection]:
    """Applies Non-Maximum Suppression (NMS) filtering overlap per class."""
    if not detections:
        return []

    # Group by class ID
    classes = set(d.class_id for d in detections)
    final_detections = []

    for cls in classes:
        cls_dets = [d for d in detections if d.class_id == cls]
        # Sort by confidence descending
        cls_dets.sort(key=lambda x: x.confidence, reverse=True)

        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in cls_dets])
        keep_indices = []

        while len(boxes) > 0:
            keep_indices.append(0)
            if len(boxes) == 1:
                break
                
            iou = compute_iou(boxes[0], boxes[1:])
            # Keep those with IoU <= threshold
            filtered_indices = np.where(iou <= iou_threshold)[0] + 1
            boxes = boxes[filtered_indices]
            
            # Map back to original list of sorted cls_dets
            # To do this safely, we actually need to filter the cls_dets list simultaneously.
            cls_dets = [cls_dets[0]] + [cls_dets[i] for i in filtered_indices]
            cls_dets.pop(0)  # We just kept the first one

        # Since we mutated cls_dets iteratively, we can just track what we kept by a boolean array or list tracking.
        # Let's do it cleaner:
    
    # Clean approach:
    final_detections = []
    for cls in classes:
        cls_dets = [d for d in detections if d.class_id == cls]
        cls_dets.sort(key=lambda x: x.confidence, reverse=True)
        
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in cls_dets])
        scores = np.array([d.confidence for d in cls_dets])
        
        keep = []
        indices = np.arange(len(boxes))
        
        while len(indices) > 0:
            current_idx = indices[0]
            keep.append(current_idx)
            
            if len(indices) == 1:
                break
                
            ious = compute_iou(boxes[current_idx], boxes[indices[1:]])
            
            # Find indices where IoU <= threshold
            filtered = np.where(ious <= iou_threshold)[0]
            # Offset by 1 because we sliced [1:]
            indices = indices[filtered + 1]
            
        final_detections.extend([cls_dets[i] for i in keep])
        
    return final_detections
