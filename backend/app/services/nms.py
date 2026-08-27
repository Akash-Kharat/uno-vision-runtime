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

def class_aware_nms(
    detections: list[RawDetection], 
    iou_threshold: float = 0.45,
    max_pre_nms: int = 300,
    max_detections: int = 100
) -> list[RawDetection]:
    """Applies Non-Maximum Suppression (NMS) filtering overlap per class."""
    if not detections:
        return []

    # Sort globally by confidence first to enforce max_pre_nms
    detections.sort(key=lambda x: x.confidence, reverse=True)
    if len(detections) > max_pre_nms:
        detections = detections[:max_pre_nms]

    # Group by class ID
    classes = set(d.class_id for d in detections)
    final_detections = []

    for cls in classes:
        cls_dets = [d for d in detections if d.class_id == cls]
        # Already sorted by confidence
        
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in cls_dets])
        
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
        
    final_detections.sort(key=lambda x: x.confidence, reverse=True)
    if len(final_detections) > max_detections:
        final_detections = final_detections[:max_detections]
        
    return final_detections
