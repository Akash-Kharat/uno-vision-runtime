"""YOLO output processor."""

import numpy as np
from app.domain.detection import RawDetection, PreprocessedInput
from app.domain.runtime import ModelRuntimeDescriptor
from app.domain.enums import BoundingBoxFormat, ConfidenceInterpretation
from app.core.exceptions import AppError

class YOLOOutputProcessor:
    name: str = "YOLO"

    def process(
        self,
        outputs: list[np.ndarray],
        descriptor: ModelRuntimeDescriptor,
        preprocessing: PreprocessedInput,
    ) -> list[RawDetection]:
        if not outputs:
            return []

        # Typically the first output is the detection tensor.
        tensor = outputs[0]
        
        # Squeeze batch dimension if present (e.g. [1, 84, 8400] -> [84, 8400])
        if len(tensor.shape) == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
            
        # Determine shape layout: we expect [features, boxes] or [boxes, features]
        # Features usually include: 4 box coords + N classes (e.g., 4 + 80 = 84).
        # Sometimes there's an objectness score (4 + 1 + 80 = 85).
        # Identify the boxes dimension (the larger one usually).
        if tensor.shape[0] < tensor.shape[1]:
            # Layout is [features, boxes], e.g. [84, 8400]
            tensor = tensor.transpose(1, 0) # Convert to [8400, 84]

        profile = descriptor.profile
        out_prof = profile.output
        bbox_format = out_prof.bbox_format
        conf_interp = out_prof.confidence_interpretation
        conf_thresh = out_prof.confidence_threshold

        num_classes = len(profile.classes)
        num_features = tensor.shape[1]
        
        if num_features < 4 + num_classes:
            raise AppError(
                code="INVALID_MODEL_OUTPUT", 
                message=f"Output tensor features ({num_features}) is less than required box(4) + classes({num_classes})", 
                status_code=500
            )

        detections = []
        
        for row in tensor:
            # Parse bounding box based on configured format
            if bbox_format == BoundingBoxFormat.CXCYWH:
                cx, cy, w, h = row[0:4]
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
            elif bbox_format == BoundingBoxFormat.XYWH:
                x1, y1, w, h = row[0:4]
                x2 = x1 + w
                y2 = y1 + h
            elif bbox_format == BoundingBoxFormat.XYXY:
                x1, y1, x2, y2 = row[0:4]
            else:
                raise AppError(code="OUTPUT_PROCESSOR_UNSUPPORTED", message=f"Unsupported bbox format: {bbox_format}", status_code=500)

            # Map from model coordinates back to original image coordinates
            # Reverse padding
            x1 -= preprocessing.pad_x
            x2 -= preprocessing.pad_x
            y1 -= preprocessing.pad_y
            y2 -= preprocessing.pad_y
            
            # Reverse scale
            x1 *= preprocessing.scale_x
            x2 *= preprocessing.scale_x
            y1 *= preprocessing.scale_y
            y2 *= preprocessing.scale_y
            
            # Clamp to original image bounds
            x1 = max(0, min(x1, preprocessing.original_width))
            x2 = max(0, min(x2, preprocessing.original_width))
            y1 = max(0, min(y1, preprocessing.original_height))
            y2 = max(0, min(y2, preprocessing.original_height))
            
            if x2 <= x1 or y2 <= y1:
                continue

            # Parse confidence and class
            if conf_interp == ConfidenceInterpretation.DIRECT:
                # Direct means the scores are mutually exclusive or no objectness
                class_scores = row[4:4+num_classes]
                class_id = int(np.argmax(class_scores))
                confidence = float(class_scores[class_id])
            elif conf_interp == ConfidenceInterpretation.SIGMOID:
                # e.g., objectness * class_score (YOLOv5 style, sometimes 85 features)
                if num_features == 5 + num_classes:
                    objectness = row[4]
                    class_scores = row[5:5+num_classes]
                    class_id = int(np.argmax(class_scores))
                    confidence = float(objectness * class_scores[class_id])
                else:
                    # No explicit objectness, just sigmoid probabilities over classes (YOLOv8 style)
                    class_scores = row[4:4+num_classes]
                    class_id = int(np.argmax(class_scores))
                    confidence = float(class_scores[class_id])
            else:
                # Default fallback
                class_scores = row[4:4+num_classes]
                class_id = int(np.argmax(class_scores))
                confidence = float(class_scores[class_id])

            if confidence >= conf_thresh:
                detections.append(RawDetection(
                    class_id=class_id,
                    confidence=confidence,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2)
                ))

        return detections
