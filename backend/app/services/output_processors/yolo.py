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
        profiler: "PerformanceProfiler" = None
    ) -> list[RawDetection]:
        if not outputs:
            return []
            
        class DummyProfiler:
            def measure(self, name):
                import contextlib
                @contextlib.contextmanager
                def dummy(): yield
                return dummy()
                
        p = profiler or DummyProfiler()

        with p.measure("output_decode_time_ms"):
            tensor = outputs[0]
            if len(tensor.shape) == 3 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.shape[0] < tensor.shape[1]:
                tensor = tensor.transpose(1, 0) # [8400, 84]

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

        with p.measure("confidence_filter_time_ms"):
            # Extract scores and find max per box
            if conf_interp == ConfidenceInterpretation.SIGMOID and num_features == 5 + num_classes:
                objectness = tensor[:, 4]
                class_scores = tensor[:, 5:5+num_classes]
                confidences = objectness * np.max(class_scores, axis=1)
                class_ids = np.argmax(class_scores, axis=1)
            else:
                class_scores = tensor[:, 4:4+num_classes]
                confidences = np.max(class_scores, axis=1)
                class_ids = np.argmax(class_scores, axis=1)
                
            mask = confidences >= conf_thresh
            
            surviving_boxes = tensor[mask, 0:4]
            surviving_confidences = confidences[mask]
            surviving_class_ids = class_ids[mask]
            
            if profiler and hasattr(profiler, "timings"):
                if "diagnostics" not in profiler.timings:
                    profiler.timings["diagnostics"] = {}
                profiler.timings["diagnostics"]["raw_candidate_count"] = tensor.shape[0]
                profiler.timings["diagnostics"]["confidence_filtered_count"] = len(surviving_boxes)

        with p.measure("box_conversion_time_ms"):
            if len(surviving_boxes) == 0:
                return []
                
            if bbox_format == BoundingBoxFormat.CXCYWH:
                cx = surviving_boxes[:, 0]
                cy = surviving_boxes[:, 1]
                w = surviving_boxes[:, 2]
                h = surviving_boxes[:, 3]
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
            elif bbox_format == BoundingBoxFormat.XYWH:
                x1 = surviving_boxes[:, 0]
                y1 = surviving_boxes[:, 1]
                w = surviving_boxes[:, 2]
                h = surviving_boxes[:, 3]
                x2 = x1 + w
                y2 = y1 + h
            elif bbox_format == BoundingBoxFormat.XYXY:
                x1 = surviving_boxes[:, 0]
                y1 = surviving_boxes[:, 1]
                x2 = surviving_boxes[:, 2]
                y2 = surviving_boxes[:, 3]
            else:
                raise AppError(code="OUTPUT_PROCESSOR_UNSUPPORTED", message=f"Unsupported bbox format: {bbox_format}", status_code=500)

        with p.measure("coordinate_restore_time_ms"):
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
            
            # Clamp
            x1 = np.clip(x1, 0, preprocessing.original_width)
            x2 = np.clip(x2, 0, preprocessing.original_width)
            y1 = np.clip(y1, 0, preprocessing.original_height)
            y2 = np.clip(y2, 0, preprocessing.original_height)
            
            # Filter invalid boxes where x2 <= x1 or y2 <= y1
            valid_mask = (x2 > x1) & (y2 > y1)
            
            x1 = x1[valid_mask]
            y1 = y1[valid_mask]
            x2 = x2[valid_mask]
            y2 = y2[valid_mask]
            surviving_confidences = surviving_confidences[valid_mask]
            surviving_class_ids = surviving_class_ids[valid_mask]

        # Construct final raw detections for NMS
        detections = []
        for i in range(len(x1)):
            detections.append(RawDetection(
                class_id=int(surviving_class_ids[i]),
                confidence=float(surviving_confidences[i]),
                x1=float(x1[i]),
                y1=float(y1[i]),
                x2=float(x2[i]),
                y2=float(y2[i])
            ))
            
        return detections
