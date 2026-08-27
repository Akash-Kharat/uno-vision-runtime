"""Detection service orchestrating inference."""

import time
import logging
import uuid
from datetime import datetime, timezone

from app.core.exceptions import AppError
from app.domain.enums import ModelStatus, ModelTask
from app.schemas.detection import DetectionResponse, DetectedObject, BoundingBox
from app.services.preprocessing import Preprocessor
from app.services.nms import class_aware_nms
from app.services.output_registry import output_registry

logger = logging.getLogger(__name__)

class DetectionService:
    def __init__(self, camera_manager, runtime_manager, registry):
        self.camera_manager = camera_manager
        self.runtime_manager = runtime_manager
        self.registry = registry
        self.preprocessor = Preprocessor()

    def detect_current_frame(self) -> DetectionResponse:
        # 1. Verify Camera
        if self.camera_manager.state != "RUNNING":
            raise AppError(code="CAMERA_NOT_RUNNING", message="Camera is not active", status_code=503)
        
        # 2. Capture Frame
        t0 = time.perf_counter()
        frame = self.camera_manager.capture_frame()
        t_capture = time.perf_counter()
        if frame is None:
            raise AppError(code="CAMERA_FRAME_UNAVAILABLE", message="Failed to capture frame", status_code=500)

        # 3. Snapshot semantics for Runtime Manager
        session, descriptor = self.runtime_manager.get_active_runtime()
        if not session or not descriptor:
            raise AppError(code="MODEL_NOT_ACTIVE", message="No active model loaded", status_code=503)

        # 4. Preprocessing
        try:
            preprocessed = self.preprocessor.preprocess_frame(frame, descriptor.profile)
        except Exception as e:
            raise AppError(code="PREPROCESSING_FAILED", message=str(e), status_code=500)
            
        t_pre = time.perf_counter()

        # 5. ONNX Inference
        try:
            input_name = session.get_inputs()[0].name
            ort_inputs = {input_name: preprocessed.tensor}
            ort_outs = session.run(None, ort_inputs)
        except Exception as e:
            logger.exception("ONNX inference failed")
            raise AppError(code="INFERENCE_FAILED", message="Model execution failed", status_code=500)
            
        t_inf = time.perf_counter()

        # 6. Dispatch Output Processor
        processor_name = descriptor.profile.output.processor
        processor = output_registry.create_processor(processor_name)
        if not processor:
            raise AppError(code="OUTPUT_PROCESSOR_UNSUPPORTED", message=f"Processor {processor_name} not implemented", status_code=501)
            
        try:
            raw_detections = processor.process(ort_outs, descriptor, preprocessed)
        except AppError:
            raise
        except Exception as e:
            logger.exception("Output processing failed")
            raise AppError(code="INVALID_MODEL_OUTPUT", message="Failed to parse model output", status_code=500)

        # 7. NMS
        nms_thresh = descriptor.profile.output.nms_threshold
        filtered = class_aware_nms(raw_detections, nms_thresh)
        
        t_post = time.perf_counter()

        # 8. Build canonical structure
        objects = []
        class_counts = {}
        for i, det in enumerate(filtered):
            cls_name = "unknown"
            if det.class_id < len(descriptor.profile.classes):
                cls_name = descriptor.profile.classes[det.class_id]
                
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            objects.append(DetectedObject(
                id=i,
                class_id=det.class_id,
                class_name=cls_name,
                confidence=det.confidence,
                bbox=BoundingBox(
                    x=det.x1,
                    y=det.y1,
                    width=det.x2 - det.x1,
                    height=det.y2 - det.y1
                )
            ))

        capture_ms = (t_capture - t0) * 1000.0
        pre_ms = (t_pre - t_capture) * 1000.0
        inf_ms = (t_inf - t_pre) * 1000.0
        post_ms = (t_post - t_inf) * 1000.0
        total_ms = (t_post - t0) * 1000.0

        from app.schemas.detection import DetectionTimings
        
        return DetectionResponse(
            request_id=f"req_{uuid.uuid4().hex[:10]}",
            success=True,
            timestamp=datetime.now(timezone.utc),
            model_id=descriptor.model_id,
            model_name=self.registry.get_metadata(descriptor.model_id).original_filename,
            inference_time_ms=inf_ms,
            timings=DetectionTimings(
                capture_time_ms=capture_ms,
                preprocessing_time_ms=pre_ms,
                inference_time_ms=inf_ms,
                postprocessing_time_ms=post_ms,
                total_time_ms=total_ms
            ),
            image_width=preprocessed.original_width,
            image_height=preprocessed.original_height,
            object_count=len(objects),
            class_counts=class_counts,
            objects=objects
        )
