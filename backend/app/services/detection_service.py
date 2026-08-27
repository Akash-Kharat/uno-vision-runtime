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
    def __init__(self, camera_manager, runtime_manager, registry, opencl_backend=None, config=None):
        self.camera_manager = camera_manager
        self.runtime_manager = runtime_manager
        self.registry = registry
        self.preprocessor = Preprocessor(backend=opencl_backend, config=config)

    def detect_current_frame(self, profiler=None) -> DetectionResponse:
        from app.domain.performance import PerformanceProfiler
        p = profiler or PerformanceProfiler(enabled=True)

        with p.measure("total_time_ms"):
            # 1. Verify Camera
            if self.camera_manager.state != "RUNNING":
                raise AppError(code="CAMERA_NOT_RUNNING", message="Camera is not active", status_code=503)
            
            # 2. Capture Frame
            with p.measure("capture_time_ms"):
                frame = self.camera_manager.capture_frame()
                if frame is None:
                    raise AppError(code="CAMERA_FRAME_UNAVAILABLE", message="Failed to capture frame", status_code=500)

            # 3. Snapshot semantics for Runtime Manager
            session, descriptor = self.runtime_manager.get_active_runtime()
            if not session or not descriptor:
                raise AppError(code="MODEL_NOT_ACTIVE", message="No active model loaded", status_code=503)

            # 4. Preprocessing
            with p.measure("preprocessing_time_ms"):
                try:
                    preprocessed = self.preprocessor.preprocess_frame(frame, descriptor.profile, profiler=p)
                except Exception as e:
                    raise AppError(code="PREPROCESSING_FAILED", message=str(e), status_code=500)
                
            # 5. ONNX Inference
            with p.measure("inference_time_ms"):
                try:
                    input_name = session.get_inputs()[0].name
                    ort_inputs = {input_name: preprocessed.tensor}
                    ort_outs = session.run(None, ort_inputs)
                except Exception as e:
                    logger.exception("ONNX inference failed")
                    raise AppError(code="INFERENCE_FAILED", message="Model execution failed", status_code=500)
                
            # 6. Dispatch Output Processor
            with p.measure("postprocessing_time_ms"):
                processor_name = descriptor.profile.output.processor
                processor = output_registry.create_processor(processor_name)
                if not processor:
                    raise AppError(code="OUTPUT_PROCESSOR_UNSUPPORTED", message=f"Processor {processor_name} not implemented", status_code=501)
                    
                try:
                    # Pass the profiler into the processor for detailed timings
                    raw_detections = processor.process(ort_outs, descriptor, preprocessed, profiler=p)
                except AppError:
                    raise
                except Exception as e:
                    logger.exception("Output processing failed")
                    raise AppError(code="INVALID_MODEL_OUTPUT", message="Failed to parse model output", status_code=500)

                # 7. NMS
                with p.measure("nms_time_ms"):
                    from app.config import get_settings
                    settings = get_settings()
                    nms_thresh = descriptor.profile.output.nms_threshold
                    filtered = class_aware_nms(
                        raw_detections, 
                        nms_thresh,
                        max_pre_nms=getattr(settings, "MAX_PRE_NMS_DETECTIONS", 300),
                        max_detections=getattr(settings, "MAX_DETECTIONS", 100)
                    )
                    
                    if p.enabled and "diagnostics" not in p.timings:
                        p.timings["diagnostics"] = {}
                    if p.enabled:
                        p.timings["diagnostics"]["pre_nms_count"] = min(len(raw_detections), getattr(settings, "MAX_PRE_NMS_DETECTIONS", 300))
                        p.timings["diagnostics"]["final_detection_count"] = len(filtered)

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

        from app.schemas.detection import DetectionTimings
        
        # Build timings mapping
        timings = p.get_timings()
        diagnostics = timings.pop("diagnostics", None)
        
        # We must extract the top-level required fields
        capture_ms = timings.pop("capture_time_ms", 0.0)
        pre_ms = timings.pop("preprocessing_time_ms", 0.0)
        inf_ms = timings.pop("inference_time_ms", 0.0)
        post_ms = timings.pop("postprocessing_time_ms", 0.0)
        total_ms = timings.pop("total_time_ms", 0.0)
        
        gpu_upload_ms = timings.pop("gpu_upload_ms", None)
        gpu_kernel_ms = timings.pop("gpu_kernel_ms", None)
        gpu_download_ms = timings.pop("gpu_download_ms", None)
        total_gpu_time_ms = timings.pop("total_gpu_time_ms", None)
        
        in_reused = timings.pop("input_buffer_reused", None)
        if in_reused is not None:
            in_reused = bool(in_reused)
        out_reused = timings.pop("output_buffer_reused", None)
        if out_reused is not None:
            out_reused = bool(out_reused)
        
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
                total_time_ms=total_ms,
                gpu_upload_ms=gpu_upload_ms,
                gpu_kernel_ms=gpu_kernel_ms,
                gpu_download_ms=gpu_download_ms,
                total_gpu_time_ms=total_gpu_time_ms,
                input_buffer_reused=in_reused,
                output_buffer_reused=out_reused,
                inner=timings if timings else None,
                diagnostics=diagnostics
            ),
            image_width=preprocessed.original_width,
            image_height=preprocessed.original_height,
            object_count=len(objects),
            class_counts=class_counts,
            objects=objects
        )
