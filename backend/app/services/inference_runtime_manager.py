import threading
import logging
import time
import asyncio
from typing import Optional, Callable
from collections import deque

from app.domain.runtime_state import InferenceState, InferenceResultSnapshot, InferenceStats
from app.services.camera_manager import CameraManager
from app.services.detection_service import DetectionService
from app.core.exceptions import AppError
from app.config import get_settings

logger = logging.getLogger(__name__)

VALID_TRANSITIONS = {
    InferenceState.STOPPED: {InferenceState.STARTING},
    InferenceState.STARTING: {InferenceState.RUNNING, InferenceState.ERROR},
    InferenceState.RUNNING: {InferenceState.INFERENCE, InferenceState.WAITING, InferenceState.DEGRADED, InferenceState.STOPPING, InferenceState.ERROR},
    InferenceState.WAITING: {InferenceState.INFERENCE, InferenceState.STOPPING, InferenceState.ERROR, InferenceState.RUNNING},
    InferenceState.INFERENCE: {InferenceState.WAITING, InferenceState.DEGRADED, InferenceState.STOPPING, InferenceState.ERROR, InferenceState.RUNNING},
    InferenceState.DEGRADED: {InferenceState.WAITING, InferenceState.INFERENCE, InferenceState.STOPPING, InferenceState.ERROR, InferenceState.RUNNING},
    InferenceState.STOPPING: {InferenceState.STOPPED},
    InferenceState.ERROR: {InferenceState.STOPPED, InferenceState.STARTING, InferenceState.RUNNING}
}

class InferenceRuntimeManager:
    """Manages continuous background inference."""
    
    def __init__(self, camera_manager: CameraManager, detection_service: DetectionService, target_fps: int = 5):
        self.camera_manager = camera_manager
        self.detection_service = detection_service
        self.target_fps = get_settings().RUNTIME_TARGET_FPS
        self._target_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.2
        
        self.lock = threading.Lock()
        self.state = InferenceState.STOPPED
        self.stats = InferenceStats()
        
        self._latest_result: InferenceResultSnapshot | None = None
        self._sequence_id = 0
        
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        
        self._result_callbacks: list[Callable[[InferenceResultSnapshot], None]] = []
        
        history_size = getattr(get_settings(), "PERFORMANCE_HISTORY_SIZE", 500)
        self.history_total = deque(maxlen=history_size)
        self.history_capture = deque(maxlen=history_size)
        self.history_pre = deque(maxlen=history_size)
        self.history_inf = deque(maxlen=history_size)
        self.history_post = deque(maxlen=history_size)

    def register_callback(self, callback: Callable[[InferenceResultSnapshot], None]) -> None:
        with self.lock:
            self._result_callbacks.append(callback)
            
    def unregister_callback(self, callback: Callable[[InferenceResultSnapshot], None]) -> None:
        with self.lock:
            if callback in self._result_callbacks:
                self._result_callbacks.remove(callback)

    def _transition(self, target: InferenceState) -> None:
        allowed = VALID_TRANSITIONS.get(self.state, set())
        if target not in allowed:
            raise AppError(code="INVALID_RUNTIME_TRANSITION", message=f"Cannot transition from {self.state.value} to {target.value}", status_code=400)
        self.state = target

    def start(self) -> None:
        with self.lock:
            if self.state not in (InferenceState.STOPPED, InferenceState.ERROR):
                raise AppError(code="RUNTIME_ALREADY_ACTIVE", message="Inference is already running.", status_code=400)
            
            if self.camera_manager.state != "RUNNING":
                raise AppError(code="CAMERA_NOT_RUNNING", message="Cannot start inference: camera stopped.", status_code=400)
            
            session, desc = self.detection_service.runtime_manager.get_active_runtime()
            if not session or not desc:
                raise AppError(code="MODEL_NOT_ACTIVE", message="Cannot start inference: no active model.", status_code=400)

            self._transition(InferenceState.STARTING)
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self.lock:
            if self.state == InferenceState.STOPPED:
                return
            self._transition(InferenceState.STOPPING)
            self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            
        with self.lock:
            self.state = InferenceState.STOPPED

    def _inference_loop(self) -> None:
        with self.lock:
            self._transition(InferenceState.RUNNING)
            
        last_frame_seq = -1
        next_deadline = time.perf_counter()
        
        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_deadline:
                with self.lock:
                    if self.state not in (InferenceState.DEGRADED, InferenceState.ERROR, InferenceState.STOPPING):
                        self.state = InferenceState.WAITING
                time.sleep(min(0.01, next_deadline - now))
                continue
                
            try:
                latest_frame = self.camera_manager.get_latest_frame()
                if not latest_frame:
                    time.sleep(0.05)
                    next_deadline = time.perf_counter() + 0.05
                    continue
                    
                self.stats.captured_frames = latest_frame.sequence_id
                    
                if latest_frame.sequence_id <= last_frame_seq:
                    # Waiting for a new frame, but don't sleep statically
                    time.sleep(0.01)
                    continue
                    
                skipped = (latest_frame.sequence_id - last_frame_seq - 1) if last_frame_seq >= 0 else 0
                self.stats.skipped_frames += skipped
                    
                # Run detection
                with self.lock:
                    self.state = InferenceState.INFERENCE
                    self.stats.inference_busy = True
                
                resp = self.detection_service.detect_current_frame()
                
                with self.lock:
                    self.stats.inferred_frames += 1
                    self.stats.successful_inference_count += 1
                    self.stats.inference_busy = False
                    
                    self._sequence_id += 1
                    
                    # Process metrics
                    timings = resp.timings
                    if timings:
                        self.history_total.append(timings.total_time_ms)
                        self.history_capture.append(timings.capture_time_ms)
                        self.history_pre.append(timings.preprocessing_time_ms)
                        self.history_inf.append(timings.inference_time_ms)
                        self.history_post.append(timings.postprocessing_time_ms)
                        
                        self.stats.last_inference_time_ms = timings.total_time_ms
                        self.stats.average_inference_time_ms = sum(self.history_inf) / len(self.history_inf) if self.history_inf else 0.0

                    self._latest_result = InferenceResultSnapshot(
                        sequence_id=self._sequence_id,
                        frame_sequence_id=latest_frame.sequence_id,
                        timestamp=time.time(),
                        frame_timestamp=latest_frame.timestamp,
                        detection_timestamp=time.time(),
                        model_id=resp.model_id,
                        response=resp
                    )
                    snapshot = self._latest_result
                    callbacks = list(self._result_callbacks)
                
                # Update loop control
                last_frame_seq = latest_frame.sequence_id
                
                # Advance next deadline
                # If inference took longer than interval, we will process the next frame immediately
                next_deadline = max(time.perf_counter(), next_deadline + self._target_interval)
                
                for cb in callbacks:
                    try:
                        cb(snapshot)
                    except Exception:
                        pass
                        
            except Exception as e:
                with self.lock:
                    self.stats.inference_busy = False
                    self.stats.inference_failures += 1
                    self.stats.failed_inference_count += 1
                    self.stats.last_error = str(e)
                    if self.state != InferenceState.STOPPING:
                        self.state = InferenceState.DEGRADED
                logger.exception(f"Exception in inference loop: {str(e)}")
                # Reset deadline to not accumulate backlog
                next_deadline = time.perf_counter() + self._target_interval

    def get_status(self) -> dict:
        def calc_stats(h) -> dict:
            import numpy as np
            if not h: return {"mean": 0, "p50": 0, "p95": 0}
            return {
                "mean": sum(h)/len(h),
                "p50": np.percentile(h, 50),
                "p95": np.percentile(h, 95)
            }
            
        with self.lock:
            session, desc = self.detection_service.runtime_manager.get_active_runtime()
            active_model = desc.model_id if desc else None
            
            # FPS calculation based on history timestamps would be better, but we use effective for now
            status = {
                "state": self.state.value,
                "target_fps": self.target_fps,
                "active_model_id": active_model,
                "available_providers": session.get_providers() if session else [],
                "active_providers": session.get_providers() if session else [],
                "latest_sequence_id": self._sequence_id,
                "last_error": self.stats.last_error,
                "stats": {
                    "total": self.stats.total_inference_count,
                    "success": self.stats.successful_inference_count,
                    "failed": self.stats.failed_inference_count,
                    "avg_total_ms": calc_stats(self.history_total)["mean"],
                    "avg_capture_ms": calc_stats(self.history_capture)["mean"],
                    "avg_preprocessing_ms": calc_stats(self.history_pre)["mean"],
                    "avg_inference_ms": calc_stats(self.history_inf)["mean"],
                    "avg_postprocessing_ms": calc_stats(self.history_post)["mean"],
                    "p50_total_ms": calc_stats(self.history_total)["p50"],
                    "p95_total_ms": calc_stats(self.history_total)["p95"],
                    "captured_frames": self.stats.captured_frames,
                    "inferred_frames": self.stats.inferred_frames,
                    "skipped_frames": self.stats.skipped_frames,
                    "inference_failures": self.stats.inference_failures,
                    "latest_detection_age_ms": ((time.time() - self._latest_result.frame_timestamp) * 1000) if self._latest_result else None,
                    "inference_busy": self.stats.inference_busy
                }
            }
        return {"success": True, "runtime": status}
        
    def get_latest_result(self) -> dict | None:
        with self.lock:
            if not self._latest_result:
                return None
            return {
                "sequence_id": self._latest_result.sequence_id,
                "frame_sequence_id": self._latest_result.frame_sequence_id,
                "frame_timestamp": self._latest_result.frame_timestamp,
                "detection_timestamp": self._latest_result.detection_timestamp,
                "detection_age_ms": (time.time() - self._latest_result.frame_timestamp) * 1000,
                "inference_duration_ms": self.stats.last_inference_time_ms,
                "model_id": self._latest_result.model_id,
                "payload": self._latest_result.response.model_dump()
            }
