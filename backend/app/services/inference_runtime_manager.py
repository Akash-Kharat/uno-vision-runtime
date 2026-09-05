"""Background continuous inference manager."""

import time
import logging
import threading
import asyncio
from typing import Optional, Callable

from app.domain.runtime_state import InferenceState, InferenceResultSnapshot, InferenceStats
from app.services.camera_manager import CameraManager
from app.services.detection_service import DetectionService
from app.core.exceptions import AppError

logger = logging.getLogger(__name__)

VALID_TRANSITIONS = {
    InferenceState.STOPPED: {InferenceState.STARTING},
    InferenceState.STARTING: {InferenceState.RUNNING, InferenceState.ERROR},
    InferenceState.RUNNING: {InferenceState.PAUSED, InferenceState.STOPPING, InferenceState.ERROR},
    InferenceState.PAUSED: {InferenceState.RUNNING, InferenceState.STOPPING},
    InferenceState.STOPPING: {InferenceState.STOPPED},
    InferenceState.ERROR: {InferenceState.STOPPED, InferenceState.STARTING}
}

class InferenceRuntimeManager:
    """Manages continuous background inference."""
    
    def __init__(self, camera_manager: CameraManager, detection_service: DetectionService, target_fps: int = 5):
        self.camera_manager = camera_manager
        self.detection_service = detection_service
        self.target_fps = target_fps
        self._target_interval = 1.0 / target_fps if target_fps > 0 else 0.2
        
        self.lock = threading.Lock()
        self.state = InferenceState.STOPPED
        self.stats = InferenceStats()
        
        self._latest_result: InferenceResultSnapshot | None = None
        self._sequence_id = 0
        
        # Thread control
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # Asyncio bridges
        self._result_callbacks: list[Callable[[InferenceResultSnapshot], None]] = []
        
        # Performance benchmarking history
        from collections import deque
        from app.config import get_settings
        history_size = getattr(get_settings(), "PERFORMANCE_HISTORY_SIZE", 500)
        self.history_total = deque(maxlen=history_size)
        self.history_capture = deque(maxlen=history_size)
        self.history_pre = deque(maxlen=history_size)
        self.history_inf = deque(maxlen=history_size)
        self.history_post = deque(maxlen=history_size)

    def register_callback(self, callback: Callable[[InferenceResultSnapshot], None]) -> None:
        """Register a callback for when a new result is ready. 
        Will be called from the background thread."""
        with self.lock:
            self._result_callbacks.append(callback)
            
    def unregister_callback(self, callback: Callable[[InferenceResultSnapshot], None]) -> None:
        """Unregister a callback."""
        with self.lock:
            if callback in self._result_callbacks:
                self._result_callbacks.remove(callback)

    def _transition(self, target: InferenceState) -> None:
        """State transition helper."""
        allowed = VALID_TRANSITIONS.get(self.state, set())
        if target not in allowed:
            raise AppError(code="INVALID_RUNTIME_TRANSITION", message=f"Cannot transition from {self.state.value} to {target.value}", status_code=400)
        self.state = target

    def start(self) -> None:
        """Start the background inference loop."""
        with self.lock:
            if self.state in (InferenceState.RUNNING, InferenceState.STARTING, InferenceState.PAUSED):
                raise AppError(code="RUNTIME_ALREADY_ACTIVE", message="Inference is already running or paused.", status_code=400)
            
            # Verify camera and model readiness before launching thread
            if self.camera_manager.state != "RUNNING":
                raise AppError(code="CAMERA_NOT_RUNNING", message="Cannot start inference: camera stopped.", status_code=400)
            
            session, desc = self.detection_service.runtime_manager.get_active_runtime()
            if not session or not desc:
                raise AppError(code="MODEL_NOT_ACTIVE", message="Cannot start inference: no active model.", status_code=400)

            self._transition(InferenceState.STARTING)
            self._stop_event.clear()
            self._pause_event.clear()
            self._thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._thread.start()

    def pause(self) -> None:
        """Pause continuous inference."""
        with self.lock:
            if self.state != InferenceState.RUNNING:
                raise AppError(code="INVALID_RUNTIME_TRANSITION", message="Can only pause from RUNNING state.", status_code=400)
            self._pause_event.set()
            self._transition(InferenceState.PAUSED)

    def resume(self) -> None:
        """Resume continuous inference."""
        with self.lock:
            if self.state != InferenceState.PAUSED:
                raise AppError(code="INVALID_RUNTIME_TRANSITION", message="Can only resume from PAUSED state.", status_code=400)
            self._pause_event.clear()
            self._transition(InferenceState.RUNNING)

    def stop(self) -> None:
        """Stop continuous inference."""
        with self.lock:
            if self.state == InferenceState.STOPPED:
                return # Idempotent
            if self.state in (InferenceState.RUNNING, InferenceState.PAUSED):
                self._transition(InferenceState.STOPPING)
            self._stop_event.set()
            
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            
        with self.lock:
            self.state = InferenceState.STOPPED

    def _inference_loop(self) -> None:
        """Main background loop."""
        with self.lock:
            self._transition(InferenceState.RUNNING)
            
        last_frame_seq = -1
        t_last_inference = time.perf_counter()
        
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.1)
                continue
                
            loop_start = time.perf_counter()
            
            # Check conditions safely
            try:
                latest_frame = self.camera_manager.get_latest_frame()
                if not latest_frame:
                    time.sleep(0.05)
                    continue
                    
                # Update total frames captured since last check
                self.stats.frames_captured = latest_frame.sequence_id
                    
                # Skip if we already processed this frame
                if latest_frame.sequence_id <= last_frame_seq:
                    time.sleep(0.01)
                    continue
                    
                # Calculate how many frames were skipped
                skipped = (latest_frame.sequence_id - last_frame_seq - 1) if last_frame_seq > 0 else 0
                self.stats.frames_skipped += skipped
                    
                # Run detection
                self.stats.inference_busy = True
                resp = self.detection_service.detect_current_frame()
                self.stats.inference_busy = False
                
                with self.lock:
                    self.stats.frames_inferred += 1
                    self._sequence_id += 1
                    self.stats.total_inference_count += 1
                    self.stats.successful_inference_count += 1
                    
                    inf_ms = resp.inference_time_ms
                    self.stats.last_inference_time_ms = inf_ms
                    
                    # Store timings directly into histories
                    timings = resp.timings
                    if timings:
                        self.history_total.append(timings.total_time_ms)
                        self.history_capture.append(timings.capture_time_ms)
                        self.history_pre.append(timings.preprocessing_time_ms)
                        self.history_inf.append(timings.inference_time_ms)
                        self.history_post.append(timings.postprocessing_time_ms)
                        
                    # Exponential moving average for time
                    if self.stats.successful_inference_count == 1:
                        self.stats.average_inference_time_ms = inf_ms
                    else:
                        self.stats.average_inference_time_ms = 0.9 * self.stats.average_inference_time_ms + 0.1 * inf_ms
                        
                    # Calculate effective FPS over recent period
                    now = time.perf_counter()
                    dt = now - t_last_inference
                    if dt > 0:
                        fps = 1.0 / dt
                        self.stats.effective_inference_fps = 0.9 * self.stats.effective_inference_fps + 0.1 * fps
                    t_last_inference = now
                    
                    self._latest_result = InferenceResultSnapshot(
                        sequence_id=self._sequence_id,
                        frame_sequence_id=latest_frame.sequence_id,
                        frame_timestamp=latest_frame.timestamp,
                        detection_timestamp=time.time(),
                        model_id=resp.model_id,
                        response=resp
                    )
                    snapshot = self._latest_result
                    callbacks = list(self._result_callbacks)
                
                # Dispatch callbacks outside lock
                for cb in callbacks:
                    try:
                        cb(snapshot)
                    except Exception as e:
                        logger.error(f"Error in result callback: {e}")
                        
                last_frame_seq = latest_frame.sequence_id

            except AppError as e:
                self.stats.inference_busy = False
                with self.lock:
                    if not self._stop_event.is_set():
                        self.stats.failed_inference_count += 1
                        self.stats.last_error = e.message
                        # Do not transition to ERROR state, just log and continue
                logger.error(f"Inference AppError: {e.message}")
                time.sleep(1.0) # Backoff
                continue
            except Exception as e:
                self.stats.inference_busy = False
                with self.lock:
                    if not self._stop_event.is_set():
                        self.stats.failed_inference_count += 1
                        self.stats.last_error = str(e)
                logger.exception("Unexpected error in inference loop")
                time.sleep(1.0) # Backoff
                continue
                
            # Pacing
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, self._target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_status(self) -> dict:
        import numpy as np
        with self.lock:
            session, desc = self.detection_service.runtime_manager.get_active_runtime()
            active_model = desc.model_id if desc else None
            
            providers = []
            if session:
                try:
                    providers = session.get_providers()
                except Exception:
                    pass
            
            def calc_stats(dq):
                if not dq:
                    return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
                arr = np.array(dq)
                return {
                    "mean": float(np.mean(arr)),
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95))
                }
            
            return {
                "success": True,
                "runtime": {
                    "state": self.state.value,
                    "target_fps": self.target_fps,
                    "active_model_id": active_model,
                    "available_providers": providers,
                    "active_providers": providers,
                    "latest_sequence_id": self._sequence_id,
                    "last_error": self.stats.last_error,
                    "stats": {
                        "total": self.stats.total_inference_count,
                        "success": self.stats.successful_inference_count,
                        "failed": self.stats.failed_inference_count,
                        "effective_fps": self.stats.effective_inference_fps,
                        "avg_total_ms": calc_stats(self.history_total)["mean"],
                        "avg_capture_ms": calc_stats(self.history_capture)["mean"],
                        "avg_preprocessing_ms": calc_stats(self.history_pre)["mean"],
                        "avg_inference_ms": calc_stats(self.history_inf)["mean"],
                        "avg_postprocessing_ms": calc_stats(self.history_post)["mean"],
                        "p50_total_ms": calc_stats(self.history_total)["p50"],
                        "p95_total_ms": calc_stats(self.history_total)["p95"],
                        "frames_captured": self.stats.frames_captured,
                        "frames_inferred": self.stats.frames_inferred,
                        "frames_skipped": self.stats.frames_skipped,
                        "dropped_frames": self.stats.dropped_frames,
                        "latest_detection_age_ms": self._latest_result.detection_age_ms if self._latest_result else None,
                        "inference_busy": self.stats.inference_busy
                    }
                }
            }
            
    def get_latest_result(self) -> dict | None:
        with self.lock:
            if not self._latest_result:
                return None
            return {
                "sequence_id": self._latest_result.sequence_id,
                "frame_sequence_id": self._latest_result.frame_sequence_id,
                "frame_timestamp": self._latest_result.frame_timestamp,
                "detection_timestamp": self._latest_result.detection_timestamp,
                "detection_age_ms": self._latest_result.detection_age_ms,
                "model_id": self._latest_result.model_id,
                "payload": self._latest_result.response.model_dump()
            }
