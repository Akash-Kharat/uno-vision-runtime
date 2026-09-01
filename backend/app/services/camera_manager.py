"""Camera Manager service."""

import logging
import platform
import threading
import time

import cv2
import numpy as np

from app.config import Settings
from app.core.exceptions import AppError
from app.schemas.camera import CameraStateDetail, CameraStatusResponse

logger = logging.getLogger(__name__)

class LatestFrame:
    def __init__(self, sequence_id: int, timestamp: float, frame: np.ndarray, jpeg_bytes: bytes | None = None):
        self.sequence_id = sequence_id
        self.timestamp = timestamp
        self.frame = frame
        self.jpeg_bytes = jpeg_bytes

class CameraManager:
    """Thread-safe manager for USB camera access providing shared frames."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.lock = threading.Lock()
        
        self.state = "STOPPED"
        self.capture: cv2.VideoCapture | None = None
        self.actual_width: int | None = None
        self.actual_height: int | None = None
        
        # Shared frame state
        self._latest_frame: LatestFrame | None = None
        self._frame_sequence_id = 0
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._condition = threading.Condition(self.lock)

    def get_status(self) -> CameraStatusResponse:
        """Get the current camera status."""
        with self.lock:
            return self._build_status()

    def _build_status(self) -> CameraStatusResponse:
        """Build status response. Must be called with lock held."""
        return CameraStatusResponse(
            success=True,
            camera=CameraStateDetail(
                state=self.state, # type: ignore[arg-type]
                index=self.settings.CAMERA_INDEX,
                device=self.settings.CAMERA_DEVICE,
                requested_width=self.settings.CAMERA_WIDTH,
                requested_height=self.settings.CAMERA_HEIGHT,
                actual_width=self.actual_width,
                actual_height=self.actual_height,
            ),
        )

    def _capture_loop(self) -> None:
        """Background thread reading frames from OpenCV."""
        while not self._stop_event.is_set():
            with self.lock:
                if self.state != "RUNNING" or not self.capture or not self.capture.isOpened():
                    break
                cap = self.capture
                
            # Read outside lock to allow concurrent get_latest_frame calls
            ret, frame = cap.read()
            
            if not ret or frame is None:
                with self.lock:
                    if not self._stop_event.is_set():
                        self.state = "ERROR"
                        logger.error("Failed to read frame from camera loop")
                with self.condition:
                    self.condition.notify_all()
                break
                
            now = time.perf_counter()
            
            # Optionally encode to JPEG here so we only do it once per frame for all MJPEG streams
            # Since JPEG encoding is somewhat costly, we can do it lazily or pre-emptively.
            # We'll pre-encode to save CPU if multiple clients connect, or lazy encode in the stream.
            # Let's lazy-encode in the stream if it's too much overhead, but doing it here guarantees
            # it's off the main HTTP thread.
            ret_jpg, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpeg_bytes = encoded.tobytes() if ret_jpg else None
            
            with self.condition:
                self._frame_sequence_id += 1
                self._latest_frame = LatestFrame(
                    sequence_id=self._frame_sequence_id,
                    timestamp=now,
                    frame=frame,
                    jpeg_bytes=jpeg_bytes
                )
                self.condition.notify_all()
                
    @property
    def condition(self) -> threading.Condition:
        return self._condition

    def start(self) -> CameraStatusResponse:
        """Start the camera."""
        with self.lock:
            if self.state == "RUNNING" and self.capture and self.capture.isOpened():
                return self._build_status()

            self.state = "STARTING"
            
            cap = None
            try:
                backend = cv2.CAP_ANY
                if platform.system() == "Linux":
                    backend = getattr(cv2, f"CAP_{self.settings.CAMERA_BACKEND.upper()}", cv2.CAP_ANY)
                    
                target = self.settings.CAMERA_DEVICE if self.settings.CAMERA_DEVICE else self.settings.CAMERA_INDEX
                cap = cv2.VideoCapture(target, backend)
                if not cap.isOpened():
                    self.state = "ERROR"
                    cap.release()
                    raise AppError(
                        code="CAMERA_START_FAILED",
                        message=f"Failed to open camera device {target}",
                        status_code=500
                    )

                # Set format and properties
                if self.settings.CAMERA_PIXEL_FORMAT:
                    fourcc = cv2.VideoWriter_fourcc(*self.settings.CAMERA_PIXEL_FORMAT)
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                    
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.CAMERA_HEIGHT)
                
                if self.settings.CAMERA_FPS > 0:
                    cap.set(cv2.CAP_PROP_FPS, self.settings.CAMERA_FPS)

                # Warmup
                start_time = time.time()
                success = False
                while (time.time() - start_time) < self.settings.CAMERA_STARTUP_TIMEOUT_SECONDS:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        pixel_range = int(np.max(frame)) - int(np.min(frame))
                        if pixel_range >= self.settings.CAMERA_MIN_PIXEL_RANGE:
                            success = True
                            break
                    time.sleep(0.1)
                
                if not success:
                    self.state = "ERROR"
                    cap.release()
                    raise AppError(
                        code="CAMERA_START_FAILED",
                        message="Camera opened but failed to provide a valid warm-up frame.",
                        status_code=500
                    )

                self.capture = cap
                self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.state = "RUNNING"
                self._stop_event.clear()
                
                # Start capture loop
                self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._capture_thread.start()
                
                logger.info("Camera %s started successfully: %sx%s", 
                            target, self.actual_width, self.actual_height)
                
                return self._build_status()

            except AppError:
                raise
            except Exception as e:
                self.state = "ERROR"
                if cap is not None:
                    cap.release()
                logger.exception("Unexpected error during camera startup")
                raise AppError(
                    code="CAMERA_START_FAILED",
                    message="Unexpected error during camera startup.",
                    status_code=500
                )

    def stop(self) -> CameraStatusResponse:
        """Stop the camera safely."""
        self._stop_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            # Wait briefly without holding lock
            self._capture_thread.join(timeout=1.0)
            
        with self.lock:
            if self.capture:
                self.capture.release()
                self.capture = None
            
            self.state = "STOPPED"
            self.actual_width = None
            self.actual_height = None
            self._latest_frame = None
            
            # Wake up anyone waiting for frames
            self.condition.notify_all()
            
            target = self.settings.CAMERA_DEVICE if self.settings.CAMERA_DEVICE else self.settings.CAMERA_INDEX
            logger.info("Camera %s stopped.", target)
            
            return self._build_status()
            
    def get_latest_frame(self) -> LatestFrame | None:
        """Returns the latest captured frame."""
        with self.lock:
            if self.state != "RUNNING" or self._latest_frame is None:
                return None
            return self._latest_frame

    def capture_frame(self, timeout=2.0) -> np.ndarray:
        """Capture a single frame from the camera. Blocks if necessary until one is ready."""
        with self.condition:
            start_time = time.time()
            while self._latest_frame is None and self.state == "RUNNING" and (time.time() - start_time) < timeout:
                self.condition.wait(timeout=0.1)

            if self.state == "ERROR":
                raise AppError(code="CAMERA_READ_FAILED", message="Camera is in error state.", status_code=500)
            if self.state != "RUNNING":
                raise AppError(code="CAMERA_NOT_RUNNING", message=f"Camera state is {self.state}.", status_code=400)
            
            if self._latest_frame:
                return self._latest_frame.frame.copy()
            else:
                raise AppError(code="CAMERA_READ_FAILED", message="Failed to read frame from camera.", status_code=500)
