"""Camera Manager service."""

import logging
import threading
import time

import cv2
import numpy as np

from app.config import Settings
from app.core.exceptions import AppError
from app.schemas.camera import CameraStateDetail, CameraStatusResponse

logger = logging.getLogger(__name__)


class CameraManager:
    """Thread-safe manager for USB camera access."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.lock = threading.Lock()
        
        self.state = "STOPPED"
        self.capture: cv2.VideoCapture | None = None
        self.actual_width: int | None = None
        self.actual_height: int | None = None

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
                requested_width=self.settings.CAMERA_WIDTH,
                requested_height=self.settings.CAMERA_HEIGHT,
                actual_width=self.actual_width,
                actual_height=self.actual_height,
            ),
        )

    def start(self) -> CameraStatusResponse:
        """Start the camera."""
        with self.lock:
            if self.state == "RUNNING" and self.capture and self.capture.isOpened():
                return self._build_status()

            self.state = "STARTING"
            
            cap = None
            try:
                cap = cv2.VideoCapture(self.settings.CAMERA_INDEX)
                if not cap.isOpened():
                    self.state = "ERROR"
                    cap.release()
                    raise AppError(
                        code="CAMERA_START_FAILED",
                        message=f"Failed to open camera index {self.settings.CAMERA_INDEX}",
                        status_code=500
                    )

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.CAMERA_HEIGHT)

                # Verify frame read
                # Note: CAMERA_STARTUP_TIMEOUT_SECONDS is a retry deadline, not a strict interrupt.
                # If cap.read() blocks indefinitely at the driver/OS level, this loop cannot preempt it.
                # It only acts as a timeout if cap.read() returns quickly with False.
                start_time = time.time()
                success = False
                while (time.time() - start_time) < self.settings.CAMERA_STARTUP_TIMEOUT_SECONDS:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        success = True
                        break
                    time.sleep(0.1)
                
                if not success:
                    self.state = "ERROR"
                    cap.release()
                    raise AppError(
                        code="CAMERA_START_FAILED",
                        message="Camera opened but failed to provide a frame.",
                        status_code=500
                    )

                self.capture = cap
                self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.state = "RUNNING"
                
                logger.info("Camera %s started successfully: %sx%s", 
                            self.settings.CAMERA_INDEX, self.actual_width, self.actual_height)
                
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
        with self.lock:
            if self.capture:
                self.capture.release()
                self.capture = None
            
            self.state = "STOPPED"
            self.actual_width = None
            self.actual_height = None
            logger.info("Camera %s stopped.", self.settings.CAMERA_INDEX)
            
            return self._build_status()

    def capture_frame(self) -> np.ndarray:
        """Capture a single frame from the camera."""
        with self.lock:
            if self.state != "RUNNING" or not self.capture or not self.capture.isOpened():
                raise AppError(
                    code="CAMERA_NOT_RUNNING",
                    message="Camera is not running.",
                    status_code=400
                )

            ret, frame = self.capture.read()
            if not ret or frame is None:
                self.state = "ERROR"
                self.capture.release()
                self.capture = None
                self.actual_width = None
                self.actual_height = None
                logger.error("Failed to read frame from camera")
                raise AppError(
                    code="CAMERA_READ_FAILED",
                    message="Failed to read frame from camera.",
                    status_code=500
                )

            return frame
