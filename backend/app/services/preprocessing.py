"""Preprocessing logic for image frames."""

import cv2
import numpy as np

from typing import Any
from app.domain.enums import ColorFormat, ResizeMethod, NormalizationType, InputLayout
from app.schemas.profile import ModelProfile
from app.domain.detection import PreprocessedInput
from app.core.exceptions import AppError

class Preprocessor:
    """Handles image preprocessing for model inference."""
    
    def __init__(self, backend=None, config=None):
        self.backend = backend
        self.config = config

    def preprocess_frame(self, frame: np.ndarray, profile: ModelProfile, profiler: Any = None, inspection=None) -> PreprocessedInput:
        """Preprocesses a BGR OpenCV frame according to the model profile."""
        if frame is None or frame.size == 0:
            raise AppError(code="PREPROCESSING_FAILED", message="Empty frame provided", status_code=500)
            
        # Try OpenCL if configured
        if self.backend and self.backend.is_available() and self.config:
            backend_mode = self.config.PREPROCESSING_BACKEND.upper()
            if backend_mode in ("AUTO", "OPENCL"):
                try:
                    return self.backend.preprocess_yolo(frame, profile, profiler)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"OpenCL preprocessing failed, falling back to CPU: {e}")
                    
        # CPU Fallback
        original_height, original_width = frame.shape[:2]

        input_prof = profile.input
        target_w = input_prof.width
        target_h = input_prof.height

        # If not set in profile, derive from ONNX inspection shape (e.g. [1, C, H, W])
        if (not target_w or not target_h) and inspection is not None:
            try:
                shape = inspection.inputs[0].shape  # e.g. [1, 3, 416, 416]
                if shape and len(shape) == 4:
                    target_h = int(shape[2])
                    target_w = int(shape[3])
            except Exception:
                pass

        if not target_w or not target_h:
            raise AppError(code="PREPROCESSING_FAILED", message="Model profile missing input dimensions", status_code=500)

        # 1. Color formatting (Assuming frame comes as BGR from CameraManager)
        img = frame
        if input_prof.color_format == ColorFormat.RGB:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif input_prof.color_format == ColorFormat.GRAYSCALE:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)

        # 2. Resizing
        resize_mode = profile.preprocessing.resize
        scale_x = 1.0
        scale_y = 1.0
        pad_x = 0
        pad_y = 0

        if resize_mode == ResizeMethod.STRETCH:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            scale_x = original_width / target_w
            scale_y = original_height / target_h

        elif resize_mode == ResizeMethod.LETTERBOX:
            scale = min(target_w / original_width, target_h / original_height)
            new_w = int(original_width * scale)
            new_h = int(original_height * scale)
            
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            
            # Pad to center
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            
            img = cv2.copyMakeBorder(
                img_resized, pad_top, pad_bottom, pad_left, pad_right, 
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
            
            scale_x = 1.0 / scale
            scale_y = 1.0 / scale
            pad_x = pad_left
            pad_y = pad_top
        else:
            # NONE or CUSTOM, do nothing, just ensure it matches bounds
            if img.shape[1] != target_w or img.shape[0] != target_h:
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            scale_x = original_width / target_w
            scale_y = original_height / target_h

        # 3. Normalization
        img = img.astype(np.float32)
        norm_type = profile.preprocessing.normalization.type
        if norm_type == NormalizationType.SCALE_0_1:
            scale_val = profile.preprocessing.normalization.scale or (1.0 / 255.0)
            img *= scale_val
        elif norm_type == NormalizationType.MEAN_STD:
            mean = profile.preprocessing.normalization.mean
            std = profile.preprocessing.normalization.std
            if mean and std:
                img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        
        # 4. Layout
        if input_prof.layout == InputLayout.NCHW:
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
            
        # 5. Add Batch Dimension
        img = np.expand_dims(img, axis=0)

        # Ensure correct type based on profile (usually float32)
        if input_prof.dtype == "tensor(float)":
            img = img.astype(np.float32)
        elif input_prof.dtype == "tensor(float16)":
            img = img.astype(np.float16)
        
        return PreprocessedInput(
            tensor=img,
            original_width=original_width,
            original_height=original_height,
            model_width=target_w,
            model_height=target_h,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_x=pad_x,
            pad_y=pad_y
        )
