"""OpenCL Compute Backend Implementation."""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

from app.schemas.profile import ModelProfile
from app.domain.detection import PreprocessedInput
from app.services.accelerators.base import ComputeBackend
from app.services.accelerators.opencl_context import OpenCLContext, HAS_OPENCL
from app.services.accelerators.opencl_buffer_pool import OpenCLBufferPool

logger = logging.getLogger(__name__)

if HAS_OPENCL:
    import pyopencl as cl

class OpenCLBackend(ComputeBackend):
    def __init__(self, config):
        self.config = config
        self.ctx_mgr = OpenCLContext(config)
        self.pool = None
        self.program = None
        self.available = False

    def initialize(self) -> None:
        if not HAS_OPENCL or not self.config.ENABLE_OPENCL:
            return
            
        self.ctx_mgr.discover_and_initialize()
        if not self.ctx_mgr.available:
            return
            
        self.pool = OpenCLBufferPool(self.ctx_mgr.context)
        
        # Load kernel
        kernel_path = Path(__file__).parent / "kernels" / "yolo_preprocess.cl"
        try:
            with open(kernel_path, "r") as f:
                kernel_src = f.read()
                
            self.program = cl.Program(self.ctx_mgr.context, kernel_src).build()
            self.available = True
            logger.info("OpenCL Backend initialized and kernels compiled.")
        except Exception as e:
            logger.exception(f"Failed to compile OpenCL kernels: {e}")
            self.available = False

    def is_available(self) -> bool:
        return self.available

    def get_device_info(self) -> Dict[str, Any]:
        return self.ctx_mgr.get_info()

    def preprocess_yolo(self, frame: np.ndarray, profile: ModelProfile, profiler: Any = None) -> PreprocessedInput:
        if not self.available:
            raise RuntimeError("OpenCL Backend is not available.")
            
        target_w = profile.input.width
        target_h = profile.input.height
        
        orig_h, orig_w = frame.shape[:2]
        
        # Determine scaling and padding for Letterbox
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        
        scale_x = orig_w / float(new_w)
        scale_y = orig_h / float(new_h)
        
        norm_scale = profile.preprocessing.normalization.scale or (1.0 / 255.0)

        # Buffers
        # Frame is uint8, target is float32
        in_size = frame.nbytes
        out_size = target_w * target_h * 3 * 4 # float32 = 4 bytes
        
        in_buf = self.pool.get_buffer("input_frame", in_size, cl.mem_flags.READ_ONLY)
        out_buf = self.pool.get_buffer("output_tensor", out_size, cl.mem_flags.WRITE_ONLY)
        
        # Host array for output (1, 3, target_h, target_w)
        out_host = np.empty((1, 3, target_h, target_w), dtype=np.float32)

        queue = self.ctx_mgr.queue

        t_start = time.perf_counter()

        # 1. Upload
        upload_event = cl.enqueue_copy(queue, in_buf, frame, is_blocking=False)
        
        # 2. Kernel execution
        global_work_size = (target_w, target_h)
        kernel = self.program.yolo_preprocess_kernel
        
        # Kernel args:
        # input, output, in_width, in_height, in_stride, out_width, out_height, pad_x, pad_y, scale_x, scale_y, norm_scale
        in_stride = orig_w * 3
        kernel.set_args(
            in_buf, 
            out_buf,
            np.int32(orig_w),
            np.int32(orig_h),
            np.int32(in_stride),
            np.int32(target_w),
            np.int32(target_h),
            np.int32(pad_left),
            np.int32(pad_top),
            np.float32(scale_x),
            np.float32(scale_y),
            np.float32(norm_scale)
        )
        
        kernel_event = cl.enqueue_nd_range_kernel(
            queue, 
            kernel, 
            global_work_size, 
            None,
            wait_for=[upload_event]
        )
        
        # 3. Download
        download_event = cl.enqueue_copy(queue, out_host, out_buf, wait_for=[kernel_event], is_blocking=True)
        
        t_end = time.perf_counter()

        # Profiling
        if self.config.ENABLE_OPENCL_PROFILING and profiler:
            try:
                # Timestamps are in nanoseconds
                upload_start = upload_event.profile.submit
                upload_end = upload_event.profile.end
                kernel_start = kernel_event.profile.submit
                kernel_end = kernel_event.profile.end
                download_start = download_event.profile.submit
                download_end = download_event.profile.end
                
                profiler.record_custom("gpu_upload_ms", (upload_end - upload_start) * 1e-6)
                profiler.record_custom("gpu_kernel_ms", (kernel_end - kernel_start) * 1e-6)
                profiler.record_custom("gpu_download_ms", (download_end - download_start) * 1e-6)
            except Exception:
                pass
                
        if profiler:
            profiler.record_custom("total_gpu_time_ms", (t_end - t_start) * 1000.0)

        # Enforce Float16 if model requests it
        if profile.input.dtype == "tensor(float16)":
            out_host = out_host.astype(np.float16)

        return PreprocessedInput(
            tensor=out_host,
            original_width=orig_w,
            original_height=orig_h,
            model_width=target_w,
            model_height=target_h,
            scale_x=1.0 / scale,
            scale_y=1.0 / scale,
            pad_x=pad_left,
            pad_y=pad_top
        )

    def shutdown(self) -> None:
        if self.pool:
            self.pool.release_all()
