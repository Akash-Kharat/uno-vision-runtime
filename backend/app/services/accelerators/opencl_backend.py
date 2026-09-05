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
            self.kernel = cl.Kernel(self.program, "yolo_preprocess_kernel")
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
            
        t_total_start = time.perf_counter()
        t_start = t_total_start
        
        target_w = profile.input.width
        target_h = profile.input.height
        
        orig_h, orig_w = frame.shape[:2]
        
        # If profile doesn't declare dims, raise so CPU fallback is triggered
        if not target_w or not target_h:
            raise RuntimeError("Profile missing input width/height — OpenCL cannot determine resize target")
        
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

        in_size = frame.nbytes
        out_size = target_w * target_h * 3 * 4 
        
        mode = getattr(self.config, "OPENCL_MEMORY_MODE", "COPY").upper()
        if mode == "AUTO":
            mode = "COPY" 
            
        is_mapped = (mode == "MAPPED")
        out_flags = cl.mem_flags.WRITE_ONLY
        if is_mapped:
            out_flags |= cl.mem_flags.ALLOC_HOST_PTR
            
        t_input_prep = time.perf_counter()
        
        in_buf, in_reused = self.pool.get_buffer("input_frame", in_size, cl.mem_flags.READ_ONLY)
        out_buf, out_reused = self.pool.get_buffer("output_tensor", out_size, out_flags)
        
        queue = self.ctx_mgr.queue
        t_buf_acquire = time.perf_counter()

        # 1. Upload Input
        upload_event = cl.enqueue_copy(queue, in_buf, frame, is_blocking=False)
        
        # 2. Kernel execution
        global_work_size = (target_w, target_h)
        in_stride = orig_w * 3
        self.kernel.set_args(
            in_buf, out_buf, np.int32(orig_w), np.int32(orig_h), np.int32(in_stride),
            np.int32(target_w), np.int32(target_h), np.int32(pad_left), np.int32(pad_top),
            np.float32(scale_x), np.float32(scale_y), np.float32(norm_scale)
        )
        
        kernel_event = cl.enqueue_nd_range_kernel(
            queue, self.kernel, global_work_size, None, wait_for=[upload_event]
        )
        
        # 3. Output Access
        if is_mapped:
            out_host, download_event = cl.enqueue_map_buffer(
                queue, out_buf, cl.map_flags.READ,
                0, (1, 3, target_h, target_w), np.float32,
                wait_for=[kernel_event], is_blocking=False
            )
        else:
            if not hasattr(self, "_host_output") or self._host_output.shape != (1, 3, target_h, target_w):
                self._host_output = np.empty((1, 3, target_h, target_w), dtype=np.float32)
            out_host = self._host_output
            
            download_event = cl.enqueue_copy(queue, out_host, out_buf, wait_for=[kernel_event], is_blocking=False)
            
        t_enqueue = time.perf_counter()
        
        # Explicit synchronization step
        download_event.wait()
        
        t_sync_wait = time.perf_counter()

        if profile.input.dtype == "tensor(float16)":
            out_host = out_host.astype(np.float16)
            
        t_output_prep = time.perf_counter()

        if profiler:
            profiler.record_custom("host_input_prepare_ms", (t_input_prep - t_start) * 1000.0)
            profiler.record_custom("host_buffer_acquire_ms", (t_buf_acquire - t_input_prep) * 1000.0)
            profiler.record_custom("host_enqueue_ms", (t_enqueue - t_buf_acquire) * 1000.0)
            profiler.record_custom("host_sync_wait_ms", (t_sync_wait - t_enqueue) * 1000.0)
            profiler.record_custom("host_output_prepare_ms", (t_output_prep - t_sync_wait) * 1000.0)
            
            if self.config.ENABLE_OPENCL_PROFILING:
                try:
                    upload_ms = (upload_event.profile.end - upload_event.profile.submit) * 1e-6
                    kernel_ms = (kernel_event.profile.end - kernel_event.profile.submit) * 1e-6
                    download_ms = (download_event.profile.end - download_event.profile.submit) * 1e-6
                    profiler.record_custom("gpu_upload_ms", upload_ms)
                    profiler.record_custom("gpu_kernel_ms", kernel_ms)
                    profiler.record_custom("gpu_download_ms", download_ms)
                except Exception:
                    pass
            profiler.record_custom("total_gpu_time_ms", (t_output_prep - t_total_start) * 1000.0)
            profiler.record_custom("input_buffer_reused", 1 if in_reused else 0)
            profiler.record_custom("output_buffer_reused", 1 if out_reused else 0)

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
