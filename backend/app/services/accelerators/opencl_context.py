"""OpenCL Context and Device Discovery."""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

class OpenCLContext:
    def __init__(self, config):
        self.config = config
        self.platform = None
        self.device = None
        self.context = None
        self.queue = None
        self.available = False
        self.fp16_supported = False
        self.device_info = {}

    def discover_and_initialize(self) -> None:
        if not HAS_OPENCL or not self.config.ENABLE_OPENCL:
            logger.info("OpenCL backend disabled or pyopencl not installed.")
            return

        try:
            platforms = cl.get_platforms()
            if not platforms:
                logger.warning("No OpenCL platforms found.")
                return

            # Match vendor preference
            preferred_vendor = self.config.OPENCL_PREFERRED_VENDOR.lower()
            selected_platform = None
            
            for p in platforms:
                if preferred_vendor in p.name.lower() or preferred_vendor in p.vendor.lower():
                    selected_platform = p
                    break
            
            if not selected_platform:
                selected_platform = platforms[0]

            self.platform = selected_platform
            
            # Match device type
            device_type = cl.device_type.ALL
            cfg_type = self.config.OPENCL_DEVICE_TYPE.upper()
            if cfg_type == "GPU":
                device_type = cl.device_type.GPU
            elif cfg_type == "CPU":
                device_type = cl.device_type.CPU
                
            devices = selected_platform.get_devices(device_type=device_type)
            if not devices:
                # Fallback to ANY device on this platform
                devices = selected_platform.get_devices(device_type=cl.device_type.ALL)
                if not devices:
                    logger.warning("No OpenCL devices found on platform.")
                    return
                    
            self.device = devices[0]
            
            # Create context and queue
            self.context = cl.Context([self.device])
            
            # Enable profiling if configured
            properties = cl.command_queue_properties.PROFILING_ENABLE if self.config.ENABLE_OPENCL_PROFILING else 0
            self.queue = cl.CommandQueue(self.context, properties=properties)
            
            # Gather diagnostics
            extensions = self.device.extensions
            self.fp16_supported = 'cl_khr_fp16' in extensions
            
            self.device_info = {
                "available": True,
                "active": True, # Will be set by backend if selected
                "platform": self.platform.name.strip(),
                "platform_vendor": self.platform.vendor.strip(),
                "platform_version": self.platform.version.strip(),
                "device": self.device.name.strip(),
                "vendor": self.device.vendor.strip(),
                "device_version": self.device.version.strip(),
                "device_type": cl.device_type.to_string(self.device.type),
                "opencl_c_version": self.device.opencl_c_version.strip(),
                "global_memory_size": self.device.global_mem_size,
                "max_memory_allocation": self.device.max_mem_alloc_size,
                "image_support": bool(self.device.image_support),
                "fp16_supported": self.fp16_supported
            }
            
            self.available = True
            logger.info(f"OpenCL initialized successfully on {self.device.name}")
            
        except Exception as e:
            logger.exception(f"OpenCL initialization failed: {e}")
            self.available = False
            self.context = None
            self.queue = None

    def get_info(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False}
        return self.device_info
