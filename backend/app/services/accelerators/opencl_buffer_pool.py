"""OpenCL Buffer Pool."""

import logging
from typing import Tuple, Any

try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

logger = logging.getLogger(__name__)

class OpenCLBufferPool:
    """Manages reusable OpenCL buffers for a specific tensor shape/dtype."""
    
    def __init__(self, context: 'cl.Context'):
        self.context = context
        self._buffers = {}
        
    def get_buffer(self, name: str, size_bytes: int, flags: Any = None) -> 'cl.Buffer':
        """
        Gets an existing buffer matching the size, or allocates a new one.
        If size changes, releases the old buffer safely and reallocates.
        """
        if not HAS_OPENCL:
            return None
            
        if flags is None:
            flags = cl.mem_flags.READ_WRITE
            
        if name in self._buffers:
            existing_buf, existing_size = self._buffers[name]
            if existing_size == size_bytes:
                return existing_buf
            else:
                # Size mismatch, release old buffer
                logger.debug(f"Reallocating OpenCL buffer '{name}' from {existing_size} to {size_bytes} bytes.")
                existing_buf.release()
                del self._buffers[name]
                
        # Allocate new buffer
        buf = cl.Buffer(self.context, flags, size=size_bytes)
        self._buffers[name] = (buf, size_bytes)
        return buf

    def release_all(self):
        """Release all allocated buffers."""
        for name, (buf, _) in self._buffers.items():
            try:
                buf.release()
            except Exception:
                pass
        self._buffers.clear()
