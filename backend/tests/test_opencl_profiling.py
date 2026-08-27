import pytest
from unittest.mock import MagicMock
from app.services.accelerators.opencl_buffer_pool import OpenCLBufferPool

def test_buffer_pool_reuses_buffers():
    # We mock HAS_OPENCL and cl.Buffer
    import app.services.accelerators.opencl_buffer_pool as pool_module
    
    # Store original
    orig_has_opencl = pool_module.HAS_OPENCL
    pool_module.HAS_OPENCL = True
    
    class DummyBuffer:
        def release(self):
            pass
            
    # Mock cl
    cl_mock = MagicMock()
    cl_mock.Buffer.return_value = DummyBuffer()
    cl_mock.mem_flags.READ_WRITE = 1
    
    orig_cl = None
    if hasattr(pool_module, 'cl'):
        orig_cl = pool_module.cl
    pool_module.cl = cl_mock
    
    try:
        context = MagicMock()
        pool = OpenCLBufferPool(context)
        
        # First allocation
        buf1, reused1 = pool.get_buffer("test", 1024)
        assert reused1 is False
        assert pool.metrics["allocations"]["test"] == 1
        
        # Second allocation (same size, should reuse)
        buf2, reused2 = pool.get_buffer("test", 1024)
        assert reused2 is True
        assert pool.metrics["reuses"]["test"] == 1
        assert buf1 is buf2
        
        # Third allocation (different size, should reallocate)
        buf3, reused3 = pool.get_buffer("test", 2048)
        assert reused3 is False
        assert pool.metrics["allocations"]["test"] == 2
        
    finally:
        # Restore
        pool_module.HAS_OPENCL = orig_has_opencl
        if orig_cl:
            pool_module.cl = orig_cl
        else:
            del pool_module.cl
