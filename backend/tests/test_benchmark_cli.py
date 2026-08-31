import pytest
import json
import subprocess
import os
from tempfile import NamedTemporaryFile

def test_cli_parsing_empty_data():
    mock_payload = {
        "success": True,
        "model_id": "test_id",
        "model_name": "test_model.onnx",
        "input_shape": [1, 3, 640, 640],
        "iterations": 100,
        "successful_iterations": 0,
        "failed_iterations": 100,
        "total_ms": None,
        "capture_ms": None,
        "preprocessing_ms": None,
        "inference_ms": None,
        "postprocessing_ms": None,
        "effective_fps": 0.0
    }
    
    test_script = f"""
import sys
import json
import requests

class MockResponse:
    def __init__(self):
        self.status_code = 200
    def json(self):
        return payload

def mock_post(*args, **kwargs):
    return MockResponse()

requests.post = mock_post

payload_str = {repr(json.dumps(mock_payload))}
payload = json.loads(payload_str)

import runpy
sys.argv = ['benchmark_detection.py']
runpy.run_path('scripts/benchmark_detection.py', run_name='__main__')
"""
    
    with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
        
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        result = subprocess.run(["python", script_path], capture_output=True, text=True, env=env)
        
        out = result.stdout
        if result.returncode != 0:
            print("STDERR:", result.stderr)
        assert "Total:" in out
        assert "Mean        N/A" in out
        assert "P50         N/A" in out
        assert result.returncode == 0
        
    finally:
        os.remove(script_path)


def test_cli_parsing_valid_data():
    mock_payload = {
        "success": True,
        "model_id": "test_id",
        "model_name": "test_model.onnx",
        "input_shape": [1, 3, 640, 640],
        "iterations": 100,
        "successful_iterations": 100,
        "failed_iterations": 0,
        "total_ms": {"mean": 10.511, "p50": 10.5, "p95": 11.0, "p99": 12.0, "min": 10.0, "max": 12.0},
        "capture_ms": {"mean": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0, "min": 1.0, "max": 1.0},
        "preprocessing_ms": {"mean": 2.0, "p50": 2.0, "p95": 2.0, "p99": 2.0, "min": 2.0, "max": 2.0},
        "inference_ms": {"mean": 3.0, "p50": 3.0, "p95": 3.0, "p99": 3.0, "min": 3.0, "max": 3.0},
        "postprocessing_ms": {"mean": 4.0, "p50": 4.0, "p95": 4.0, "p99": 4.0, "min": 4.0, "max": 4.0},
        "effective_fps": 95.5
    }
    
    test_script = f"""
import sys
import json
import requests

class MockResponse:
    def __init__(self):
        self.status_code = 200
    def json(self):
        return payload

def mock_post(*args, **kwargs):
    return MockResponse()

requests.post = mock_post

payload_str = {repr(json.dumps(mock_payload))}
payload = json.loads(payload_str)

import runpy
sys.argv = ['benchmark_detection.py']
runpy.run_path('scripts/benchmark_detection.py', run_name='__main__')
"""
    
    with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
        
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        result = subprocess.run(["python", script_path], capture_output=True, text=True, env=env)
        
        out = result.stdout
        if result.returncode != 0:
            print("STDERR:", result.stderr)
        assert "Total:" in out
        assert "Mean        10.51 ms" in out
        assert "P95         11.00 ms" in out
        assert result.returncode == 0
        
    finally:
        os.remove(script_path)
