import sys
import subprocess
import pytest
from pathlib import Path

def test_sweep_requires_model():
    script_path = Path(__file__).parent.parent / "scripts" / "benchmark_ort_cpu_sweep.py"
    
    res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    assert res.returncode != 0
    assert "the following arguments are required: --model" in res.stderr

def test_sweep_invalid_model_path():
    script_path = Path(__file__).parent.parent / "scripts" / "benchmark_ort_cpu_sweep.py"
    
    res = subprocess.run([sys.executable, str(script_path), "--model", "nonexistent/model.onnx"], capture_output=True, text=True)
    assert res.returncode != 0
    assert "Error: Model file not found at nonexistent/model.onnx" in res.stdout or "Error: Model file not found" in res.stdout
    
def test_sweep_propagates_model_path(tmp_path):
    # Create a dummy model file just to pass the exists() check
    dummy_model = tmp_path / "dummy_model.onnx"
    dummy_model.touch()
    
    script_path = Path(__file__).parent.parent / "scripts" / "benchmark_ort_cpu_sweep.py"
    
    # We can mock subprocess.run in the script, or just patch it
    import unittest.mock as mock
    import importlib.util
    
    # Load the script as a module
    spec = importlib.util.spec_from_file_location("sweep", str(script_path))
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)
    
    # Mock subprocess.run and argparse
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.MagicMock(stdout="Mean: 10.0\nP50: 10.0\nP95: 11.0\nP99: 12.0\nEffective FPS: 100.0\n")
        
        with mock.patch("sys.argv", ["benchmark_ort_cpu_sweep.py", "--model", str(dummy_model)]):
            sweep_module.run_sweep()
            
            # Verify subprocess.run was called with the model path
            assert mock_run.call_count == 8  # 4 intra * 2 inter
            
            # Check the first call
            args, kwargs = mock_run.call_args_list[0]
            cmd = args[0]
            
            assert "--model" in cmd
            model_index = cmd.index("--model")
            assert cmd[model_index + 1] == str(dummy_model)
