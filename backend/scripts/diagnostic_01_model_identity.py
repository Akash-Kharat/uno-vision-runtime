import hashlib
import json
import sys
from pathlib import Path

import onnxruntime as ort

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.services.execution_provider_manager import ExecutionProviderManager
from app.services.onnx_session_factory import ONNXSessionFactory

def get_file_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

def main():
    print("==================================================")
    print("DIAGNOSTIC 01: MODEL IDENTITY & ORT SESSION OPTIONS")
    print("==================================================")
    
    settings = get_settings()
    model_path = Path(settings.MODEL_STORAGE_PATH) / "models" / "mdl_0f92eb15da2d" / "model.onnx"
    
    if not model_path.exists():
        # Fallback to default local dev model
        model_path = Path(__file__).resolve().parent.parent / "data" / "models" / "yolov8n.onnx"
        
    print(f"Target Model: {model_path}")
    print(f"File Exists:  {model_path.exists()}")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"File Size:    {size_mb:.2f} MB")
        print(f"SHA-256:      {get_file_sha256(str(model_path))}")
        
        try:
            # Direct ORT inspection
            sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            inputs = sess.get_inputs()
            outputs = sess.get_outputs()
            
            print("\n--- ONNX Model Signature ---")
            for i in inputs:
                print(f"Input:  {i.name} | Shape: {i.shape} | Type: {i.type}")
            for o in outputs:
                print(f"Output: {o.name} | Shape: {o.shape} | Type: {o.type}")
                
            meta = sess.get_modelmeta()
            print("\n--- ONNX Model Metadata ---")
            print(f"Producer: {meta.producer_name}")
            print(f"Graph:    {meta.graph_name}")
            print(f"Version:  {meta.version}")
            
        except Exception as e:
            print(f"\nFailed to load model in ORT: {e}")
            
    print("\n==================================================")
    print("FACTORY SESSION OPTIONS VERIFICATION")
    print("==================================================")
    
    ep_manager = ExecutionProviderManager()
    factory = ONNXSessionFactory(ep_manager, settings)
    
    try:
        res = factory.create(model_path, override_providers=["CPUExecutionProvider"])
        opts = res.get("options", {})
        
        print("\n--- Active SessionOptions Dictionary ---")
        print(json.dumps(opts, indent=2))
        
        print("\n--- Actual SessionProviders ---")
        print(res.get("actual_providers"))
        
    except Exception as e:
        print(f"Failed to generate factory session: {e}")
        
if __name__ == "__main__":
    main()
