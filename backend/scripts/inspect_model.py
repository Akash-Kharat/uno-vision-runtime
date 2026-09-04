import hashlib
from pathlib import Path
import onnxruntime as ort

def get_file_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()

def inspect_model(model_path):
    print("==================================================")
    print("PART 1 — MODEL INSPECTION")
    print("==================================================")
    print(f"Model Path: {model_path}")
    path = Path(model_path)
    
    if not path.exists():
        print("Model file does not exist!")
        return
        
    print(f"File Size: {path.stat().st_size / (1024*1024):.2f} MB")
    print(f"SHA-256: {get_file_sha256(str(path))}")
    
    # Load ONNX model using ONNX Runtime
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        
        print("\n--- Inputs ---")
        for inp in sess.get_inputs():
            print(f"Name: {inp.name} | Shape: {inp.shape} | Type: {inp.type}")
            
        print("\n--- Outputs ---")
        for out in sess.get_outputs():
            print(f"Name: {out.name} | Shape: {out.shape} | Type: {out.type}")
            
        meta = sess.get_modelmeta()
        print("\n--- Model Metadata ---")
        print(f"Producer: {meta.producer_name}")
        print(f"Graph: {meta.graph_name}")
        print(f"Version: {meta.version}")
        
    except Exception as e:
        print(f"Failed to inspect model with ONNX Runtime: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/yolov8n.onnx", help="Model path")
    args = parser.parse_args()
    inspect_model(args.model)
