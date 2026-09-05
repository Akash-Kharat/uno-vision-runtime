import torch
from ultralytics import YOLO

def check_qat_feasibility():
    print("Loading YOLOv8n for QAT feasibility check...")
    model = YOLO("yolov8n.pt")
    
    pytorch_model = model.model
    pytorch_model.train()
    
    # Try to set qconfig
    pytorch_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    print("\nAttempting torch.quantization.prepare_qat...")
    try:
        torch.quantization.prepare_qat(pytorch_model, inplace=True)
        print("prepare_qat succeeded! The model architecture is natively supported by PyTorch QAT.")
        return True
    except Exception as e:
        print(f"prepare_qat FAILED with error:\n{e}")
        return False

if __name__ == "__main__":
    check_qat_feasibility()
