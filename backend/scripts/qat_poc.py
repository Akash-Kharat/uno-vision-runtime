import torch
from ultralytics import YOLO
import onnx
import sys
import os

def check_graph(model_path):
    print(f"--- Inspecting exported QAT model: {model_path} ---")
    model = onnx.load(model_path)
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        
    print(f"Total Nodes: {len(model.graph.node)}")
    print(f"Conv: {op_counts.get('Conv', 0)}")
    print(f"QLinearConv: {op_counts.get('QLinearConv', 0)}")
    print(f"QuantizeLinear: {op_counts.get('QuantizeLinear', 0)}")
    print(f"DequantizeLinear: {op_counts.get('DequantizeLinear', 0)}")
    print(f"MatMul: {op_counts.get('MatMul', 0)}")

def main():
    print("Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    pytorch_model = model.model
    pytorch_model.train()
    
    # Set qconfig
    pytorch_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(pytorch_model, inplace=True)
    
    print("Simulating 1 step of training...")
    # Just a dummy forward pass
    dummy_input = torch.randn(1, 3, 640, 640)
    pytorch_model(dummy_input)
    
    print("Converting to quantized model...")
    pytorch_model.eval()
    torch.quantization.convert(pytorch_model, inplace=True)
    
    os.makedirs("data/models/experiments/qat", exist_ok=True)
    out_path = "data/models/experiments/qat/yolov8n_qat_poc.onnx"
    
    print(f"Exporting to {out_path}...")
    try:
        torch.onnx.export(
            pytorch_model, 
            dummy_input, 
            out_path,
            opset_version=13,
            input_names=["images"],
            output_names=["output"]
        )
        print("Export successful!")
        check_graph(out_path)
    except Exception as e:
        print(f"Export FAILED: {e}")

if __name__ == "__main__":
    main()
