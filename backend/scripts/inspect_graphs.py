import onnx
import sys

def inspect_model(model_path):
    print(f"--- Inspecting {model_path} ---")
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"Failed to load: {e}")
        return
        
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        
    print(f"Total Nodes: {len(model.graph.node)}")
    print(f"Conv: {op_counts.get('Conv', 0)}")
    print(f"QLinearConv: {op_counts.get('QLinearConv', 0)}")
    print(f"QuantizeLinear: {op_counts.get('QuantizeLinear', 0)}")
    print(f"DequantizeLinear: {op_counts.get('DequantizeLinear', 0)}")
    print(f"MatMul: {op_counts.get('MatMul', 0)}")
    print(f"QLinearMatMul: {op_counts.get('QLinearMatMul', 0)}")
    print(f"Concat: {op_counts.get('Concat', 0)}")
    print(f"Mul: {op_counts.get('Mul', 0)}")
    print(f"Add: {op_counts.get('Add', 0)}")
    print()

def main():
    models = [
        "model.onnx",
        "data/models/experiments/yolov8n_int8.onnx", # Task 17 naive minmax
        "data/models/experiments/yolov8n_int8_percentile.onnx" # Task 18 percentile exclude head
    ]
    for m in models:
        inspect_model(m)

if __name__ == "__main__":
    main()
