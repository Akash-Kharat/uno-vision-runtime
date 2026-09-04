import sys
import os
import time
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
import glob

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod
)

class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, model_path: str, image_dir: str, limit: int = 100):
        self.enum_data_dicts = iter([])
        
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = session.get_inputs()[0].name
        
        img_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        img_paths = sorted(img_paths)[:limit]
        
        self.datas = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (640, 640))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = img.transpose(2, 0, 1) # HWC to CHW
                img = np.expand_dims(img, axis=0) # NCHW
                self.datas.append({self.input_name: img})
                
        self.enum_data_dicts = iter(self.datas)
        
    def get_next(self):
        return next(self.enum_data_dicts, None)

def get_nodes_to_exclude(model_path):
    # For YOLOv8, excluding output layers (Concat, Split, Mul, Sigmoid etc near the end)
    # can recover accuracy. We can identify them by names or just exclude non-Conv ops.
    import onnx
    model = onnx.load(model_path)
    exclude_nodes = []
    for node in model.graph.node:
        if node.op_type in ["Concat", "Split", "Sigmoid", "Mul", "Add", "Softmax"]:
            exclude_nodes.append(node.name)
    return exclude_nodes

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calib_dir", required=True)
    parser.add_argument("--calib_count", type=int, default=100)
    parser.add_argument("--method", choices=["minmax", "entropy", "percentile"], default="minmax")
    parser.add_argument("--exclude_head", action="store_true")
    args = parser.parse_args()
    
    calib_reader = ImageCalibrationDataReader(args.input, args.calib_dir, args.calib_count)
    
    if args.method == "entropy":
        calib_method = CalibrationMethod.Entropy
    elif args.method == "percentile":
        calib_method = CalibrationMethod.Percentile
    else:
        calib_method = CalibrationMethod.MinMax
        
    nodes_to_exclude = []
    if args.exclude_head:
        nodes_to_exclude = get_nodes_to_exclude(args.input)
        
    print(f"Quantizing {args.input} to {args.output} using {args.method} (Calib: {args.calib_count} images)")
    
    t0 = time.time()
    quantize_static(
        model_input=args.input,
        model_output=args.output,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=calib_method,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=True # Per-channel weight quant often better for accuracy
    )
    t1 = time.time()
    print(f"Quantization finished in {t1-t0:.2f}s")

if __name__ == "__main__":
    main()
