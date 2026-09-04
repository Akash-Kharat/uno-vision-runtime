import sys
import os
import time
import urllib.request
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat
)

class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, model_path: str, image_paths: list[str]):
        self.image_paths = image_paths
        self.enum_data_dicts = iter([])
        self.input_name = ""
        
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = session.get_inputs()[0].name
        self.input_shape = session.get_inputs()[0].shape
        
        # Preprocess images
        self.datas = []
        for img_path in self.image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                # YOLOv8 typical preprocess
                # Letterbox / resize to 640x640
                img = cv2.resize(img, (640, 640))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = img.transpose(2, 0, 1) # HWC to CHW
                img = np.expand_dims(img, axis=0) # NCHW
                self.datas.append({self.input_name: img})
                
        self.enum_data_dicts = iter(self.datas)
        
    def get_next(self):
        return next(self.enum_data_dicts, None)

def download_sample_images(out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    urls = [
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
    ]
    paths = []
    for i, url in enumerate(urls):
        p = os.path.join(out_dir, f"sample_{i}.jpg")
        if not os.path.exists(p):
            print(f"Downloading {url} to {p}")
            urllib.request.urlretrieve(url, p)
        paths.append(p)
    return paths

def main():
    print("==================================================")
    print("TASK 017: STATIC INT8 QUANTIZATION (YOLOv8)")
    print("==================================================")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input FP32 model")
    parser.add_argument("--output", required=True, help="Output INT8 model")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input model {args.input} not found.")
        sys.exit(1)
        
    img_dir = os.path.join(os.path.dirname(args.output), "calib_images")
    img_paths = download_sample_images(img_dir)
    
    calib_reader = ImageCalibrationDataReader(args.input, img_paths)
    
    print(f"Quantizing {args.input} to {args.output}...")
    
    t0 = time.time()
    # For CNNs, static quantization is necessary to quantize Conv layers.
    quantize_static(
        model_input=args.input,
        model_output=args.output,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ, # QDQ is recommended for modern ORT
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8
    )
    t1 = time.time()
    
    print(f"Quantization finished in {t1-t0:.2f}s")
    if os.path.exists(args.output):
        orig_size = os.path.getsize(args.input) / (1024*1024)
        new_size = os.path.getsize(args.output) / (1024*1024)
        print(f"Original Size: {orig_size:.2f} MB")
        print(f"Quantized Size: {new_size:.2f} MB")

if __name__ == "__main__":
    main()
