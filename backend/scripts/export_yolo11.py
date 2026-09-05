import os
from ultralytics import YOLO

def main():
    os.makedirs("data/models/experiments/yolo11", exist_ok=True)
    model = YOLO("yolo11n.pt")
    
    resolutions = [640, 416]
    for res in resolutions:
        print(f"Exporting YOLO11n at {res}x{res}...")
        out_name = f"yolo11n_{res}"
        path = model.export(format="onnx", imgsz=res, half=False, simplify=True)
        os.rename(path, f"data/models/experiments/yolo11/{out_name}.onnx")

if __name__ == "__main__":
    main()
