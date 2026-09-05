import os
from ultralytics import YOLO

def main():
    os.makedirs("data/models/experiments/resolutions", exist_ok=True)
    model = YOLO("yolov8n.pt")
    
    resolutions = [640, 512, 416, 384, 320]
    for res in resolutions:
        print(f"Exporting YOLOv8n at {res}x{res}...")
        out_name = f"yolov8n_{res}"
        # Ultralytics export
        path = model.export(format="onnx", imgsz=res, half=False, simplify=True)
        
        # move to our experimental dir
        os.rename(path, f"data/models/experiments/resolutions/{out_name}.onnx")
        print(f"Saved to data/models/experiments/resolutions/{out_name}.onnx")

if __name__ == "__main__":
    main()
