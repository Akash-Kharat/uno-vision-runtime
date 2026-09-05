from ultralytics import YOLO

def main():
    print("=========================================")
    print("VALIDATING YOLOv8n (640) - BASELINE")
    print("=========================================")
    baseline = YOLO("data/models/experiments/resolutions/yolov8n_640.onnx", task='detect')
    res_base = baseline.val(data="coco128.yaml", imgsz=640)
    
    print("\n=========================================")
    print("VALIDATING YOLO11n (416) - CANDIDATE")
    print("=========================================")
    candidate = YOLO("data/models/experiments/yolo11/yolo11n_416.onnx", task='detect')
    res_cand = candidate.val(data="coco128.yaml", imgsz=416)
    
    print("\n--- RESULTS COMPARISON ---")
    print(f"YOLOv8n @ 640: mAP50: {res_base.box.map50:.3f}, mAP50-95: {res_base.box.map:.3f}, P: {res_base.box.mp:.3f}, R: {res_base.box.mr:.3f}")
    print(f"YOLO11n @ 416: mAP50: {res_cand.box.map50:.3f}, mAP50-95: {res_cand.box.map:.3f}, P: {res_cand.box.mp:.3f}, R: {res_cand.box.mr:.3f}")

if __name__ == "__main__":
    main()
