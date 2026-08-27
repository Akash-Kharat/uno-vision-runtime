# Hardware Validation Procedure (TASK 006.1)

This document describes the steps required to validate the Generic Object Detection Inference Engine natively on the Arduino UNO Q target running Linux aarch64.

## Prerequisites

1. An Arduino UNO Q booted into its standard Linux distribution.
2. The `uno-vision-runtime` backend running natively (not inside Docker) on port `8000`.
3. A USB camera connected and mapped to `/dev/video0`.
4. A known valid ONNX object detection model (e.g., `yolov8n.onnx` or equivalent).

## Validation Steps

### 1. Start the Backend
Execute the standard start command in the terminal on the UNO Q:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Start the Camera
```bash
curl -X POST http://localhost:8000/api/v1/camera/start
```

### 3. Upload the Model
```bash
curl -X POST -F "file=@yolov8n.onnx" http://localhost:8000/api/v1/models/upload
```
*Note the returned `model_id` for subsequent steps.*

### 4. Inspect Model (Optional but recommended)
```bash
curl http://localhost:8000/api/v1/models/<model_id>/inspect
```
Review the input shapes and classes detected to ensure ONNX parsed successfully.

### 5. Configure the Profile
Upload the required `profile.json` setting up the execution metrics.
```bash
curl -X PUT -H "Content-Type: application/json" -d '{
  "task": "OBJECT_DETECTION",
  "input": {
    "layout": "NCHW",
    "color_format": "RGB"
  },
  "preprocessing": {
    "resize": "LETTERBOX",
    "normalization": {
      "type": "SCALE_0_1"
    }
  },
  "output": {
    "processor": "YOLO",
    "bbox_format": "CXCYWH",
    "confidence_interpretation": "DIRECT",
    "confidence_threshold": 0.5,
    "nms_threshold": 0.45
  },
  "classes": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
}' http://localhost:8000/api/v1/models/<model_id>/profile
```

### 6. Activate the Model
```bash
curl -X POST http://localhost:8000/api/v1/models/<model_id>/activate
```
Wait for successful atomic swap confirming the runtime mapped it cleanly.

### 7. Run Detections & Validate Timings
Utilize the testing script to fire repeated executions and calculate latency averages.
```bash
python scripts/validate_detection.py --iterations 100
```
Review the console output verifying stable bounding boxes without severe memory leaks (verify using `htop` in another terminal).

### 8. Visual Bounding Box Verification
Utilize the internal debug tool to download an annotated JPEG bounding box image.
```bash
curl -o debug_frame.jpg http://localhost:8000/api/v1/detect/debug/frame
```
Download `debug_frame.jpg` and visually confirm that bounding boxes properly align to the physical objects in the frame, proving that letterbox mapping is physically correct.
