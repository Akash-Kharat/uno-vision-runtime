import cv2
import time
import numpy as np
import onnxruntime as ort

def profile_pipeline():
    model_path = "data/models/experiments/yolo11/yolo11n_416.onnx"
    print(f"Profiling Pipeline for {model_path}...")
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2
    session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 1. Capture (simulated with random numpy or video read)
    # We will simulate a 720p camera frame
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    times = {'capture': [], 'preprocess': [], 'inference': [], 'postprocess': []}
    
    for _ in range(25): # 5 warmup, 20 measure
        # Capture (stub time, realistically ~15ms for v4l2)
        t0 = time.perf_counter()
        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (416, 416))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        t1 = time.perf_counter()
        
        # Inference
        outputs = session.run(None, {input_name: img})
        t2 = time.perf_counter()
        
        # Postprocess (NMS simulation)
        # YOLO11n outputs [1, 84, 3549] for 416x416
        preds = outputs[0][0]
        # Transpose to [3549, 84]
        preds = preds.transpose(1, 0)
        boxes = preds[:, :4]
        scores = np.max(preds[:, 4:], axis=1)
        classes = np.argmax(preds[:, 4:], axis=1)
        
        # Filter by conf
        mask = scores > 0.25
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]
        
        # OpenCV NMS
        if len(boxes) > 0:
            # xywh to xyxy for NMS
            # But OpenCV NMSBoxes expects xywh
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.25, 0.45)
        t3 = time.perf_counter()
        
        if _ >= 5:
            times['capture'].append(15.0) # stubbed
            times['preprocess'].append((t1 - t0) * 1000)
            times['inference'].append((t2 - t1) * 1000)
            times['postprocess'].append((t3 - t2) * 1000)
            
    print(f"Capture:      {np.mean(times['capture']):.2f} ms")
    print(f"Preprocess:   {np.mean(times['preprocess']):.2f} ms")
    print(f"Inference:    {np.mean(times['inference']):.2f} ms")
    print(f"Postprocess:  {np.mean(times['postprocess']):.2f} ms")
    
if __name__ == "__main__":
    profile_pipeline()
