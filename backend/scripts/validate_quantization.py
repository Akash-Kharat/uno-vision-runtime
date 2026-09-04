import sys
import os
import urllib.request
import numpy as np
import cv2
import onnxruntime as ort

def download_image(url, out_path):
    if not os.path.exists(out_path):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, out_path)

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

def postprocess(preds, conf_thres=0.25):
    # preds: [1, 84, 8400]
    preds = preds[0].T # [8400, 84]
    boxes = preds[:, :4]
    scores = preds[:, 4:]
    
    class_ids = np.argmax(scores, axis=1)
    class_scores = np.max(scores, axis=1)
    
    mask = class_scores > conf_thres
    boxes = boxes[mask]
    class_scores = class_scores[mask]
    class_ids = class_ids[mask]
    
    xyxy = xywh2xyxy(boxes)
    
    # Simple NMS
    indices = cv2.dnn.NMSBoxes(xyxy.tolist(), class_scores.tolist(), conf_thres, 0.45)
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "box": xyxy[i],
                "score": class_scores[i],
                "class": class_ids[i]
            })
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", default="data/models/models/mdl_0f92eb15da2d/model.onnx")
    parser.add_argument("--int8", default="data/models/experiments/yolov8n_int8.onnx")
    args = parser.parse_args()
    
    img_path = "val_image.jpg"
    download_image("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", img_path)
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    sess_fp32 = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(args.int8, providers=["CPUExecutionProvider"])
    
    out_fp32 = sess_fp32.run(None, {sess_fp32.get_inputs()[0].name: img_tensor})[0]
    out_int8 = sess_int8.run(None, {sess_int8.get_inputs()[0].name: img_tensor})[0]
    
    det_fp32 = postprocess(out_fp32)
    det_int8 = postprocess(out_int8)
    
    print(f"FP32 Detections: {len(det_fp32)}")
    print(f"INT8 Detections: {len(det_int8)}")
    
    # Match detections (greedy)
    matched = 0
    deltas = []
    ious = []
    for d1 in det_fp32:
        best_iou = 0
        best_d2 = None
        for d2 in det_int8:
            if d1["class"] == d2["class"]:
                iou = box_iou(d1["box"], d2["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_d2 = d2
        if best_iou > 0.5:
            matched += 1
            ious.append(best_iou)
            deltas.append(best_d2["score"] - d1["score"])
            
    print(f"Matched Detections: {matched}")
    if matched > 0:
        print(f"Mean IoU: {np.mean(ious):.4f}")
        print(f"Mean Confidence Delta (INT8 - FP32): {np.mean(deltas):.4f}")
        print(f"Max Confidence Drop: {np.min(deltas):.4f}")

if __name__ == "__main__":
    main()
