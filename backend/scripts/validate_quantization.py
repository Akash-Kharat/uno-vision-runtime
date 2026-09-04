import os
import glob
import numpy as np
import cv2
import onnxruntime as ort
import argparse

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def box_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

def postprocess(preds, conf_thres=0.25):
    preds = preds[0].T
    boxes = preds[:, :4]
    scores = preds[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    class_scores = np.max(scores, axis=1)
    mask = class_scores > conf_thres
    boxes = boxes[mask]
    class_scores = class_scores[mask]
    class_ids = class_ids[mask]
    xyxy = xywh2xyxy(boxes)
    indices = cv2.dnn.NMSBoxes(xyxy.tolist(), class_scores.tolist(), conf_thres, 0.45)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({"box": xyxy[i], "score": class_scores[i], "class": class_ids[i]})
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", required=True)
    parser.add_argument("--int8", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--val_count", type=int, default=28)
    args = parser.parse_args()
    
    img_paths = sorted(glob.glob(os.path.join(args.val_dir, "*.jpg")))[-args.val_count:]
    
    sess_fp32 = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(args.int8, providers=["CPUExecutionProvider"])
    
    total_fp32_dets = 0
    total_int8_dets = 0
    total_matched = 0
    ious = []
    deltas = []
    
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        out_fp32 = sess_fp32.run(None, {sess_fp32.get_inputs()[0].name: img_tensor})[0]
        out_int8 = sess_int8.run(None, {sess_int8.get_inputs()[0].name: img_tensor})[0]
        
        det_fp32 = postprocess(out_fp32)
        det_int8 = postprocess(out_int8)
        
        total_fp32_dets += len(det_fp32)
        total_int8_dets += len(det_int8)
        
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
                total_matched += 1
                ious.append(best_iou)
                deltas.append(best_d2["score"] - d1["score"])
                
    print(f"FP32 Detections: {total_fp32_dets}")
    print(f"INT8 Detections: {total_int8_dets}")
    print(f"Matched Detections: {total_matched}")
    
    if total_matched > 0:
        print(f"Mean IoU: {np.mean(ious):.4f}")
        print(f"Mean Confidence Delta (INT8 - FP32): {np.mean(deltas):.4f}")
    
    if total_fp32_dets > 0:
        recovery = total_matched / total_fp32_dets
        print(f"Detection Recovery Rate: {recovery*100:.1f}%")
        
if __name__ == "__main__":
    main()
