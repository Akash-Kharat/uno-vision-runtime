import os
import glob
from ultralytics import YOLO
import numpy as np

def calculate_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def validate_model(baseline_model_path, test_model_path, val_dir):
    print(f"\n--- Validating {test_model_path} against {baseline_model_path} ---")
    
    baseline = YOLO(baseline_model_path, task='detect')
    test_model = YOLO(test_model_path, task='detect')
    
    images = glob.glob(os.path.join(val_dir, "*.jpg"))[:30]  # use 30 images
    
    total_baseline_det = 0
    total_test_det = 0
    matched_det = 0
    iou_sum = 0.0
    conf_delta_sum = 0.0
    
    for img_path in images:
        res_base = baseline(img_path, verbose=False)[0]
        res_test = test_model(img_path, verbose=False)[0]
        
        boxes_base = res_base.boxes.xyxy.cpu().numpy()
        conf_base = res_base.boxes.conf.cpu().numpy()
        cls_base = res_base.boxes.cls.cpu().numpy()
        
        boxes_test = res_test.boxes.xyxy.cpu().numpy()
        conf_test = res_test.boxes.conf.cpu().numpy()
        cls_test = res_test.boxes.cls.cpu().numpy()
        
        total_baseline_det += len(boxes_base)
        total_test_det += len(boxes_test)
        
        # Greedy match
        matched_this_image = set()
        for i, bb in enumerate(boxes_base):
            best_iou = 0.5
            best_j = -1
            for j, tb in enumerate(boxes_test):
                if j in matched_this_image:
                    continue
                if cls_base[i] != cls_test[j]:
                    continue
                iou = calculate_iou(bb, tb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            
            if best_j != -1:
                matched_this_image.add(best_j)
                matched_det += 1
                iou_sum += best_iou
                conf_delta_sum += (conf_test[best_j] - conf_base[i])
                
    recovery_rate = matched_det / max(total_baseline_det, 1) * 100
    mean_iou = iou_sum / max(matched_det, 1)
    mean_conf_delta = conf_delta_sum / max(matched_det, 1)
    
    print(f"Baseline Detections: {total_baseline_det}")
    print(f"Test Detections: {total_test_det}")
    print(f"Matched Detections: {matched_det}")
    print(f"Detection Recovery Rate: {recovery_rate:.1f}%")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Confidence Delta: {mean_conf_delta:.4f}")

def main():
    val_dir = "data/coco128/images/train2017"
    baseline = "data/models/experiments/resolutions/yolov8n_640.onnx"
    
    resolutions = [512, 416, 384, 320]
    for res in resolutions:
        test = f"data/models/experiments/resolutions/yolov8n_{res}.onnx"
        validate_model(baseline, test, val_dir)
        
    y11_resolutions = [640, 416]
    for res in y11_resolutions:
        test = f"data/models/experiments/yolo11/yolo11n_{res}.onnx"
        validate_model(baseline, test, val_dir)

if __name__ == "__main__":
    main()
