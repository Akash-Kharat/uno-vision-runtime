import cv2
import time
import psutil
import onnxruntime as ort
import numpy as np
import argparse
import sys
import threading

class CameraThread:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened() and src == 2:
            print("Failed to open 2, trying 0")
            self.cap = cv2.VideoCapture(0)
            
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            
    def read(self):
        return self.ret, self.frame
        
    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--model", type=str, default="data/models/experiments/yolo11/yolo11n_416.onnx")
    parser.add_argument("--camera", type=int, default=2)
    args = parser.parse_args()
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2
    session = ort.InferenceSession(args.model, sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    cam = CameraThread(args.camera)
    if not cam.cap.isOpened():
        print("Failed to open any camera.")
        return
        
    target_interval = 1.0 / args.fps
    start_time = time.time()
    last_infer_time = 0
    inferences_run = 0
    
    try:
        while (time.time() - start_time) < args.duration:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue
                
            current_time = time.time()
            if (current_time - last_infer_time) >= target_interval:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (416, 416))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0)
                
                session.run(None, {input_name: img})
                inferences_run += 1
                last_infer_time = current_time
                
                cpu = psutil.cpu_percent()
                try:
                    temps = psutil.sensors_temperatures()
                    temp = list(temps.values())[0][0].current if temps else 0
                except:
                    temp = 0
                sys.stdout.write(f"\rInferences: {inferences_run} | CPU: {cpu}% | Temp: {temp}C   ")
                sys.stdout.flush()
                
    except KeyboardInterrupt:
        pass
        
    cam.release()
    print(f"\n\nTest completed. Ran {inferences_run} inferences.")
    print(f"Effective Inference FPS: {inferences_run / args.duration:.2f}")

if __name__ == "__main__":
    main()
