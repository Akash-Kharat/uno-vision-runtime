import numpy as np
import onnxruntime as ort

sess1 = ort.InferenceSession("data/models/models/mdl_0f92eb15da2d/model.onnx", providers=["CPUExecutionProvider"])
sess2 = ort.InferenceSession("data/models/experiments/yolov8n_int8.onnx", providers=["CPUExecutionProvider"])

t = np.random.randn(1,3,640,640).astype(np.float32)

out1 = sess1.run(None, {"images":t})[0]
out2 = sess2.run(None, {"images":t})[0]

print("FP32 max:", out1.max())
print("INT8 max:", out2.max())
