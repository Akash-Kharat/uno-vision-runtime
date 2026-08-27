import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Opened:", cap.isOpened())
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS reported:", cap.get(cv2.CAP_PROP_FPS))

fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

print(
    "FOURCC:",
    "".join(
        chr((fourcc >> (8 * i)) & 0xFF)
        for i in range(4)
    )
)

start = time.perf_counter()

for i in range(30):

    ok, frame = cap.read()

    if not ok:
        print(f"Frame {i}: FAILED")
        continue

    print(
        f"Frame {i}: "
        f"shape={frame.shape}, "
        f"min={frame.min()}, "
        f"max={frame.max()}, "
        f"mean={frame.mean():.2f}"
    )

    if i in (0, 10, 29):
        cv2.imwrite(f"mjpeg_frame_{i}.jpg", frame)

elapsed = time.perf_counter() - start

print(f"Elapsed: {elapsed:.3f}s")
print(f"Actual capture FPS: {30 / elapsed:.2f}")

cap.release()