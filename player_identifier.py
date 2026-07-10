import os
import cv2
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "runs/detect/train-2/weights/best.pt"
IMAGE_DIR = "frames"
OUTPUT_FILE = "gt/gt.txt"

CONF = 0.25

model = YOLO(MODEL_PATH)

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith((".jpg", ".png", ".jpeg"))
])

all_lines = []

# -----------------------------
# BOt-SORT TRACKING (CRITICAL PART)
# -----------------------------
results_stream = model.track(
    source=IMAGE_DIR,
    tracker="botsort.yaml",   # 🔥 KEY CHANGE
    stream=True,
    persist=True,
    conf=CONF,
    verbose=False
)

# -----------------------------
# EXPORT LOOP
# -----------------------------
for frame_id, results in enumerate(results_stream):

    if results.boxes is None:
        continue

    for box in results.boxes:

        conf = float(box.conf[0])
        if conf < CONF:
            continue

        if box.id is None:
            continue  # ✅ critical fix

        track_id = int(box.id[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        w = x2 - x1
        h = y2 - y1

        all_lines.append(
            f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.3f},-1,-1,-1"
        )
        
# -----------------------------
# SAVE MOT FILE
# -----------------------------
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(all_lines))

print("Done: MOT file exported for CVAT.")