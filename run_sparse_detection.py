import cv2
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "runs/detect/train-2/weights/best.pt"
VIDEO_PATH = "test3.mp4"
OUTPUT_VIDEO = "tracked_output.mp4"

CONF = 0.25

model = YOLO(MODEL_PATH)

# -----------------------------
# VIDEO SETUP
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

# -----------------------------
# BOt-SORT STREAM TRACKING
# -----------------------------
results_stream = model.track(
    source=VIDEO_PATH,
    tracker="botsort.yaml",
    stream=True,
    persist=True,
    conf=CONF,
    verbose=False
)

# -----------------------------
# DRAW LOOP
# -----------------------------
for results in results_stream:

    frame = results.orig_img.copy()

    if results.boxes is not None:

        for box in results.boxes:

            conf = float(box.conf[0])
            if conf < CONF:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            track_id = int(box.id[0]) if box.id is not None else -1

            # -----------------------------
            # DRAW BOX
            # -----------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"ID {track_id} {conf:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    out.write(frame)

cap.release()
out.release()

print("Done: tracked video saved as", OUTPUT_VIDEO)