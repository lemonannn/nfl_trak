from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # fast baseline

cap = cv2.VideoCapture("all22.mp4")

frames_detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])

        # COCO class 0 = person
        if cls == 0:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])

    frames_detections.append(detections)