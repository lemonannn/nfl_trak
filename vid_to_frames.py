import cv2
import os

video_path = "test.mp4"
output_dir = "frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite(f"{output_dir}/frame_{i:06d}.jpg", frame)
    i += 1

cap.release()
print("Done extracting frames")