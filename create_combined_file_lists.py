import os
import glob
import random

# ---------------------------------------
# CONFIG
# ---------------------------------------

video_dirs = [
    "player_tracking_training_data/obj_Train_data",
    "player_tracking_training_data_2/obj_train_data",
    "player_tracking_training_data_3/obj_Train_data"
]

TRAIN_RATIO = 0.8

IMAGE_EXTS = ["*.jpg", "*.jpeg", "*.png"]

random.seed(42)

# ---------------------------------------
# VIDEO-LEVEL SPLIT
# ---------------------------------------

random.shuffle(video_dirs)

split_idx = int(len(video_dirs) * TRAIN_RATIO)

train_videos = video_dirs[:split_idx]
val_videos = video_dirs[split_idx:]

print("TRAIN VIDEOS:")
for v in train_videos:
    print(" ", v)

print("\nVAL VIDEOS:")
for v in val_videos:
    print(" ", v)

# ---------------------------------------
# COLLECT VALID IMAGES
# ---------------------------------------

def collect_images(video_dir):
    images = []

    for ext in IMAGE_EXTS:
        images.extend(glob.glob(os.path.join(video_dir, ext)))

    valid = []

    for img_path in images:
        label_path = os.path.splitext(img_path)[0] + ".txt"

        if os.path.exists(label_path):
            valid.append(img_path)

    return sorted(valid)

train_images = []
val_images = []

for vid in train_videos:
    train_images.extend(collect_images(vid))

for vid in val_videos:
    val_images.extend(collect_images(vid))

# ---------------------------------------
# WRITE SPLITS
# ---------------------------------------

with open("train.txt", "w") as f:
    for img in train_images:
        f.write(img + "\n")

with open("val.txt", "w") as f:
    for img in val_images:
        f.write(img + "\n")

print("\nDONE")
print("Train images:", len(train_images))
print("Val images:", len(val_images))