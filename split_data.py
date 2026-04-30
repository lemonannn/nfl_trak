import os
import shutil
import random

data = "player_tracking_training_data_2"
src = data + "/obj_train_data"

images = [f for f in os.listdir(src) if f.endswith(".png")]
random.shuffle(images)

split = int(len(images) * 0.8)

train = images[:split]
val = images[split:]

for split_name, split_files in [("train", train), ("val", val)]:
    os.makedirs(f"{data}/dataset/images/{split_name}", exist_ok=True)
    os.makedirs(f"{data}/dataset/labels/{split_name}", exist_ok=True)

    for img in split_files:
        label = img.replace(".png", ".txt")

        shutil.copy(f"{src}/{img}", f"{data}/dataset/images/{split_name}/{img}")
        shutil.copy(f"{src}/{label}", f"{data}/dataset/labels/{split_name}/{label}")