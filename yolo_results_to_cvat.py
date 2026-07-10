import os
import zipfile

# ===== CONFIG =====
IMAGES_DIR = "frames"
LABELS_DIR = "labels"
SUBSET_NAME = "obj_train_data"
OUTPUT_ZIP = "archive.zip"

CLASSES = ["player"]  # modify if needed

# ===================

def build_train_txt():
    lines = []
    for img in sorted(os.listdir(IMAGES_DIR)):
        if img.lower().endswith((".jpg", ".png", ".jpeg")):
            lines.append(f"{SUBSET_NAME}/{img}")
    return "\n".join(lines)


def clean_labels():
    """
    Ensures labels are Darknet format:
    class x y w h
    """
    cleaned = {}

    for file in os.listdir(LABELS_DIR):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(LABELS_DIR, file)

        with open(path, "r") as f:
            lines = f.readlines()

        fixed = []
        for line in lines:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            # drop YOLOv8 confidence if present
            if len(parts) == 6:
                parts = parts[:5]

            fixed.append(" ".join(parts))

        cleaned[file] = "\n".join(fixed)

    return cleaned


labels = clean_labels()
train_txt = build_train_txt()

obj_data = f"""classes = {len(CLASSES)}
train = train.txt
names = obj.names
backup = backup/
"""

obj_names = "\n".join(CLASSES) + "\n"

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:

    # --- core files ---
    zipf.writestr("obj.data", obj_data)
    zipf.writestr("obj.names", obj_names)
    zipf.writestr("train.txt", train_txt)

    # --- subset folder ---
    for img in os.listdir(IMAGES_DIR):
        if img.lower().endswith((".jpg", ".png", ".jpeg")):
            zipf.write(os.path.join(IMAGES_DIR, img),
                       f"{SUBSET_NAME}/{img}")

    # --- labels ---
    for name, content in labels.items():
        zipf.writestr(f"{SUBSET_NAME}/{name}", content)

print("✅ archive.zip created successfully")