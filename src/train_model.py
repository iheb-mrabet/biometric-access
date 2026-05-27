import cv2
import os
import numpy as np
import pickle
from pathlib import Path

DATASET_DIR = "data/faces"
MODEL_PATH = "models/face_model.yml"
LABELS_PATH = "models/labels.pkl"


def remove_stale_model_files():
    for file_path in [Path(MODEL_PATH), Path(LABELS_PATH)]:
        if file_path.exists():
            file_path.unlink()
            print(f"Deleted stale model file: {file_path}")


recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_names = {}
current_label = 0

print("Loading dataset...")

if not os.path.isdir(DATASET_DIR):
    print("No face dataset found. Removing old trained model and labels.")
    remove_stale_model_files()
    raise SystemExit(0)

for user_name in os.listdir(DATASET_DIR):
    user_folder = os.path.join(DATASET_DIR, user_name)

    if not os.path.isdir(user_folder):
        continue

    user_images = []

    for image_name in os.listdir(user_folder):
        image_path = os.path.join(user_folder, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        image = cv2.resize(image, (200, 200))
        user_images.append(image)

    if not user_images:
        print(f"Skipped empty user folder: {user_name}")
        continue

    label_names[current_label] = user_name

    for image in user_images:
        faces.append(image)
        labels.append(current_label)

    print(f"Loaded user: {user_name} with label {current_label}")
    current_label += 1

if len(faces) == 0:
    print("No face images found. Removing old trained model and labels.")
    remove_stale_model_files()
    raise SystemExit(0)

print(f"Training model with {len(faces)} images...")

recognizer.train(faces, np.array(labels))

Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
recognizer.save(MODEL_PATH)

with open(LABELS_PATH, "wb") as f:
    pickle.dump(label_names, f)

print("Training completed.")
print(f"Model saved to: {MODEL_PATH}")
print(f"Labels saved to: {LABELS_PATH}")
print("Labels:", label_names)
