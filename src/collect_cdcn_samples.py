import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "spoofgate_cdcn_dataset"
IMAGE_SIZE = 224


def ensure_src_imports():
    src = str(PROJECT_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def draw_panel(frame, lines, color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (1180, 260), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    y = 58
    for line in lines:
        cv2.putText(frame, line, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        y += 38


def detect_largest_face(face_cascade, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(95, 95),
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def crop_face(frame, box, pad_ratio=0.16):
    x, y, w, h = box
    size = max(w, h) * (1.0 + 2.0 * pad_ratio)
    cx = x + w / 2.0
    cy = y + h / 2.0
    x1 = max(0, int(cx - size / 2.0))
    y1 = max(0, int(cy - size / 2.0))
    x2 = min(frame.shape[1], int(cx + size / 2.0))
    y2 = min(frame.shape[0], int(cy + size / 2.0))
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def quality_check(face_crop):
    if face_crop.size == 0:
        return False, "EMPTY_FACE_CROP"

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < 35:
        return False, f"TOO_DARK brightness={brightness:.1f}"
    if brightness > 230:
        return False, f"TOO_BRIGHT brightness={brightness:.1f}"
    if sharpness < 16:
        return False, f"TOO_BLURRY sharpness={sharpness:.1f}"

    return True, f"OK brightness={brightness:.1f} sharpness={sharpness:.1f}"


def save_crop(face_crop, output_dir, prefix, index):
    resized = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    filename = f"{prefix}_{index:05d}_{datetime.now().strftime('%H%M%S%f')}.jpg"
    path = output_dir / filename
    cv2.imwrite(str(path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["real", "spoof"], required=True)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--interval", type=float, default=0.08)
    args = parser.parse_args()

    ensure_src_imports()
    from camera_utils import open_camo_camera

    output_dir = DATASET_DIR / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camo_camera()
    if not cap.isOpened():
        print("Could not open camera.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    window_name = f"Collect CDCN Samples - {args.label.upper()}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 720)
    cv2.moveWindow(window_name, 120, 80)

    print(f"Collecting {args.count} {args.label.upper()} samples.")
    if args.label == "real":
        print("Show your real face. Slowly vary distance, angle, and lighting.")
    else:
        print("Show spoof attacks: printed photo, phone screen, laptop screen, replay/photo.")
    print("Press Q to stop early.")

    saved = 0
    last_save = 0.0
    last_quality = "WAITING"

    while saved < args.count:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        box = detect_largest_face(face_cascade, frame)
        color = (0, 255, 255)
        status = "Waiting for face..."

        if box is not None:
            face_crop, crop_box = crop_face(frame, box)
            quality_ok, last_quality = quality_check(face_crop)
            x1, y1, x2, y2 = crop_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            if quality_ok and time.time() - last_save >= args.interval:
                save_crop(face_crop, output_dir, args.label, saved + 1)
                saved += 1
                last_save = time.time()
                color = (0, 255, 0)
                status = f"Saved {saved}/{args.count}"
            elif not quality_ok:
                color = (0, 0, 255)
                status = last_quality
            else:
                status = f"Ready... saved {saved}/{args.count}"

        lines = [
            f"CDCN DATASET COLLECTION: {args.label.upper()}",
            status,
            f"Saved: {saved}/{args.count}",
            "Move/angle slowly. Use multiple spoof sources for SPOOF.",
            last_quality,
        ]
        draw_panel(frame, lines, color)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} samples to {output_dir}")


if __name__ == "__main__":
    main()
