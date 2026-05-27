import argparse
import cv2
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from config import CAMERA_INDEX, CAMERA_BACKEND
from camera_utils import open_camo_camera
from database import (
    add_or_update_worker,
    get_departments,
    mark_worker_registration_success,
    get_worker
)

MAX_IMAGES = 100
STATUS_DIR = Path("data/registration_status")
LOCK_DIR = Path("data/process_locks")
PREVIEW_DIR = Path("data/worker_previews")
MODEL_PATH = Path("models/face_model.yml")
LABELS_PATH = Path("models/labels.pkl")
DUPLICATE_THRESHOLD = 75


def write_status(username, message):
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    status_file = STATUS_DIR / f"{username}.txt"
    status_file.write_text(message, encoding="utf-8")


def remove_lock(username):
    lock_file = LOCK_DIR / f"register_{username}.lock"
    if lock_file.exists():
        lock_file.unlink()


def save_clean_face_preview(frame, x, y, w, h, username):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    height, width = frame.shape[:2]

    pad_x = int(w * 0.70)
    pad_y_top = int(h * 0.90)
    pad_y_bottom = int(h * 0.55)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y_top)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y_bottom)

    preview = frame[y1:y2, x1:x2]

    if preview.size == 0:
        return None

    preview_path = PREVIEW_DIR / f"{username}.jpg"
    cv2.imwrite(str(preview_path), preview)

    return str(preview_path)


def draw_panel(frame, title, lines, progress=None, color=(0, 255, 255)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (980, 280), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, title, (45, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

    y = 105
    for line in lines:
        cv2.putText(frame, line, (45, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        y += 35

    if progress is not None:
        bar_x, bar_y, bar_w, bar_h = 45, 235, 760, 30
        filled_w = int((progress / 100) * bar_w)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)

        cv2.putText(frame, f"{progress}%", (830, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)


def check_face_already_registered(username):
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return False, None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    with open(LABELS_PATH, "rb") as f:
        label_names = pickle.load(f)

    cap = open_camo_camera()

    if not cap.isOpened():
        print("ERROR: Could not open Camo for duplicate face check.")
        return False, None

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    window_name = "Checking Face Before Registration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 720)

    scan_seconds_after_face = 7
    required_matches = 3
    scan_started_at = None
    match_counts = {}
    best_by_username = {}

    print("Checking if this face is already registered.")
    print("The 7 second check starts when a face is detected.")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(90, 90),
        )

        panel_lines = [
            "Waiting for a face.",
            "The 7 second duplicate check starts after detection.",
            "Keep your face visible.",
            "Result: waiting...",
        ]
        panel_color = (0, 255, 255)

        if len(faces) > 0:
            if scan_started_at is None:
                scan_started_at = time.time()

            elapsed = time.time() - scan_started_at
            remaining = max(0, int(scan_seconds_after_face - elapsed + 0.99))

            faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
            x, y, w, h = [int(v) for v in faces[0]]

            face_gray = gray[y:y + h, x:x + w]
            face_gray = cv2.resize(face_gray, (200, 200))

            label, confidence = recognizer.predict(face_gray)
            predicted_username = label_names.get(label, "unknown")
            worker = get_worker(predicted_username)

            if confidence < DUPLICATE_THRESHOLD and worker is not None:
                match_counts[predicted_username] = match_counts.get(predicted_username, 0) + 1
                best_for_user = best_by_username.get(predicted_username)
                if best_for_user is None or confidence < best_for_user:
                    best_by_username[predicted_username] = confidence

                panel_lines = [
                    "Registered face candidate found.",
                    f"Matched username: {predicted_username}",
                    f"Confirming frames: {match_counts[predicted_username]}",
                    f"Check closes in: {remaining}s",
                ]
                panel_color = (0, 0, 255)

            elif confidence < DUPLICATE_THRESHOLD and worker is None:
                panel_lines = [
                    "Stale deleted-worker model label ignored.",
                    f"Old label: {predicted_username}",
                    f"Confidence: {confidence:.2f}",
                    f"Check closes in: {remaining}s",
                ]
                panel_color = (0, 255, 255)
            else:
                panel_lines = [
                    "Face detected.",
                    "No strong active-worker match.",
                    f"Confidence: {confidence:.2f}",
                    f"Check closes in: {remaining}s",
                ]
                panel_color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), panel_color, 3)

        draw_panel(frame, "PRE-REGISTRATION FACE CHECK", panel_lines, None, panel_color)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        if scan_started_at is not None and time.time() - scan_started_at >= scan_seconds_after_face:
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(3):
        cv2.waitKey(1)

    if not match_counts:
        return False, None

    ranked = sorted(
        match_counts,
        key=lambda name: (-match_counts[name], best_by_username.get(name, 9999.0)),
    )
    best_match = ranked[0]
    best_count = match_counts[best_match]
    best_confidence = best_by_username.get(best_match, 9999.0)

    if best_count >= required_matches:
        return True, best_match

    if best_count >= 2 and best_confidence < DUPLICATE_THRESHOLD - 8:
        return True, best_match

    return False, None



def collect_face_dataset(username, full_name, job_title):
    save_dir = Path("data/faces") / username

    if save_dir.exists():
        shutil.rmtree(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camo_camera()

    if not cap.isOpened():
        print(f"ERROR: Could not open Camo camera index={CAMERA_INDEX}, backend={CAMERA_BACKEND}")
        return False, None

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    preview_path = None
    last_capture_time = 0
    capture_delay = 0.08
    window_name = "Face Registration - Camo Camera"

    print("=" * 70)
    print("FACE REGISTRATION MODE")
    print(f"Worker: {full_name}")
    print(f"Username: {username}")
    print(f"Job: {job_title}")
    print("Press Q once to cancel. The window closes automatically at 100%.")
    print("=" * 70)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while count < MAX_IMAGES:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Waiting for Camo frame...")
            time.sleep(0.2)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(90, 90)
        )

        current_time = time.time()

        for (x, y, w, h) in faces:
            if current_time - last_capture_time < capture_delay:
                break

            face_gray = gray[y:y+h, x:x+w]
            face_gray = cv2.resize(face_gray, (200, 200))

            image_path = save_dir / f"{username}_{count}.jpg"
            cv2.imwrite(str(image_path), face_gray)

            if preview_path is None and count >= 10:
                preview_path = save_clean_face_preview(frame, x, y, w, h, username)

            count += 1
            last_capture_time = current_time

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

            cv2.putText(frame, "Face detected - sample saved", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            break

        progress = int((count / MAX_IMAGES) * 100)

        draw_panel(
            frame,
            "FACE REGISTRATION MODE",
            [
                f"Worker: {full_name}",
                f"Job: {job_title}",
                f"Username: {username}",
                "Move face slowly: front, left, right, up, down"
            ],
            progress,
            (0, 255, 255)
        )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Registration cancelled by user.")
            break

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Registration window closed by user.")
                break
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Saved {count} face images for {username}")

    return count >= MAX_IMAGES, preview_path


def retrain_model():
    print("Retraining face recognition model...")

    result = subprocess.run(
        [sys.executable, "src/train_model.py"],
        cwd=".",
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--full-name", required=True)
    parser.add_argument("--job-title", required=True)
    parser.add_argument("--role", required=False, default="worker")
    parser.add_argument("--departments", nargs="+", required=True)

    args = parser.parse_args()

    username = args.username.strip().lower()
    full_name = args.full_name.strip()
    job_title = args.job_title.strip()
    role = args.role.strip().lower()
    allowed_departments = args.departments

    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = LOCK_DIR / f"register_{username}.lock"

    if lock_file.exists():
        print("Registration is already running for this user.")
        return

    lock_file.write_text("running", encoding="utf-8")

    try:
        if get_worker(username) is not None:
            message = f"FAILED: Username '{username}' is already registered. Delete the worker first if you want to re-register."
            print(message)
            write_status(username, message)
            return

        existing_departments = get_departments()

        if role == "admin":
            allowed_departments = existing_departments

        for department in allowed_departments:
            if department not in existing_departments:
                message = f"FAILED: Unknown department: {department}"
                print(message)
                write_status(username, message)
                return

        duplicate, predicted_username = check_face_already_registered(username)

        if duplicate:
            message = f"FAILED: This face is already registered as '{predicted_username}'."
            print(message)
            write_status(username, message)
            return

        print("Starting face registration scan...")

        ok, preview_path = collect_face_dataset(username, full_name, job_title)

        if not ok:
            message = f"FAILED: Registration for {full_name} was not completed. Face scan did not reach 100%."
            print(message)
            write_status(username, message)
            return

        add_or_update_worker(
            username=username,
            full_name=full_name,
            job_title=job_title,
            allowed_departments=allowed_departments,
            role=role
        )

        mark_worker_registration_success(username, preview_path)

        train_ok = retrain_model()

        if not train_ok:
            message = f"FAILED: {full_name} face was scanned, but model retraining failed."
            print(message)
            write_status(username, message)
            return

        message = f"SUCCESS: {full_name} is successfully registered."
        print(message)
        write_status(username, message)

    finally:
        remove_lock(username)


if __name__ == "__main__":
    main()
