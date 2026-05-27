import cv2
import pickle
import time
from pathlib import Path

from camera_utils import open_camo_camera
from database import get_worker

MODEL_PATH = Path("models/face_model.yml")
LABELS_PATH = Path("models/labels.pkl")
STATUS_FILE = Path("data/face_check_result.txt")

DUPLICATE_THRESHOLD = 85
SCAN_SECONDS_AFTER_FACE = 7
RESULT_DISPLAY_SECONDS = 1.2
REQUIRED_MATCHES = 3


def write_result(message):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(message, encoding="utf-8")
    print(message)


def close_camera(cap):
    cap.release()
    cv2.destroyAllWindows()
    for _ in range(3):
        cv2.waitKey(1)


def draw_panel(frame, title, lines, color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (1120, 285), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(
        frame,
        title,
        (45, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        color,
        2,
    )

    y = 110
    for line in lines:
        cv2.putText(
            frame,
            line,
            (45, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.76,
            (255, 255, 255),
            2,
        )
        y += 36


def show_final_result(cap, window_name, title, lines, color):
    start_time = time.time()

    while time.time() - start_time < RESULT_DISPLAY_SECONDS:
        ret, frame = cap.read()

        if not ret or frame is None:
            frame = 255 * cv2.UMat(720, 1280, cv2.CV_8UC3).get()

        draw_panel(frame, title, lines, color)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break


def choose_best_match(match_counts, best_by_username):
    if not match_counts:
        return None, None, 9999.0

    ranked = sorted(
        match_counts,
        key=lambda name: (-match_counts[name], best_by_username[name]["confidence"]),
    )
    username = ranked[0]
    confidence = best_by_username[username]["confidence"]
    full_name = best_by_username[username]["full_name"]
    count = match_counts[username]

    if count >= REQUIRED_MATCHES:
        return username, full_name, confidence

    if count >= 2 and confidence < DUPLICATE_THRESHOLD - 8:
        return username, full_name, confidence

    return None, None, confidence


def main():
    write_result("RUNNING: Face verification opened. Waiting for a face.")

    cap = open_camo_camera()

    if not cap.isOpened():
        write_result("FAILED: Could not open Camo camera.")
        return

    window_name = "Pre-Registration Face Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 720)

    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        result_message = "NOT_FOUND: No trained model yet. No face is registered."
        show_final_result(
            cap,
            window_name,
            "PRE-REGISTRATION FACE VERIFICATION",
            [
                "No trained face model was found.",
                "This means no face is registered yet.",
                "You can continue with registration.",
            ],
            (0, 255, 0),
        )
        write_result(result_message)
        close_camera(cap)
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    with open(LABELS_PATH, "rb") as f:
        label_names = pickle.load(f)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    scan_started_at = None
    match_counts = {}
    best_by_username = {}
    best_weak_username = None
    best_weak_confidence = 9999.0
    best_weak_full_name = None

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

        panel_color = (0, 255, 255)

        if len(faces) == 0:
            panel_lines = [
                "Waiting for a face.",
                "The 7 second scan starts only after a face is detected.",
                "Keep your face visible in front of the camera.",
                "Result: waiting...",
            ]
        else:
            if scan_started_at is None:
                scan_started_at = time.time()
                write_result("RUNNING: Face detected. Scanning for 7 seconds.")

            elapsed = time.time() - scan_started_at
            remaining = max(0, int(SCAN_SECONDS_AFTER_FACE - elapsed + 0.99))

            faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
            x, y, w, h = [int(v) for v in faces[0]]

            face_gray = gray[y:y + h, x:x + w]
            face_gray = cv2.resize(face_gray, (200, 200))

            label, confidence = recognizer.predict(face_gray)
            predicted_username = label_names.get(label, "unknown")
            worker = get_worker(predicted_username)

            if confidence < best_weak_confidence:
                best_weak_confidence = confidence
                best_weak_username = predicted_username
                best_weak_full_name = predicted_username

                if worker is not None:
                    _worker_id, _username, full_name, _job_title, _role, _is_active = worker
                    best_weak_full_name = full_name

            if confidence < DUPLICATE_THRESHOLD and worker is not None:
                match_counts[predicted_username] = match_counts.get(predicted_username, 0) + 1
                _worker_id, _username, full_name, _job_title, _role, _is_active = worker

                best_for_user = best_by_username.get(predicted_username)
                if best_for_user is None or confidence < best_for_user["confidence"]:
                    best_by_username[predicted_username] = {
                        "confidence": confidence,
                        "full_name": full_name,
                    }

                panel_lines = [
                    "Registered face candidate found.",
                    f"Matched username: {predicted_username}",
                    f"Confirming frames: {match_counts[predicted_username]}",
                    f"Scan closes in: {remaining}s",
                ]
                panel_color = (0, 0, 255)

            elif confidence < DUPLICATE_THRESHOLD and worker is None:
                panel_lines = [
                    "Old deleted-worker face label ignored.",
                    f"Old label: {predicted_username}",
                    f"Confidence: {confidence:.2f}",
                    f"Scan closes in: {remaining}s",
                ]
                panel_color = (0, 255, 255)
            else:
                panel_lines = [
                    "Face detected.",
                    "No strong active-worker match yet.",
                    f"Confidence: {confidence:.2f}",
                    f"Scan closes in: {remaining}s",
                ]
                panel_color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), panel_color, 3)

        draw_panel(
            frame,
            "PRE-REGISTRATION FACE VERIFICATION",
            panel_lines,
            panel_color,
        )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            write_result("CANCELLED: Face verification cancelled by user.")
            close_camera(cap)
            return

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                write_result("CANCELLED: Face verification window closed by user.")
                close_camera(cap)
                return
        except cv2.error:
            break

        if scan_started_at is not None and time.time() - scan_started_at >= SCAN_SECONDS_AFTER_FACE:
            break

    best_match_username, best_match_full_name, best_confidence = choose_best_match(
        match_counts,
        best_by_username,
    )

    if best_match_username is not None:
        result_message = (
            f"FOUND: Face already registered as {best_match_full_name} "
            f"({best_match_username}) | confidence={best_confidence:.2f}"
        )
        show_final_result(
            cap,
            window_name,
            "FACE ALREADY REGISTERED",
            [
                f"Registered as: {best_match_full_name}",
                f"Username: {best_match_username}",
                f"Best confidence: {best_confidence:.2f}",
                "Registration should be blocked.",
            ],
            (0, 0, 255),
        )
        write_result(result_message)
    else:
        if best_weak_username is None:
            result_message = "NOT_FOUND: No usable face match was found after the 7 second scan."
            final_lines = [
                "No registered face detected.",
                "No active worker matched this face.",
                "You may continue registration.",
                "Try better lighting if needed.",
            ]
        else:
            result_message = (
                f"NOT_FOUND: Face is not registered. "
                f"Best weak match={best_weak_username}, confidence={best_weak_confidence:.2f}"
            )
            final_lines = [
                "This face is not registered.",
                f"Best weak match: {best_weak_full_name}",
                f"Confidence: {best_weak_confidence:.2f}",
                "You can continue registration.",
            ]

        show_final_result(
            cap,
            window_name,
            "NEW FACE / NOT REGISTERED",
            final_lines,
            (0, 255, 0),
        )
        write_result(result_message)

    close_camera(cap)


if __name__ == "__main__":
    main()
