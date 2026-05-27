import argparse
import json
import cv2
import pickle
import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np

from config import CAMERA_INDEX, CAMERA_BACKEND
from camera_utils import open_camo_camera
from watermark_logs import write_watermarked_log
from database import (
    init_database,
    get_departments,
    get_worker,
    is_worker_allowed,
    save_access_attempt,
)
from email_alerts import send_security_alert
from cdcn_liveness_gate import CDCNLivenessGate, MIN_FRAMES

MODEL_PATH = "models/face_model.yml"
LABELS_PATH = "models/labels.pkl"

CONFIDENCE_THRESHOLD = 85
EMAIL_COOLDOWN_SECONDS = 60
CDCN_LIVENESS_THRESHOLD = 0.42
CDCN_LIVENESS_MARGIN = 0.04
CDCN_CLEAR_SPOOF_SCORE = 0.20
SCAN_SECONDS_AFTER_FACE = 7
FINAL_DISPLAY_SECONDS = 2.0
CLEAR_SPOOF_CLOSE_SECONDS = 5

LOCK_DIR = Path("data/process_locks")
CAPTURE_DIR = Path("captures")
ACCESS_LIVE_STATUS_FILE = Path("data/access_live_status.json")

os.makedirs("logs", exist_ok=True)
CAPTURE_DIR.mkdir(exist_ok=True)


def choose_department():
    departments = get_departments()

    print("\nAvailable departments:")
    for index, department in enumerate(departments, start=1):
        print(f"{index}. {department}")

    while True:
        choice = input("Choose department number: ").strip()

        if choice.isdigit():
            choice = int(choice)

            if 1 <= choice <= len(departments):
                return departments[choice - 1]

        for department in departments:
            if choice.lower() == department.lower():
                return department

        print("Invalid choice. Try again.")


def save_security_capture(frame, username, status, department):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_department = department.replace(" ", "_")
    filename = f"{timestamp}_{username}_{status}_{safe_department}.jpg"
    capture_path = CAPTURE_DIR / filename

    cv2.imwrite(str(capture_path), frame)

    return str(capture_path)


def send_alert_if_needed(username, full_name, job_title, department, status, confidence, capture_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if status == "ACCESS_DENIED_NO_PERMISSION":
        subject = f"[SECURITY ALERT] Known unauthorized user tried to access {department}"

        body = f"""
Security alert: known registered worker tried to access a restricted department.

Time: {timestamp}
Recognized user: {full_name}
Username: {username}
Job title: {job_title}
Target department: {department}
Decision: ACCESS DENIED
Reason: Worker is registered but not authorized for this department.
Confidence: {confidence:.2f}

A captured image is attached as evidence.
"""

        send_security_alert(subject, body, capture_path)

    elif status == "ACCESS_DENIED_UNKNOWN_FACE":
        subject = f"[ALARM] Unknown impostor detected near {department}"

        body = f"""
Security alarm: unknown face detected.

Time: {timestamp}
Recognized user: Unknown
Target department: {department}
Decision: ACCESS DENIED
Reason: Face is not registered in the biometric database.
Confidence: {confidence:.2f}

A captured image is attached as evidence.
"""

        send_security_alert(subject, body, capture_path)

    elif status == "ACCESS_DENIED_SPOOF":
        subject = f"[ALARM] Spoof attempt blocked near {department}"

        body = f"""
Security alarm: anti-spoofing blocked a presentation attack.

Time: {timestamp}
Target department: {department}
Decision: ACCESS DENIED
Reason: CDCN SpoofGate classified the face as spoof.
Spoof confidence: {confidence:.2f}

A captured image is attached as evidence.
"""

        send_security_alert(subject, body, capture_path)


def write_access_live_status(
    full_name,
    job_title,
    department,
    access_text,
    decision_text,
    status,
    confidence,
    liveness_decision,
    liveness_score,
):
    ACCESS_LIVE_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "face_recognized": full_name,
        "job_title": job_title,
        "target_department": department,
        "access_to_department": access_text,
        "decision": decision_text,
        "status": status,
        "confidence": round(float(confidence), 2),
        "liveness_decision": liveness_decision,
        "liveness_score": round(float(liveness_score), 3),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    ACCESS_LIVE_STATUS_FILE.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def draw_status_panel(frame, lines, color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (1120, 305), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    y = 58
    for line in lines:
        cv2.putText(
            frame,
            line,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            color,
            2,
        )
        y += 35


def close_camera(cap):
    cap.release()
    cv2.destroyAllWindows()
    for _ in range(3):
        cv2.waitKey(1)


def decide_liveness(liveness_gate):
    scores = list(liveness_gate.scores)

    if not scores:
        return {
            "decision": "BLOCK",
            "reason": "CDCN_NO_LIVENESS_SIGNAL",
            "median_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "frames": 0,
        }

    median_score = float(np.median(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    live_frames = sum(score >= liveness_gate.threshold for score in scores)
    strong_live = sum(score >= liveness_gate.threshold + liveness_gate.margin for score in scores)
    clear_spoof_frames = sum(score <= liveness_gate.clear_spoof_score for score in scores)

    if len(scores) < MIN_FRAMES:
        return {
            "decision": "BLOCK",
            "reason": "CDCN_NOT_ENOUGH_LIVENESS_FRAMES",
            "median_score": median_score,
            "min_score": min_score,
            "max_score": max_score,
            "frames": len(scores),
        }

    if median_score <= liveness_gate.clear_spoof_score and clear_spoof_frames >= MIN_FRAMES:
        reason = "CDCN_CLEAR_SPOOF"
        decision = "BLOCK"
    elif median_score >= liveness_gate.threshold and live_frames >= MIN_FRAMES and strong_live >= 2:
        reason = "CDCN_LIVE_FACE"
        decision = "ALLOW"
    elif median_score >= liveness_gate.threshold - liveness_gate.margin and max_score >= liveness_gate.threshold + liveness_gate.margin:
        reason = "CDCN_BORDERLINE_LIVE_FACE"
        decision = "ALLOW"
    else:
        reason = "CDCN_UNCERTAIN"
        decision = "BLOCK"

    return {
        "decision": decision,
        "reason": reason,
        "median_score": median_score,
        "min_score": min_score,
        "max_score": max_score,
        "frames": len(scores),
    }


def choose_identity(candidates):
    active = [candidate for candidate in candidates if candidate["worker"] is not None]

    if not active:
        best_confidence = min(
            [candidate["confidence"] for candidate in candidates],
            default=100.0,
        )
        return None, best_confidence

    grouped = {}
    for candidate in active:
        grouped.setdefault(candidate["username"], []).append(candidate)

    def rank(username):
        rows = grouped[username]
        confidences = [row["confidence"] for row in rows]
        return (-len(rows), float(np.median(confidences)), min(confidences))

    username = sorted(grouped, key=rank)[0]
    rows = grouped[username]
    confidences = [row["confidence"] for row in rows]
    best_confidence = min(confidences)
    median_confidence = float(np.median(confidences))

    if len(rows) < 2 and best_confidence >= CONFIDENCE_THRESHOLD - 10:
        return None, best_confidence

    if median_confidence >= CONFIDENCE_THRESHOLD and best_confidence >= CONFIDENCE_THRESHOLD - 8:
        return None, best_confidence

    worker = rows[0]["worker"]
    worker_id, username, full_name, job_title, role, is_active = worker

    return {
        "username": username,
        "full_name": full_name,
        "job_title": job_title,
        "confidence": median_confidence,
    }, median_confidence


def build_decision(department, liveness, candidates):
    if liveness["decision"] == "BLOCK":
        is_clear_spoof = liveness["reason"] == "CDCN_CLEAR_SPOOF"
        username = "spoof_attempt" if is_clear_spoof else "liveness_uncertain"
        full_name = "Spoof attempt" if is_clear_spoof else "Liveness uncertain"
        job_title = "Anti-spoof gate"
        confidence = max(0.0, min(100.0, (1.0 - liveness["median_score"]) * 100.0))
        status = "ACCESS_DENIED_SPOOF" if is_clear_spoof else "ACCESS_DENIED_LIVENESS_UNCERTAIN"
        decision_text = "ACCESS DENIED / SPOOF" if is_clear_spoof else "ACCESS DENIED / LIVENESS"

        return {
            "username": username,
            "full_name": full_name,
            "job_title": job_title,
            "status": status,
            "access_text": "NO",
            "decision_text": decision_text,
            "confidence": confidence,
            "color": (0, 0, 255),
        }

    identity, confidence = choose_identity(candidates)

    if identity is None:
        return {
            "username": "unknown",
            "full_name": "Unknown",
            "job_title": "Unknown",
            "status": "ACCESS_DENIED_UNKNOWN_FACE",
            "access_text": "NO",
            "decision_text": "ACCESS DENIED",
            "confidence": confidence,
            "color": (0, 0, 255),
        }

    username = identity["username"]
    allowed = is_worker_allowed(username, department)

    if allowed:
        status = "ACCESS_GRANTED"
        access_text = "YES"
        decision_text = "ACCESS GRANTED"
        color = (0, 255, 0)
    else:
        status = "ACCESS_DENIED_NO_PERMISSION"
        access_text = "NO"
        decision_text = "ACCESS DENIED"
        color = (0, 0, 255)

    return {
        "username": username,
        "full_name": identity["full_name"],
        "job_title": identity["job_title"],
        "status": status,
        "access_text": access_text,
        "decision_text": decision_text,
        "confidence": identity["confidence"],
        "color": color,
    }


def record_final_decision(frame, department, decision, liveness):
    capture_statuses = {
        "ACCESS_DENIED_NO_PERMISSION",
        "ACCESS_DENIED_UNKNOWN_FACE",
        "ACCESS_DENIED_SPOOF",
        "ACCESS_DENIED_LIVENESS_UNCERTAIN",
    }
    email_statuses = {
        "ACCESS_DENIED_NO_PERMISSION",
        "ACCESS_DENIED_UNKNOWN_FACE",
        "ACCESS_DENIED_SPOOF",
    }

    capture_path = None
    if decision["status"] in capture_statuses:
        capture_path = save_security_capture(
            frame,
            decision["username"],
            decision["status"],
            department,
        )

    write_watermarked_log(decision["username"], decision["status"], decision["confidence"])
    save_access_attempt(
        decision["username"],
        decision["full_name"],
        department,
        decision["status"],
        decision["confidence"],
    )

    if decision["status"] in email_statuses:
        send_alert_if_needed(
            username=decision["username"],
            full_name=decision["full_name"],
            job_title=decision["job_title"],
            department=department,
            status=decision["status"],
            confidence=decision["confidence"],
            capture_path=capture_path,
        )

    write_access_live_status(
        full_name=decision["full_name"],
        job_title=decision["job_title"],
        department=department,
        access_text=decision["access_text"],
        decision_text=decision["decision_text"],
        status=decision["status"],
        confidence=decision["confidence"],
        liveness_decision=liveness["decision"],
        liveness_score=liveness["median_score"],
    )

    print("-" * 80)
    print(f"Liveness: {liveness['decision']} | reason={liveness['reason']} | score={liveness['median_score']:.3f}")
    print(f"Face recognized: {decision['full_name']}")
    print(f"Job title: {decision['job_title']}")
    print(f"Target department: {department}")
    print(f"Access to department: {decision['access_text']}")
    print(f"Decision: {decision['decision_text']}")
    print(f"Status: {decision['status']}")
    print(f"Confidence: {decision['confidence']:.2f}")
    if capture_path:
        print(f"Capture saved: {capture_path}")


def panel_lines_for_decision(department, decision, liveness):
    return [
        f"Liveness: {liveness['decision']} score={liveness['median_score']:.3f}",
        f"Face recognized: {decision['full_name']}",
        f"Job title: {decision['job_title']}",
        f"Target department: {department}",
        f"Access to department: {decision['access_text']}",
        f"Decision: {decision['decision_text']}",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--department", required=False)
    args = parser.parse_args()

    init_database()

    if args.department:
        department = args.department
    else:
        department = choose_department()

    if department not in get_departments():
        print(f"ERROR: Unknown department: {department}")
        return

    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = LOCK_DIR / f"access_{department.replace(' ', '_')}_cdcn.lock"

    if lock_file.exists():
        print(f"CDCN access control is already running for department: {department}")
        return

    lock_file.write_text("running", encoding="utf-8")

    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            print("ERROR: Model not found. Register a worker and train the model first.")
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)

        with open(LABELS_PATH, "rb") as f:
            label_names = pickle.load(f)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        try:
            liveness_gate = CDCNLivenessGate(
                threshold=CDCN_LIVENESS_THRESHOLD,
                margin=CDCN_LIVENESS_MARGIN,
                clear_spoof_score=CDCN_CLEAR_SPOOF_SCORE,
                cpu=True,
            )
        except Exception as error:
            print(f"ERROR: Could not load CDCN SpoofGate: {error}")
            return

        cap = open_camo_camera()

        if not cap.isOpened():
            print(f"ERROR: Could not open Camo index={CAMERA_INDEX}, backend={CAMERA_BACKEND}")
            return

        window_name = f"CDCN Access Control - {department}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1100, 720)
        cv2.moveWindow(window_name, 120, 80)

        print("Company biometric access control started with CDCN SpoofGate.")
        print(f"Target department: {department}")
        print(f"Liveness threshold: {liveness_gate.threshold:.3f}")
        print("Waiting for a face. The decision window starts after detection.")

        write_access_live_status(
            full_name="-",
            job_title="-",
            department=department,
            access_text="-",
            decision_text="WAITING FOR FACE",
            status="WAITING_FOR_FACE",
            confidence=0.0,
            liveness_decision="PENDING",
            liveness_score=0.0,
        )

        scan_started_at = None
        candidates = []
        final_frame = None
        last_status_write = 0
        clear_spoof_announced = False

        while True:
            scan_finished_early = False
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Waiting for Camo frame...")
                time.sleep(0.1)
                continue

            final_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(90, 90),
            )

            panel_color = (255, 255, 255)
            panel_lines = [
                "Waiting for face...",
                f"Target department: {department}",
                "The 7 second decision starts after detection.",
                "Decision: -",
            ]

            if len(faces) > 0:
                faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
                x, y, w, h = [int(v) for v in faces[0]]

                if scan_started_at is None:
                    scan_started_at = time.time()
                    liveness_gate.reset()
                    print("Face detected. Starting 7 second decision window.")

                elapsed = time.time() - scan_started_at
                remaining = max(0, int(SCAN_SECONDS_AFTER_FACE - elapsed + 0.99))

                live = liveness_gate.update(frame, (x, y, w, h))

                face_gray = gray[y:y + h, x:x + w]
                face_gray = cv2.resize(face_gray, (200, 200))
                label, confidence = recognizer.predict(face_gray)
                predicted_username = label_names.get(label, "unknown")
                worker = get_worker(predicted_username) if confidence < CONFIDENCE_THRESHOLD else None
                candidates.append(
                    {
                        "username": predicted_username,
                        "confidence": float(confidence),
                        "worker": worker,
                    }
                )

                panel_color = (0, 255, 255)
                panel_lines = [
                    "Scanning face...",
                    f"Target department: {department}",
                    f"Liveness score: {live.last_score:.3f} | frames: {live.frames}",
                    f"Best face candidate: {predicted_username} ({confidence:.2f})",
                    f"Decision closes in: {remaining}s",
                ]

                cv2.rectangle(frame, (x, y), (x + w, y + h), panel_color, 3)

                now = time.time()
                if now - last_status_write >= 0.5:
                    write_access_live_status(
                        full_name=predicted_username if predicted_username != "unknown" else "-",
                        job_title="-",
                        department=department,
                        access_text="-",
                        decision_text=f"SCANNING ({remaining}s)",
                        status="SCANNING",
                        confidence=confidence,
                        liveness_decision=live.decision,
                        liveness_score=live.last_score,
                    )
                    last_status_write = now

                if (
                    live.decision == "BLOCK"
                    and live.reason == "CDCN_CLEAR_SPOOF"
                    and elapsed >= CLEAR_SPOOF_CLOSE_SECONDS
                ):
                    if not clear_spoof_announced:
                        print("Clear spoof detected. Closing access window with SPOOF decision.")
                        clear_spoof_announced = True
                    scan_finished_early = True
                    break

            elif scan_started_at is not None:
                elapsed = time.time() - scan_started_at
                remaining = max(0, int(SCAN_SECONDS_AFTER_FACE - elapsed + 0.99))
                panel_color = (0, 255, 255)
                panel_lines = [
                    "Face temporarily lost.",
                    f"Target department: {department}",
                    "Keep your face visible.",
                    f"Decision closes in: {remaining}s",
                ]

            draw_status_panel(frame, panel_lines, panel_color)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Access window closed by Q.")
                write_access_live_status(
                    full_name="Unknown",
                    job_title="Unknown",
                    department=department,
                    access_text="NO",
                    decision_text="CANCELLED",
                    status="CANCELLED",
                    confidence=0.0,
                    liveness_decision="CANCELLED",
                    liveness_score=0.0,
                )
                close_camera(cap)
                return

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Access window closed by user.")
                    write_access_live_status(
                        full_name="Unknown",
                        job_title="Unknown",
                        department=department,
                        access_text="NO",
                        decision_text="CANCELLED",
                        status="CANCELLED",
                        confidence=0.0,
                        liveness_decision="CANCELLED",
                        liveness_score=0.0,
                    )
                    close_camera(cap)
                    return
            except cv2.error:
                break

            if scan_finished_early:
                break

            if scan_started_at is not None and time.time() - scan_started_at >= SCAN_SECONDS_AFTER_FACE:
                break

        if final_frame is None:
            final_frame = 255 * cv2.UMat(720, 1280, cv2.CV_8UC3).get()

        liveness = decide_liveness(liveness_gate)
        decision = build_decision(department, liveness, candidates)
        record_final_decision(final_frame, department, decision, liveness)

        final_start = time.time()
        while time.time() - final_start < FINAL_DISPLAY_SECONDS:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = final_frame.copy()

            draw_status_panel(
                frame,
                panel_lines_for_decision(department, decision, liveness),
                decision["color"],
            )
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        close_camera(cap)

    finally:
        if lock_file.exists():
            lock_file.unlink()


if __name__ == "__main__":
    main()
