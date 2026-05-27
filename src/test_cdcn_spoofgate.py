import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "spoofgate_cdcn" / "spoofgate_cdcn.pt"
RESULT_FILE = PROJECT_ROOT / "data" / "cdcn_spoofgate_result.txt"

WARMUP_SECONDS = 0.8
STABLE_FACE_FRAMES = 4
TARGET_VALID_FRAMES = 5
MIN_VALID_FRAMES = 3
MAX_SCAN_SECONDS = 1.2


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


def preprocess(face_crop, image_size):
    resized = cv2.resize(face_crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    chw = np.transpose(rgb, (2, 0, 1))[None, ...]
    return torch.from_numpy(chw).float()


@torch.no_grad()
def score_frame(model, face_crop, image_size, device):
    tensor = preprocess(face_crop, image_size).to(device)
    logit, live_map = model(tensor)
    score = float(torch.sigmoid(logit.view(-1))[0].detach().cpu().item())
    return score


def decide(scores, threshold, margin):
    if len(scores) < MIN_VALID_FRAMES:
        return "BLOCK", "NOT_ENOUGH_VALID_FRAMES"

    median_score = float(np.median(scores))
    min_score = float(np.min(scores))
    strong_live = sum(score >= threshold + margin for score in scores)
    spoof_frames = sum(score < threshold for score in scores)

    if spoof_frames >= 3:
        return "BLOCK", "CDCN_SPOOF_MAJORITY"

    if median_score >= threshold and strong_live >= 2:
        return "ALLOW", "CDCN_LIVE_FACE"

    return "BLOCK", "CDCN_UNCERTAIN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--margin", type=float, default=0.04)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    ensure_src_imports()
    from camera_utils import open_camo_camera
    from spoofgate_cdcn_model import load_checkpoint

    if not MODEL_PATH.exists():
        print(f"Missing trained model: {MODEL_PATH}")
        print("Run: python src\\train_cdcn_spoofgate.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, trained_threshold, image_size, checkpoint = load_checkpoint(MODEL_PATH, device)
    threshold = args.threshold if args.threshold is not None else trained_threshold

    cap = open_camo_camera()
    if not cap.isOpened():
        message = "BLOCK | CAMERA_ERROR | Could not open camera"
        RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        RESULT_FILE.write_text(message, encoding="utf-8")
        print(message)
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    window_name = "CDCN SpoofGate Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 720)
    cv2.moveWindow(window_name, 120, 80)

    print("CDCN SpoofGate test started.")
    print("It waits for a stable face before scanning.")
    print(f"Threshold: {threshold:.3f} | Device: {device}")
    print("Press Q to cancel.")

    warmup_start = time.time()
    while time.time() - warmup_start < WARMUP_SECONDS:
        cap.read()
        cv2.waitKey(1)

    stable_face_count = 0
    capture_started = False
    capture_started_at = None
    scores = []
    raw_lines = []
    quality_failures = 0
    no_face_after_start = 0
    last_quality = "WAITING"
    last_score = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        box = detect_largest_face(face_cascade, frame)
        color = (0, 255, 255)
        status = "Waiting for face..."

        if box is None:
            if capture_started:
                no_face_after_start += 1
            else:
                stable_face_count = 0
        else:
            face_crop, crop_box = crop_face(frame, box)
            quality_ok, last_quality = quality_check(face_crop)
            x1, y1, x2, y2 = crop_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            if not quality_ok:
                if capture_started:
                    quality_failures += 1
                else:
                    stable_face_count = 0
                color = (0, 0, 255)
                status = last_quality
            elif not capture_started:
                stable_face_count += 1
                status = f"Stable face detected... {stable_face_count}/{STABLE_FACE_FRAMES}"
                if stable_face_count >= STABLE_FACE_FRAMES:
                    capture_started = True
                    capture_started_at = time.time()
                    status = "CDCN scan started"
                    color = (0, 255, 0)
            elif len(scores) < TARGET_VALID_FRAMES:
                last_score = score_frame(model, face_crop, image_size, device)
                scores.append(last_score)
                raw_lines.append(f"frame{len(scores)} score={last_score:.3f}")
                status = f"Scoring CDCN crop {len(scores)}/{TARGET_VALID_FRAMES}"
                color = (0, 255, 0) if last_score >= threshold else (0, 0, 255)

        if capture_started:
            elapsed = time.time() - capture_started_at
            if len(scores) >= TARGET_VALID_FRAMES:
                break
            if elapsed >= MAX_SCAN_SECONDS and len(scores) >= MIN_VALID_FRAMES:
                break
            if elapsed >= MAX_SCAN_SECONDS + 0.8:
                break
        else:
            elapsed = 0.0

        lines = [
            "CDCN SPOOFGATE TEST",
            status,
            f"Score: {last_score:.3f} | Threshold: {threshold:.3f}",
            f"Scan started: {capture_started} | Scores: {len(scores)}/{TARGET_VALID_FRAMES}",
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

    decision, reason = decide(scores, threshold, args.margin)
    median_score = float(np.median(scores)) if scores else 0.0
    min_score = float(np.min(scores)) if scores else 0.0
    max_score = float(np.max(scores)) if scores else 0.0
    elapsed = time.time() - capture_started_at if capture_started_at else 0.0

    final = (
        f"{decision} | {reason} | "
        f"median_score={median_score:.3f} | min_score={min_score:.3f} | max_score={max_score:.3f} | "
        f"threshold={threshold:.3f} | valid_frames={len(scores)} | quality_failures={quality_failures} | "
        f"no_face_after_start={no_face_after_start} | elapsed={elapsed:.2f}s | "
        + " | ".join(raw_lines)
    )

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(final, encoding="utf-8")
    print("=" * 80)
    print(final)
    print("=" * 80)

    display_start = time.time()
    while time.time() - display_start < 3:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        final_color = (0, 255, 0) if decision == "ALLOW" else (0, 0, 255)
        title = "FINAL DECISION: LIVE FACE" if decision == "ALLOW" else "FINAL DECISION: BLOCK ACCESS"
        lines = [
            title,
            f"Reason: {reason}",
            f"Median score: {median_score:.3f} | Threshold: {threshold:.3f}",
            f"Frames: {len(scores)} | Scan time: {elapsed:.2f}s",
            "CDCN-style local SpoofGate",
        ]
        draw_panel(frame, lines, final_color)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
