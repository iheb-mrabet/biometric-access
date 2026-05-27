import cv2
import time
import numpy as np
from config import CAMERA_INDEX, CAMERA_BACKEND


def get_backend_value():
    if CAMERA_BACKEND == "MSMF":
        return cv2.CAP_MSMF
    if CAMERA_BACKEND == "DSHOW":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def open_camo_camera():
    backend_value = get_backend_value()

    cap = cv2.VideoCapture(CAMERA_INDEX, backend_value)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    return cap


def read_valid_frame(cap, timeout_seconds=8):
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        ret, frame = cap.read()

        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_value = np.mean(gray)
            std_value = np.std(gray)

            # Avoid black/empty frames
            if mean_value > 10 and std_value > 5:
                return True, frame

        time.sleep(0.1)

    return False, None
