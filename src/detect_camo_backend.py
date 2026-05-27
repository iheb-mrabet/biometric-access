import cv2
import time
import numpy as np

CAMERA_INDEX = 0

BACKENDS = [
    ("MSMF", cv2.CAP_MSMF),
    ("DSHOW", cv2.CAP_DSHOW),
    ("ANY", cv2.CAP_ANY),
]

def is_real_frame(frame):
    if frame is None:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_value = np.mean(gray)
    std_value = np.std(gray)

    print(f"Frame check -> mean={mean_value:.2f}, std={std_value:.2f}")

    # black screen usually has very low mean and low variation
    return mean_value > 10 and std_value > 5

for backend_name, backend_value in BACKENDS:
    print("=" * 60)
    print(f"Testing Camo index {CAMERA_INDEX} with backend: {backend_name}")

    cap = cv2.VideoCapture(CAMERA_INDEX, backend_value)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    time.sleep(1)

    if not cap.isOpened():
        print(f"[FAILED] Could not open Camo with backend {backend_name}")
        cap.release()
        continue

    working = False

    for i in range(30):
        ret, frame = cap.read()

        if not ret:
            print(f"[FAILED] No frame with backend {backend_name}")
            break

        if is_real_frame(frame):
            working = True

            cv2.putText(
                frame,
                f"CAMO WORKING - BACKEND: {backend_name}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Camo backend test", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

            print(f"[SUCCESS] Camo works with backend: {backend_name}")

            config_content = f'''CAMERA_INDEX = 0
CAMERA_BACKEND = "{backend_name}"
'''
            with open("src/config.py", "w", encoding="utf-8") as f:
                f.write(config_content)

            print("Saved working backend to src/config.py")
            break

    cap.release()
    cv2.destroyAllWindows()

    if working:
        break

else:
    print("No backend produced a real Camo image.")
    print("Camo is visible in Camo Studio, but OpenCV receives black frames.")
    print("Restart Camo Studio, keep Diffuser active, then run this script again.")
