import cv2

# Try 0 first. If it is not Camo, change it to 1.
CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Could not open camera index {CAMERA_INDEX}")
    exit()

print(f"Using camera index {CAMERA_INDEX}. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    cv2.putText(
        frame,
        f"Camera index: {CAMERA_INDEX}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Camo Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()