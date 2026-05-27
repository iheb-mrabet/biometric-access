import cv2

print("Testing only camera indexes 0 and 1")
print("Close this window with Q.")

for index in [0, 1]:
    print(f"Trying camera index {index}...")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"[NO] Camera {index} not opened")
        continue

    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"[ERROR] Camera {index} opened but no frame")
            break

        cv2.putText(
            frame,
            f"Camera index {index}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(f"Camera test index {index}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

print("Camera test finished.")
