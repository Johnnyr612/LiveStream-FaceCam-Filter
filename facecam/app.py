import time
import cv2

from .tracker import FaceTracker


def main() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try index 1 or 2.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    tracker = FaceTracker()

    win = "FaceCam MVP - Tracking (Press Q to Quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps = 0.0
    frame_count = 0
    fps_t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        frame_count += 1
        now = time.perf_counter()
        if now - fps_t0 >= 0.5:
            fps = frame_count / (now - fps_t0)
            fps_t0 = now
            frame_count = 0

        signals, debug = tracker.process(frame)

        cv2.putText(
            frame,
            f"FPS: {fps:0.1f} | M2 tracking | OBS capture this window",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if signals is None:
            cv2.putText(frame, "No face detected", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            tracker.draw_debug(frame, signals, debug)

        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
