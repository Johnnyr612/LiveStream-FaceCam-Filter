import time
import cv2

from .tracker import FaceTracker
from .background import Background
from .puppet.renderer import PuppetRenderer


def main() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try index 1 or 2.")

    out_w, out_h = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, out_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, out_h)
    cap.set(cv2.CAP_PROP_FPS, 30)

    tracker = FaceTracker()
    puppet = PuppetRenderer(puppets_root="assets/puppets")
    puppet.load("plush_01")

    bg = Background("assets/backgrounds/bg.png")
    bg_img = bg.render((out_w, out_h))

    win = "FaceCam MVP - Avatar Output (No Face) | Q quit | P toggle puppet"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps = 0.0
    frame_count = 0
    fps_t0 = time.perf_counter()

    while True:
        ok, cam = cap.read()  # camera is tracking-only
        if not ok or cam is None:
            continue

        frame_count += 1
        now = time.perf_counter()
        if now - fps_t0 >= 0.5:
            fps = frame_count / (now - fps_t0)
            fps_t0 = now
            frame_count = 0

        # Output canvas: background only (face hidden)
        out = bg_img.copy()

        signals, debug = tracker.process(cam)

        # Overlay UI text onto output
        cv2.putText(
            out,
            f"FPS: {fps:0.1f} | Output: BG + Puppet (No Face) | OBS capture this window",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            "Hotkeys: P toggle puppet | Q quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        if signals is None:
            cv2.putText(out, "No face detected (tracking input only)", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # Anchor puppet to face position (nose)
            ax = int(debug.get("anchor_x", out_w // 2))
            ay = int(debug.get("anchor_y", out_h // 2))

            puppet.render_on(out, anchor_xy=(ax, ay), signals=signals)

        cv2.imshow(win, out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("p"), ord("P")):
            puppet.toggle()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
