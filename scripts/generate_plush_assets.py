from pathlib import Path
import json
import numpy as np
import cv2


def save_rgba(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def circle_rgba(canvas, center, radius, color_bgra, thickness=-1):
    cv2.circle(canvas, center, radius, color_bgra, thickness, lineType=cv2.LINE_AA)


def ellipse_rgba(canvas, center, axes, angle, color_bgra, thickness=-1):
    cv2.ellipse(canvas, center, axes, angle, 0, 360, color_bgra, thickness, lineType=cv2.LINE_AA)


def main():
    out_dir = Path("assets/puppets/plush_01")
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = 512, 512
    pivot = [256, 300]

    # Base transparent canvas
    def blank():
        return np.zeros((H, W, 4), dtype=np.uint8)

    # BODY
    body = blank()
    # big head
    circle_rgba(body, (256, 260), 170, (170, 140, 220, 255))  # BGRA "plush purple-ish"
    # ears
    circle_rgba(body, (135, 150), 65, (160, 130, 210, 255))
    circle_rgba(body, (377, 150), 65, (160, 130, 210, 255))
    # inner face patch
    ellipse_rgba(body, (256, 310), (120, 95), 0, (200, 180, 240, 255))
    save_rgba(out_dir / "body.png", body)

    # EYES OPEN
    eyes_open = blank()
    # left eye
    circle_rgba(eyes_open, (205, 240), 22, (0, 0, 0, 255))
    circle_rgba(eyes_open, (212, 232), 6, (255, 255, 255, 255))
    # right eye
    circle_rgba(eyes_open, (307, 240), 22, (0, 0, 0, 255))
    circle_rgba(eyes_open, (314, 232), 6, (255, 255, 255, 255))
    save_rgba(out_dir / "eyes_open.png", eyes_open)

    # EYES CLOSED
    eyes_closed = blank()
    cv2.line(eyes_closed, (180, 240), (230, 240), (0, 0, 0, 255), 6, lineType=cv2.LINE_AA)
    cv2.line(eyes_closed, (282, 240), (332, 240), (0, 0, 0, 255), 6, lineType=cv2.LINE_AA)
    save_rgba(out_dir / "eyes_closed.png", eyes_closed)

    # MOUTH CLOSED
    mouth_closed = blank()
    cv2.ellipse(mouth_closed, (256, 330), (35, 18), 0, 0, 180, (20, 20, 60, 255), 5, lineType=cv2.LINE_AA)
    save_rgba(out_dir / "mouth_closed.png", mouth_closed)

    # MOUTH OPEN
    mouth_open = blank()
    ellipse_rgba(mouth_open, (256, 335), (34, 26), 0, (20, 20, 60, 255), thickness=-1)
    ellipse_rgba(mouth_open, (256, 342), (22, 14), 0, (60, 60, 180, 255), thickness=-1)
    save_rgba(out_dir / "mouth_open.png", mouth_open)

    rig = {
        "name": "plush_01",
        "base_scale": 1.0,
        "canvas_size": [W, H],
        "pivot": pivot,
        "layers": {
            "body": "body.png",
            "eyes_open": "eyes_open.png",
            "eyes_closed": "eyes_closed.png",
            "mouth_closed": "mouth_closed.png",
            "mouth_open": "mouth_open.png",
        },
    }
    (out_dir / "rig.json").write_text(json.dumps(rig, indent=2), encoding="utf-8")
    print("Generated plush_01 puppet pack:", out_dir)


if __name__ == "__main__":
    main()
