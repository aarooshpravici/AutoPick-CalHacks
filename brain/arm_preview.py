"""Preview window for the arm cameras (task A14): shows <folder>/*.jpg side by side.

Started by brain.arm_camera after the RealSense switches to its grey IR view, so the
person sees what the arm policy sees. It only reads image files: it opens no camera
and moves nothing. Q or Esc closes the window; the pick carries on regardless.
"""
import sys
import time
from pathlib import Path

TITLE = "Arm cameras (RealSense IR + wrist) | Q or Esc closes"


def compose(images, height=360):
    import cv2
    import numpy as np
    tiles = [cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height)) for image in images]
    return np.hstack(tiles)


def main(argv=None):
    import cv2
    folder = Path((argv or sys.argv[1:])[0])
    while True:
        images = [image for image in (cv2.imread(str(p)) for p in sorted(folder.glob("*.jpg"))
                                      if not p.name.endswith(".tmp.jpg")) if image is not None]
        if images:
            cv2.imshow(TITLE, compose(images))
        if cv2.waitKey(100) & 0xFF in (ord("q"), ord("Q"), 27):
            break
        if not images:
            time.sleep(.1)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
