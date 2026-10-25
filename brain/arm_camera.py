"""Switch the RealSense from the colour detector to the grey IR arm camera (task A14).

After parking, the person presses "Switch to arm camera" on the page. Then, in order:
1. the colour publisher (perception.publish, a separate sudo process) is asked over
   UDP to release the RealSense, and must confirm it closed the camera;
2. the ACT checkpoint opens its webcams by robot.yaml `cameras` index: on macOS
   the RealSense then appears as a plain webcam that gives the grey infrared view
   the grasp_v1_ir demos were recorded with (plan: "RealSense D435 on macOS");
3. a preview window (brain.arm_preview, its own process) shows those frames, and
   the arm policy starts only after the window has a frame from every camera;
4. afterwards the cameras close and the publisher is asked to resume colour.

Nothing here moves the base or the arm.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from perception.publish import address

ROOT = Path(__file__).resolve().parents[1]

RELEASE, RELEASED, RESUME = b"release", b"released", b"resume"


class DetectorControl:
    """The publisher's control port (robot.yaml perception.control_udp)."""

    def __init__(self, value):
        self.destination = address(value)

    def release(self, timeout=5.0):
        """Ask the publisher to close the RealSense; return once it confirms."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(.25)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                sock.sendto(RELEASE, self.destination)
                try:
                    reply, _ = sock.recvfrom(64)
                except (socket.timeout, ConnectionError):
                    continue
                if reply == RELEASED:
                    return
        raise RuntimeError("colour detector did not release the RealSense; is perception.publish running?")

    def resume(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(RESUME, self.destination)


class FrameTap:
    """Wraps a policy camera; saves its latest frame as <folder>/<name>.jpg for the preview."""

    def __init__(self, camera, path: Path, every_s=.1):
        self.camera, self.path, self.every_s = camera, Path(path), every_s
        self.saved = 0.0
        self.lock = threading.Lock()

    def read(self):
        frame = self.camera.read()                          # RGB, as the policy gets it
        now = time.monotonic()
        if now - self.saved >= self.every_s:
            import cv2
            with self.lock:
                partial = self.path.with_suffix(".tmp.jpg")
                cv2.imwrite(str(partial), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                partial.replace(self.path)                  # the window never reads half a file
                self.saved = now
        return frame

    def close(self):
        self.camera.close()


class SimArmCamera:
    """Simulation: the same button and states, no camera, no checkpoint (scripted pick)."""

    def open(self, log=print, timeout=10.0):
        log("simulated switch to the arm camera (no camera opened)")
        return None

    def close(self):
        return []


class ArmCamera:
    """The real switch. Mission calls open(), then the pick runs, then close()."""

    def __init__(self, config, *, preview=True, load_policy=None):
        path = Path(config["handoff"]["arm_policy"])
        self.path = path if path.is_absolute() else ROOT / path
        self.control = DetectorControl(config["perception"]["control_udp"])
        self.preview = preview
        self.load_policy = load_policy
        self.window = self.folder = None
        self.policy = None

    def open(self, log=print, timeout=10.0):
        """Release colour, open the IR + wrist cameras, show them; returns the loaded policy."""
        if self.load_policy is None:
            from arm.pick import load_policy
            self.load_policy = load_policy
        self.control.release()
        log("colour detector released the RealSense")
        policy = self.load_policy(self.path)                 # cameras by robot.yaml index
        self.policy = policy
        self.folder = Path(tempfile.mkdtemp(prefix="arm-camera-"))
        policy.cameras = {name: FrameTap(cam, self.folder / f"{name}.jpg") for name, cam in policy.cameras.items()}
        deadline = time.monotonic() + timeout
        waiting = set(policy.cameras)
        while waiting:                                      # a frame from every camera before any arm motion
            for name in list(waiting):
                try:
                    policy.cameras[name].read()
                    waiting.discard(name)
                except TimeoutError:
                    pass
            if waiting and time.monotonic() > deadline:
                raise RuntimeError(f"no frames from arm cameras {sorted(waiting)}")
            time.sleep(.05)
        if self.preview:
            self.window = subprocess.Popen([sys.executable, "-m", "brain.arm_preview", str(self.folder)])
        log(f"arm cameras open ({', '.join(sorted(policy.cameras))}); preview window "
            f"{'open' if self.preview else 'off'}")
        return policy

    def close(self):
        """Close everything open() opened, then hand the RealSense back to colour."""
        notes = []
        if self.window is not None:
            self.window.terminate()
            try:
                self.window.wait(2)
            except subprocess.TimeoutExpired:
                self.window.kill()
            self.window = None
        if self.policy is not None:
            from arm.pick import close_cameras
            notes += close_cameras(self.policy.cameras)
            self.policy = None
        try:
            self.control.resume()
        except OSError as error:
            notes.append(f"resume colour detector failed: {error}")
        return notes
