"""Nonblocking section 9.2 sources; no cameras or model imports."""
import copy
import json
import math
import socket
import time


def validate(message):
    if not isinstance(message, dict) or not isinstance(message.get("items"), list):
        raise ValueError("expected a 9.2 message")
    if type(message.get("frame")) is not int or message["frame"] < 0:
        raise ValueError("invalid frame")
    if not isinstance(message.get("t"), (int, float)) or not math.isfinite(message["t"]):
        raise ValueError("invalid timestamp")
    for item in message["items"]:
        if not isinstance(item.get("label"), str) or not 0 <= item.get("conf", -1) <= 1:
            raise ValueError("invalid detection")
        for key, size in (("bbox_px", 4), ("centre_px", 2), ("floor_m", 2)):
            value = item[key]
            if key == "floor_m" and value is None:
                continue
            if len(value) != size or not all(math.isfinite(v) for v in value):
                raise ValueError(f"invalid {key}")
        if item["depth_m"] is not None and not (math.isfinite(item["depth_m"]) and item["depth_m"] > 0):
            raise ValueError("invalid depth")
    return message


class FakeDetections:
    def __init__(self, label="phone", labels=None):
        self.labels, self.frame = list(labels or [label]), 0

    def poll(self):
        self.frame += 1
        return dict(t=time.time(), frame=self.frame, items=[dict(
            label=label, conf=.9, bbox_px=[576, 432, 704, 504],
            centre_px=[640, 468], floor_m=None, depth_m=1.0) for label in self.labels])

    def close(self):
        pass


class ReplayDetections:
    """Replay one saved 9.2 message per tick, with a fresh reception timestamp."""
    def __init__(self, path):
        self.messages = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            message = row.get("message", row)
            if "items" in message:
                self.messages.append(validate(message))
        self.index = 0

    def poll(self):
        if self.index == len(self.messages):
            return None
        message = copy.deepcopy(self.messages[self.index])
        self.index += 1
        message["t"] = time.time()
        return message

    def close(self):
        pass


class UdpDetections:
    def __init__(self, address):
        host, port = address.rsplit(":", 1)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind((host, int(port)))
            self.socket.setblocking(False)
        except BaseException:
            self.socket.close()
            raise
        self.rejected = 0

    def poll(self):
        latest = None
        # Bounded work even under a packet flood; malformed traffic never refreshes
        # camera health. Consume the newest valid message from this batch.
        for _ in range(64):
            try:
                raw = self.socket.recv(65535)
            except BlockingIOError:
                break
            try:
                message = validate(json.loads(raw))
                if -0.5 <= time.time() - message["t"] <= 1:
                    latest = message
                else:
                    self.rejected += 1
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                self.rejected += 1
        return latest

    def close(self):
        self.socket.close()
