"""Measured navigation adapter; no direct hardware access."""
import math
import time
from base.parking import Parking
from base.search import Search


class RealNavigation:
    def __init__(self, config, floor):
        self.parking = Parking(config)
        self.turns = Search(config)
        self.floor = floor
        self.frame = self.status = None
        self.age = math.inf

    def observe(self, frame, age, status):
        self.frame, self.age, self.status = frame, age, status

    def search(self):
        command = self.turns.update(self.status, now=time.monotonic())
        if command.reason and not command.done:
            raise RuntimeError(command.reason)
        return command.done, (command.left, command.right)

    def approach(self, detection):
        x1, _, x2, y2 = detection['bbox_px']
        floor = self.floor.map([(x1+x2)/2, y2])
        reported = detection['floor_m']
        if floor is None or reported is None or math.dist(floor, reported) > .001:
            floor = None  # Missing/different publisher calibration must never park.
        command = self.parking.update(floor, frame=self.frame, age=self.age, now=time.monotonic())
        if self.parking.failure:
            raise RuntimeError(command.reason)
        return command.done, (command.left, command.right)
