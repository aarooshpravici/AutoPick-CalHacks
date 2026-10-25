"""Return-to-start (A15): retrace the outbound path backwards from commanded steps.

The mission records the firmware's commanded half-step counters (STAT left/right)
while it searches and approaches. Driving those (left, right) waypoints in reverse
order retraces a differential drive's path exactly in commanded steps: the same
turns and straights, undone in reverse. Wheel slip is not measured (commanded steps,
not encoders), so the end point is approximate. No serial I/O here.
"""
from __future__ import annotations

import math

from base.parking import Command


class Path:
    """Waypoints kept while driving out: a point whenever either wheel moved `spacing` steps."""

    def __init__(self, spacing=100):
        self.spacing = spacing
        self.points: list[tuple[int, int]] = []

    def add(self, left, right):
        if not self.points:
            self.points.append((left, right))
            return
        last = self.points[-1]
        if max(abs(left - last[0]), abs(right - last[1])) >= self.spacing:
            self.points.append((left, right))

    def length_mm(self, steps_per_mm):
        """Wheel travel the firmware LIMIT will count: mean of both wheels' absolute steps."""
        total = sum((abs(b[0] - a[0]) + abs(b[1] - a[1])) / 2 for a, b in zip(self.points, self.points[1:]))
        return total / steps_per_mm


class Retrace:
    """Drives the recorded waypoints backwards, last to first. update() -> base.parking.Command."""

    def __init__(self, points, last, *, steps_per_mm, speed_mm_s, tolerance_steps=40, slow_mm=20):
        if not points:
            raise ValueError("no recorded path to retrace")
        if not (math.isfinite(speed_mm_s) and speed_mm_s > 0 and steps_per_mm > 0):
            raise ValueError("retrace speed and step size must be positive")
        # Targets in driving order: from where the base is now back to the first point.
        self.targets = list(reversed(points))
        self.previous = tuple(last)
        self.steps_per_mm, self.speed = steps_per_mm, speed_mm_s
        self.tolerance, self.slow_steps = tolerance_steps, slow_mm * steps_per_mm
        self.index = 0

    def update(self, left, right) -> Command:
        while self.index < len(self.targets):
            target = self.targets[self.index]
            d_left, d_right = target[0] - left, target[1] - right
            seg = (target[0] - self.previous[0], target[1] - self.previous[1])
            seg_sq = seg[0] ** 2 + seg[1] ** 2
            # Reached: close enough, or already past the end of this segment.
            past = seg_sq > 0 and ((left - self.previous[0]) * seg[0] + (right - self.previous[1]) * seg[1]) >= seg_sq
            if max(abs(d_left), abs(d_right)) <= self.tolerance or (past and self.index < len(self.targets) - 1):
                self.previous = target
                self.index += 1
                continue
            largest = max(abs(d_left), abs(d_right))
            speed = self.speed
            if self.index == len(self.targets) - 1:      # slow into the start point
                speed = max(3.0, min(speed, self.speed * largest / self.slow_steps))
            return Command(speed * d_left / largest, speed * d_right / largest)
        return Command(done=True)
