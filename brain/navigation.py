"""Measured navigation adapter; no direct hardware access."""
import math
import time
from base.parking import Parking
from base.search import Search


class RealNavigation:
    def __init__(self, config, floor):
        base, perception, nav = config['base'], config['perception'], config['navigation']
        positive = [base[key] for key in ('max_speed_mm_s', 'track_width_mm',
                    'wheel_diameter_mm', 'half_steps_per_rev')]
        positive += [nav[key] for key in ('detection_max_age_s', 'settle_frames',
                     'approach_timeout_s', 'approach_speed_mm_s', 'linear_gain',
                     'angular_gain', 'aligned_angle_deg', 'search_speed_mm_s', 'search_step_deg')]
        positive.append(perception['parking_tolerance_m'])
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in positive):
            raise ValueError('navigation dimensions and limits must be positive and finite')
        if (not 0 < nav['detection_max_age_s'] <= .3 or
                type(nav['settle_frames']) is not int or nav['settle_frames'] < 3 or
                not 0 < perception['parking_tolerance_m'] <= .03 or
                not math.isfinite(nav['search_pause_s']) or nav['search_pause_s'] < 0):
            raise ValueError('invalid freshness, settling or parking tolerance')
        target = perception['parking_spot_floor_m']
        if len(target) != 2 or not all(math.isfinite(v) for v in target) or target[0] <= 0:
            raise ValueError('pickup target must be finite and ahead of the camera')
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
