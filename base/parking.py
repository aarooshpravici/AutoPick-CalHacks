"""B5 pure parking controller. No serial I/O; distances are metres."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Command:
    left: float = 0
    right: float = 0
    done: bool = False
    reason: str = ""


class Parking:
    def __init__(self, config):
        self.target = tuple(config['perception']['parking_spot_floor_m'])
        self.tolerance = config['perception']['parking_tolerance_m']
        self.track = config['base']['track_width_mm']
        self.limit = config['base']['max_speed_mm_s']
        self.cfg = config['navigation']
        self.pickup_max = self.cfg.get('pickup_max_distance_m', .33)
        if not math.isfinite(self.pickup_max) or self.pickup_max <= 0:
            raise ValueError('pickup range must be positive and finite')
        if math.hypot(*self.target) > self.pickup_max:
            raise ValueError('measured parking target exceeds pickup range; verify camera/arm reference')
        self.progress_timeout = self.cfg.get('approach_progress_timeout_s', 10)
        if not isinstance(self.progress_timeout, (int, float)) or not math.isfinite(self.progress_timeout) or self.progress_timeout <= 0:
            raise ValueError('approach progress timeout must be positive and finite')
        self.best_error = math.inf
        self.progress_at = None
        self.started = None
        self.last_frame = None
        self.settled = 0
        self.failure = None

    def update(self, floor, *, frame, age, now):
        if self.started is None:
            self.started = now
        if self.failure:
            return Command(reason=self.failure)
        if now - self.started > self.cfg['approach_timeout_s']:
            self.failure = 'approach timed out'
            return Command(reason=self.failure)
        if (floor is None or len(floor) != 2 or
                not all(math.isfinite(v) for v in floor) or
                not math.isfinite(age) or not 0 <= age <= self.cfg['detection_max_age_s']):
            self.settled = 0
            return Command(reason='waiting for fresh calibrated detection')
        x, y = floor
        dx, dy = x - self.target[0], y - self.target[1]
        if math.hypot(dx, dy) <= self.tolerance and math.hypot(x, y) <= self.pickup_max:
            if frame != self.last_frame:
                self.settled += 1
            self.last_frame = frame
            return Command(done=self.settled >= self.cfg['settle_frames'])
        error = math.hypot(dx, dy)
        if error < self.best_error - .005:
            self.best_error, self.progress_at = error, now
        elif self.progress_at is not None and now - self.progress_at >= self.progress_timeout:
            self.failure = 'approach made no visual progress; check traction or obstruction'
            return Command(reason=self.failure)
        self.settled = 0
        self.last_frame = frame
        if dx < -self.tolerance or x <= 0:
            self.failure = 'target passed; reposition manually'
            return Command(reason=self.failure)
        # Match the bearing of the measured pickup target, including an offset
        # arm mount. Positive angular speed turns left (right wheel faster).
        angle = math.atan2(y, x) - math.atan2(self.target[1], self.target[0])
        angle = math.atan2(math.sin(angle), math.cos(angle))
        turn = self.cfg['angular_gain'] * angle * self.track / 2
        forward = min(self.cfg['approach_speed_mm_s'], max(0, dx * 1000 * self.cfg['linear_gain']))
        if abs(angle) > math.radians(self.cfg['aligned_angle_deg']):
            forward = 0
        left, right = forward - turn, forward + turn
        scale = max(1, abs(left) / self.limit, abs(right) / self.limit)
        return Command(left / scale, right / scale)
