"""B6 bounded turn/pause search using commanded steps, not wheel encoders."""
import math
from base.parking import Command


class Search:
    def __init__(self, config):
        base, nav = config['base'], config['navigation']
        self.speed = min(nav['search_speed_mm_s'], base['max_speed_mm_s'])
        self.step_angle = math.radians(nav['search_step_deg'])
        self.pause = nav['search_pause_s']
        self.mm_per_step = math.pi * base['wheel_diameter_mm'] / base['half_steps_per_rev']
        self.track = base['track_width_mm']
        if not 0 < self.step_angle <= math.pi or self.speed <= 0 or self.pause < 0:
            raise ValueError('invalid search settings')
        self.timeout = self.step_angle * self.track / (2 * self.speed) * 3 + 2
        self.origin = None
        self.started = None
        self.pause_until = None
        self.total = 0
        self.failure = None

    def update(self, status, *, now):
        if self.failure:
            return Command(reason=self.failure)
        if status is None or not 0 <= now - status.received_at <= .3 or status.estop:
            self.failure = 'search lost base status or E-stop'
            return Command(reason=self.failure)
        if self.total >= 4 * math.pi - 1e-6:
            return Command(done=True, reason='two commanded turns exhausted')
        if self.pause_until is not None:
            if now < self.pause_until:
                return Command()
            self.pause_until = None
            self.origin = None
        if self.origin is None:
            self.origin = status.left_steps, status.right_steps
            self.started = now
        angle = ((status.right_steps - self.origin[1]) -
                 (status.left_steps - self.origin[0])) * self.mm_per_step / self.track
        target = min(self.step_angle, 4 * math.pi - self.total)
        if angle >= target:
            self.total += angle
            self.pause_until = now + self.pause
            return Command()
        if now - self.started > self.timeout or angle < -.05:
            self.failure = 'search steps stalled or reversed'
            return Command(reason=self.failure)
        return Command(-self.speed, self.speed)
