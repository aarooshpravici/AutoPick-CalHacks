"""Measured floor homography with independent validation and bounded coverage."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import numpy as np


def points(values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != 2 or not np.isfinite(result).all():
        raise ValueError('points must be finite [x,y] pairs')
    return result


def project(matrix, pixels):
    p = points(pixels)
    h = np.c_[p, np.ones(len(p))] @ np.asarray(matrix).T
    if not np.isfinite(h).all() or np.any(np.abs(h[:, 2]) < 1e-10):
        raise ValueError('homography maps a point to infinity')
    return h[:, :2] / h[:, 2, None]


def inside(polygon, pixel):
    # Convex hull vertices must be ordered. Boundary points count as inside.
    vertices = points(polygon)
    pixel = points([pixel])[0]
    edges = np.roll(vertices, -1, axis=0) - vertices
    delta = pixel - vertices
    cross = edges[:, 0]*delta[:, 1] - edges[:, 1]*delta[:, 0]
    return bool(np.all(cross >= -1e-5) or np.all(cross <= 1e-5))


def fit(observations):
    import cv2
    train, checks = observations['fit'], observations['check']
    if len(train) < 4 or len(checks) < 3:
        raise ValueError('need >=4 fit points and >=3 independent check points')
    ids = [row['id'] for row in train + checks]
    if len(set(ids)) != len(ids):
        raise ValueError('fit/check marker IDs must be distinct')
    px = points([row['pixel'] for row in train])
    floor = points([row['floor_m'] for row in train])
    check_px = points([row['pixel'] for row in checks])
    truth = points([row['floor_m'] for row in checks])
    if len(np.unique(np.vstack([px, check_px]), axis=0)) != len(ids):
        raise ValueError('check pixels must be distinct from fit points')
    for data in (px, floor):
        if np.linalg.matrix_rank(data - data.mean(axis=0)) < 2:
            raise ValueError('calibration points must cover an area, not a line')
    matrix, _ = cv2.findHomography(px, floor, method=0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError('degenerate homography')
    region = cv2.convexHull(px.astype(np.float32)).reshape(-1, 2).tolist()
    if not all(inside(region, point) for point in check_px):
        raise ValueError('check points must lie within the calibrated region')
    errors = np.linalg.norm(project(matrix, check_px) - truth, axis=1)
    fit_errors = np.linalg.norm(project(matrix, px) - floor, axis=1)
    if max(errors.max(), fit_errors.max()) > .02:
        raise ValueError(f'calibration rejected: max fit/check error {max(errors.max(), fit_errors.max()):.4f} m exceeds 0.02 m')
    return dict(matrix=matrix.tolist(), colour_size=list(observations['colour_size']),
                flip=observations['flip'], calibration_id=str(uuid.uuid4()),
                calibrated_at=datetime.now(timezone.utc).isoformat(),
                mount_note=observations['mount_note'], pixel_region=region,
                fit=train, check=checks,
                validation=dict(max_error_m=float(errors.max()),
                                check_errors_m=errors.tolist(), fit_max_error_m=float(fit_errors.max()),
                                passed=True))


class FloorMap:
    def __init__(self, data, *, colour_size, flip):
        if tuple(data['colour_size']) != tuple(colour_size) or data['flip'] != flip:
            raise ValueError('camera size/flip changed; redo P4')
        self.data = data
        self.matrix = np.asarray(data['matrix'], dtype=float)
        if self.matrix.shape != (3, 3) or not np.isfinite(self.matrix).all() or np.linalg.matrix_rank(self.matrix) != 3:
            raise ValueError('invalid floor matrix')
        self.region = points(data['pixel_region'])
        if len(self.region) < 3 or np.linalg.matrix_rank(self.region-self.region.mean(axis=0)) < 2:
            raise ValueError('invalid calibrated region')
        # Recompute independent errors rather than trusting a stored passed flag.
        checks = data['check']
        if len(checks) < 3 or not data['validation']['passed']:
            raise ValueError('P4 needs three passed independent checks')
        expected = fit(data)
        actual = project(self.matrix, [row['pixel'] for row in checks])
        errors = np.linalg.norm(actual-points([row['floor_m'] for row in checks]), axis=1)
        if errors.max() > .02 or not np.allclose(self.matrix/self.matrix[2,2],
                                               np.asarray(expected['matrix'])/expected['matrix'][2][2], rtol=1e-5, atol=1e-7):
            raise ValueError('stored matrix fails calibration evidence')
        if not np.allclose(self.region, expected['pixel_region']):
            raise ValueError('calibrated region differs from measured points')

    @classmethod
    def load(cls, path, **kwargs):
        return cls(json.loads(Path(path).read_text()), **kwargs)

    def map(self, pixel):
        if not inside(self.region, pixel):
            return None
        return project(self.matrix, [pixel])[0].tolist()

    def parking(self):
        measured = self.map(self.data['parking_pixel'])
        if measured is None or not np.allclose(measured, self.data['parking_spot_floor_m'], atol=1e-6):
            raise ValueError('parking spot must be recorded inside the validated map')
        return measured
