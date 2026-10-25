"""P4: generate markers, collect measured correspondences, fit and record grab spot."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import uuid
import numpy as np

from perception.floor import FloorMap, fit
from perception.realsense import ROOT, RealSense, Settings, make_output_dir, return_ownership


def write_json(path, data, *, replace=False):
    path = Path(path)
    make_output_dir(path.parent)
    if path.exists():
        if not replace:
            raise FileExistsError(f'{path} exists; use --replace to back it up and replace')
        backup_dir = ROOT/'data/calibration/backups'
        make_output_dir(backup_dir)
        backup = backup_dir/f'{path.stem}_{uuid.uuid4().hex}{path.suffix}'
        shutil.copy2(path, backup); return_ownership(backup)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    return_ownership(temporary)
    temporary.replace(path)


def marker_centre(corners):
    # Intersection of diagonals is the projected centre, unlike corner averaging.
    p = np.c_[np.asarray(corners).reshape(4, 2), np.ones(4)]
    centre = np.cross(np.cross(p[0], p[2]), np.cross(p[1], p[3]))
    if abs(centre[2]) < 1e-10:
        raise ValueError('degenerate marker')
    return (centre[:2]/centre[2]).tolist()


def detect(colour):
    import cv2
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(colour)
    return {} if ids is None else {int(i): marker_centre(c) for i, c in zip(ids.flatten(), corners)}


def make_markers(directory):
    import cv2
    make_output_dir(directory)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id in range(7):
        grid = cv2.aruco.generateImageMarker(dictionary, marker_id, 6)
        # Physical black-square bounds are 100x100 mm, with a 10 mm white margin.
        rects = ''.join(f'<rect x="{10+x*100/6}" y="{10+y*100/6}" width="{100/6}" height="{100/6}"/>'
                        for y in range(6) for x in range(6) if grid[y,x] == 0)
        path = directory/f'marker_{marker_id}.svg'
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="120mm" height="130mm" viewBox="0 0 120 130"><rect width="120" height="130" fill="white"/><g fill="black">{rects}</g><text x="10" y="124" font-size="5">ID {marker_id} - black square 100 mm</text></svg>')
        return_ownership(path)
    layout = dict(mount_note='', fit=[dict(id=i, floor_m=[None,None]) for i in range(4)],
                  check=[dict(id=i, floor_m=[None,None]) for i in range(4,7)])
    path = directory/'layout.json'
    if not path.exists(): write_json(path, layout)
    print(f'Print SVGs at 100% size; measure black square = 10 cm. Fill measured centre positions in {path}')


def collect(layout, settings, colour):
    if colour.shape[:2] != (settings.colour[1], settings.colour[0]):
        raise ValueError('image resolution does not match robot.yaml')
    if not layout.get('mount_note'):
        raise ValueError('record camera height/tilt/final mount in mount_note first')
    centres = detect(colour)
    result = dict(colour_size=list(settings.colour), flip=settings.flip,
                  mount_note=layout['mount_note'], captured_at=datetime.now(timezone.utc).isoformat())
    for group in ('fit','check'):
        result[group] = []
        for row in layout[group]:
            world = np.asarray(row['floor_m'], dtype=float)
            if world.shape != (2,) or not np.isfinite(world).all():
                raise ValueError(f'measure floor centre for marker {row["id"]}; x forward, y left, metres')
            if row['id'] not in centres:
                raise ValueError(f'marker {row["id"]} is not visible; adjust marker placement, not the mounted camera')
            result[group].append(dict(id=row['id'], floor_m=world.tolist(), pixel=centres[row['id']]))
    return result


def record_parking(path, config, pixel):
    import yaml
    settings = Settings.load(config)
    floor = FloorMap.load(path, colour_size=settings.colour, flip=settings.flip)
    target = floor.map(pixel)
    if target is None: raise ValueError('grab pixel is outside the measured region')
    # Preserve comments and SW1 calibration settings: only replace this existing key.
    original = config.read_text()
    pattern = r'(?m)^(\s*parking_spot_floor_m:)\s*\[[^\n]*$'
    updated, count = re.subn(pattern, lambda m: m[1]+f' [{target[0]:.6f}, {target[1]:.6f}]  # Measured by P4.', original)
    if count != 1: raise ValueError('expected exactly one parking_spot_floor_m key')
    yaml.safe_load(updated)
    data = dict(floor.data, parking_pixel=list(pixel), parking_spot_floor_m=target)
    write_json(path, data, replace=True)
    backup = ROOT/'data/calibration/backups'/f'robot_{uuid.uuid4().hex}.yaml'
    make_output_dir(backup.parent); shutil.copy2(config, backup); return_ownership(backup)
    temp = config.with_name(config.name+'.tmp')
    temp.write_text(updated); return_ownership(temp); temp.replace(config)
    print(json.dumps(dict(parking_spot_floor_m=target, calibration_id=data['calibration_id'])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/robot.yaml')
    sub = parser.add_subparsers(dest='command', required=True)
    markers = sub.add_parser('markers'); markers.add_argument('--output', type=Path, default=ROOT/'data/calibration/markers')
    capture = sub.add_parser('collect'); capture.add_argument('--layout', type=Path, required=True)
    capture.add_argument('--image', type=Path, help='use an existing P1 colour image instead of opening RealSense')
    capture.add_argument('--output', type=Path, default=ROOT/'data/calibration/floor_observations.json')
    capture.add_argument('--replace', action='store_true')
    solve = sub.add_parser('fit'); solve.add_argument('--observations', type=Path, required=True)
    solve.add_argument('--output', type=Path, default=ROOT/'config/floor_homography.json')
    solve.add_argument('--replace', action='store_true')
    park = sub.add_parser('parking'); park.add_argument('--pixel', nargs=2, type=float, required=True)
    park.add_argument('--homography', type=Path, default=ROOT/'config/floor_homography.json')
    args = parser.parse_args(argv)
    if args.command == 'markers': make_markers(args.output)
    elif args.command == 'fit':
        result = fit(json.loads(args.observations.read_text()))
        write_json(args.output, result, replace=args.replace)
        print(json.dumps(result['validation']))
    elif args.command == 'parking': record_parking(args.homography, args.config, args.pixel)
    else:
        import cv2
        settings = Settings.load(args.config)
        layout = json.loads(args.layout.read_text())
        if args.image:
            colour = cv2.imread(str(args.image))
            if colour is None: raise ValueError('image cannot be read')
            result = collect(layout, settings, colour)
        else:
            print('Camera only. SPACE captures visible markers; Q/Esc cancels. Keep robot and camera fixed.')
            result = None
            try:
                with RealSense(settings) as camera:
                    while True:
                        frame = camera.read(); view = frame.colour.copy()
                        for i, (x,y) in detect(view).items():
                            cv2.putText(view,str(i),(round(x),round(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        cv2.imshow('P4 markers: SPACE to capture, Q to cancel',view)
                        key = cv2.waitKey(1) & 255
                        if key in (ord('q'),27): return 1
                        if key == 32:
                            try: result = collect(layout, settings, frame.colour)
                            except ValueError as error: print(error); continue
                            break
            finally: cv2.destroyAllWindows()
        write_json(args.output, result, replace=args.replace)
        print(args.output)
    return 0


if __name__ == '__main__': raise SystemExit(main())
