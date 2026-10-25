"""Audit phone/remote/sock YOLO data before optional training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import yaml

NAMES = ['phone', 'remote', 'sock']


def audit(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    names = cfg['names']
    if isinstance(names, dict):
        if set(names) != {0, 1, 2}:
            raise ValueError("class IDs must be 0, 1, 2")
        names = [names[key] for key in sorted(names)]
    if names != NAMES:
        raise ValueError('class order must be phone, remote, sock (0, 1, 2)')
    root = Path(cfg.get('path', path.parent))
    if not root.is_absolute():
        root = path.parent / root
    seen = {}
    report = {}
    for split in ('train', 'val'):
        counts = dict.fromkeys(NAMES, 0)
        directories = cfg[split] if isinstance(cfg[split], list) else [cfg[split]]
        images = []
        for directory in directories:
            directory = Path(directory)
            directory = directory if directory.is_absolute() else root / directory
            if not directory.is_dir():
                raise ValueError(f'missing image directory: {directory}')
            images.extend(p for p in directory.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        if not images:
            raise ValueError(f'{split} contains no images')
        for image in images:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise ValueError(f'train/val image leakage: {image}')
            seen[digest] = split
            parts = list(image.parts)
            if 'images' not in parts:
                raise ValueError('use YOLO images/train and labels/train directories')
            parts[len(parts)-1-parts[::-1].index('images')] = 'labels'
            label = Path(*parts).with_suffix('.txt')
            if not label.is_file():
                raise ValueError(f'missing label (empty file required for negatives): {label}')
            classes = set()
            for line in label.read_text().splitlines():
                values = line.split()
                if len(values) != 5:
                    raise ValueError(f'invalid YOLO box: {label}')
                cls, x, y, w, h = map(float, values)
                if (not all(math.isfinite(v) for v in (cls, x, y, w, h)) or
                        cls not in (0, 1, 2) or not 0 < w <= 1 or not 0 < h <= 1 or
                        not w/2 <= x <= 1-w/2 or not h/2 <= y <= 1-h/2):
                    raise ValueError(f'invalid class or normalized box: {label}')
                classes.add(int(cls))
            for cls in classes:
                counts[NAMES[cls]] += 1
        if not all(counts.values()):
            raise ValueError(f'each split needs all three classes; {split}: {counts}')
        report[split] = dict(images=len(images), images_per_class=counts)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--weights', default='data/models/phone_v2.pt')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--name', default='objects_v1')
    args = parser.parse_args(argv)
    report = audit(args.data)
    print(json.dumps(report, indent=2))
    if args.train:
        if args.epochs <= 0 or Path(args.name).name != args.name:
            parser.error('positive epochs and a simple run name required')
        if any(report['train']['images_per_class'][label] < 100 for label in ('remote', 'sock')):
            parser.error('collect and label at least 100 training images each for remote and sock')
        if (Path('data/runs') / args.name).exists():
            parser.error('run exists; choose a new name to preserve checkpoints')
        from ultralytics import YOLO
        YOLO(args.weights).train(data=str(args.data.resolve()), imgsz=1280,
                                 epochs=args.epochs, device=args.device,
                                 project='data/runs', name=args.name, exist_ok=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
