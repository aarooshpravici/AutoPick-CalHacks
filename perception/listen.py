"""Print UDP detections and record one labelled live-check placement per run."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import time

from perception.publish import address
from perception.realsense import ROOT


def summary(messages, elapsed, placement):
    phones = [max((item for item in message["items"] if item["label"] == "phone"),
                  key=lambda item: item["conf"], default=None) for message in messages]
    seen = [item for item in phones if item is not None]
    return dict(placement=placement, frames=len(messages), detected_frames=len(seen),
                detected=bool(seen), hz=len(messages) / elapsed if elapsed else 0,
                max_conf=max((item["conf"] for item in seen), default=None),
                mean_pixel_height=(sum(item["bbox_px"][3] - item["bbox_px"][1] for item in seen)
                                   / len(seen) if seen else None))


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/robot.yaml")
    parser.add_argument("--placement", required=True, help="e.g. 0.75m-left or empty")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "data/logs/p3-live.jsonl")
    args = parser.parse_args(argv)
    if not 0 < args.seconds < 3600:
        parser.error("seconds must be between 0 and 3600")
    destination = address(yaml.safe_load(args.config.read_text())["perception"]["detections_udp"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    messages = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, args.output.open("a") as log:
        sock.bind(destination)
        sock.settimeout(.2)
        start = time.monotonic()
        while time.monotonic() - start < args.seconds:
            try:
                raw, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            message = json.loads(raw)
            messages.append(message)
            log.write(json.dumps(dict(placement=args.placement, message=message)) + "\n")
            print(json.dumps(message), flush=True)
        result = summary(messages, time.monotonic() - start, args.placement)
        result["recorded_at"] = datetime.now().isoformat()
        log.write(json.dumps(dict(summary=result)) + "\n")
        print("SUMMARY " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
