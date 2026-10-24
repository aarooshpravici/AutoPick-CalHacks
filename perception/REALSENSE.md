# RealSense reader (P1)

Use the S4 environment in README.md and S6's `config/robot.yaml` (PR #8).
All stream modes, alignment and flip settings come from `realsense` in that file.
Colour is BGR uint8; depth is uint16 sensor units. Multiply depth by
`Frame.depth_scale` to obtain metres; zero is invalid. `Frame.t` is Unix time.
When alignment is disabled, do not index depth using colour pixel coordinates.

No-hardware checks, from the repository root:

```sh
.venv/perception/bin/python -m pytest tests/test_realsense.py -q
.venv/perception/bin/python -m perception.realsense --fake snapshot --count 3 --every 0.1
```

Synthetic snapshots go to `data/simulated/`, have `sim_` filenames and metadata
with `simulated: true`. They must not be mixed into the training dataset.
Snapshots save colour PNG, lossless 16-bit depth PNG and JSON with the depth
scale, timestamp and frame number. Every saved file and newly created output
directory is handed back to SUDO_UID/SUDO_GID when running under sudo.

With the camera available, the person runs these commands on the robot Mac:

```sh
sudo .venv/perception/bin/python -m perception.realsense preview
sudo .venv/perception/bin/python -m perception.realsense snapshot --count 10 --every 1
sudo .venv/perception/bin/python -m perception.realsense benchmark --count 300
```

Quit preview with q or Escape. Use the final mount for dataset photos. Snapshot
capture keeps reading between saves so snapshots do not come from stale queues.
Five attempts with one-second delays handle SDK errors; persistent failures are
reported rather than silently yielding stale frames. No other process may own
the camera. The agent never runs sudo.

P1 is not hardware-complete until the real colour/depth benchmark reaches
25+ fps. The benchmark warms up for 30 frames and reports whether alignment is
enabled. Fake benchmark results are explicitly marked simulated and are not a
camera performance measurement. Post the real result to issue #1.
