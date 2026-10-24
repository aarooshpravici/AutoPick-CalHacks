"""Dataset convert step (task T2). Tiny synthetic LeRobot datasets in tmp_path, shaped like the
ones lerobot-record writes for the SO-101: 2 episodes of 10 frames at 64x36, converted to 32x18.
Nothing is read from or written to the repo's data/ or the Hugging Face caches in ~/.cache."""

import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("lerobot.datasets")

from arm.poses import JOINTS  # noqa: E402
from training import convert  # noqa: E402

NAMES = [f"{j}.pos" for j in JOINTS]
TASKS = ["Pick up the object from the floor", "Pick up the other object"]


@pytest.fixture(autouse=True)
def hf_cache(tmp_path, monkeypatch):
    """Keep the datasets library's parquet cache under tmp_path."""
    import datasets

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf_datasets_cache")


def picture(i: int, width: int, height: int) -> np.ndarray:
    """A frame with detail at every scale, different in each frame."""
    y, x = np.mgrid[0:height, 0:width]
    img = np.stack([(x * 255 // width + i * 7) % 256, (y * 255 // height) % 256, ((x + y + i) * 13) % 256], -1)
    return img.astype(np.uint8)


def make_dataset(root, size=(64, 36), dtype="video", cameras=("wrist",), episodes=2, frames=10):
    from lerobot.datasets import LeRobotDataset

    w, h = size
    features = {"action": {"dtype": "float32", "shape": (6,), "names": NAMES},
                "observation.state": {"dtype": "float32", "shape": (6,), "names": NAMES}}
    for cam in cameras:
        features[f"observation.images.{cam}"] = {"dtype": dtype, "shape": (h, w, 3),
                                                 "names": ["height", "width", "channels"],
                                                 "info": {"is_depth_map": False}}
    ds = LeRobotDataset.create("local/src", 30, features, root=root, robot_type="so_follower",
                               use_videos=dtype == "video")
    rng = np.random.default_rng(0)
    for e in range(episodes):
        for i in range(frames):
            frame = {"action": rng.uniform(-90, 90, 6).astype(np.float32),
                     "observation.state": rng.uniform(-90, 90, 6).astype(np.float32), "task": TASKS[e % 2]}
            for k, cam in enumerate(cameras):
                frame[f"observation.images.{cam}"] = picture(10 * e + i + k, w, h)
            ds.add_frame(frame)
        ds.save_episode(parallel_encoding=False)
    ds.finalize()
    return root


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    """Read-only source datasets, shared: convert never writes to its source (checked below)."""
    import datasets

    base = tmp_path_factory.mktemp("sources")
    old = datasets.config.HF_DATASETS_CACHE
    datasets.config.HF_DATASETS_CACHE = base / "hf_datasets_cache"
    try:
        return {"video": make_dataset(base / "video"),
                "image": make_dataset(base / "image", dtype="image"),
                "two": make_dataset(base / "two", cameras=("wrist", "scene"), episodes=1, frames=6)}
    finally:
        datasets.config.HF_DATASETS_CACHE = old


def snapshot(root):
    """Every file under root: its contents' hash and modification time."""
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def load(root):
    from lerobot.datasets import LeRobotDataset

    return LeRobotDataset("local/x", root=root)


def column(ds, key):
    import torch

    return torch.stack([ds[i][key] for i in range(len(ds))])


def info(root):
    return json.loads((root / "meta" / "info.json").read_text())


def no_leftovers(parent):
    return [p.name for p in parent.iterdir() if p.name.endswith(".partial")] == []


@pytest.mark.parametrize("kind", ["video", "image"])
def test_convert_resizes_images_and_keeps_everything_else(sources, tmp_path, kind):
    import torch

    src, dst = sources[kind], tmp_path / "out_32"
    before = snapshot(src)
    summary = convert.convert(src, dst, 32, 18, log=lambda m: None)
    assert snapshot(src) == before                                         # the source is never touched
    assert summary == {"episodes": 2, "frames": 20,
                       "cameras": {"observation.images.wrist": ((64, 36), (32, 18))}}

    a, b = load(src), load(dst)
    assert (b.meta.total_episodes, b.meta.total_frames, b.meta.fps) == (2, 20, 30)
    assert b.meta.robot_type == "so_follower"
    for key in ("action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"):
        assert torch.equal(column(a, key), column(b, key)), key
    assert [a[i]["task"] for i in range(20)] == [b[i]["task"] for i in range(20)]
    assert list(b.meta.tasks.index) == list(a.meta.tasks.index)

    feature = info(dst)["features"]["observation.images.wrist"]
    assert (feature["dtype"], feature["shape"]) == (kind, [18, 32, 3])
    assert info(dst)["features"]["action"] == info(src)["features"]["action"]
    assert info(dst)["features"]["observation.state"] == info(src)["features"]["observation.state"]
    if kind == "video":
        assert (feature["info"]["video.width"], feature["info"]["video.height"]) == (32, 18)
        source = info(src)["features"]["observation.images.wrist"]["info"]
        assert feature["info"]["video.codec"] == source["video.codec"]
    assert tuple(b[7]["observation.images.wrist"].shape) == (3, 18, 32)
    stats = b.meta.stats
    assert np.asarray(stats["observation.images.wrist"]["mean"]).shape == (3, 1, 1)
    assert np.allclose(stats["action"]["mean"], a.meta.stats["action"]["mean"])
    assert no_leftovers(tmp_path)


def test_converted_frames_are_what_the_policy_would_see(sources, tmp_path):
    """convert resizes exactly like arm/policy.py does a live frame (cv2.INTER_AREA). Image
    datasets are lossless, so the stored frame equals the runtime resize of the source frame."""
    from arm.policy import ActGrasp

    src, dst = sources["image"], tmp_path / "out"
    convert.convert(src, dst, 32, 18, log=lambda m: None)
    a, b = load(src), load(dst)

    class Frames:
        def read(self):
            return convert.to_hwc_uint8(a[13]["observation.images.wrist"])

    live = ActGrasp.__new__(ActGrasp)              # only its frame resize, no checkpoint
    live.cameras, live.image_sizes = {"wrist": Frames()}, {"wrist": (32, 18)}
    stored = convert.to_hwc_uint8(b[13]["observation.images.wrist"])
    assert np.array_equal(stored, live._frame("wrist"))


def test_two_cameras(sources, tmp_path):
    summary = convert.convert(sources["two"], tmp_path / "out", 32, 18, log=lambda m: None)
    assert set(summary["cameras"]) == {"observation.images.wrist", "observation.images.scene"}
    features = info(tmp_path / "out")["features"]
    assert features["observation.images.scene"]["shape"] == [18, 32, 3]
    assert load(tmp_path / "out").meta.total_frames == 6


@pytest.mark.parametrize("size", [(32, 24), (36, 64), (40, 18)])
def test_refuses_to_change_the_aspect_ratio(sources, tmp_path, size):
    with pytest.raises(SystemExit, match="aspect ratio"):
        convert.convert(sources["video"], tmp_path / "out", *size)
    assert not (tmp_path / "out").exists() and no_leftovers(tmp_path)


def test_refuses_an_odd_size_for_video(sources, tmp_path):
    with pytest.raises(SystemExit, match="even"):
        convert.convert(sources["video"], tmp_path / "out", 16, 9)
    assert not (tmp_path / "out").exists()


def test_never_overwrites_dst(sources, tmp_path):
    dst = tmp_path / "out"
    dst.mkdir()
    (dst / "keep.txt").write_text("mine")
    with pytest.raises(SystemExit, match="already exists"):
        convert.convert(sources["video"], dst, 32, 18)
    assert [p.name for p in dst.iterdir()] == ["keep.txt"]


def test_refuses_dst_inside_src(sources, tmp_path):
    src = tmp_path / "src"
    make_dataset(src, episodes=1, frames=2)
    before = snapshot(src)
    with pytest.raises(SystemExit, match="outside"):
        convert.convert(src, src / "resized", 32, 18)
    assert snapshot(src) == before


def test_refuses_a_folder_that_is_not_a_dataset(tmp_path):
    with pytest.raises(SystemExit, match="not a LeRobot dataset"):
        convert.convert(tmp_path, tmp_path.parent / "out", 32, 18)


def test_a_failure_leaves_no_dst(sources, tmp_path, monkeypatch):
    calls = []
    resize = convert.resize

    def flaky(image, width, height):
        calls.append(1)
        if len(calls) == 13:                       # in the second episode
            raise RuntimeError("disk full")
        return resize(image, width, height)

    monkeypatch.setattr(convert, "resize", flaky)
    with pytest.raises(RuntimeError, match="disk full"):
        convert.convert(sources["video"], tmp_path / "out", 32, 18, log=lambda m: None)
    assert not (tmp_path / "out").exists() and no_leftovers(tmp_path)


def test_cli(sources, tmp_path, capsys):
    assert convert.main(["--src", str(sources["video"]), "--dst", str(tmp_path / "out"),
                         "--width", "32", "--height", "18"]) == 0
    out = capsys.readouterr().out
    assert "observation.images.wrist: 64x36 -> 32x18" in out
    assert "2 episodes, 20 frames" in out



def test_default_size_is_what_act_trains_on(tmp_path):
    from training import record

    src = tmp_path / "src"
    make_dataset(src, size=(640, 360), episodes=1, frames=2)
    convert.main(["--src", str(src), "--dst", str(tmp_path / "out")])
    assert info(tmp_path / "out")["features"]["observation.images.wrist"]["shape"] == [record.HEIGHT,
                                                                                       record.WIDTH, 3]
