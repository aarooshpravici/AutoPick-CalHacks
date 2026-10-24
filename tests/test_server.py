"""Status page tests (task U2, plan 9.7 / 10.8 / 17). No hardware, no network, no sockets.

Concurrency tests use `with TestClient(app)`, which runs one event loop and one
worker-thread pool like uvicorn does, and send requests from several threads.
"""

import asyncio
import builtins
import json
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import anyio
import pytest
import uvicorn
from fastapi.testclient import TestClient

from ui import request as request_mod
from ui import server
from ui.request import parse_request
from ui.server import PAGE, STATUS_FIELDS, Brain, FakeBrain, create_app

# Same keys as config/robot.yaml from task S6 (origin/task/S6-robot-config), with
# a seated hand height that differs from the plan so tests can see it was read.
ROBOT_YAML = """\
arm:
  follower_port: /dev/tty.usbmodemXXXX
  leader_port: /dev/tty.usbmodemYYYY
  follower_id: so101_follower_arm
  leader_id: so101_leader_arm
  max_relative_target_deg: 10
  move_speed_deg_s: 45
cameras:
  wrist: 1
  scene: 2
  mast_top: 3
realsense:
  colour: [1280, 720]
  depth: [848, 480]
  fps: 30
  align_depth_to_colour: true
  flip: false
base:
  serial_port: /dev/tty.usbserial-XXXX
  baud: 115200
  wheel_diameter_mm: 72
  track_width_mm: 260
  max_speed_mm_s: 200
perception:
  detections_udp: 127.0.0.1:5601
  hand_udp: 127.0.0.1:5602
  parking_spot_floor_m: [0.22, 0.0]
  parking_tolerance_m: 0.02
lift:
  seated_hand_mm: 650
  standing_hand_mm: 1000
  taken_weight_drop_g: 50
items: [phone, glasses, pill_bottle, remote, pen, sock]
"""

STOP_LIMIT_S = 0.5      # plan 10.8 / 17


def offline_parse(text):
    return parse_request(text, use_api=False)


class RecordingBrain(FakeBrain):
    def __init__(self, lift=None):
        super().__init__(lift)
        self.calls = []

    def request(self, item, hand):
        self.calls.append(("request", item, hand))
        super().request(item, hand)

    def stop(self):
        self.calls.append(("stop",))
        super().stop()

    def thumbnail(self, name):
        return b"\xff\xd8fakejpeg" if name == "wrist" else None


def wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.005)


def in_thread(fn):
    """Run fn in a thread; returns (thread, result dict filled with 'value')."""
    result = {}
    t = threading.Thread(target=lambda: result.update(value=fn()), daemon=True)
    t.start()
    return t, result


def timed(fn):
    t0 = time.monotonic()
    value = fn()
    return value, time.monotonic() - t0


@pytest.fixture
def brain():
    return RecordingBrain()


@pytest.fixture
def client(brain):
    return TestClient(create_app(brain, parse=offline_parse))


@pytest.fixture
def robot_yaml(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text(ROBOT_YAML)
    return path


# --- routes ------------------------------------------------------------------

def test_page_has_stop_button(client):
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="stop"' in r.text and 'id="stopfail"' in r.text


def test_stop_route(client, brain):
    r = client.post("/stop")
    assert r.json()["stopped"] is True
    assert brain.calls == [("stop",)]
    assert client.get("/status").json()["state"] == "Stopped"


def test_request_fetches_with_u1_return_shape(client, brain):
    r = client.post("/request", json={"text": "I dropped my phone", "hand": "standing"})
    assert r.json() == {"parsed": {"action": "fetch", "item": "phone"}}     # U1: no "via" key
    assert brain.calls == [("request", "phone", "standing")]
    assert client.get("/status").json()["state"] == "Search"


def test_typed_stop_stops(client, brain):
    client.post("/request", json={"text": "I dropped my phone"})
    client.post("/request", json={"text": "STOP!"})
    assert brain.calls[-1] == ("stop",)


def test_unknown_item_does_nothing(client, brain):
    assert client.post("/request", json={"text": "find my keys"}).json()["parsed"]["action"] == "none"
    assert brain.calls == []


def test_bad_hand_rejected(client, brain):
    assert client.post("/request", json={"text": "my pen fell", "hand": "flying"}).status_code == 422
    assert brain.calls == []


def test_typed_stop_wins_over_a_bad_hand_and_a_broken_parser(brain):
    def broken_parse(text):
        raise RuntimeError("parser down")

    c = TestClient(create_app(brain, parse=broken_parse))
    for hand in ("flying", 5, None):
        r = c.post("/request", json={"text": "stop", "hand": hand})
        assert r.status_code == 200 and r.json()["parsed"] == {"action": "stop", "item": None}
    assert c.post("/stop").json()["stopped"] is True
    assert brain.calls == [("stop",)] * 4


def test_status_fields(client):
    s = client.get("/status").json()
    for key in STATUS_FIELDS:
        assert key in s


def test_thumbnails(client):
    r = client.get("/thumb/wrist")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/thumb/scene").status_code == 404


def test_fakebrain_follows_the_brain_protocol():
    assert isinstance(FakeBrain(), Brain)


def test_shutdown_stops_the_brain(brain):
    with TestClient(create_app(brain, parse=offline_parse)) as c:
        c.post("/request", json={"text": "my phone"})
    assert brain.calls[-1] == ("stop",)


# --- STOP beats a request that is still being parsed -------------------------

def test_stop_during_parse_drops_the_request(brain):
    entered, release = threading.Event(), threading.Event()

    def slow_parse(text):
        entered.set()
        release.wait(10)
        return offline_parse(text)

    with TestClient(create_app(brain, parse=slow_parse)) as c:
        t, result = in_thread(lambda: c.post("/request", json={"text": "I dropped my phone"}))
        assert entered.wait(5)
        stop, elapsed = timed(lambda: c.post("/stop"))
        release.set()
        t.join(10)
        assert stop.json()["stopped"] is True
        assert elapsed < STOP_LIMIT_S, f"STOP took {elapsed:.2f} s"
        assert result["value"].json() == {"parsed": {"action": "fetch", "item": "phone"}, "dropped": "stopped"}
        assert not any(call[0] == "request" for call in brain.calls)
        assert c.get("/status").json()["state"] == "Stopped"


def test_typed_stop_during_parse_drops_the_request(brain):
    entered, release = threading.Event(), threading.Event()

    def slow_parse(text):
        if not request_mod.is_stop(text):
            entered.set()
            release.wait(10)
        return offline_parse(text)

    with TestClient(create_app(brain, parse=slow_parse)) as c:
        t, result = in_thread(lambda: c.post("/request", json={"text": "my glasses fell"}))
        assert entered.wait(5)
        assert c.post("/request", json={"text": "stop"}).json()["parsed"]["action"] == "stop"
        release.set()
        t.join(10)
        assert result["value"].json()["dropped"] == "stopped"
        assert c.get("/status").json()["state"] == "Stopped"


def test_stop_racing_brain_request_still_wins():
    class SlowRequestBrain(RecordingBrain):
        def __init__(self):
            super().__init__()
            self.entered, self.release = threading.Event(), threading.Event()

        def request(self, item, hand):
            self.entered.set()
            self.release.wait(10)          # STOP arrives here, before the brain records the goal
            super().request(item, hand)

    brain = SlowRequestBrain()
    with TestClient(create_app(brain, parse=offline_parse)) as c:
        t, result = in_thread(lambda: c.post("/request", json={"text": "my pen fell"}))
        assert brain.entered.wait(5)
        c.post("/stop")
        brain.release.set()
        t.join(10)
        assert result["value"].json()["dropped"] == "stopped"
        assert brain.calls[-2:] == [("request", "pen", "seated"), ("stop",)]
        assert c.get("/status").json()["state"] == "Stopped"


def test_a_request_after_stop_still_runs(client, brain):
    client.post("/stop")
    r = client.post("/request", json={"text": "my remote fell"})
    assert "dropped" not in r.json()
    assert client.get("/status").json()["state"] == "Search"


# --- STOP is fast and needs no worker thread ---------------------------------

def test_stop_route_is_async():
    app = create_app(FakeBrain(), parse=offline_parse)
    endpoints = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}
    assert asyncio.iscoroutinefunction(endpoints["/stop"])
    assert asyncio.iscoroutinefunction(endpoints["/request"])     # its STOP check runs on the event loop


def test_stop_is_fast_while_a_request_is_slow(brain):
    """STOP within 0.5 s (plan 10.8) while the parser is stuck on a slow API call."""
    release = threading.Event()

    def slow_parse(text):
        release.wait(10)
        return offline_parse(text)

    with TestClient(create_app(brain, parse=slow_parse)) as c:
        t, _ = in_thread(lambda: c.post("/request", json={"text": "my phone"}))
        time.sleep(0.2)
        stop, elapsed = timed(lambda: c.post("/stop"))
        release.set()
        t.join(10)
    assert stop.json()["stopped"] is True
    assert elapsed < STOP_LIMIT_S, f"STOP took {elapsed:.2f} s"


def test_stop_is_fast_while_every_worker_thread_is_busy():
    """Measured from the client side, with every worker thread stuck in brain.status()."""
    class StuckStatusBrain(RecordingBrain):
        def __init__(self):
            super().__init__()
            self.release = threading.Event()
            self._n_lock = threading.Lock()
            self.inside = 0

        def status(self):
            with self._n_lock:
                self.inside += 1
            self.release.wait(10)
            return super().status()

    brain = StuckStatusBrain()
    with TestClient(create_app(brain, parse=offline_parse)) as c:
        workers = int(c.portal.call(lambda: anyio.to_thread.current_default_thread_limiter().total_tokens))
        threads = [in_thread(lambda: c.get("/status"))[0] for _ in range(workers + 5)]
        try:
            wait_until(lambda: brain.inside >= workers)          # the pool is full; 5 more are queued
            stop, elapsed = timed(lambda: c.post("/stop"))
            typed, typed_elapsed = timed(lambda: c.post("/request", json={"text": "halt"}))
        finally:
            brain.release.set()
            for t in threads:
                t.join(10)
    assert stop.json()["stopped"] is True
    assert elapsed < STOP_LIMIT_S, f"STOP took {elapsed:.2f} s with the worker pool full"
    assert typed.json()["parsed"]["action"] == "stop"
    assert typed_elapsed < STOP_LIMIT_S, f"typed stop took {typed_elapsed:.2f} s with the worker pool full"
    assert brain.calls[:2] == [("stop",), ("stop",)]


# --- FakeBrain -------------------------------------------------------------------

def test_fakebrain_reads_hand_heights_and_reports_pick_detail():
    b = FakeBrain(lift={"seated_hand_mm": 650, "standing_hand_mm": 1050, "taken_weight_drop_g": 50})
    b.request("phone", "standing")
    b._started -= 2 + 3 + 4 + 2 + 0.5           # into WaitForTake
    s = b.status()
    assert s["state"] == "WaitForTake" and s["lift_mm"] == 1050
    assert set(s["last_pick"]) == {"result", "grasp", "attempts", "seconds", "detail"}     # plan 9.4


# --- the page's script (run in node against a stub DOM, if node is installed) --

NODE = shutil.which("node")
HARNESS = Path(__file__).with_name("page_harness.js")


def test_page_uses_the_parsers_stop_words():
    assert json.dumps(request_mod.STOP_WORDS.pattern) in PAGE
    assert "__STOP_PATTERN__" not in PAGE


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("scenario", [
    "stop_server_error_shows_banner_and_retries",
    "stop_no_connection_shows_banner",
    "stop_with_no_answer_fails_after_2_s",
    "ask_server_error_is_not_reported_as_no_connection",
    "ask_is_blocked_while_a_request_is_pending",
    "typed_stop_goes_to_stop_even_while_a_request_is_pending",
    "dropped_request_is_reported",
    "only_one_status_request_at_a_time",
    "hung_status_times_out_and_shows_no_connection",
])
def test_page_script(tmp_path, scenario):
    script = tmp_path / "page.js"
    script.write_text(PAGE.split("<script>", 1)[1].split("</script>", 1)[0])
    r = subprocess.run([NODE, str(HARNESS), str(script), scenario], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr + r.stdout


# --- main(): safe defaults ---------------------------------------------------

@pytest.fixture
def no_uvicorn(monkeypatch):
    ran = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: ran.update(app=app, host=host, port=port))
    return ran


@pytest.fixture
def no_input(monkeypatch):
    def refuse(prompt=""):
        raise AssertionError(f"unexpected prompt: {prompt}")
    monkeypatch.setattr(builtins, "input", refuse)


@pytest.fixture
def fake_mission(monkeypatch):
    built = []

    def build(config, sim):
        b = RecordingBrain()
        built.append((config, sim, b))
        return b

    module = types.ModuleType("brain.mission")
    module.build = build
    monkeypatch.setitem(sys.modules, "brain.mission", module)
    return built


def test_main_defaults_to_fakebrain_on_localhost(robot_yaml, no_uvicorn, no_input, monkeypatch, capsys):
    made = []

    class SpyBrain(RecordingBrain):
        def __init__(self, lift=None):
            super().__init__(lift)
            made.append(self)

    monkeypatch.setattr(server, "FakeBrain", SpyBrain)
    monkeypatch.setitem(sys.modules, "brain.mission", None)    # the real brain must not even be imported
    assert server.main(["--config", str(robot_yaml)]) == 0
    assert no_uvicorn["host"] == "127.0.0.1" and no_uvicorn["port"] == 8000
    assert made[0]._hand_mm["seated"] == 650                    # read from robot.yaml
    assert made[0].calls == [("stop",)]                         # stopped when the server exits
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8000" in out and "WARNING" not in out


def test_main_lan_host_is_explicit_and_warns(robot_yaml, no_uvicorn, capsys):
    assert server.main(["--config", str(robot_yaml), "--host", "0.0.0.0"]) == 0
    assert no_uvicorn["host"] == "0.0.0.0"
    out = capsys.readouterr().out
    assert "no password" in out and ".local:8000" in out


def test_main_real_without_m1_says_so(robot_yaml, no_uvicorn, no_input, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "brain.mission", None)
    assert server.main(["--real", "--config", str(robot_yaml)]) == 2
    assert "task M1" in capsys.readouterr().err
    assert no_uvicorn == {}


def test_main_shows_a_real_import_error_inside_m1(tmp_path, robot_yaml, no_uvicorn, no_input, monkeypatch):
    pkg = tmp_path / "brain"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mission.py").write_text("import serial_driver_not_installed_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("brain", "brain.mission"):
        monkeypatch.setitem(sys.modules, name, None)   # records the original; restored after the test
        monkeypatch.delitem(sys.modules, name)
    with pytest.raises(ModuleNotFoundError, match="serial_driver_not_installed_xyz"):
        server.main(["--sim", "--config", str(robot_yaml)])
    assert no_uvicorn == {}


def test_main_real_needs_robot_yaml(tmp_path, fake_mission, no_uvicorn, no_input, capsys):
    assert server.main(["--real", "--config", str(tmp_path / "missing.yaml")]) == 2
    assert "robot.yaml" in capsys.readouterr().err
    assert fake_mission == [] and no_uvicorn == {}


def test_main_real_asks_before_building(robot_yaml, fake_mission, no_uvicorn, monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt="": "no")
    assert server.main(["--real", "--config", str(robot_yaml)]) == 1
    assert fake_mission == [] and no_uvicorn == {}

    monkeypatch.setattr(builtins, "input", lambda prompt="": "yes")
    assert server.main(["--real", "--config", str(robot_yaml)]) == 0
    (config, sim, brain), = fake_mission
    assert sim is False and config["lift"]["seated_hand_mm"] == 650
    assert no_uvicorn["host"] == "127.0.0.1"
    assert brain.calls[-1] == ("stop",)


def test_main_sim_builds_the_simulated_brain_without_asking(robot_yaml, fake_mission, no_uvicorn, no_input):
    assert server.main(["--sim", "--config", str(robot_yaml)]) == 0
    assert [sim for _, sim, _ in fake_mission] == [True]


def test_main_modes_are_exclusive_and_help_works(capsys):
    with pytest.raises(SystemExit) as e:
        server.main(["--real", "--fake"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        server.main(["--help"])
    assert e.value.code == 0
    assert "--real" in capsys.readouterr().out
