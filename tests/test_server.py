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
ITEMS = ["phone", "glasses", "pill_bottle", "remote", "pen", "sock"]    # as in ROBOT_YAML above


def offline_parse(text):
    # The items above, not config/robot.yaml's (that list is only [phone] since A7).
    return parse_request(text, ITEMS, use_api=False)


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


# --- clarifying questions ----------------------------------------------------

def offline_clarify(text, dialog=(), hand=None):
    return request_mod.clarify_request(text, dialog, hand, ITEMS, use_api=False)


@pytest.fixture
def cclient(brain):
    return TestClient(create_app(brain, parse=offline_parse, clarify=offline_clarify))


def test_clarify_asks_without_calling_the_brain(cclient, brain):
    r = cclient.post("/request", json={"text": "my phone fell", "hand": "", "clarify": True})
    assert r.status_code == 200
    parsed = r.json()["parsed"]
    assert parsed["action"] == "ask" and parsed["question"] == request_mod.HAND_QUESTION
    assert set(parsed) == {"action", "item", "hand", "question", "options"}
    assert brain.calls == []


def test_clarify_fetches_after_an_answer_with_the_resolved_hand(cclient, brain):
    r = cclient.post("/request", json={"text": "my phone fell", "hand": "", "clarify": True,
                                       "dialog": [{"question": request_mod.HAND_QUESTION, "answer": "Standing"}]})
    assert r.json()["parsed"]["action"] == "fetch"
    assert brain.calls == [("request", "phone", "standing")]
    cclient.post("/request", json={"text": "my pen fell", "hand": "seated", "clarify": True})
    assert brain.calls[-1] == ("request", "pen", "seated")


def test_more_than_four_questions_is_rejected(cclient, brain):
    dialog = [{"question": "q", "answer": "a"}] * 5
    assert cclient.post("/request", json={"text": "my pen", "hand": "", "clarify": True, "dialog": dialog}).status_code == 422
    assert brain.calls == []


def test_four_questions_are_allowed_and_never_a_fifth(cclient, brain):
    dialog = [{"question": "q", "answer": "hm"}] * 4
    r = cclient.post("/request", json={"text": "my thing", "hand": "", "clarify": True, "dialog": dialog})
    assert r.json()["parsed"]["action"] == "none"
    assert brain.calls == []


def test_stop_word_in_an_answer_stops_first(brain):
    def broken(*a):
        raise RuntimeError("parser down")

    c = TestClient(create_app(brain, parse=broken, clarify=broken))
    dialog = [{"question": "q", "answer": "a"}] * 5 + [{"question": "q", "answer": "stop!"}]
    r = c.post("/request", json={"text": "my pen", "hand": "flying", "clarify": True, "dialog": dialog})
    assert r.status_code == 200 and r.json()["parsed"]["action"] == "stop"
    assert brain.calls == [("stop",)]


def test_empty_hand_needs_clarify(cclient, brain):
    for hand in ("", None):
        assert cclient.post("/request", json={"text": "my pen fell", "hand": hand, "clarify": True}).status_code == 200
        assert cclient.post("/request", json={"text": "my pen fell", "hand": hand}).status_code == 422
    assert [c[0] for c in brain.calls] == []        # the clarify ones only asked (hand unknown)


def test_clarify_stop_during_parse_drops_the_request(brain):
    started, release = threading.Event(), threading.Event()

    def slow_clarify(text, dialog, hand):
        started.set()
        release.wait(5)
        return request_mod.clarify_request(text, dialog, hand, ITEMS, use_api=False)

    with TestClient(create_app(brain, clarify=slow_clarify)) as c:
        t, result = in_thread(lambda: c.post("/request", json={"text": "my pen", "hand": "seated", "clarify": True}))
        assert started.wait(5)
        c.post("/stop")
        release.set()
        t.join(5)
    assert result["value"].json()["dropped"] == "stopped"
    assert all(call == ("stop",) for call in brain.calls)       # no request() (the last stop is shutdown's)


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


@pytest.mark.parametrize("direct", [False, True])
def test_stop_racing_brain_request_still_wins(direct):
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
        endpoint = "/fetch" if direct else "/request"
        payload = {"item": "phone"} if direct else {"text": "my pen fell"}
        t, result = in_thread(lambda: c.post(endpoint, json=payload))
        assert brain.entered.wait(5)
        c.post("/stop")
        brain.release.set()
        t.join(10)
        assert result["value"].json()["dropped"] == "stopped"
        assert brain.calls[-2:] == [("request", "phone" if direct else "pen", "standing"), ("stop",)]
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

HANDOFF = [("Search", "Looking for: phone"), ("Approach", "Driving to the phone"),
           ("Pick", "Picking it up"), ("Present", "Holding it up for you"),
           ("WaitForTake", "Take your phone from the claw")]


def fake_at(item="phone", seconds=0.0, hand="seated"):
    """A FakeBrain `seconds` into a request."""
    b = FakeBrain()
    b.request(item, hand)
    b._started -= seconds
    return b


def test_fakebrain_walks_the_handoff_states():
    assert [state for state, _ in FakeBrain.STEPS] == [state for state, _ in HANDOFF]    # no Raise, no Lower
    b = FakeBrain()
    assert b.status()["state"] == "Idle"
    b.request("phone", "seated")
    spent = 0.0
    for (state, message), (_, seconds) in zip(HANDOFF, FakeBrain.STEPS):
        b._started = time.monotonic() - (spent + 0.5)            # half a second into this state
        s = b.status()
        assert (s["state"], s["message"]) == (state, message)
        spent += seconds
    b._started = time.monotonic() - (spent + 0.5)
    s = b.status()
    assert (s["state"], s["message"]) == ("Idle", "Done")
    assert b.status()["state"] == "Idle"                         # and stays there


def test_fakebrain_messages_use_plain_item_names():
    s = fake_at("pill_bottle", 2 + 3 + 4 + 2 + 0.5).status()
    assert s["state"] == "WaitForTake" and s["message"] == "Take your pill bottle from the claw"
    assert fake_at("pill_bottle").status()["message"] == "Looking for: pill bottle"


def test_fakebrain_has_no_lift_or_tray():
    b = FakeBrain(lift={"seated_hand_mm": 650, "standing_hand_mm": 1050})    # a robot.yaml lift section is ignored
    b.request("phone", "standing")
    spent = 0.0
    for state, seconds in FakeBrain.STEPS + [("Idle", 1.0)]:
        b._started = time.monotonic() - (spent + 0.5)
        s = b.status()
        assert s["state"] == state and s["lift_mm"] is None and s["tray_g"] is None
        spent += seconds


def test_status_json_has_null_lift_and_tray(client):
    client.post("/request", json={"text": "I dropped my phone"})
    raw = client.get("/status").text
    s = json.loads(raw)
    assert s["lift_mm"] is None and s["tray_g"] is None
    assert '"lift_mm":null' in raw.replace(" ", "") and '"tray_g":null' in raw.replace(" ", "")


def test_fakebrain_last_pick_is_a_pick_to_hand_result_once_taken():
    b = fake_at("phone", 2 + 3 + 4 + 2 + 0.5)                    # in WaitForTake: not taken yet
    assert b.status()["last_pick"] is None
    b._started -= 4
    s = b.status()
    assert s["state"] == "Idle"
    assert set(s["last_pick"]) == {"result", "grasp", "attempts", "seconds", "detail"}     # plan 9.4
    assert s["last_pick"]["result"] == "taken" and s["last_pick"]["grasp"] == "scripted"
    b.request("pen", "seated")
    assert b.status()["last_pick"] is None                       # a new request clears it


def test_fakebrain_stop_still_stops_mid_handoff():
    b = fake_at("phone", 2 + 3 + 4 + 0.5, hand="standing")       # Present
    assert b.status()["state"] == "Present"
    b.stop()
    b.stop()                                                     # idempotent
    s = b.status()
    assert (s["state"], s["message"], s["last_pick"]) == ("Stopped", "Stopped", None)
    b._started = None
    assert b.status()["state"] == "Stopped"                      # time does not move it on
    b.request("phone", "seated")
    assert b.status()["state"] == "Search"                       # until the next request


def test_page_shows_no_tray_flow():
    for old in ("Raise", "Lower"):
        assert old not in PAGE and old not in [state for state, _ in FakeBrain.STEPS]
    assert 'id="liftrow" hidden' in PAGE and 'id="trayrow" hidden' in PAGE     # hidden until there is a value
    assert 'id="stop"' in PAGE and 'id="stopfail"' in PAGE


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
    "phone_button_sends_structured_fetch_without_parser",
    "typed_stop_goes_to_stop_even_while_a_request_is_pending",
    "dropped_request_is_reported",
    "only_one_status_request_at_a_time",
    "hung_status_times_out_and_shows_no_connection",
    "lift_and_tray_rows_are_hidden_when_null",
    "lift_and_tray_rows_show_when_there_are_numbers",
    "wait_for_take_is_prominent",
    "handoff_steps_follow_the_state",
    "question_is_shown_with_option_buttons",
    "tapping_an_option_reposts_with_the_dialog",
    "typing_an_answer_posts_it_as_the_answer_to_the_question",
    "stop_clears_the_conversation",
    "cancel_clears_the_conversation_without_stopping",
    "clarify_setting_off_sends_no_dialog_and_hand_standing",
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
    assert len(made) == 1
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


def test_object_dropdown_contains_selectable_phone():
    from html.parser import HTMLParser
    from ui.server import PAGE

    class Options(HTMLParser):
        in_objects = False
        values = []
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'select':
                self.in_objects = attrs.get('id') == 'object'
            if tag == 'option' and self.in_objects:
                self.values.append(attrs.get('value'))
        def handle_endtag(self, tag):
            if tag == 'select':
                self.in_objects = False

    parser = Options()
    parser.feed(PAGE)
    assert parser.values == ['', 'phone']
