"""Request parser tests (task U1, plan 9.6 and 10.8).

Offline: the Claude API is faked, or the real SDK is given a mock transport. The
one live test also needs RUN_LIVE_API_TESTS=1 and ANTHROPIC_API_KEY, because it
makes about 20 real, paid API calls."""

import json
import logging
import os
import sys
import types

import anthropic
import httpx2 as httpx
import pytest

from ui import request
from ui.request import parse_request

ITEMS = ["phone", "glasses", "pill_bottle", "remote", "pen", "sock"]

# Same keys as config/robot.yaml from task S6 (origin/task/S6-robot-config).
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
  seated_hand_mm: 700
  standing_hand_mm: 1000
  taken_weight_drop_g: 50
items: [phone, glasses, pill_bottle, remote, pen, sock]
"""

PHRASES = [
    ("I dropped my phone", "phone"),
    ("can you get my phone", "phone"),
    ("my glasses fell", "glasses"),
    ("I can't find my reading glasses, they're on the floor", "glasses"),
    ("could you pick up my specs please", "glasses"),
    ("my pills rolled under the table", "pill_bottle"),
    ("I dropped my medicine bottle", "pill_bottle"),
    ("grab my meds off the floor", "pill_bottle"),
    ("the remote fell on the floor", "remote"),
    ("pass me the TV remote", "remote"),
    ("I dropped the clicker", "remote"),
    ("my pen fell", "pen"),
    ("can you pick up that pen for me", "pen"),
    ("I dropped a sock", "sock"),
    ("pick up my socks", "sock"),
    ("my mobile slipped off the couch", "phone"),
    ("please bring me my iPhone", "phone"),
    ("Phone! Down there!", "phone"),
    ("oops, there goes my cell", "phone"),
    ("my eyeglasses are on the ground", "glasses"),
]

STOPS = [
    "stop", "stop!", "STOP STOP", "halt", "freeze!", "please stop now", "Stoppp", "no no stop it",
    "stopping", "it hasn't stopped", "halting now", "halt!", "freezing", "stoooop", "STOP, forget the phone",
    "stooop",
]

# Near misses, decided: "stop" inside a word is not a stop ("doorstep", "nonstop",
# "unstoppable"); a word that starts with a stop stem is, since a false stop is cheap.
NOT_STOPS = [
    ("my phone is on the doorstep", "phone"),
    ("my phone keeps ringing nonstop", "phone"),
    ("this pen is unstoppable", "pen"),
    ("I can't stoop down, can you get my phone?", "phone"),
    ("stooping hurts my back, my glasses fell", "glasses"),
]
STEM_STOPS = ["the stoplight toy fell", "my pills are by the freezer"]


def _message(answer, stop_reason="end_turn"):
    """A Messages API response body, as the real API returns it with structured outputs."""
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": json.dumps(answer)}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 50, "output_tokens": 10},
    }


class DeadClient:
    """Every call fails as if the network is off. No `beta` attribute, so a beta call breaks the test."""

    def __init__(self):
        self.calls = 0
        self.messages = types.SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls += 1
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


class CannedClient:
    """Answers like the GA API would with structured outputs."""

    def __init__(self, answer, stop_reason="end_turn"):
        self.answer = answer
        self.stop_reason = stop_reason
        self.kwargs = None
        self.messages = types.SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.kwargs = kwargs
        block = types.SimpleNamespace(type="text", text=json.dumps(self.answer))
        return types.SimpleNamespace(stop_reason=self.stop_reason, content=[block])


class RaisingClient:
    def __init__(self, exc):
        self.messages = types.SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(exc))


def _no_network_sdk_client(seen, answer=None, **kwargs):
    """The real anthropic 2.3.0 client, with a mock transport in place of the network."""

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json=_message(answer or {"action": "fetch", "item": "phone"}))

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return anthropic.Anthropic(http_client=http, max_retries=0, **kwargs)


# --- return shape (plan 9.6) ---

def test_plan_examples_exactly():
    assert parse_request("I dropped my phone", ITEMS, use_api=False) == {"action": "fetch", "item": "phone"}
    assert parse_request("stop!", ITEMS, client=DeadClient()) == {"action": "stop", "item": None}
    assert parse_request("I dropped my phone", ITEMS, client=CannedClient({"action": "fetch", "item": "phone"})) == {
        "action": "fetch", "item": "phone"}


@pytest.mark.parametrize("text, client", [
    ("stop", DeadClient()),
    ("my pen fell", DeadClient()),
    ("I dropped my keys", DeadClient()),
    ("my mobile fell", CannedClient({"action": "fetch", "item": "phone"})),
    ("wait, cancel that", CannedClient({"action": "stop", "item": "none"})),
    ("what time is it", CannedClient({"action": "none", "item": "none"})),
    ("my pen fell", CannedClient({"action": "none", "item": "none"}, stop_reason="refusal")),
])
def test_result_has_only_action_and_item(text, client):
    assert set(parse_request(text, ITEMS, client=client)) == {"action", "item"}


def test_path_is_logged_not_returned(caplog):
    with caplog.at_level(logging.INFO, logger="ui.request"):
        assert parse_request("my mobile fell", ITEMS, client=CannedClient({"action": "fetch", "item": "phone"})) == {
            "action": "fetch", "item": "phone"}
    assert "fetch phone (via claude)" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ui.request"):
        parse_request("my glasses fell", ITEMS, client=DeadClient())
    assert "fetch glasses (via keyword)" in caplog.text


# --- keyword fallback and STOP ---

@pytest.mark.parametrize("text, item", PHRASES)
def test_keyword_fallback_maps_phrases(text, item):
    assert parse_request(text, ITEMS, use_api=False) == {"action": "fetch", "item": item}


@pytest.mark.parametrize("text", STOPS)
def test_stop_wins_with_network_off(text):
    client = DeadClient()
    assert parse_request(text, ITEMS, client=client) == {"action": "stop", "item": None}
    assert client.calls == 0                    # decided before any network call


@pytest.mark.parametrize("text, item", NOT_STOPS)
def test_stop_inside_a_word_is_not_stop(text, item):
    assert parse_request(text, ITEMS, use_api=False) == {"action": "fetch", "item": item}


@pytest.mark.parametrize("text", STEM_STOPS)
def test_word_starting_with_a_stop_stem_is_stop(text):
    assert parse_request(text, ITEMS, client=DeadClient())["action"] == "stop"


def test_stop_needs_no_config_or_sdk(monkeypatch):
    monkeypatch.setattr(request, "load_items", lambda *a: pytest.fail("config read before the STOP check"))
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert parse_request("halt") == {"action": "stop", "item": None}


def test_network_off_falls_back_to_keywords():
    client = DeadClient()
    assert parse_request("my glasses fell", ITEMS, client=client) == {"action": "fetch", "item": "glasses"}
    assert client.calls == 1


def test_unknown_object_is_none():
    assert parse_request("I dropped my keys", ITEMS, use_api=False) == {"action": "none", "item": None}


# --- Claude path ---

def test_claude_path_and_request_shape():
    client = CannedClient({"action": "fetch", "item": "phone"})
    assert parse_request("my mobile fell", ITEMS, client=client) == {"action": "fetch", "item": "phone"}
    kw = client.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert "betas" not in kw and "fallbacks" not in kw
    assert kw["output_config"]["format"]["type"] == "json_schema"
    schema = kw["output_config"]["format"]["schema"]
    assert schema["properties"]["item"]["enum"] == ITEMS + ["none"]
    assert schema["additionalProperties"] is False


def test_ga_call_shape_through_the_real_sdk():
    """The installed anthropic 2.3.0 accepts the call and sends a GA request: no beta, no fallbacks."""
    seen = []
    client = _no_network_sdk_client(seen, api_key="test-not-a-real-key")
    assert request.ask_claude("my mobile fell", ITEMS, client) == {"action": "fetch", "item": "phone"}
    (req,) = seen
    assert req.method == "POST" and req.url.path == "/v1/messages"
    assert "beta" not in req.url.params
    assert "anthropic-beta" not in req.headers
    body = json.loads(req.content)
    assert set(body) == {"model", "max_tokens", "system", "output_config", "messages"}
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["format"]["schema"]["properties"]["item"]["enum"] == ITEMS + ["none"]


def test_claude_none_and_stop():
    assert parse_request("what time is it", ITEMS, client=CannedClient({"action": "none", "item": "none"})) == {
        "action": "none", "item": None}
    assert parse_request("wait, cancel that", ITEMS, client=CannedClient({"action": "stop", "item": "none"})) == {
        "action": "stop", "item": None}


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_unfinished_answer_falls_back_to_keywords(stop_reason, caplog):
    client = CannedClient({"action": "none", "item": "none"}, stop_reason=stop_reason)
    with caplog.at_level(logging.INFO, logger="ui.request"):
        assert parse_request("my pen fell", ITEMS, client=client) == {"action": "fetch", "item": "pen"}
    assert stop_reason in caplog.text and "(via keyword)" in caplog.text


def test_client_settings(monkeypatch):
    monkeypatch.setattr(request, "_client", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    client = request._get_client()                  # builds the client; sends nothing
    assert client.timeout == request.TIMEOUT_S <= 5
    assert client.max_retries == 0


# --- failures that must still answer from keywords ---

def test_no_api_key_falls_back(monkeypatch, caplog):
    """The real SDK client with no credentials at all: fails before sending anything."""
    for var in [v for v in os.environ if v.startswith("ANTHROPIC_")]:
        monkeypatch.delenv(var)
    monkeypatch.setenv("HOME", "/nonexistent")       # no saved profile either
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    seen = []
    client = _no_network_sdk_client(seen)
    with caplog.at_level(logging.WARNING, logger="ui.request"):
        assert parse_request("can you grab my specs", ITEMS, client=client) == {"action": "fetch", "item": "glasses"}
    assert seen == []
    assert "no API credentials" in caplog.text


def test_other_type_error_is_not_reported_as_missing_key(caplog):
    client = RaisingClient(TypeError("create() got an unexpected keyword argument 'fallbacks'"))
    with caplog.at_level(logging.WARNING, logger="ui.request"):
        assert parse_request("my pen fell", ITEMS, client=client) == {"action": "fetch", "item": "pen"}
    assert "no API credentials" not in caplog.text
    (record,) = [r for r in caplog.records if "bad SDK call" in r.getMessage()]
    assert record.levelno == logging.ERROR


def test_missing_sdk_falls_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)     # import anthropic now raises ImportError
    assert parse_request("my pen fell", ITEMS) == {"action": "fetch", "item": "pen"}


def test_non_object_answer_falls_back():
    assert parse_request("my pen fell", ITEMS, client=CannedClient(["fetch", "phone"])) == {"action": "fetch", "item": "pen"}


# --- items from robot.yaml ---

def test_items_come_from_robot_yaml(tmp_path):
    cfg = tmp_path / "robot.yaml"
    cfg.write_text(ROBOT_YAML)
    assert request.load_items(cfg) == ITEMS
    cfg.write_text(ROBOT_YAML.replace("items: [phone, glasses, pill_bottle, remote, pen, sock]", "items: [phone, keys]"))
    assert request.load_items(cfg) == ["phone", "keys"]


def test_missing_robot_yaml_warns_and_uses_plan_items(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="ui.request"):
        assert request.load_items(tmp_path / "missing.yaml") == ITEMS
    assert "not found" in caplog.text


@pytest.mark.parametrize("items_line", ["", "items: []\n", "items: phone\n", "items: [phone, 3]\n"])
def test_robot_yaml_without_usable_items_is_an_error(tmp_path, items_line):
    cfg = tmp_path / "robot.yaml"
    cfg.write_text(ROBOT_YAML.replace("items: [phone, glasses, pill_bottle, remote, pen, sock]\n", items_line))
    with pytest.raises(ValueError, match="items"):
        request.load_items(cfg)


# --- live (opt-in) ---

@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_API_TESTS") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="live API test: set RUN_LIVE_API_TESTS=1 and ANTHROPIC_API_KEY",
)
def test_live_claude_maps_phrases():
    # ask_claude, not parse_request, so a keyword fallback can't hide an API failure.
    wrong = []
    for text, item in PHRASES:
        r = request.ask_claude(text, ITEMS)
        if r != {"action": "fetch", "item": item}:
            wrong.append((text, r))
    assert not wrong, wrong
