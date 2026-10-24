"""Request parser tests (task U1, plan 10.8). Offline except the live test, which
runs only when ANTHROPIC_API_KEY is set."""

import json
import os
import types

import anthropic
import httpx2 as httpx
import pytest

from ui import request
from ui.request import DEFAULT_ITEMS, parse_request

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

STOPS = ["stop", "stop!", "STOP STOP", "halt", "freeze!", "please stop now", "Stoppp", "no no stop it"]


class DeadClient:
    """Every call fails as if the network is off."""

    def __init__(self):
        self.calls = 0
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


class CannedClient:
    """Answers like the API would with structured outputs."""

    def __init__(self, answer, stop_reason="end_turn"):
        self.answer = answer
        self.stop_reason = stop_reason
        self.kwargs = None
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.kwargs = kwargs
        block = types.SimpleNamespace(type="text", text=json.dumps(self.answer))
        return types.SimpleNamespace(stop_reason=self.stop_reason, content=[block])


@pytest.mark.parametrize("text, item", PHRASES)
def test_keyword_fallback_maps_phrases(text, item):
    assert parse_request(text, DEFAULT_ITEMS, use_api=False) == {"action": "fetch", "item": item, "via": "keyword"}


@pytest.mark.parametrize("text", STOPS)
def test_stop_wins_with_network_off(text):
    client = DeadClient()
    assert parse_request(text, DEFAULT_ITEMS, client=client)["action"] == "stop"
    assert client.calls == 0                    # decided before any network call


def test_stop_wins_even_with_an_item():
    assert parse_request("stop, forget the phone", DEFAULT_ITEMS, client=DeadClient())["action"] == "stop"


def test_words_containing_stop_letters_are_not_stop():
    assert parse_request("my phone is on the doorstep", DEFAULT_ITEMS, use_api=False)["action"] == "fetch"


def test_network_off_falls_back_to_keywords():
    client = DeadClient()
    r = parse_request("my glasses fell", DEFAULT_ITEMS, client=client)
    assert r == {"action": "fetch", "item": "glasses", "via": "keyword"}
    assert client.calls == 1


def test_unknown_object_is_none():
    assert parse_request("I dropped my keys", DEFAULT_ITEMS, use_api=False) == {"action": "none", "item": None, "via": "keyword"}


def test_claude_path_and_request_shape():
    client = CannedClient({"action": "fetch", "item": "phone"})
    assert parse_request("my mobile fell", DEFAULT_ITEMS, client=client) == {"action": "fetch", "item": "phone", "via": "claude"}
    kw = client.kwargs
    assert kw["model"] == "claude-opus-5-5"
    schema = kw["output_config"]["format"]["schema"]
    assert schema["properties"]["item"]["enum"] == DEFAULT_ITEMS + ["none"]
    assert schema["additionalProperties"] is False


def test_claude_none_and_stop():
    assert parse_request("what time is it", DEFAULT_ITEMS, client=CannedClient({"action": "none", "item": "none"}))["action"] == "none"
    assert parse_request("wait, cancel that", DEFAULT_ITEMS, client=CannedClient({"action": "stop", "item": "none"}))["action"] == "stop"


def test_refusal_falls_back_to_keywords():
    client = CannedClient({"action": "none", "item": "none"}, stop_reason="refusal")
    assert parse_request("my pen fell", DEFAULT_ITEMS, client=client)["via"] == "keyword"


def test_items_come_from_robot_yaml(tmp_path):
    cfg = tmp_path / "robot.yaml"
    cfg.write_text("items: [phone, keys]\n")
    assert request.load_items(cfg) == ["phone", "keys"]
    assert request.load_items(tmp_path / "missing.yaml") == DEFAULT_ITEMS


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_live_claude_maps_phrases():
    wrong = []
    for text, item in PHRASES:
        r = parse_request(text, DEFAULT_ITEMS)
        if r["via"] != "claude" or r["item"] != item:
            wrong.append((text, r))
    assert not wrong, wrong


def test_no_api_key_falls_back(monkeypatch):
    """The real SDK client with no credentials at all."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(request, "_client", None)
    monkeypatch.setenv("HOME", "/nonexistent")       # no saved profile either
    r = parse_request("can you grab my specs", DEFAULT_ITEMS)
    assert r == {"action": "fetch", "item": "glasses", "via": "keyword"}
