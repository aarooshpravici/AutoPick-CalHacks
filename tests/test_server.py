"""Status page tests (task U2, plan 9.7 / 10.8). No hardware, no network."""

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from ui.request import parse_request
from ui.server import FakeBrain, create_app


def offline_parse(text):
    return parse_request(text, use_api=False)


class RecordingBrain(FakeBrain):
    def __init__(self):
        super().__init__()
        self.calls = []

    def request(self, item, hand):
        self.calls.append(("request", item, hand))
        super().request(item, hand)

    def stop(self):
        self.calls.append(("stop",))
        super().stop()

    def thumbnail(self, name):
        return b"\xff\xd8fakejpeg" if name == "wrist" else None


@pytest.fixture
def brain():
    return RecordingBrain()


@pytest.fixture
def client(brain):
    return TestClient(create_app(brain, parse=offline_parse))


def test_page_has_stop_button(client):
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="stop"' in r.text


def test_stop_route(client, brain):
    r = client.post("/stop")
    assert r.json()["stopped"] is True
    assert brain.calls == [("stop",)]
    assert client.get("/status").json()["state"] == "Stopped"


def test_request_fetches(client, brain):
    r = client.post("/request", json={"text": "I dropped my phone", "hand": "standing"})
    assert r.json()["parsed"]["item"] == "phone"
    assert brain.calls == [("request", "phone", "standing")]
    assert client.get("/status").json()["state"] == "Search"


def test_typed_stop_stops(client, brain):
    client.post("/request", json={"text": "I dropped my phone"})
    client.post("/request", json={"text": "STOP!"})
    assert brain.calls[-1] == ("stop",)


def test_unknown_item_does_nothing(client, brain):
    assert client.post("/request", json={"text": "find my keys"}).json()["parsed"]["action"] == "none"
    assert brain.calls == []


def test_bad_hand_rejected(client):
    assert client.post("/request", json={"text": "my pen fell", "hand": "flying"}).status_code == 422


def test_status_fields(client):
    s = client.get("/status").json()
    for key in ("state", "item", "message", "detection", "lift_mm", "tray_g", "last_pick", "cameras"):
        assert key in s


def test_thumbnails(client):
    r = client.get("/thumb/wrist")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/thumb/scene").status_code == 404


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_stop_is_fast_while_a_request_is_slow(brain):
    """STOP within 0.5 s (plan 10.8) even while the parser is stuck on a slow API call."""
    def slow_parse(text):
        time.sleep(3)
        return offline_parse(text)

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(brain, parse=slow_parse), host="127.0.0.1",
                                           port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(base + "/status", timeout=0.2)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    try:
        slow = threading.Thread(target=lambda: httpx.post(base + "/request", json={"text": "my phone"}, timeout=10))
        slow.start()
        time.sleep(0.3)                                  # the slow request is now in progress
        t0 = time.monotonic()
        r = httpx.post(base + "/stop", timeout=2)
        elapsed = time.monotonic() - t0
        assert r.json()["stopped"] is True
        assert elapsed < 0.5, f"STOP took {elapsed:.2f} s"
        slow.join()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
