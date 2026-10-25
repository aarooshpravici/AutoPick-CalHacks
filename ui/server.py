"""Status page with STOP (task U2, plan 9.7). FastAPI on port 8000.

    python -m ui.server                    # FakeBrain, no hardware (the default)
    python -m ui.server --sim              # the brain (task M1) with simulated arm, base and camera
    python -m ui.server --real             # the brain on the robot: asks you to confirm first
    python -m ui.server --host 0.0.0.0     # let a phone on the same Wi-Fi open it (no password)

| Route           | Does                                                              |
|-----------------|-------------------------------------------------------------------|
| GET  /          | The page: text box, big STOP button, current state, thumbnails    |
| POST /request   | {"text": "...", "hand": "seated"|"standing"}: parsed, sent on     |
|                 | + "clarify": true, "dialog": [{"question", "answer"}], hand ""   |
|                 | allowed: may answer {"parsed": {"action": "ask", ...}} instead    |
| POST /stop      | Stop everything                                                   |
| GET  /status    | Brain state, last detection, last pick; lift and tray (null: none) |
| GET  /thumb/{n} | Latest JPEG from camera n, if the brain provides one              |

The brain is anything that follows the `Brain` protocol below.

STOP never waits on the parser, the network or the worker threads:
- /stop is an async route, so it runs on the event loop and calls brain.stop()
  at once even when every worker thread is busy with slow parses or status calls.
- "stop" typed in the box, or in any answer to a question, is caught by the
  keyword check before the hand is checked or the parser runs, also on the
  event loop.
- A request still being parsed when STOP arrives is dropped, never started.
- Shutting the server down calls brain.stop().
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from . import request as request_mod

HANDS = request_mod.HANDS
STATUS_FIELDS = ("state", "item", "message", "detection", "lift_mm", "tray_g", "last_pick", "cameras", "t")


@runtime_checkable
class Brain(Protocol):
    """What the status page needs from the brain (task M1). SW1 proposes this to SW2.

    The server calls these from several threads at once: request(), status() and
    thumbnail() from its worker threads, stop() from the event loop. Every method
    must be thread-safe. stop() can arrive at any moment, including while
    request() or status() is running.

    How M1 plugs in, either way works:
    - M1 imports `create_app(brain)` and runs uvicorn in the brain's own process, or
    - brain/mission.py provides `build(config: dict, sim: bool) -> Brain`, where
      config is the parsed config/robot.yaml and sim=True means simulated arm,
      fake serial port and replayed detections (plan 10.7). `python -m ui.server
      --sim` and `--real` call it. build() must not move anything.
    """

    def request(self, item: str, hand: str) -> None:
        """Start fetching `item` (one of robot.yaml `items`) for a person whose hand
        is at `hand` height: "seated" or "standing" (robot.yaml lift.seated_hand_mm
        or lift.standing_hand_mm).

        Return at once: record the goal and let the brain's own 10 Hz loop do the
        work. Replaces any mission in progress, and leaves Stopped. The server never
        calls request() for a request that arrived before the latest stop(), and if
        a stop() races with request() the server calls stop() again afterwards, so
        the brain does not need its own ordering check.
        """
        ...

    def stop(self) -> None:
        """Stop everything now: wheels stop (serial STOP), arm holds its pose, lift
        holds. State becomes "Stopped" until the next request().

        Must return at once (well under 50 ms): it runs on the web server's event
        loop, so it must not sleep, wait for a reply, join a thread, or take a lock
        that is held during serial, camera or arm I/O. Set a flag or send the stop
        commands and return. Must be idempotent: the server may call it twice.
        If it raises, the page shows STOP FAILED and tells people to hit the E-stop.
        """
        ...

    def status(self) -> dict:
        """A JSON-safe snapshot, cheap to make (the page asks twice a second):

            state      str    plan 10.7 state: "Idle", "Search", "Approach", "Pick", "Present",
                              "WaitForTake" or "Failed", or "Stopped"
            item       str | None    what it is fetching
            message    str    one line for people ("Looking for: phone")
            detection  dict | None   latest plan 9.2 item: {"label", "conf", "floor_m", ...}
            lift_mm    float | None  tray height above the bottom switch; None without the lift (9.7)
            tray_g     float | None  load-cell reading; None without the lift (9.7)
            last_pick  dict | None   the plan 9.4 pick_to_hand result dict
            cameras    list[str]     camera names that thumbnail() can serve
            t          float  time.time() of the snapshot
        """
        ...

    def thumbnail(self, name: str) -> bytes | None:
        """Latest JPEG from camera `name` (a robot.yaml `cameras` name), or None.

        Return a frame the brain already holds: never open a camera or a serial
        port, and never wait for a new frame. A brain without pictures returns None
        (the server also copes if the method is missing).
        """
        ...


class Turn(BaseModel):
    question: str
    answer: str


class RequestBody(BaseModel):
    text: str
    hand: Any = "seated"      # checked by hand after the STOP check, so "stop" always wins
    clarify: bool = False     # True: the robot may ask questions first; hand may then be "" or None
    dialog: list[Turn] = []   # questions already asked for this request and the answers, oldest first


class FetchBody(BaseModel):
    """A selected UI item, never arbitrary serial instructions or model output."""
    item: Literal["phone"]
    hand: Literal["seated", "standing"] = "seated"


class _StopCount:
    """How many STOPs so far. A request parsed across a STOP is dropped."""

    def __init__(self):
        self._lock = threading.Lock()      # held only for the increment, never during a brain call
        self._n = 0

    def bump(self) -> None:
        with self._lock:
            self._n += 1

    @property
    def value(self) -> int:
        with self._lock:
            return self._n


def create_app(brain: Brain, parse=request_mod.parse_request, clarify=request_mod.clarify_request) -> FastAPI:
    stops = _StopCount()

    def stop_now() -> None:
        stops.bump()                       # first, so in-flight requests are dropped even if stop() raises
        brain.stop()

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            brain.stop()                   # no page, no STOP button: the robot stops too

    app = FastAPI(title="Floor Picker", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def page():
        simulated = isinstance(brain, FakeBrain) or getattr(brain, "simulated", False)
        return PAGE.replace("__MODE__", "Simulation" if simulated else "Robot control").replace(
            "__MODE_NOTE__", "Simulation · no hardware moves" if simulated else "Robot control · stay within reach of the E-stop")

    # async: runs on the event loop, never queued behind the worker threads.
    @app.post("/stop")
    async def stop():
        t0 = time.monotonic()
        stop_now()
        return {"stopped": True, "brain_ms": round((time.monotonic() - t0) * 1000, 1)}

    async def dispatch_fetch(parsed, hand, seen):
        def begin():
            # Recheck after waiting for a thread: a queued request must not undo STOP.
            if stops.value != seen:
                return False
            brain.request(parsed["item"], hand)
            return True
        try:
            accepted = await run_in_threadpool(begin)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        if not accepted or stops.value != seen:
            if accepted:
                brain.stop()
            return {"parsed": parsed, "dropped": "stopped"}
        return {"parsed": parsed}

    @app.post("/fetch")
    async def fetch_selected(body: FetchBody):
        # A Phone click needs no natural-language model or external API.
        parsed = {"action": "fetch", "item": body.item, "hand": body.hand}
        return await dispatch_fetch(parsed, body.hand, stops.value)

    @app.post("/request")
    async def post_request(body: RequestBody):
        # stop wins, before the hand check, the dialog check or the parser
        if any(request_mod.is_stop(t) for t in (body.text, *(turn.answer for turn in body.dialog))):
            stop_now()
            return {"parsed": {"action": "stop", "item": None}}
        if len(body.dialog) > request_mod.MAX_QUESTIONS:
            raise HTTPException(422, f"at most {request_mod.MAX_QUESTIONS} questions per request")
        if body.hand not in HANDS and not (body.clarify and body.hand in ("", None)):
            raise HTTPException(422, f"hand must be one of {HANDS}")
        seen = stops.value
        if body.clarify:
            dialog = [turn.model_dump() for turn in body.dialog]
            parsed = await run_in_threadpool(clarify, body.text, dialog, body.hand or None)
            if parsed["action"] == "ask":              # nothing moves until the person has answered
                return {"parsed": parsed}
            hand = parsed["hand"]
        else:
            parsed = await run_in_threadpool(parse, body.text)    # may take seconds (Claude API)
            hand = body.hand
        if parsed["action"] == "stop":
            stop_now()
        elif parsed["action"] == "fetch":
            return await dispatch_fetch(parsed, hand, seen)
        return {"parsed": parsed}

    @app.get("/status")
    def status():
        return brain.status()

    @app.get("/thumb/{name}")
    def thumb(name: str):
        getter = getattr(brain, "thumbnail", None)
        jpeg = getter(name) if getter else None
        if not jpeg:
            raise HTTPException(404, "no picture")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return app


class FakeBrain:
    """Walks through the hand-off states on a timer, for trying the page without hardware.

    Idle, Search, Approach, Pick, Present, WaitForTake, then Idle again (plan 10.7).
    There is no tray lift, so lift_mm and tray_g are None (plan 9.7). `lift` is
    accepted and ignored, so a robot.yaml with or without a lift section works.
    Nothing here moves.
    """

    STEPS = [("Search", 2.0), ("Approach", 3.0), ("Pick", 4.0), ("Present", 2.0), ("WaitForTake", 4.0)]
    MESSAGES = {"Search": "Looking for: {item}", "Approach": "Driving to the {item}",
                "Pick": "Picking it up", "Present": "Holding it up for you",
                "WaitForTake": "Take your {item} from the claw"}

    def __init__(self, lift: dict | None = None):
        self._lock = threading.Lock()
        self._state = "Idle"
        self._item = None
        self._hand = "seated"
        self._started = None
        self._last_pick = None
        self._message = "Ready"

    def request(self, item: str, hand: str) -> None:
        with self._lock:
            self._state, self._item, self._hand = "Search", item, hand
            self._started = time.monotonic()
            self._last_pick = None
            self._message = self.MESSAGES["Search"].format(item=item.replace("_", " "))

    def stop(self) -> None:
        with self._lock:
            self._state, self._started = "Stopped", None
            self._message = "Stopped"

    def _advance(self) -> None:
        if self._started is None:
            return
        elapsed = time.monotonic() - self._started
        for state, seconds in self.STEPS:
            if elapsed < seconds:
                self._state = state
                self._message = self.MESSAGES[state].format(item=self._item.replace("_", " "))
                return
            elapsed -= seconds
        # Taken: the last_pick a real pick_to_hand returns (plan 9.4).
        self._last_pick = {"result": "taken", "grasp": "scripted", "attempts": 1,
                           "seconds": sum(sec for state, sec in self.STEPS if state in ("Pick", "Present", "WaitForTake")),
                           "detail": "FakeBrain: no arm"}
        self._state, self._started, self._message = "Idle", None, "Done"

    def status(self) -> dict:
        with self._lock:
            self._advance()
            seen = self._state in ("Search", "Approach") and self._item
            return {
                "state": self._state,
                "item": self._item,
                "message": self._message,
                "detection": ({"label": self._item, "conf": 0.9, "floor_m": [0.5, 0.0]} if seen else None),
                "lift_mm": None,
                "tray_g": None,
                "last_pick": self._last_pick,
                "cameras": [],
                "t": time.time(),
            }

    def thumbnail(self, name: str) -> bytes | None:
        return None


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Floor Picker</title>
<style>
:root { color-scheme:light; --bg:#f5f4ef; --card:#fffefa; --ink:#243a32; --muted:#69756d; --line:#dddeda;
        --go:#245641; --stop:#b83832; --stop-dark:#912923; }
* { box-sizing:border-box; }
html { -webkit-text-size-adjust:100%; text-size-adjust:100%; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
button,input,select { font:inherit; } button,select { touch-action:manipulation; } button { cursor:pointer; }
button:focus-visible,input:focus-visible,select:focus-visible { outline:3px solid #a96a30; outline-offset:4px; }
[hidden] { display:none!important; }
main { max-width:1240px; margin:auto; padding:0 48px 30px; }
.topbar { display:flex; align-items:center; justify-content:space-between; gap:20px; padding:28px 0; border-bottom:1px solid var(--line); }
.brand { display:flex; align-items:center; gap:11px; }
.mark { display:grid; place-items:center; width:36px; height:36px; border-radius:12px; background:var(--go); color:#f4f6dc; font-size:24px; }
h1 { font-size:21px; margin:0; font-weight:650; letter-spacing:-.7px; }
.brand-note { border-left:1px solid #cdd2ca; margin-left:8px; padding-left:18px; font-size:12px; color:var(--muted); }
.top-status { display:flex; gap:16px; align-items:center; font-size:12px; }
.mode { padding:6px 12px; border-radius:30px; border:1px solid #d3d8cb; background:#ecefe5; white-space:nowrap; }
#connection { display:flex; align-items:center; gap:7px; color:var(--muted); }
#connection::before { content:""; width:6px; height:6px; background:#75856a; border-radius:50%; }
.intro { padding:46px 0 32px; display:flex; justify-content:space-between; align-items:end; gap:24px; }
.eyebrow { text-transform:uppercase; letter-spacing:2px; font-size:10px; font-weight:700; color:var(--muted); }
h2 { font-family:Georgia,"Times New Roman",serif; font-weight:400; font-size:52px; line-height:1.12; letter-spacing:-2px; margin:12px 0 14px; }
h2 em { color:#64805c; font-weight:400; }
.intro p { color:var(--muted); font-size:14px; margin:0; }
.intro-note { text-align:right; max-width:185px; font-size:12px!important; line-height:1.7; }
.workspace { display:grid; grid-template-columns:minmax(0,1.45fr) minmax(300px,1fr); gap:22px; align-items:start; }
.card { background:var(--card); border:1px solid var(--line); border-radius:20px; padding:28px; }
.card-label { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:24px; }
.card-label h3 { margin:0; font-size:13px; font-weight:600; }
.card-number { color:#8b9487; font-size:11px; font-variant-numeric:tabular-nums; }
.request-title { font-family:Georgia,"Times New Roman",serif; font-size:29px; font-weight:400; letter-spacing:-.7px; margin:0 0 8px; }
.help { font-size:13px; color:var(--muted); margin:0 0 24px; }
form { display:grid; grid-template-columns:1fr auto; gap:12px; }
.input-label { grid-column:1/-1; color:var(--muted); font-size:12px; margin-bottom:-5px; }
#text { grid-column:1/-1; min-width:0; width:100%; border:1px solid #ccd3c7; border-radius:12px; background:#f8f9f3; color:var(--ink); padding:18px; font-size:18px; }
#text::placeholder { color:#7c897a; }
.select-wrap { display:flex; gap:12px; align-items:center; color:var(--muted); font-size:12px; }
select { appearance:none; -webkit-appearance:none; height:48px; background-image:linear-gradient(45deg,transparent 50%,#69756d 50%),linear-gradient(135deg,#69756d 50%,transparent 50%); background-position:calc(100% - 15px) 21px,calc(100% - 11px) 21px; background-size:4px 4px; background-repeat:no-repeat; min-height:48px; padding:10px 28px 10px 12px; border:1px solid var(--line); border-radius:10px; background-color:var(--card); color:var(--ink); max-width:100%; }
button.go { min-height:48px; padding:12px 22px; background:var(--go); border:1px solid var(--go); border-radius:10px; color:#fff; font-size:14px; font-weight:600; }
button.go:hover { background:#183f2e; }
button[aria-disabled="true"] { opacity:.5; cursor:wait; }
#setting { display:flex; align-items:center; gap:10px; margin-top:18px; font-size:12px; color:var(--muted); min-height:44px; }
#setting input { width:18px; height:18px; accent-color:var(--go); }
#question { border-top:1px solid var(--line); padding-top:20px; margin-top:14px; font-size:20px; }
#question small { display:block; margin-top:6px; font-size:12px; color:var(--muted); }
#opts { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
button.opt { padding:10px 18px; min-height:48px; border:1px solid #a3b49b; background:#edf2e6; color:var(--ink); border-radius:10px; }
#cancel { padding:10px 14px; min-height:44px; margin-top:10px; border:1px solid var(--line); border-radius:10px; background:var(--card); color:var(--muted); }
#reply { font-size:13px; color:var(--muted); min-height:20px; margin-top:10px; overflow-wrap:anywhere; }
.request-foot { border-top:1px solid var(--line); padding-top:18px; margin-top:16px; display:flex; gap:10px; align-items:center; color:var(--muted); font-size:11px; }
.request-foot svg { flex:none; }
.side { display:grid; gap:18px; }
#statecard { background:#e9eedf; border-color:#dce3d1; padding:26px; transition:background .2s; }
#state { font-family:Georgia,"Times New Roman",serif; font-size:36px; line-height:1.2; margin:6px 0 10px; letter-spacing:-1px; }
#message { color:#53654d; font-size:13px; min-height:20px; }
#steps { display:flex; list-style:none; gap:5px; padding:0; margin:30px 0 0; counter-reset:step; }
#steps li { flex:1; min-width:0; font-size:10px; color:#5f7056; text-align:center; position:relative; padding-top:34px; }
#steps li::before { counter-increment:step; content:counter(step); display:grid; place-items:center; width:24px; height:24px; border-radius:50%; background:#e9eedf; border:1px solid #aebba1; position:absolute; top:0; left:calc(50% - 12px); font-size:10px; z-index:1; }
#steps li:not(:last-child)::after { content:""; height:1px; background:#bdc9b0; position:absolute; top:12px; left:calc(50% + 12px); width:calc(100% - 19px); }
#steps li[data-step="now"] { color:var(--go); font-weight:700; }
#steps li[data-step="now"]::before { background:var(--go); border-color:var(--go); color:#fff; box-shadow:0 0 0 4px #d7e0cb; }
#steps li[data-step="done"]::before { content:"✓"; background:#c8d7bc; border-color:#a9bd9a; }
#statecard[data-state="WaitForTake"] { background:var(--go); color:#fff; }
#statecard[data-state="WaitForTake"] #message,#statecard[data-state="WaitForTake"] li,#statecard[data-state="WaitForTake"] .eyebrow,#statecard[data-state="WaitForTake"] h3 { color:#fff; }
#statecard[data-state="WaitForTake"] li::before { color:var(--ink); }
#statecard[data-state="WaitForTake"] li[data-step="now"]::before { background:#fff; }
.stop-panel { background:#fcf3ef; border:1px solid #e8d5cf; border-radius:18px; padding:19px; }
#stop { display:flex; align-items:center; justify-content:center; gap:12px; width:100%; min-height:76px; background:var(--stop); border:0; border-radius:11px; color:#fff; font-size:24px; font-weight:650; letter-spacing:2px; box-shadow:0 3px 0 var(--stop-dark); }
#stop:hover { background:#a52f29; } #stop:active { transform:translateY(2px); box-shadow:0 1px 0 var(--stop-dark); }
.stop-symbol { width:15px; height:15px; border:2px solid currentColor; border-radius:3px; }
.stop-caption { display:flex; align-items:center; justify-content:space-between; gap:8px; margin:13px 0 0; color:#805a52; font-size:11px; }
kbd { font:10px inherit; border:1px solid #d8beb6; padding:1px 5px; border-radius:4px; }
#stopfail { background:#a52f29; color:white; padding:18px; border-radius:12px; margin-bottom:20px; }
#stopfail b { display:block; font-size:20px; }
.telemetry { margin-top:22px; display:grid; grid-template-columns:1fr 1fr; padding:0; overflow:hidden; }
.detail-panel { padding:25px 28px; }
.detail-panel + .detail-panel { border-left:1px solid var(--line); }
.telemetry .card-label { margin-bottom:18px; }
dl { display:grid; grid-template-columns:90px 1fr; gap:10px; margin:0; font-size:13px; }
dt { color:var(--muted); } dd { margin:0; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }
.row { display:contents; }
#thumbs { display:flex; flex-wrap:wrap; gap:8px; } #thumbs img { width:calc(50% - 4px); border-radius:8px; }
#cameraempty { display:flex; align-items:center; gap:14px; padding:12px 0; color:var(--muted); font-size:12px; }
.camera-icon { display:grid; place-items:center; width:46px; height:46px; border:1px dashed #bfc9b7; border-radius:13px; flex:none; }
#cameraempty strong { display:block; font-weight:500; font-size:13px; color:var(--ink); margin-bottom:4px; }
footer { display:flex; justify-content:space-between; gap:16px; padding-top:24px; color:#7a8377; font-size:10px; letter-spacing:.3px; }
#connection[data-online="false"] { color:var(--stop); } #connection[data-online="false"]::before { background:var(--stop); }
.stale #state::after { content:" · Offline"; font:14px -apple-system,sans-serif; color:var(--stop); }
button.phone-choice { grid-column:1/-1; grid-row:1; display:flex; align-items:center; gap:18px; min-height:110px; padding:24px; background:#edf2e6; color:var(--go); border:1px solid #a3b49b; font-size:21px; text-align:left; border-radius:14px; }
button.phone-choice:hover { background:#dce7d1; border-color:var(--go); }
.choice-arrow { margin-left:auto; font-size:23px; }
#ask .select-wrap { grid-column:1/-1; margin-top:10px; }
@media (min-width:761px) { .side { position:sticky; top:20px; } }
@media (max-width:900px) { main { padding:0 28px 24px; } .brand-note { display:none; } h2 { font-size:46px; } .workspace { grid-template-columns:minmax(0,1.2fr) minmax(280px,1fr); } .card { padding:22px; } .select-wrap { gap:6px; } button.go { padding:12px 15px; } }
@media (max-width:760px) {
 main { padding:0 20px 150px; } .topbar { padding:22px 0; } .top-status { gap:8px; } #connection { font-size:0; gap:0; }
 h1 { font-size:19px; } .intro { padding:32px 0 26px; } h2 { font-size:43px; } .intro-note { display:none; }
 .workspace { display:flex; flex-direction:column; gap:16px; } .request-card,.side { width:100%; } .side { display:contents; }
 #statecard { width:100%; } .stop-panel { position:fixed; bottom:0; left:0; right:0; padding:12px 20px max(12px,env(safe-area-inset-bottom)); border-radius:18px 18px 0 0; z-index:10; box-shadow:0 -6px 24px #243a3210; }
 #stop { min-height:60px; font-size:23px; } .stop-caption { margin-top:10px; } .telemetry { grid-template-columns:1fr; margin-top:16px; }
 .detail-panel + .detail-panel { border-left:0; border-top:1px solid var(--line); } footer { font-size:10px; } .request-title { font-size:27px; }
 #stopfail { position:sticky; top:8px; z-index:11; } .select-wrap { flex-wrap:wrap; } .select-wrap label { display:none; }
}
@media (max-width:380px) { form { grid-template-columns:1fr; } .select-wrap label { display:inline; } #go { width:100%; } .mode { padding:5px 9px; } }
@media (prefers-reduced-motion:reduce) { *,*::before,*::after { transition:none!important; scroll-behavior:auto!important; } }

.tutorial-button { border:1px solid #cbd3c3; background:var(--card); color:var(--go); border-radius:30px; padding:7px 13px; min-height:40px; font-size:12px; white-space:nowrap; }
.tutorial-button:hover { background:#e9eedf; }
#tour { position:fixed; inset:0; z-index:1000; }
#tour-shade { position:absolute; inset:0; background:#15281fc9; transition:background .35s; }
#tour[data-guided="true"] #tour-shade { background:transparent; }
#tour-ring { position:fixed; pointer-events:none; border:2px solid #c8e69c; border-radius:16px; box-shadow:0 0 0 9999px #15281fc9,0 0 0 5px #c8e69c33; transition:top .4s ease,left .4s ease,width .4s ease,height .4s ease,opacity .3s; opacity:0; }
#tour[data-guided="true"] #tour-ring { opacity:1; }
#tour-card { position:fixed; width:min(410px,calc(100vw - 32px)); padding:25px; background:var(--card); color:var(--ink); border:1px solid #d9dfcf; border-radius:20px; box-shadow:0 20px 70px #10201540; transition:top .4s ease,left .4s ease; max-height:calc(100dvh - 32px); overflow:auto; }
.tour-heading { display:flex; justify-content:space-between; align-items:center; margin-bottom:18px; }
#tour-count { font-size:10px; letter-spacing:1.5px; color:var(--muted); text-transform:uppercase; }
#tour-close { width:44px; height:44px; border:0; background:#eef1e8; border-radius:50%; color:var(--ink); font-size:23px; margin:-10px -10px -10px 0; }
#tour-title { font:400 28px/1.15 Georgia,serif; letter-spacing:-.6px; margin:0 0 12px; }
#tour-description { color:var(--muted); font-size:14px; line-height:1.6; margin:0; }
.tour-copy-enter { animation:tour-copy .35s ease both; }
@keyframes tour-copy { from { opacity:0; transform:translateY(-10px); } to { opacity:1; transform:translateY(0); } }
.tour-actions { display:flex; gap:10px; margin-top:24px; align-items:center; }
#tour-next { margin-left:auto; background:var(--go); color:#fff; border:1px solid var(--go); }
#tour-back { background:transparent; color:var(--muted); border:1px solid var(--line); }
#tour-next,#tour-back { min-height:44px; border-radius:9px; padding:10px 16px; font-size:13px; }
#tour-stop { margin-top:17px; width:100%; min-height:44px; border:1px solid #d7aaa2; border-radius:9px; background:#fcf0ed; color:#a02c28; font-size:12px; font-weight:650; }
@media (prefers-reduced-motion:reduce) { .tour-copy-enter { animation:none; } }
@media (max-width:760px) { .topbar { flex-wrap:wrap; gap:14px; } .top-status { margin-left:auto; } }
@media (max-width:380px) { .top-status { width:100%; justify-content:flex-end; } }
</style></head>
<body><main>
  <header class="topbar">
    <div class="brand"><span class="mark" aria-hidden="true"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 4v8a7 7 0 0 0 14 0V4M5 8h4m6 0h4M9 4v4m6-4v4M12 19v3"/><circle cx="12" cy="13" r="2"/></svg></span><h1>floor picker<span style="color:#8eaa72">.</span></h1><span class="brand-note">A little help, within reach.</span></div>
    <div class="top-status"><button id="tutorial" type="button" class="tutorial-button">Tutorial <span aria-hidden="true">↗</span></button><span class="mode">__MODE__</span><span id="connection" role="status">Connecting</span></div>
  </header>
  <section class="intro">
    <div><div class="eyebrow">Your everyday helping hand</div><h2>Leave the reaching<br><em>to us.</em></h2><p>Choose what you dropped. We'll take it from here.</p></div>
    <p class="intro-note">From the floor to your hand.<br>A little more independence.</p>
  </section>
  <div id="stopfail" role="alert" hidden><b>STOP FAILED: hit the E-stop on the robot.</b><span id="stopwhy"></span> Trying again every second.</div>
  <div class="workspace">
    <section class="card request-card">
      <div class="card-label"><h3>Make a request</h3><span class="card-number">01 / REQUEST</span></div>
      <h3 class="request-title">What did you drop?</h3><p class="help">Choose an item and we’ll bring it to you.</p>
      <form id="ask">
        <input id="text" type="hidden" value="phone">
        <div class="select-wrap"><label for="hand">Hand-off height</label><select id="hand" aria-label="Hand height"><option value="seated">Seated</option><option value="standing">Standing</option></select></div>
        <button class="go phone-choice" id="go" type="button" aria-disabled="false"><svg width="28" height="36" viewBox="0 0 28 36" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="5" y="2" width="18" height="32" rx="4"/><path d="M10 6h8M11 29h6"/></svg><span>Phone</span><span class="choice-arrow" aria-hidden="true">↗</span></button>
      </form>
      <label id="setting" hidden><input type="checkbox" id="clarify"> Ask me questions first</label>
      <div id="question" role="status" aria-live="polite" hidden></div><div id="opts"></div>
      <button id="cancel" type="button" hidden>Cancel</button><div id="reply" role="status" aria-live="polite"></div>
      <div class="request-foot"><svg width="16" height="18" viewBox="0 0 16 18" fill="none" aria-hidden="true"><path d="M8 1 14 3v5c0 4-6 8-6 8S2 12 2 8V3l6-2Z" stroke="currentColor"/><path d="m5 8 2 2 4-4" stroke="currentColor"/></svg> You're in control. Stop a request at any time.</div>
    </section>
    <aside class="side" aria-label="Mission and stop controls">
      <section class="card" id="statecard">
        <div class="card-label"><h3>Mission status</h3><span class="card-number">02 / PROGRESS</span></div>
        <div id="state" aria-live="polite">Connecting…</div><div id="message"></div>
        <ol id="steps" aria-label="Hand-off steps"><li id="step-Search">Search</li><li id="step-Approach">Drive</li><li id="step-Pick">Pick</li><li id="step-Present">Present</li><li id="step-WaitForTake">Take</li></ol>
      </section>
      <section class="stop-panel" aria-label="Stop control"><button id="stop" type="button" aria-label="STOP"><span class="stop-symbol" aria-hidden="true"></span>STOP</button><div class="stop-caption"><span>Stop motion · hold the arm</span><span>or press <kbd>Esc</kbd></span></div></section>
    </aside>
  </div>
  <section class="card telemetry" aria-label="Robot details">
    <div class="detail-panel"><div class="card-label"><h3>At a glance</h3><span class="card-number">03 / DETAILS</span></div>
      <dl><dt>Item</dt><dd id="item">–</dd><dt>Seen</dt><dd id="detection">–</dd><div class="row" id="liftrow" hidden><dt>Tray height</dt><dd id="lift"></dd></div><div class="row" id="trayrow" hidden><dt>Tray weight</dt><dd id="tray"></dd></div><dt>Last pick</dt><dd id="pick">–</dd></dl>
    </div>
    <div class="detail-panel"><div class="card-label"><h3>Camera view</h3><span class="card-number">04 / VISION</span></div>
      <div id="cameraempty"><span class="camera-icon"><svg width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3" aria-hidden="true"><rect x="3" y="6" width="18" height="14" rx="3"/><path d="m8 6 1-3h6l1 3"/><circle cx="12" cy="13" r="3.5"/></svg></span><span><strong>No camera feed available</strong>A live view appears when the robot provides one.</span></div><div id="thumbs"></div>
    </div>
  </section>
  <footer><span>FLOOR PICKER / Made for a little more ease.</span><span>__MODE_NOTE__</span></footer>
</main>
<div id="tour" hidden>
  <div id="tour-shade"></div><div id="tour-ring" aria-hidden="true"></div>
  <section id="tour-card" role="dialog" aria-modal="true" aria-labelledby="tour-title" aria-describedby="tour-description" tabindex="-1">
    <div class="tour-heading"><span id="tour-count"></span><button id="tour-close" type="button" aria-label="Close tutorial">×</button></div>
    <div id="tour-copy" aria-live="polite"><h2 id="tour-title"></h2><p id="tour-description"></p></div>
    <div class="tour-actions"><button id="tour-back" type="button">Back</button><button id="tour-next" type="button">Let’s begin ↗</button></div>
    <button id="tour-stop" type="button">STOP motion · available any time</button>
  </section>
</div>

<script>
const $ = id => document.getElementById(id);
const STOP_WORDS = new RegExp(__STOP_PATTERN__, "i");   // the same check as ui/request.py
const label = s => s.replaceAll("_", " ");

// The hand-off, in order (plan 10.7), and what the big heading says for each state.
const STEPS = ["Search", "Approach", "Pick", "Present", "WaitForTake"];
const TITLES = {Search: "Searching", Approach: "Driving over", Pick: "Picking it up", Present: "Holding it up"};
const title = s => s.state === "WaitForTake" ? (s.item ? `Take your ${label(s.item)}` : "Take it")
                                              : TITLES[s.state] || s.state;

// fetch with a time limit; reads the status before trying to read JSON.
async function call(url, opts, ms) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  try {
    const r = await fetch(url, {...opts, signal: ctl.signal, cache: "no-store"});
    let body = null;
    try { body = await r.json(); } catch (e) { if (e.name === "AbortError") throw e; }
    return {ok: r.ok, status: r.status, body};
  } finally { clearTimeout(timer); }
}
const why = e => e.name === "AbortError" ? "no answer in time" : "no connection";

let stopFailing = false, stopRetry = null, asking = false;
// The questions in progress: the original request, the answers so far, the question on screen.
let convo = {text: "", dialog: [], pending: null};
const MAX_QUESTIONS = __MAX_QUESTIONS__;      // the server enforces it too
function setAskBusy() { $("go").setAttribute("aria-disabled", String(asking || stopFailing)); }

function showQuestion() {
  const q = $("question"), pending = convo.pending;
  q.hidden = pending === null;
  q.textContent = pending === null ? "" : pending;
  $("opts").innerHTML = "";
  $("cancel").hidden = pending === null;
  $("text").placeholder = pending === null ? "I dropped my phone" : "Your answer";
  if (pending === null) return;
  const n = document.createElement("small");
  n.textContent = `Question ${convo.dialog.length + 1} of up to ${MAX_QUESTIONS}`;
  q.append(n);
  for (const o of convo.options || []) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "opt"; b.textContent = o;
    b.addEventListener("click", () => send(o));
    $("opts").append(b);
  }
}
function endConversation() {
  convo = {text: "", dialog: [], pending: null};
  showQuestion();
}
$("cancel").addEventListener("click", () => { endConversation(); $("reply").textContent = "Cancelled"; });

// The "Ask me questions first" setting; the page works without localStorage.
let handTouched = false;
try { const v = localStorage.getItem("clarify"); if (v !== null) $("clarify").checked = v === "1"; } catch (e) {}
const syncHand = () => { if (!handTouched) $("hand").value = "seated"; };
syncHand();
$("hand").addEventListener("change", () => { handTouched = true; });
$("clarify").addEventListener("change", () => {
  try { localStorage.setItem("clarify", $("clarify").checked ? "1" : "0"); } catch (e) {}
  syncHand();
  endConversation();
});

async function stop() {
  clearTimeout(stopRetry);
  endConversation();
  $("state").textContent = "Stopping…";
  let problem;
  try {
    const r = await call("/stop", {method: "POST"}, 2000);
    if (r.ok) {
      stopFailing = false; $("stopfail").hidden = true; setAskBusy();
      refresh();
      return;
    }
    problem = `The robot did not confirm (server error ${r.status}).`;
  } catch (e) { problem = `STOP was not delivered (${why(e)}).`; }
  stopFailing = true;
  $("stopwhy").textContent = problem; $("stopfail").hidden = false; setAskBusy();
  stopRetry = setTimeout(stop, 1000);
}
$("stop").addEventListener("click", stop);
document.addEventListener("keydown", e => { if (e.key === "Escape") stop(); });

$("ask").addEventListener("submit", e => {
  e.preventDefault();
  return send($("text").value);
});

// Send the box text (or a tapped option): a new request, or the answer to the pending question.
$("go").addEventListener("click", () => { endConversation(); return send("phone", true); });

async function send(raw, direct = false) {
  const text = raw.trim(); if (!text) return;
  if (STOP_WORDS.test(text)) {           // typed stop goes straight to /stop, even mid-request
    $("text").value = ""; $("reply").textContent = "Stopping";
    stop();
    return;
  }
  if (stopFailing) { $("reply").textContent = "STOP has not gone through yet."; return; }
  if (asking) return;
  const clarify = !direct && $("clarify").checked;
  const answering = clarify && convo.pending !== null;      // else this starts a new request
  const sent = answering ? {text: convo.text, dialog: [...convo.dialog, {question: convo.pending, answer: text}]}
                         : {text, dialog: []};
  const hand = $("hand").value;
  const payload = direct ? {item: "phone", hand: hand || "seated"}
                  : clarify ? {text: sent.text, hand, clarify: true, dialog: sent.dialog}
                          : {text: sent.text, hand: hand || "seated"};
  asking = true; setAskBusy(); $("reply").textContent = "…";
  try {
    const r = await call(direct ? "/fetch" : "/request", {method: "POST", headers: {"Content-Type": "application/json"},
                                      body: JSON.stringify(payload)}, 30000);
    if (!r.ok || !r.body || !r.body.parsed) {
      const detail = r.body && typeof r.body.detail === "string" ? `: ${r.body.detail}` : "";
      $("reply").textContent = `Not done (server error ${r.status}${detail})`;
    } else {
      const p = r.body.parsed;
      if (p.action === "ask") {
        convo = {...sent, pending: p.question, options: p.options};
        showQuestion();
        $("reply").textContent = "";
      } else {
        endConversation();
        $("reply").textContent = r.body.dropped ? "Cancelled: STOP was pressed"
                               : p.action === "fetch" ? `Fetching: ${label(p.item)}${p.hand ? ` (${p.hand})` : ""}`
                               : p.action === "stop" ? "Stopped" : "I can't fetch that one.";
      }
      $("text").value = "";
    }
  } catch (e) { $("reply").textContent = `Not sent (${why(e)})`; }
  finally { asking = false; setAskBusy(); }
  refresh();
}

let shown = "", refreshing = false;
async function refresh() {
  if (refreshing) return;                // one /status at a time, so STOP never queues behind them
  refreshing = true;
  try {
    const r = await call("/status", {}, 1500);
    if (!r.ok || !r.body) throw new Error(`status ${r.status}`);
    const s = r.body;
    $("statecard").parentElement.classList.remove("stale");
    if ($("connection").textContent !== "Connected") $("connection").textContent = "Connected"; $("connection").setAttribute("data-online", "true");
    if ($("state").textContent !== title(s)) $("state").textContent = title(s);
    $("message").textContent = s.message || "";
    $("statecard").dataset.state = s.state;
    const at = STEPS.indexOf(s.state);
    STEPS.forEach((n, i) => $("step-" + n).setAttribute("data-step", i < at ? "done" : i === at ? "now" : "todo"));
    $("item").textContent = s.item ? label(s.item) : "–";
    $("detection").textContent = s.detection ? `${s.detection.label} ${Math.round(s.detection.conf * 100)}%` : "–";
    $("liftrow").hidden = s.lift_mm == null;       // no lift: no row, not a dash
    $("trayrow").hidden = s.tray_g == null;
    if (s.lift_mm != null) $("lift").textContent = `${Math.round(s.lift_mm)} mm`;
    if (s.tray_g != null) $("tray").textContent = `${Math.round(s.tray_g)} g`;
    $("pick").textContent = s.last_pick ? `${s.last_pick.result} (${s.last_pick.grasp}, ${s.last_pick.seconds} s)` : "–";
    $("cameraempty").hidden = (s.cameras || []).length > 0;
    const names = (s.cameras || []).join(",");
    if (names !== shown) {
      shown = names;
      $("thumbs").innerHTML = "";
      for (const n of s.cameras || []) {
        const img = new Image(); img.alt = n; img.dataset.name = n;
        img.onload = img.onerror = () => { img.dataset.busy = ""; };
        $("thumbs").append(img);
      }
    }
    for (const img of $("thumbs").children) {
      if (img.dataset.busy) continue;    // at most one picture request per camera
      img.dataset.busy = "1";
      img.src = `/thumb/${encodeURIComponent(img.dataset.name)}?t=${Date.now()}`;
    }
  } catch (e) { $("statecard").parentElement.classList.add("stale"); if ($("connection").textContent !== "Disconnected") $("connection").textContent = "Disconnected"; $("connection").setAttribute("data-online", "false"); }
  finally { refreshing = false; }
}

// The tutorial never activates a target control or sends a fetch request.
const TOUR_STEPS = [
  {target:null, title:"Welcome, let’s help you use Floor Picker", text:"A quick tour of choosing your item, setting your hand-off height and staying in control."},
  {target:"go", title:"This is your object option", text:"Phone is the item Floor Picker can find and bring to you. This is where you choose what you dropped."},
  {target:"hand", title:"Choose your hand-off height", text:"Choose Seated or Standing to tell Floor Picker how you’d like the item handed to you. Set this before starting your request."},
  {target:"go", title:"Tap Phone to get started", text:"Once your height is set, click or tap Phone to start the request. Just looking at this tutorial won’t start anything."},
  {target:"stop", title:"You can stop at any time", text:"Watch Mission status to follow along. Hit STOP whenever you need to stop motion and hold the arm. You can also press Esc."}
];
let tourIndex = -1, tourFrame = null, tourUntil = 0, tourReturnFocus = null;
function placeTour() {
  if (tourIndex < 0) return;
  const step = TOUR_STEPS[tourIndex], card = $("tour-card"), ring = $("tour-ring");
  const vw = window.innerWidth, vh = window.innerHeight, margin = 16;
  const width = Math.min(410, vw - margin * 2);
  card.style.width = width + "px";
  const height = card.offsetHeight;
  let left = (vw - width) / 2, top = Math.max(margin, (vh - height) / 2);
  if (step.target) {
    const box = $(step.target).getBoundingClientRect();
    ring.style.left = (box.left - 6) + "px"; ring.style.top = (box.top - 6) + "px";
    ring.style.width = (box.width + 12) + "px"; ring.style.height = (box.height + 12) + "px";
    left = Math.max(margin, Math.min(vw - width - margin, box.left));
    top = box.bottom + 20;
    if (top + height > vh - margin) top = box.top - height - 20;
    top = Math.max(margin, Math.min(vh - height - margin, top));
  } else {
    ring.style.left = (vw / 2 - 20) + "px"; ring.style.top = (vh / 2 - 20) + "px";
    ring.style.width = "40px"; ring.style.height = "40px";
  }
  card.style.left = left + "px"; card.style.top = top + "px";
}
function trackTour() {
  cancelAnimationFrame(tourFrame);
  tourUntil = performance.now() + 750;
  const tick = () => { placeTour(); if (tourIndex >= 0 && performance.now() < tourUntil) tourFrame = requestAnimationFrame(tick); };
  tourFrame = requestAnimationFrame(tick);
}
function showTourStep(index) {
  tourIndex = index;
  const step = TOUR_STEPS[index];
  $("tour").hidden = false;
  $("tour").dataset.guided = String(Boolean(step.target));
  $("tour-title").textContent = step.title; $("tour-description").textContent = step.text;
  $("tour-count").textContent = `Quick tour / ${index + 1} of ${TOUR_STEPS.length}`;
  $("tour-back").hidden = index === 0;
  $("tour-next").textContent = index === 0 ? "Let’s begin ↗" : index === TOUR_STEPS.length - 1 ? "Got it ✓" : "Next →";
  $("tour-copy").classList.remove("tour-copy-enter");
  void $("tour-copy").offsetWidth; $("tour-copy").classList.add("tour-copy-enter");
  if (step.target) {
    const behavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth";
    if (window.innerWidth <= 760 && step.target !== "stop") {
      window.scrollTo({top:Math.max(0, $(step.target).getBoundingClientRect().top + window.scrollY - 90), behavior});
    } else $(step.target).scrollIntoView({block:"center", behavior});
  }
  placeTour(); trackTour();
  $("tour-next").focus({preventScroll:true});
}
function closeTour() {
  if (tourIndex < 0) return;
  tourIndex = -1; cancelAnimationFrame(tourFrame); $("tour").hidden = true;
  if (tourReturnFocus && tourReturnFocus.isConnected) tourReturnFocus.focus({preventScroll:true});
}
$("tutorial").addEventListener("click", () => { tourReturnFocus = document.activeElement; showTourStep(0); });
$("tour-next").addEventListener("click", () => tourIndex === TOUR_STEPS.length - 1 ? closeTour() : showTourStep(tourIndex + 1));
$("tour-back").addEventListener("click", () => { if (tourIndex > 0) showTourStep(tourIndex - 1); });
$("tour-close").addEventListener("click", closeTour);
$("tour-stop").addEventListener("click", () => { closeTour(); stop(); });
document.addEventListener("scroll", () => { if (tourIndex >= 0) trackTour(); }, true);
globalThis.addEventListener?.("resize", () => { if (tourIndex >= 0) trackTour(); });
document.addEventListener("keydown", e => {
  if (tourIndex < 0) return;
  if (e.key === "Escape") { closeTour(); return; } // Existing Escape handler still sends STOP.
  if (e.key !== "Tab") return;
  const controls = [...$("tour-card").querySelectorAll("button")].filter(b => !b.hidden && !b.disabled);
  const first = controls[0], last = controls[controls.length - 1];
  if (e.shiftKey && (document.activeElement === first || !controls.includes(document.activeElement))) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && (document.activeElement === last || !controls.includes(document.activeElement))) { e.preventDefault(); first.focus(); }
});

setInterval(refresh, 500); refresh();
</script></body></html>
""".replace("__STOP_PATTERN__", json.dumps(request_mod.STOP_WORDS.pattern)).replace(
    "__MAX_QUESTIONS__", str(request_mod.MAX_QUESTIONS))


def load_config(path: Path) -> dict | None:
    """Parsed robot.yaml, or None if the file does not exist."""
    if not path.exists():
        return None
    import yaml

    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a mapping")
    return config


def page_urls(host: str, port: int) -> list[str]:
    """Where to open the page. Uses only the local hostname: no network lookup."""
    if host in ("0.0.0.0", "::"):
        name = socket.gethostname()
        local = name if name.endswith(".local") else f"{name}.local"
        return [f"http://127.0.0.1:{port}", f"http://{local}:{port}"]
    return [f"http://{host}:{port}"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ui.server", description="Floor Picker status page with STOP.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fake", action="store_true", help="FakeBrain, no hardware (the default)")
    mode.add_argument("--sim", action="store_true",
                      help="the brain (task M1) with simulated arm, fake serial port and replayed detections")
    mode.add_argument("--real", action="store_true", help="the brain on the robot hardware; asks you to confirm first")
    parser.add_argument("--config", type=Path, default=request_mod.CONFIG_PATH, help="robot.yaml (default: %(default)s)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 (default): this computer only. 0.0.0.0: any device on the Wi-Fi, with no password")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if not (args.sim or args.real):
        brain = FakeBrain(lift=(config or {}).get("lift"))
        print("FakeBrain: nothing will move. Use --sim or --real for the brain (task M1).")
    else:
        if config is None:
            print(f"error: {args.config} not found; --sim and --real need robot.yaml (task S6)", file=sys.stderr)
            return 2
        try:
            from brain.mission import build          # task M1 (SW2): see the Brain protocol above
        except ImportError as e:
            if e.name not in ("brain", "brain.mission"):
                raise   # M1 exists but something it imports is missing: show the real error
            print("error: brain not available yet (task M1: brain/mission.py with build(config, sim)). "
                  "Run without --sim/--real for FakeBrain.", file=sys.stderr)
            return 2
        if args.real:
            print("--real: requests from this page will drive the wheels, arm and lift.")
            try:
                answer = input("Is the area clear and is someone at the E-stop? Type yes to start: ")
            except EOFError:
                answer = ""
            if answer.strip().lower() != "yes":
                print("Not started.")
                return 1
        brain = build(config, sim=args.sim)

    if args.host in ("0.0.0.0", "::"):
        print("WARNING: anyone on this network can open the page and send the robot. There is no password.")
    for url in page_urls(args.host, args.port):
        print(f"Open {url}")

    import uvicorn

    try:
        uvicorn.run(create_app(brain), host=args.host, port=args.port)
    finally:
        brain.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
