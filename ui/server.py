"""Status page with STOP (task U2, plan 9.7). FastAPI on port 8000.

    python -m ui.server                    # FakeBrain, no hardware (the default)
    python -m ui.server --sim              # the brain (task M1) with simulated arm, base and camera
    python -m ui.server --real             # the brain on the robot: asks you to confirm first
    python -m ui.server --host 0.0.0.0     # let a phone on the same Wi-Fi open it (no password)

| Route           | Does                                                              |
|-----------------|-------------------------------------------------------------------|
| GET  /          | The page: text box, big STOP button, current state, thumbnails    |
| POST /request   | {"text": "...", "hand": "seated"|"standing"}: parsed, sent on     |
| POST /stop      | Stop everything                                                   |
| GET  /status    | Brain state, last detection, last pick; lift and tray (null: none) |
| GET  /thumb/{n} | Latest JPEG from camera n, if the brain provides one              |

The brain is anything that follows the `Brain` protocol below.

STOP never waits on the parser, the network or the worker threads:
- /stop is an async route, so it runs on the event loop and calls brain.stop()
  at once even when every worker thread is busy with slow parses or status calls.
- "stop" typed in the box is caught by the keyword check before the hand is
  checked or the parser runs, also on the event loop.
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
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from . import request as request_mod

HANDS = ("seated", "standing")
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


class RequestBody(BaseModel):
    text: str
    hand: Any = "seated"      # checked by hand after the STOP check, so "stop" always wins


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


def create_app(brain: Brain, parse=request_mod.parse_request) -> FastAPI:
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
        return PAGE

    # async: runs on the event loop, never queued behind the worker threads.
    @app.post("/stop")
    async def stop():
        t0 = time.monotonic()
        stop_now()
        return {"stopped": True, "brain_ms": round((time.monotonic() - t0) * 1000, 1)}

    @app.post("/request")
    async def post_request(body: RequestBody):
        if request_mod.is_stop(body.text):            # stop wins, before the hand check or the parser
            stop_now()
            return {"parsed": {"action": "stop", "item": None}}
        if body.hand not in HANDS:
            raise HTTPException(422, f"hand must be one of {HANDS}")
        seen = stops.value
        parsed = await run_in_threadpool(parse, body.text)    # may take seconds (Claude API)
        if parsed["action"] == "stop":
            stop_now()
        elif parsed["action"] == "fetch":
            if stops.value != seen:                    # STOP arrived while parsing
                return {"parsed": parsed, "dropped": "stopped"}
            await run_in_threadpool(brain.request, parsed["item"], body.hand)
            if stops.value != seen:                    # STOP raced with request(): make sure it wins
                brain.stop()
                return {"parsed": parsed, "dropped": "stopped"}
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
  :root { --bg:#f6f5f2; --card:#fff; --ink:#1d1d1b; --muted:#6b6a66; --line:#e2e0da;
          --stop:#c62828; --stop-dark:#8e1c1c; --go:#1f6f4a; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#151514; --card:#1f1f1d; --ink:#f1f0ec; --muted:#a3a29c; --line:#34332f; }
  }
  * { box-sizing:border-box; }
  html { -webkit-text-size-adjust:100%; text-size-adjust:100%; }   /* no iOS text inflation in landscape */
  button, select { touch-action:manipulation; }                   /* fast taps; a double tap on STOP doesn't zoom */
  body { margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; padding:16px; }
  main { max-width:640px; margin:0 auto; display:grid; gap:14px; }
  h1 { font-size:18px; margin:0; color:var(--muted); font-weight:600; letter-spacing:.02em; }
  #stop { width:100%; min-height:120px; border:0; border-radius:16px; background:var(--stop);
          color:#fff; font-size:44px; font-weight:800; letter-spacing:.08em; cursor:pointer;
          box-shadow:0 6px 0 var(--stop-dark); }
  #stop:active { transform:translateY(4px); box-shadow:0 2px 0 var(--stop-dark); }
  #stopfail { background:var(--stop); color:#fff; border-radius:14px; padding:14px; font-size:20px; }
  #stopfail b { display:block; font-size:26px; }
  #stopfail[hidden] { display:none; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px; }
  #state { font-size:28px; font-weight:700; }
  #message { color:var(--muted); margin-top:2px; }
  #steps { display:flex; gap:6px; list-style:none; margin:12px 0 0; padding:0; font-size:13px; }
  #steps li { flex:1; text-align:center; padding:4px 2px; border-radius:8px; color:var(--muted);
              border:1px solid var(--line); }
  #steps li[data-step="done"] { color:var(--ink); }
  #steps li[data-step="now"] { background:var(--go); border-color:var(--go); color:#fff; font-weight:700; }
  #statecard[data-state="WaitForTake"] { background:var(--go); border-color:var(--go); color:#fff;
          text-align:center; padding:24px 14px; animation:take 1.4s ease-in-out infinite; }
  #statecard[data-state="WaitForTake"] #state { font-size:40px; }
  #statecard[data-state="WaitForTake"] #message { color:#fff; font-size:20px; }
  #statecard[data-state="WaitForTake"] #steps li { color:#fff; border-color:rgba(255,255,255,.5); }
  #statecard[data-state="WaitForTake"] #steps li[data-step="now"] { background:#fff; color:var(--go); }
  .stale #statecard[data-state="WaitForTake"] #state::after { color:#fff; }
  @keyframes take { 50% { box-shadow:0 0 0 8px rgba(31,111,74,.35); } }
  @media (prefers-reduced-motion: reduce) { #statecard[data-state="WaitForTake"] { animation:none; } }
  form { display:flex; gap:8px; flex-wrap:wrap; }
  #text { flex:1 1 220px; font:inherit; font-size:18px; padding:12px; border-radius:10px;
          border:1px solid var(--line); background:var(--bg); color:var(--ink); }
  select, button.go { font:inherit; font-size:17px; padding:12px 14px; border-radius:10px;
          border:1px solid var(--line); background:var(--bg); color:var(--ink); }
  button.go { background:var(--go); color:#fff; border-color:var(--go); font-weight:600; }
  button.go[aria-disabled="true"] { opacity:.45; cursor:wait; }
  dl { display:grid; grid-template-columns:auto 1fr; gap:6px 14px; margin:0; }
  dt { color:var(--muted); } dd { margin:0; font-variant-numeric:tabular-nums; }
  .row { display:contents; } .row[hidden] { display:none; }
  #thumbs { display:flex; gap:8px; flex-wrap:wrap; } #thumbs img { width:150px; max-width:100%; border-radius:8px; }
  @media (max-width:480px) { #thumbs img { width:calc(50% - 4px); } }   /* phones: two per row, full width */
  #reply { color:var(--muted); min-height:1.4em; }
  .stale #state::after { content:" (no connection)"; font-size:16px; color:var(--stop); }
</style></head>
<body><main>
  <h1>FLOOR PICKER</h1>
  <button id="stop" type="button">STOP</button>
  <div id="stopfail" role="alert" hidden>
    <b>STOP FAILED: hit the E-stop on the robot.</b>
    <span id="stopwhy"></span> Trying again every second.
  </div>
  <section class="card" id="statecard">
    <div id="state">…</div><div id="message"></div>
    <ol id="steps" aria-label="Hand-off steps">
      <li id="step-Search">Search</li><li id="step-Approach">Drive</li><li id="step-Pick">Pick</li>
      <li id="step-Present">Present</li><li id="step-WaitForTake">Take</li>
    </ol>
  </section>
  <section class="card">
    <form id="ask">
      <input id="text" autocomplete="off" placeholder="I dropped my phone">
      <select id="hand" aria-label="Hand height"><option value="seated">Seated</option><option value="standing">Standing</option></select>
      <button class="go" id="go" type="submit" aria-disabled="false">Ask</button>
    </form>
    <div id="reply"></div>
  </section>
  <section class="card">
    <dl>
      <dt>Item</dt><dd id="item">–</dd>
      <dt>Seen</dt><dd id="detection">–</dd>
      <div class="row" id="liftrow" hidden><dt>Tray height</dt><dd id="lift"></dd></div>
      <div class="row" id="trayrow" hidden><dt>Tray weight</dt><dd id="tray"></dd></div>
      <dt>Last pick</dt><dd id="pick">–</dd>
    </dl>
    <div id="thumbs"></div>
  </section>
</main>
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
function setAskBusy() { $("go").setAttribute("aria-disabled", String(asking || stopFailing)); }

async function stop() {
  clearTimeout(stopRetry);
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

$("ask").addEventListener("submit", async e => {
  e.preventDefault();
  const text = $("text").value.trim(); if (!text) return;
  if (STOP_WORDS.test(text)) {           // typed stop goes straight to /stop, even mid-request
    $("text").value = ""; $("reply").textContent = "Stopping";
    stop();
    return;
  }
  if (stopFailing) { $("reply").textContent = "STOP has not gone through yet."; return; }
  if (asking) return;
  asking = true; setAskBusy(); $("reply").textContent = "…";
  try {
    const r = await call("/request", {method: "POST", headers: {"Content-Type": "application/json"},
                                      body: JSON.stringify({text, hand: $("hand").value})}, 30000);
    if (!r.ok || !r.body || !r.body.parsed) {
      const detail = r.body && typeof r.body.detail === "string" ? `: ${r.body.detail}` : "";
      $("reply").textContent = `Not done (server error ${r.status}${detail})`;
    } else {
      const p = r.body.parsed;
      $("reply").textContent = r.body.dropped ? "Cancelled: STOP was pressed"
                             : p.action === "fetch" ? `Fetching: ${label(p.item)}`
                             : p.action === "stop" ? "Stopped" : "I can't fetch that one.";
      $("text").value = "";
    }
  } catch (e) { $("reply").textContent = `Not sent (${why(e)})`; }
  finally { asking = false; setAskBusy(); }
  refresh();
});

let shown = "", refreshing = false;
async function refresh() {
  if (refreshing) return;                // one /status at a time, so STOP never queues behind them
  refreshing = true;
  try {
    const r = await call("/status", {}, 1500);
    if (!r.ok || !r.body) throw new Error(`status ${r.status}`);
    const s = r.body;
    $("statecard").parentElement.classList.remove("stale");
    $("state").textContent = title(s); $("message").textContent = s.message || "";
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
  } catch (e) { $("statecard").parentElement.classList.add("stale"); }
  finally { refreshing = false; }
}
setInterval(refresh, 500); refresh();
</script></body></html>
""".replace("__STOP_PATTERN__", json.dumps(request_mod.STOP_WORDS.pattern))


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
