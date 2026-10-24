"""Status page with STOP (task U2, plan 9.7). FastAPI on port 8000.

    python -m ui.server            # with the real brain (task M1)
    python -m ui.server --fake     # with FakeBrain, no hardware

| Route           | Does                                                              |
|-----------------|-------------------------------------------------------------------|
| GET  /          | The page: text box, big STOP button, current state, thumbnails    |
| POST /request   | {"text": "...", "hand": "seated"|"standing"}: parsed, sent on     |
| POST /stop      | Stop everything                                                   |
| GET  /status    | Brain state, last detection, lift height, tray weight, last pick  |
| GET  /thumb/{n} | Latest JPEG from camera n, if the brain provides one              |

The brain is any object with:

    request(item: str, hand: str) -> None   start fetching item; hand is "seated" or "standing"
    stop() -> None                          stop now: wheels stop, arm holds, lift holds
    status() -> dict                        see FakeBrain.status for the fields
    thumbnail(name: str) -> bytes | None    optional, JPEG bytes

STOP never waits on the parser or the network: /stop calls brain.stop() at once,
and "stop" typed in the box is caught by the parser's keyword check.
"""

from __future__ import annotations

import argparse
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from . import request as request_mod

HANDS = ("seated", "standing")


class RequestBody(BaseModel):
    text: str
    hand: str = "seated"


def create_app(brain, parse=request_mod.parse_request) -> FastAPI:
    app = FastAPI(title="Floor Picker")

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE

    # Plain `def` routes run in a thread pool, so a slow parse never blocks /stop.
    @app.post("/stop")
    def stop():
        t0 = time.monotonic()
        brain.stop()
        return {"stopped": True, "ms": round((time.monotonic() - t0) * 1000, 1)}

    @app.post("/request")
    def post_request(body: RequestBody):
        if body.hand not in HANDS:
            raise HTTPException(422, f"hand must be one of {HANDS}")
        parsed = parse(body.text)
        if parsed["action"] == "stop":
            brain.stop()
        elif parsed["action"] == "fetch":
            brain.request(parsed["item"], body.hand)
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
    """Walks through the mission states on a timer, for trying the page without hardware."""

    STEPS = [("Search", 2.0), ("Approach", 3.0), ("Pick", 4.0), ("Raise", 2.0),
             ("WaitForTake", 4.0), ("Lower", 2.0)]

    def __init__(self):
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
            self._message = f"Looking for: {item.replace('_', ' ')}"

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
                if state in ("Raise", "WaitForTake", "Lower") and self._last_pick is None:
                    self._last_pick = {"result": "in_tray", "grasp": "scripted", "attempts": 1, "seconds": 4.0}
                self._state = state
                return
            elapsed -= seconds
        self._state, self._started, self._message = "Idle", None, "Done"

    def status(self) -> dict:
        with self._lock:
            self._advance()
            lift = {"Raise": 400, "WaitForTake": 700 if self._hand == "seated" else 1000, "Lower": 300}.get(self._state, 0)
            seen = self._state in ("Search", "Approach") and self._item
            return {
                "state": self._state,                  # brain state (plan 10.7) or "Stopped"
                "item": self._item,                    # what it's fetching
                "message": self._message,              # one line for people
                "detection": ({"label": self._item, "conf": 0.9, "floor_m": [0.5, 0.0]} if seen else None),
                "lift_mm": lift,
                "tray_g": 180.0 if self._state in ("Raise", "WaitForTake") else 0.0,
                "last_pick": self._last_pick,          # arm.pick result dict
                "cameras": [],                         # names with /thumb pictures
                "t": time.time(),
            }


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
  body { margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; padding:16px; }
  main { max-width:640px; margin:0 auto; display:grid; gap:14px; }
  h1 { font-size:18px; margin:0; color:var(--muted); font-weight:600; letter-spacing:.02em; }
  #stop { width:100%; min-height:120px; border:0; border-radius:16px; background:var(--stop);
          color:#fff; font-size:44px; font-weight:800; letter-spacing:.08em; cursor:pointer;
          box-shadow:0 6px 0 var(--stop-dark); }
  #stop:active { transform:translateY(4px); box-shadow:0 2px 0 var(--stop-dark); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px; }
  #state { font-size:28px; font-weight:700; }
  #message { color:var(--muted); margin-top:2px; }
  form { display:flex; gap:8px; flex-wrap:wrap; }
  #text { flex:1 1 220px; font:inherit; font-size:18px; padding:12px; border-radius:10px;
          border:1px solid var(--line); background:var(--bg); color:var(--ink); }
  select, button.go { font:inherit; font-size:17px; padding:12px 14px; border-radius:10px;
          border:1px solid var(--line); background:var(--bg); color:var(--ink); }
  button.go { background:var(--go); color:#fff; border-color:var(--go); font-weight:600; }
  dl { display:grid; grid-template-columns:auto 1fr; gap:6px 14px; margin:0; }
  dt { color:var(--muted); } dd { margin:0; font-variant-numeric:tabular-nums; }
  #thumbs { display:flex; gap:8px; flex-wrap:wrap; } #thumbs img { width:150px; border-radius:8px; }
  #reply { color:var(--muted); min-height:1.4em; }
  .stale #state::after { content:" (no connection)"; font-size:16px; color:var(--stop); }
</style></head>
<body><main>
  <h1>FLOOR PICKER</h1>
  <button id="stop" type="button">STOP</button>
  <section class="card" id="statecard">
    <div id="state">…</div><div id="message"></div>
  </section>
  <section class="card">
    <form id="ask">
      <input id="text" autocomplete="off" placeholder="I dropped my phone">
      <select id="hand" aria-label="Hand height"><option value="seated">Seated</option><option value="standing">Standing</option></select>
      <button class="go" type="submit">Ask</button>
    </form>
    <div id="reply"></div>
  </section>
  <section class="card">
    <dl>
      <dt>Item</dt><dd id="item">–</dd>
      <dt>Seen</dt><dd id="detection">–</dd>
      <dt>Tray height</dt><dd id="lift">–</dd>
      <dt>Tray weight</dt><dd id="tray">–</dd>
      <dt>Last pick</dt><dd id="pick">–</dd>
    </dl>
    <div id="thumbs"></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
async function stop() {
  $("state").textContent = "Stopping…";
  try { await fetch("/stop", {method: "POST"}); } catch (e) { $("reply").textContent = "STOP not sent: no connection"; }
  refresh();
}
$("stop").addEventListener("click", stop);
document.addEventListener("keydown", e => { if (e.key === "Escape") stop(); });
$("ask").addEventListener("submit", async e => {
  e.preventDefault();
  const text = $("text").value.trim(); if (!text) return;
  $("reply").textContent = "…";
  try {
    const r = await fetch("/request", {method: "POST", headers: {"Content-Type": "application/json"},
                                       body: JSON.stringify({text, hand: $("hand").value})});
    const p = (await r.json()).parsed;
    $("reply").textContent = p.action === "fetch" ? `Fetching: ${p.item.replace("_", " ")}`
                           : p.action === "stop" ? "Stopped" : "I can't fetch that one.";
    $("text").value = "";
  } catch (e) { $("reply").textContent = "Not sent: no connection"; }
  refresh();
});
let shown = "";
async function refresh() {
  try {
    const s = await (await fetch("/status")).json();
    $("statecard").parentElement.classList.remove("stale");
    $("state").textContent = s.state; $("message").textContent = s.message || "";
    $("item").textContent = s.item ? s.item.replace("_", " ") : "–";
    $("detection").textContent = s.detection ? `${s.detection.label} ${Math.round(s.detection.conf * 100)}%` : "–";
    $("lift").textContent = s.lift_mm != null ? `${Math.round(s.lift_mm)} mm` : "–";
    $("tray").textContent = s.tray_g != null ? `${Math.round(s.tray_g)} g` : "–";
    $("pick").textContent = s.last_pick ? `${s.last_pick.result} (${s.last_pick.grasp}, ${s.last_pick.seconds} s)` : "–";
    const names = (s.cameras || []).join(",");
    if (names !== shown) {
      shown = names;
      $("thumbs").innerHTML = "";
      for (const n of s.cameras || []) { const img = new Image(); img.alt = n; img.dataset.name = n; $("thumbs").append(img); }
    }
    for (const img of $("thumbs").children) img.src = `/thumb/${img.dataset.name}?t=${Date.now()}`;
  } catch (e) { $("statecard").parentElement.classList.add("stale"); }
}
setInterval(refresh, 500); refresh();
</script></body></html>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ui.server")
    parser.add_argument("--fake", action="store_true", help="FakeBrain, no hardware")
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 so a phone on the same Wi-Fi can open it")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.fake:
        brain = FakeBrain()
    else:
        from brain.mission import Brain          # task M1 (SW2)
        brain = Brain()
    import uvicorn

    uvicorn.run(create_app(brain), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
