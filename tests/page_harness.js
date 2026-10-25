// Runs the status page's <script> against a stub DOM and a scripted fetch.
// Used by tests/test_server.py:  node tests/page_harness.js <script.js> <scenario>
// Prints OK and exits 0 when the scenario's checks pass. No network, no browser.
"use strict";
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

const [scriptPath, scenario] = process.argv.slice(2);

const els = {};
const classes = new Set();
function el(id) {
  if (!els[id]) {
    const listeners = {};
    els[id] = {
      id, textContent: "", value: "", hidden: id === "stopfail", attrs: {}, children: [], placeholder: "",
      checked: id === "clarify", className: "",         // the setting is on by default
      dataset: {}, listeners, focus() {},
      set innerHTML(v) { this.children = []; },
      get innerHTML() { return ""; },
      addEventListener(type, fn) { listeners[type] = fn; },
      setAttribute(k, v) { this.attrs[k] = v; },
      append(child) { this.children.push(child); },
      get parentElement() { return {classList: {add: c => classes.add(c), remove: c => classes.delete(c)}}; },
    };
  }
  return els[id];
}
globalThis.document = {getElementById: el, addEventListener() {},
                       createElement(tag) { const b = el("new-" + tag + "-" + Math.random()); b.tag = tag; return b; }};
globalThis.Image = class { constructor() { this.dataset = {}; } };
globalThis.setInterval = () => 0;          // scenarios call refresh() themselves

// Scripted server: routes[path] answers each fetch.
const seen = [];
const routes = {};
const reply = (status, body) => ({ok: status >= 200 && status < 300, status, json: async () => body});
const hang = signal => new Promise((_, reject) =>
  signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))));
globalThis.fetch = (url, opts = {}) => {
  const path = url.split("?")[0];
  seen.push(path);
  const handler = routes[path] || (() => reply(404, {detail: "Not Found"}));
  return Promise.resolve().then(() => handler(opts));
};
const count = path => seen.filter(p => p === path).length;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const submit = text => send(text);
const clickStop = () => el("stop").listeners.click();
// The question page shows: its option buttons' labels, and a tap on one.
const options = () => el("opts").children.map(b => b.textContent);
const tap = label => el("opts").children.find(b => b.textContent === label).listeners.click();
const bodies = [];      // every /request body sent, parsed
const ASK = {action: "ask", item: null, hand: null, question: "What did you drop?", options: ["phone", "pen"]};
const FETCH = {action: "fetch", item: "phone", hand: "standing", question: null, options: []};
// One /status round, after the page's own first refresh() has finished (it drops overlapping calls).
const poll = async () => { await sleep(10); await refresh(); };

const STATUS = {state: "Idle", item: null, message: "", detection: null, lift_mm: null, tray_g: null,
                last_pick: null, cameras: [], t: 0};
routes["/status"] = () => reply(200, STATUS);

vm.runInThisContext(fs.readFileSync(scriptPath, "utf8"));

const scenarios = {
  async phone_button_sends_structured_fetch_without_parser() {
    routes["/fetch"] = opts => { bodies.push(JSON.parse(opts.body)); return reply(200, {parsed: FETCH}); };
    for (const value of ["", "0", "-1", "Infinity", "21", "abc"]) {
      el("distance").value = value; el("object").value = "sock";
      await el("go").listeners.click();
    }
    el("distance").value = "3.5"; el("object").value = "";
    await el("go").listeners.click();
    assert.strictEqual(count("/fetch"), 0);
    el("object").value = "sock";
    await el("ask").listeners.submit({preventDefault() {}});
    assert.deepStrictEqual(bodies, [{item:"sock", hand:"standing", distance_ft:3.5}]);
    assert.strictEqual(count("/request"), 0);
    assert.strictEqual(count("/fetch"), 1);
  },
  async stop_server_error_shows_banner_and_retries() {
    routes["/stop"] = () => reply(500, {detail: "Internal Server Error"});
    clickStop();
    await sleep(50);
    assert.strictEqual(el("stopfail").hidden, false, "STOP FAILED banner shown");
    assert.match(el("stopwhy").textContent, /server error 500/);
    assert.strictEqual(el("go").attrs["aria-disabled"], "true", "Ask blocked while STOP is failing");
    routes["/stop"] = () => reply(200, {stopped: true});
    await sleep(1400);                       // the page retries every second
    assert.ok(count("/stop") >= 2, "STOP retried");
    assert.strictEqual(el("stopfail").hidden, true, "banner cleared once STOP went through");
    assert.strictEqual(el("go").attrs["aria-disabled"], "false");
  },
  async stop_no_connection_shows_banner() {
    routes["/stop"] = () => { throw new TypeError("Failed to fetch"); };
    clickStop();
    await sleep(50);
    assert.strictEqual(el("stopfail").hidden, false);
    assert.match(el("stopwhy").textContent, /not delivered \(no connection\)/);
  },
  async stop_with_no_answer_fails_after_2_s() {
    routes["/stop"] = opts => hang(opts.signal);
    clickStop();
    await sleep(1000);
    assert.strictEqual(el("stopfail").hidden, true, "still within the 2 s limit");
    await sleep(1600);
    assert.strictEqual(el("stopfail").hidden, false, "failed after the 2 s limit");
    assert.match(el("stopwhy").textContent, /no answer in time/);
  },
  async ask_server_error_is_not_reported_as_no_connection() {
    routes["/request"] = () => reply(422, {detail: "hand must be one of ('seated', 'standing')"});
    await submit("my pen fell");
    assert.match(el("reply").textContent, /server error 422: hand must be/);
    assert.doesNotMatch(el("reply").textContent, /no connection/);
  },
  async ask_is_blocked_while_a_request_is_pending() {
    let finish;
    routes["/request"] = () => new Promise(r => {
      finish = () => r(reply(200, {parsed: {action: "fetch", item: "pill_bottle"}}));
    });
    const first = submit("my pills fell");
    await sleep(10);
    assert.strictEqual(el("go").attrs["aria-disabled"], "true");
    await submit("my pen fell");             // ignored while the first is pending
    assert.strictEqual(count("/request"), 1);
    finish();
    await first;
    assert.strictEqual(el("reply").textContent, "Fetching: pill bottle");
    assert.strictEqual(el("go").attrs["aria-disabled"], "false");
  },
  async typed_stop_goes_to_stop_even_while_a_request_is_pending() {
    routes["/request"] = opts => hang(opts.signal);
    routes["/stop"] = () => reply(200, {stopped: true});
    submit("my pills fell");
    await sleep(10);
    submit("Stoooop!");
    await sleep(20);
    assert.strictEqual(count("/stop"), 1);
    assert.strictEqual(count("/request"), 1);
  },
  async dropped_request_is_reported() {
    routes["/request"] = () => reply(200, {parsed: {action: "fetch", item: "phone"}, dropped: "stopped"});
    await submit("my phone");
    assert.strictEqual(el("reply").textContent, "Cancelled: STOP was pressed");
  },
  async question_is_shown_with_option_buttons() {
    routes["/request"] = opts => { bodies.push(JSON.parse(opts.body)); return reply(200, {parsed: ASK}); };
    await submit("I dropped something");
    assert.deepStrictEqual(bodies, [{text: "I dropped something", hand: "standing", clarify: true, dialog: []}]);
    assert.strictEqual(el("question").hidden, false);
    assert.match(el("question").textContent, /What did you drop\?/);
    assert.match(el("question").children[0].textContent, /Question 1 of up to 4/);
    assert.deepStrictEqual(options(), ["phone", "pen"]);
    assert.ok(el("opts").children.every(b => b.className === "opt" && b.tag === "button"));
    assert.strictEqual(el("text").placeholder, "Your answer");
    assert.strictEqual(el("cancel").hidden, false);
    assert.strictEqual(el("text").value, "", "box cleared for the answer");
  },
  async tapping_an_option_reposts_with_the_dialog() {
    let n = 0;
    routes["/request"] = opts => { bodies.push(JSON.parse(opts.body)); return reply(200, {parsed: n++ ? FETCH : ASK}); };
    await submit("I dropped something");
    await tap("phone");
    assert.deepStrictEqual(bodies[1], {text: "I dropped something", hand: "standing", clarify: true,
                                       dialog: [{question: "What did you drop?", answer: "phone"}]});
    assert.strictEqual(el("reply").textContent, "Fetching: phone (standing)");
    assert.strictEqual(el("question").hidden, true, "the conversation is over");
    assert.deepStrictEqual(options(), []);
    assert.strictEqual(el("text").placeholder, "I dropped my phone");
    await submit("my pen");                  // a new request starts with no dialog
    assert.deepStrictEqual(bodies[2].dialog, []);
    assert.strictEqual(bodies[2].text, "my pen");
  },
  async typing_an_answer_posts_it_as_the_answer_to_the_question() {
    let n = 0;
    routes["/request"] = opts => { bodies.push(JSON.parse(opts.body)); return reply(200, {parsed: n++ ? FETCH : ASK}); };
    await submit("I dropped something");
    await submit("my phone");
    assert.deepStrictEqual(bodies[1].dialog, [{question: "What did you drop?", answer: "my phone"}]);
    assert.strictEqual(bodies[1].text, "I dropped something");
  },
  async stop_clears_the_conversation() {
    routes["/request"] = opts => { bodies.push(JSON.parse(opts.body)); return reply(200, {parsed: ASK}); };
    routes["/stop"] = () => reply(200, {stopped: true});
    await submit("I dropped something");
    assert.strictEqual(el("question").hidden, false);
    clickStop();
    await sleep(20);
    assert.strictEqual(el("question").hidden, true);
    assert.deepStrictEqual(options(), []);
    assert.strictEqual(count("/stop"), 1);
    await submit("my pen");                  // starts afresh, not as an answer
    assert.deepStrictEqual(bodies[1], {text: "my pen", hand: "standing", clarify: true, dialog: []});
    await submit("stop");                    // typed stop words clear it too
    await sleep(20);
    assert.strictEqual(el("question").hidden, true);
    assert.strictEqual(count("/stop"), 2);
  },
  async cancel_clears_the_conversation_without_stopping() {
    routes["/request"] = opts => { bodies.push(JSON.parse(opts.body)); return reply(200, {parsed: ASK}); };
    await submit("I dropped something");
    el("cancel").listeners.click();
    assert.strictEqual(el("question").hidden, true);
    assert.strictEqual(count("/stop"), 0);
    await submit("my pen");
    assert.deepStrictEqual(bodies[1].dialog, []);
  },
  async clarify_setting_off_sends_no_dialog_and_hand_standing() {
    el("clarify").checked = false;
    routes["/request"] = opts => { bodies.push(JSON.parse(opts.body)); return reply(200, {parsed: {action: "fetch", item: "pen"}}); };
    await submit("my pen fell");
    assert.deepStrictEqual(bodies, [{text: "my pen fell", hand: "standing"}]);
    assert.strictEqual(el("reply").textContent, "Fetching: pen");
  },
  async only_one_status_request_at_a_time() {
    await sleep(10);
    const before = count("/status");
    let release;
    routes["/status"] = () => new Promise(r => { release = () => r(reply(200, STATUS)); });
    refresh(); refresh(); refresh();
    await sleep(10);
    assert.strictEqual(count("/status"), before + 1);
    release();
    await sleep(10);
    refresh();
    await sleep(10);
    assert.strictEqual(count("/status"), before + 2);
  },
  async hung_status_times_out_and_shows_no_connection() {
    await sleep(10);
    routes["/status"] = opts => hang(opts.signal);
    refresh();
    await sleep(1800);
    assert.ok(classes.has("stale"), "page marked as no connection");
    refresh();
    await sleep(10);
    assert.ok(count("/status") >= 3, "a new /status is sent once the hung one timed out");
  },
  async lift_and_tray_rows_are_hidden_when_null() {
    routes["/status"] = () => reply(200, {...STATUS, state: "Pick", item: "phone"});
    await poll();
    assert.strictEqual(el("liftrow").hidden, true, "no lift row without the lift");
    assert.strictEqual(el("trayrow").hidden, true, "no tray row without the lift");
    assert.strictEqual(el("lift").textContent, "", "no dash in the hidden row");
    assert.strictEqual(el("tray").textContent, "");
  },
  async lift_and_tray_rows_show_when_there_are_numbers() {
    routes["/status"] = () => reply(200, {...STATUS, lift_mm: 650.4, tray_g: 180});
    await poll();
    assert.strictEqual(el("liftrow").hidden, false);
    assert.strictEqual(el("trayrow").hidden, false);
    assert.strictEqual(el("lift").textContent, "650 mm");
    assert.strictEqual(el("tray").textContent, "180 g");
    routes["/status"] = () => reply(200, STATUS);          // the lift goes away again
    await poll();
    assert.strictEqual(el("liftrow").hidden, true);
    assert.strictEqual(el("trayrow").hidden, true);
  },
  async wait_for_take_is_prominent() {
    routes["/status"] = () => reply(200, {...STATUS, state: "WaitForTake", item: "pill_bottle",
                                          message: "Take your pill bottle from the claw"});
    await poll();
    assert.strictEqual(el("state").textContent, "Take your pill bottle");
    assert.strictEqual(el("message").textContent, "Take your pill bottle from the claw");
    assert.strictEqual(el("statecard").dataset.state, "WaitForTake", "the card styles itself from this");
    assert.strictEqual(el("step-WaitForTake").attrs["data-step"], "now");
  },
  async handoff_steps_follow_the_state() {
    const marks = () => ["Search", "Approach", "Pick", "Present", "WaitForTake"]
      .map(n => el("step-" + n).attrs["data-step"]).join(" ");
    routes["/status"] = () => reply(200, {...STATUS, state: "Present", item: "phone"});
    await poll();
    assert.strictEqual(el("state").textContent, "Holding it up");
    assert.strictEqual(marks(), "done done done now todo");
    routes["/status"] = () => reply(200, {...STATUS, state: "Idle"});
    await poll();
    assert.strictEqual(el("state").textContent, "Idle");
    assert.strictEqual(marks(), "todo todo todo todo todo");
    routes["/status"] = () => reply(200, {...STATUS, state: "Stopped", message: "Stopped"});
    await poll();
    assert.strictEqual(el("state").textContent, "Stopped");
  },
};

(async () => {
  try {
    await scenarios[scenario]();
    console.log("OK");
    process.exit(0);
  } catch (e) {
    console.error((e && e.stack) || e);
    process.exit(1);
  }
})();
