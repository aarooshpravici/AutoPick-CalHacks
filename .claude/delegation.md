# SW1 delegation protocol (main Claude session only)

Goal: finish SW1's tasks without running out of weekly usage. The main session plans, decides and reviews. Cheaper agents do the bulk reading, writing and test runs, and hand back short results.

## Before every task

1. Make sure the board watcher is running (below).
2. Post a delegation plan in chat, at most 6 lines: the task id and its "done when" check, the route, what goes to which model, and what you review yourself. Then start; don't wait for an answer unless the plan needs the person.

| Route | Use when | How |
|---|---|---|
| **Direct** | A handful of tool calls, or the person has to act (calibration, teaching, `sudo`, Enter prompts, anything that moves hardware) | Main session, no agents |
| **Scout** | Facts from docs, logs, the web, or large files | One Agent call with `model: "haiku"`, asking for an answer of at most 150 words |
| **Workflow** | Writing a module and its tests | The `sw1-task` workflow (below) |

Keep these in the main session: section 9 interfaces, any code that commands real motion (speed caps, dry-run defaults, STOP paths), the safety review, instructions to the person, and board replies.

## Context hygiene

- Never read all of `plan.md` (about 27k tokens). Find the section with `grep -n "^### 10.1" plan.md`, then read just that range.
- Brief agents with file paths and plan section numbers, not pasted file contents.
- Ask agents for summaries, not transcripts or full logs.
- Don't re-read a file you just wrote, or one an agent has already summarised.
- Run long jobs (installs, training, firmware flashes) in the background. Don't poll them.

## Board watcher

Replaces the `/loop 15m` in plan section 15. Start it in the background at the start of each session:

    python3 .claude/board_watch.py

It polls issue #2 every 3 minutes using `gh`, which costs no tokens. It exits, waking the main session, only when a REQ addressed to `claude` has no `RE` reply. Handle it as `CLAUDE.md` says, then restart the watcher. To set aside a request the person has deferred, add `--skip REQ-codex-3`. For a one-off check, add `--once`.

## The sw1-task workflow

Create the task branch first, then run the workflow `sw1-task` with args:

    {"task": "A2",
     "brief": "what to build, which files and folders, the interface to match, the done-when check",
     "testCmd": "python -m pytest tests/test_pick.py -q",
     "plan_sections": ["9.4", "10.1"]}

Sonnet implements. Haiku runs `testCmd`. Sonnet fixes failures, for up to `max_fix_rounds` rounds (default 2), then reviews the diff. That's at most 7 agents. The workflow edits the working tree but doesn't commit. Read its summary, read the diff yourself only for code that moves hardware, then commit and open the PR.
