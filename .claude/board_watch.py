#!/usr/bin/env python3
"""Wait for team-board requests addressed to claude (plan.md section 15).

Polls issue #2 with gh and exits when a REQ addressed to claude has no
"RE <id>" reply yet, printing it. Run it in the background so the main
session is only woken when there is something to answer.

    python3 .claude/board_watch.py [--every 180] [--skip REQ-codex-1,...] [--once]
"""
import argparse
import json
import re
import subprocess
import sys
import time

REPO = "aarooshpravici/AutoPick-CalHacks"
NAME = "claude"
MAX_FAILURES = 5

REQ = re.compile(r"\b(REQ-[A-Za-z0-9]+-\d+)\s*(?:→|->)\s*([A-Za-z0-9_-]+)")
# A reply is a line that starts with "RE REQ-…", so a request quoting the
# reply format ("reply with `RE REQ-claude-1`") doesn't count as answered.
REPLY = re.compile(r"^\s*RE\s+(REQ-[A-Za-z0-9]+-\d+)", re.MULTILINE)


def pending_requests(comments, name=NAME, skip=()):
    """Return (req_id, comment) for each REQ to `name` with no RE reply."""
    answered = {rid for c in comments for rid in REPLY.findall(c["body"])}
    pending = []
    for c in comments:
        for rid, to in REQ.findall(c["body"]):
            if to.lower() == name and rid not in answered and rid not in skip:
                pending.append((rid, c))
    return pending


def fetch_comments():
    out = subprocess.run(
        ["gh", "api", f"repos/{REPO}/issues/2/comments", "--paginate",
         "--jq", ".[] | {body, user: .user.login, url: .html_url}"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--every", type=int, default=180, help="seconds between polls")
    p.add_argument("--skip", default="", help="comma-separated REQ ids to ignore")
    p.add_argument("--once", action="store_true", help="check once and exit")
    args = p.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    failures = 0
    while True:
        try:
            pending = pending_requests(fetch_comments(), skip=skip)
            failures = 0
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            failures += 1
            print(f"board_watch: poll failed ({failures}/{MAX_FAILURES}): {e}", file=sys.stderr)
            if failures >= MAX_FAILURES:
                print("board_watch: giving up; check gh auth and the network")
                sys.exit(1)
            pending = []
        if pending:
            for rid, c in pending:
                print(f"{rid} from @{c['user']} ({c['url']}):\n{c['body'][:800]}\n")
            return
        if args.once:
            print("No unanswered requests for claude.")
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
