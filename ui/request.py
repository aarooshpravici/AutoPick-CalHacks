"""Request parser (task U1, plan 9.6).

    parse_request("I dropped my phone")  # -> {"action": "fetch", "item": "phone"}
    parse_request("stop!")               # -> {"action": "stop", "item": None}
    parse_request("I dropped my keys")   # -> {"action": "none", "item": None}

The result always has exactly two keys. `action` is "fetch", "stop" or "none";
`item` is a robot.yaml item when fetching, otherwise None. "none" means the text
names no listed item. Plan 9.6 shows only fetch and stop; U2's page uses "none",
and SW1 and SW2 are agreeing it as an amendment to 9.6 on the board.

1. STOP check first: any word that starts with "stop" (or "stooop"), "halt" or
   "freez" returns stop, before any import, config read or network call. A false
   stop costs little; a missed one costs more.
2. Otherwise ask Claude (claude-opus-5-5, GA Messages API) with structured outputs:
   `item` is an enum of robot.yaml `items` plus "none", so it can only name a
   listed item.
3. If that fails (no key, network, timeout, refusal, unusable answer), plain
   keyword matching against the item list.

Which path answered is logged at INFO, not returned. The API key comes from the
environment (ANTHROPIC_API_KEY) and is never logged.

    python -m ui.request "my glasses fell"
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "robot.yaml"
# Plan 9.1's list, used only while config/robot.yaml does not exist (task S6).
DEFAULT_ITEMS = ["phone", "glasses", "pill_bottle", "remote", "pen", "sock"]

MODEL = "claude-opus-5-5"
MAX_TOKENS = 1024
TIMEOUT_S = 5.0     # the keyword fallback is cheap, so don't keep a person waiting
MAX_RETRIES = 0

# Word-start stems, so "stopping", "halted", "freezing" and "stoooop" all count. Two o's
# ("stoop") don't: this robot's users can't easily stoop, and will say so.
STOP_WORDS = re.compile(r"\b(?:stop|sto{3,}p|halt|freez)", re.IGNORECASE)

# Words people use for each item, for the offline fallback. Items not listed
# here match on their own name (underscores read as spaces).
SYNONYMS = {
    "phone": ["phone", "iphone", "cellphone", "cell", "mobile", "smartphone", "android"],
    "glasses": ["glasses", "spectacles", "specs", "eyeglasses", "reading glasses", "sunglasses"],
    "pill_bottle": ["pill bottle", "pills", "pill", "medicine", "medication", "meds", "tablets", "prescription"],
    "remote": ["remote", "remote control", "controller", "clicker", "tv remote"],
    "pen": ["pen", "biro", "ballpoint", "marker"],
    "sock": ["sock", "socks"],
}

SYSTEM = (
    "You read short requests spoken or typed to a home robot that picks things up off the floor "
    "for people who can't easily bend down. Decide whether the person wants an object fetched, "
    "and which of the robot's known items it is.\n"
    "- action \"fetch\": they want an item picked up or brought to them (\"I dropped my phone\", "
    "\"can you grab my specs\", \"my pills rolled under the chair\").\n"
    "- action \"stop\": they want the robot to stop, wait or cancel.\n"
    "- action \"none\": anything else, including objects the robot doesn't know. Then item is \"none\".\n"
    "Map everyday names to the closest known item (\"mobile\" is phone, \"meds\" is pill_bottle). "
    "If the object isn't one of the known items, answer item \"none\" rather than guessing."
)


def load_items(config_path: Path = CONFIG_PATH) -> list[str]:
    """The item enum from robot.yaml `items`. Plan 9.1's list only if the file is missing."""
    if not config_path.exists():
        log.warning("request parser: %s not found; using plan 9.1's items %s", config_path, DEFAULT_ITEMS)
        return list(DEFAULT_ITEMS)
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)
    items = config.get("items") if isinstance(config, dict) else None
    if not isinstance(items, list) or not items or not all(isinstance(i, str) and i for i in items):
        raise ValueError(f"{config_path}: `items` must be a non-empty list of item names")
    return list(items)


def is_stop(text: str) -> bool:
    return bool(STOP_WORDS.search(text))


def keyword_match(text: str, items: list[str]) -> dict:
    lowered = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
    best = None
    for item in items:
        for word in SYNONYMS.get(item, [item.replace("_", " ")]):
            if f" {word} " in lowered and (best is None or len(word) > len(best[1])):
                best = (item, word)           # longest match wins ("tv remote" over "remote")
    if best:
        return {"action": "fetch", "item": best[0]}
    return {"action": "none", "item": None}


def _schema(items: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["fetch", "stop", "none"]},
            "item": {"type": "string", "enum": items + ["none"]},
        },
        "required": ["action", "item"],
        "additionalProperties": False,
    }


_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=MAX_RETRIES)
    return _client


def ask_claude(text: str, items: list[str], client=None) -> dict:
    """One structured-output call on the GA Messages API (no betas, no fallbacks).

    Raises on any API problem or unusable answer; parse_request then falls back.
    """
    client = client or _get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _schema(items)}},
        messages=[{"role": "user", "content": f"Known items: {', '.join(items)}\n\nRequest: {text}"}],
    )
    if response.stop_reason != "end_turn":         # refusal, max_tokens (JSON cut short), ...
        raise RuntimeError(f"Claude stopped with {response.stop_reason!r}")
    texts = [b.text for b in response.content if b.type == "text"]
    if len(texts) != 1:
        raise ValueError(f"expected one text block, got {len(texts)}")
    answer = json.loads(texts[0])
    if not isinstance(answer, dict):
        raise ValueError(f"expected a JSON object, got {type(answer).__name__}")
    action, item = answer["action"], answer["item"]
    if action == "fetch" and item in items:
        return {"action": "fetch", "item": item}
    if action == "stop":
        return {"action": "stop", "item": None}
    return {"action": "none", "item": None}


def _try_claude(text: str, items: list[str], client=None) -> dict | None:
    """ask_claude, or None (with the reason logged) if the keyword fallback should answer."""
    try:
        import anthropic
    except ImportError:
        log.warning("request parser: anthropic package not installed; using keywords")
        return None
    try:
        return ask_claude(text, items, client)
    except anthropic.AuthenticationError:
        log.warning("request parser: API key rejected; using keywords")
    except anthropic.RateLimitError:
        log.warning("request parser: rate limited; using keywords")
    except anthropic.APITimeoutError:
        log.warning("request parser: API timed out; using keywords")
    except anthropic.APIConnectionError:
        log.warning("request parser: no network; using keywords")
    except anthropic.APIStatusError as e:
        log.warning("request parser: API error %s; using keywords", e.status_code)
    except anthropic.AnthropicError as e:
        log.warning("request parser: %s; using keywords", type(e).__name__)
    except TypeError as e:
        if "authentication method" in str(e):      # the SDK's "Could not resolve authentication method"
            log.warning("request parser: no API credentials (set ANTHROPIC_API_KEY); using keywords")
        else:                                      # a code or SDK-version bug, not a missing key
            log.error("request parser: bad SDK call (TypeError: %s); using keywords", e)
    except (RuntimeError, ValueError, KeyError) as e:
        log.warning("request parser: unusable answer (%s: %s); using keywords", type(e).__name__, e)
    return None


def parse_request(text: str, items: list[str] | None = None, client=None, use_api: bool = True) -> dict:
    """Plan 9.6. Returns exactly {"action", "item"}."""
    if is_stop(text):
        log.info("request parser: stop (keyword check)")
        return {"action": "stop", "item": None}
    items = items or load_items()
    result = _try_claude(text, items, client) if use_api else None
    via = "claude"
    if result is None:
        result, via = keyword_match(text, items), "keyword"
    log.info("request parser: %s %s (via %s)", result["action"], result["item"], via)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(json.dumps(parse_request(" ".join(sys.argv[1:]))))
