"""Request parser (task U1, plan 9.6).

    parse_request("I dropped my phone")  # -> {"action": "fetch", "item": "phone", "via": "claude"}
    parse_request("stop!")               # -> {"action": "stop", "item": None, "via": "keyword"}

1. STOP check first: "stop", "halt" or "freeze" anywhere returns stop, before any
   network call.
2. Otherwise ask Claude (claude-opus-5-5) with structured outputs: `item` is an
   enum of robot.yaml `items` plus "none", so it can only name a listed item.
3. If the API fails (no key, network, timeout, refusal), plain keyword matching.

`action` is "fetch", "stop" or "none" (nothing we can fetch); `item` is None
unless fetching. `via` says which path answered. The API key comes from the
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
DEFAULT_ITEMS = ["phone", "glasses", "pill_bottle", "remote", "pen", "sock"]   # plan 9.1

MODEL = "claude-opus-5-5"
TIMEOUT_S = 10.0
MAX_RETRIES = 1

STOP_WORDS = re.compile(r"\b(stop+|halt|freeze)\b", re.IGNORECASE)

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
    if config_path.exists():
        import yaml

        with open(config_path) as f:
            items = (yaml.safe_load(f) or {}).get("items")
        if items:
            return list(items)
    return list(DEFAULT_ITEMS)


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
        return {"action": "fetch", "item": best[0], "via": "keyword"}
    return {"action": "none", "item": None, "via": "keyword"}


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
    """One structured-output call. Raises on any API problem; parse_request falls back."""
    client = client or _get_client()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=1024,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _schema(items)}},
        messages=[{"role": "user", "content": f"Known items: {', '.join(items)}\n\nRequest: {text}"}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request")
    answer = json.loads(next(b.text for b in response.content if b.type == "text"))
    action, item = answer["action"], answer["item"]
    if action == "fetch" and item in items:
        return {"action": "fetch", "item": item, "via": "claude"}
    if action == "stop":
        return {"action": "stop", "item": None, "via": "claude"}
    return {"action": "none", "item": None, "via": "claude"}


def parse_request(text: str, items: list[str] | None = None, client=None, use_api: bool = True) -> dict:
    if is_stop(text):
        return {"action": "stop", "item": None, "via": "keyword"}
    items = items or load_items()
    if use_api:
        import anthropic

        try:
            return ask_claude(text, items, client)
        except anthropic.AuthenticationError:
            log.warning("request parser: API key invalid; using keywords")
        except TypeError:       # the SDK's "Could not resolve authentication method"
            log.warning("request parser: no API key (ANTHROPIC_API_KEY); using keywords")
        except anthropic.RateLimitError:
            log.warning("request parser: rate limited; using keywords")
        except anthropic.APITimeoutError:
            log.warning("request parser: API timed out; using keywords")
        except anthropic.APIConnectionError:
            log.warning("request parser: no network; using keywords")
        except anthropic.APIStatusError as e:
            log.warning("request parser: API error %s; using keywords", e.status_code)
        except (anthropic.AnthropicError, RuntimeError, ValueError, KeyError, StopIteration) as e:
            log.warning("request parser: %s; using keywords", type(e).__name__)
    return keyword_match(text, items)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(json.dumps(parse_request(" ".join(sys.argv[1:]))))
