"""Repeatable prompt-cache workloads that drive a Strata server and produce trace logs.

Each scenario is a fixed sequence of chat turns chosen to exercise one prompt-cache
behavior: shared-prefix reuse, a last-message edit, branch switching, the periodic
checkpoint gap, pinned siblings, and branch pressure.  Run them against a server
started with STRATA_PROMPT_CACHE_TRACE=1; the engine's stderr "prompt cache:" lines
are what tools/analyze_prompt_cache_log.py summarizes.

    python tools/prompt_cache_workloads.py                       # dry run: print the plan
    python tools/prompt_cache_workloads.py --list
    python tools/prompt_cache_workloads.py --run --url http://127.0.0.1:11434
    python tools/prompt_cache_workloads.py --emit-trace out.log  # synthetic trace, no server
    python tools/prompt_cache_workloads.py --scenario long_document_many_questions --run
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


TRACE_PREFIX = "strata serve: prompt cache:"
REPLY_PREFIX = "{{reply:"


def _tokens(text: str) -> int:
    """A stable stand-in for the tokenizer's count, only used for the plan and the synthetic trace."""
    return max(1, len(text) // 4)


def _turn(messages, strata_prefix=None):
    return {"messages": list(messages), "strata_prefix": strata_prefix}


def _reply(request: int) -> str:
    return REPLY_PREFIX + str(request) + "}}"


def _document(marker: str, lines: int) -> str:
    body = "\n".join(f"{marker} record {i}: blue square, green triangle, red circle." for i in range(lines))
    return f"{marker} document.\n{body}\n"


def long_document_many_questions() -> list[dict]:
    """One long document, then questions that append to the same conversation: the shared prefix only grows."""
    doc = _document("ldq", 120)
    turns = [_turn([{"role": "user", "content": doc + "Question 0: name one color. Answer in one word."}])]
    for q in range(1, 5):
        messages = [dict(m) for m in turns[-1]["messages"]]   # keep the whole history, then append
        messages.append({"role": "assistant", "content": _reply(q)})
        messages.append({"role": "user", "content": f"Question {q}: name one color. Answer in one word."})
        turns.append(_turn(messages))
    return turns


def edited_last_message() -> list[dict]:
    """A conversation, then the same conversation with only the last user message changed: reuse up to the edit."""
    doc = _document("edit", 80)
    first = [{"role": "user", "content": doc + "Question: name one shape. Answer in one word."}]
    base = [dict(first[0]), {"role": "assistant", "content": _reply(1)},
            {"role": "user", "content": "Question: name one color. Answer in one word."}]
    edited = [dict(m) for m in base]
    edited[-1] = {"role": "user", "content": "Question: name one color. Answer in two words."}
    return [_turn(first), _turn(base), _turn(edited)]


def branch_switching() -> list[dict]:
    """From one shared document, alternate two different continuations: each switch restores a parked branch."""
    doc = _document("branch", 90)
    common = [{"role": "system", "content": doc}]
    a0 = [*common, {"role": "user", "content": "Branch A step 0: name one color. One word."}]
    b0 = [*common, {"role": "user", "content": "Branch B step 0: name one color. One word."}]
    a1 = [*a0, {"role": "assistant", "content": _reply(1)},
          {"role": "user", "content": "Branch A step 1: name one shape. One word."}]
    b1 = [*b0, {"role": "assistant", "content": _reply(2)},
          {"role": "user", "content": "Branch B step 1: name one shape. One word."}]
    a2 = [*a1, {"role": "assistant", "content": _reply(3)},
          {"role": "user", "content": "Branch A step 2: name one color. One word."}]
    b2 = [*b1, {"role": "assistant", "content": _reply(4)},
          {"role": "user", "content": "Branch B step 2: name one color. One word."}]
    return [_turn(messages) for messages in (a0, b0, a1, b1, a2, b2)]


def periodic_checkpoint_gap() -> list[dict]:
    """A long document whose follow-up shares a prefix that ends between periodic checkpoints: a reread gap."""
    doc = _document("gap", 200)
    turns = [_turn([{"role": "user", "content": doc + "Read the whole document and wait."}])]
    turns.append(_turn([{"role": "user", "content": doc + "Read the whole document and wait."},
                        {"role": "assistant", "content": _reply(1)},
                        {"role": "user", "content": "Name one color. Answer in one word."}]))
    return turns


def pin_siblings() -> list[dict]:
    """Several sibling queries that share the first message as a pinned prefix (strata_prefix -> engine pin=N)."""
    doc = _document("pin", 100)
    turns = []
    for q in range(4):
        messages = [{"role": "system", "content": doc},
                    {"role": "user", "content": f"Sibling {q}: name one color. Answer in one word."}]
        turns.append(_turn(messages, {"messages": 1}))
    return turns


def branch_pressure() -> list[dict]:
    """More distinct branches than the cache holds: forces parked-branch eviction. Needs a small slot count."""
    turns = []
    for b in range(8):
        doc = _document(f"pressure{b}", 60)
        turns.append(_turn([{"role": "system", "content": doc},
                            {"role": "user", "content": "Name one color. Answer in one word."}]))
    first = [dict(m) for m in turns[0]["messages"]]
    first.extend([{"role": "assistant", "content": _reply(1)},
                  {"role": "user", "content": "Now name one shape. Answer in one word."}])
    turns.append(_turn(first))
    return turns


SCENARIOS = {
    "long_document_many_questions": long_document_many_questions,
    "edited_last_message": edited_last_message,
    "branch_switching": branch_switching,
    "periodic_checkpoint_gap": periodic_checkpoint_gap,
    "pin_siblings": pin_siblings,
    "branch_pressure": branch_pressure,
}


def _prompt_tokens(turn: dict) -> int:
    return sum(_tokens(m["content"]) for m in turn["messages"])


def _shared_prefix_tokens(previous: dict, current: dict) -> int:
    """Tokens the two turns share from the start: whole messages, then a partial last message."""
    shared = 0
    for a, b in zip(previous["messages"], current["messages"]):
        if a["role"] != b["role"]:
            break
        if a["content"] == b["content"]:
            shared += _tokens(a["content"])
            continue
        common = 0
        for ca, cb in zip(a["content"], b["content"]):
            if ca != cb:
                break
            common += 1
        shared += _tokens(a["content"][:common])
        break
    return shared


def synthetic_trace(name: str, turns: list[dict]) -> list[str]:
    """Representative prompt-cache trace lines for a scenario, in the engine's format, for offline analysis."""
    lines = [f"# synthetic prompt-cache trace for scenario {name} (not from a live engine)"]
    for req, turn in enumerate(turns, 1):
        prompt = _prompt_tokens(turn)
        resume = _shared_prefix_tokens(turns[req - 2], turn) if req > 1 else 0
        if resume >= prompt:
            resume = max(0, prompt - 1)
        parked = resume + 1 if resume else 0
        source = "checkpoint" if resume else "none"
        lines.append(f"{TRACE_PREFIX} decision req={req} prompt={prompt} live=0 checkpoint={resume} slot=0 "
                     f"parked={parked} source={source} resume={resume} read_from={resume} scan_entries=1 "
                     f"scan_checkpoints={1 if resume else 0} scan_ms={0.5 * req:.1f} cvec_match=1 images=0 "
                     f"pin={1 if turn['strata_prefix'] else 0} ckpt=1")
        if resume:
            lines.append(f"{TRACE_PREFIX} checkpoint req={req} kind=turn tokens={resume} save_ms={1.0 * req:.1f} "
                         f"kept=2 evicted_kind=none evicted_tokens=0")
        if resume:
            lines.append(f"{TRACE_PREFIX} park req={req} tokens={resume} estimate_bytes={resume * 512} "
                         f"fresh_estimate_bytes={resume * 512} retained_bytes=0 additional_bytes={resume * 512} "
                         f"held_bytes=0 save_ms={2.0 * req:.1f} stored=1")
            lines.append(f"{TRACE_PREFIX} restore req={req} tokens={resume} source=parked/checkpoint "
                         f"restore_ms={3.0 * req:.1f} bytes={resume * 512}")
    return lines


class Client:
    def __init__(self, url: str, model: str = "strata"):
        self.url = url.rstrip("/")
        self.model = model

    def chat(self, turn: dict, max_tokens: int = 16):
        body = {"model": self.model, "messages": turn["messages"], "max_tokens": max_tokens,
                "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        if turn.get("strata_prefix"):
            body["strata_prefix"] = turn["strata_prefix"]
        request = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)


def _resolve_turn(turn: dict, replies: list[str]) -> dict:
    resolved = _turn([dict(message) for message in turn["messages"]], turn.get("strata_prefix"))
    for message in resolved["messages"]:
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith(REPLY_PREFIX) or not content.endswith("}}"):
            continue
        number = content[len(REPLY_PREFIX):-2]
        if not number.isdigit() or not 1 <= int(number) <= len(replies):
            raise RuntimeError(f"unresolved workload reply placeholder: {content}")
        message["content"] = replies[int(number) - 1]
    return resolved


def run_scenario(client: Client, name: str, turns: list[dict]) -> int:
    replies = []
    for req, turn in enumerate(turns, 1):
        result = client.chat(_resolve_turn(turn, replies))
        if not result.get("choices"):
            raise RuntimeError(f"{name}: request {req} returned no completion")
        content = result["choices"][0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"{name}: request {req} returned no text completion")
        replies.append(content)
    return len(turns)


def print_plan(name: str, turns: list[dict]) -> None:
    print(f"scenario {name}: {len(turns)} turns")
    for req, turn in enumerate(turns, 1):
        shared = _shared_prefix_tokens(turns[req - 2], turn) if req > 1 else 0
        prefix = " pin" if turn.get("strata_prefix") else ""
        print(f"  turn {req}: prompt~{_prompt_tokens(turn)} tokens, shared~{shared}{prefix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list scenario names and exit")
    parser.add_argument("--scenario", help="run only this scenario (default: all)")
    parser.add_argument("--run", action="store_true", help="send the turns to a live server")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="strata")
    parser.add_argument("--emit-trace", type=Path, metavar="PATH", help="write a synthetic trace log and exit")
    args = parser.parse_args(argv)

    names = [args.scenario] if args.scenario else list(SCENARIOS)
    for name in names:
        if name not in SCENARIOS:
            parser.error(f"unknown scenario {name!r}; use --list")

    if args.list:
        for name in SCENARIOS:
            print(name)
        return 0

    if args.emit_trace:
        lines = []
        for name in names:
            lines.extend(synthetic_trace(name, SCENARIOS[name]()))
        args.emit_trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {len(lines)} synthetic trace lines to {args.emit_trace}")
        return 0

    client = Client(args.url, args.model) if args.run else None
    for name in names:
        turns = SCENARIOS[name]()
        if client is None:
            print_plan(name, turns)
        else:
            sent = run_scenario(client, name, turns)
            print(f"scenario {name}: sent {sent} turns to {client.url}")
    if client is None:
        print("\nDry run. Start the server with STRATA_PROMPT_CACHE_TRACE=1, then add --run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
