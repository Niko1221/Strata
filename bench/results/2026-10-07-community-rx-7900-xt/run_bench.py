"""Community speed runs against a local Strata server.

Reads prompt and decode rates from GET /v1/status last_timings (the engine
clock). Does not divide generated tokens by the whole request time.

Warm-up requests are stored and excluded from the summary table.
Each user message starts with a fresh run id so the prefix is not reused.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8080"
OUT = Path(__file__).resolve().parent
MAX_TOKENS = 256
TIMEOUT_S = 900
PASSPHRASE = "amber-keel-2904"
SHORT = "Write a short story about a lighthouse keeper."


def long_prompt() -> str:
    """1100 log lines, fixed seed. The passphrase sits on line 550, not at the end."""
    import random

    rng = random.Random(42)
    status = ("OK", "OK", "OK", "RETRY", "SLOW")
    lines = []
    for i in range(1, 1101):
        if i == 550:
            lines.append(f"Entry 550: NOTE the deployment passphrase is {PASSPHRASE}.")
            continue
        lines.append(
            f"Entry {i}: service-{rng.randrange(1, 50)} handled request "
            f"{rng.randrange(100000, 999999)} in {rng.randrange(5, 900)} ms "
            f"with status {status[rng.randrange(0, 5)]} on node-{rng.randrange(1, 20)}."
        )
    lines.append(
        "Question: What is the deployment passphrase mentioned in the log above? "
        "Quote it exactly, then in one sentence say what the log mostly shows."
    )
    return "\n".join(lines)


def post_json(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(path: str) -> dict:
    req = urllib.request.Request(BASE + path, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def one(label: str, text: str, warmup: bool, model: str) -> dict:
    run_id = str(uuid.uuid4())
    content = f"Run id: {run_id}\n{text}"
    body = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": content}],
    }
    started = time.perf_counter()
    try:
        chat = post_json("/v1/chat/completions", body)
        error = None
    except urllib.error.HTTPError as exc:
        chat = {"error": exc.read().decode("utf-8", errors="replace")[:2000]}
        error = f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 — record the failure and keep the row
        chat = {}
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    status = get_json("/v1/status")
    timings = status.get("last_timings") or {}
    choice = ((chat.get("choices") or [{}])[0]) if isinstance(chat, dict) else {}
    message = choice.get("message") or {}
    answer = message.get("content") or ""
    draft_n = timings.get("draft_n")
    draft_acc = timings.get("draft_n_accepted")
    row = {
        "label": label,
        "warmup": warmup,
        "run_id": run_id,
        "error": error,
        "finish_reason": choice.get("finish_reason"),
        "answer_has_passphrase": PASSPHRASE in answer if label.startswith("long") else None,
        "answer_preview": answer[:400],
        "wall_s": round(elapsed, 1),
        "ram_used_gib": (status.get("machine") or {}).get("ram", {}).get("used_gib"),
        "prompt_n": timings.get("prompt_n"),
        "cache_n": timings.get("cache_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "draft_n": draft_n,
        "draft_n_accepted": draft_acc,
        "last_timings": timings,
    }
    print(
        f"{label} warmup={warmup} err={error} prompt_n={row['prompt_n']} "
        f"cache_n={row['cache_n']} prompt_tps={row['prompt_per_second']} "
        f"out={row['predicted_n']} decode_tps={row['predicted_per_second']} "
        f"wall={row['wall_s']}s",
        flush=True,
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.8-flash-next-iq3_s")
    ap.add_argument("--out", default="runs.json")
    args = ap.parse_args()
    status = get_json("/v1/status")
    if status.get("model") != args.model:
        raise SystemExit(f"expected {args.model}, server says {status.get('model')!r}")
    long_text = long_prompt()
    plan = [
        ("short", SHORT, True),
        ("long", long_text, True),
        ("short", SHORT, False),
        ("short", SHORT, False),
        ("short", SHORT, False),
        ("long", long_text, False),
        ("long", long_text, False),
        ("long", long_text, False),
    ]
    rows = []
    path = OUT / args.out
    for label, text, warmup in plan:
        rows.append(one(label, text, warmup, args.model))
        path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        if rows[-1]["error"]:
            raise SystemExit(f"stopped after {label}: {rows[-1]['error']}")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
