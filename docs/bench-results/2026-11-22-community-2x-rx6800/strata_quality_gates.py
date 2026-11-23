#!/usr/bin/env python3
"""Quality gates for a Strata model arm (E6 IQ2_XS) — the host.

Three gates via /v1/chat/completions, temperature 0, plus capture of the raw
answers for the operator's adoption decision:
  1. needle retrieval  (~12K-token haystack, needle at ~60% depth)
  2. one coding prompt (bracket-balance function; output captured, not executed)
  3. one vision prompt (in-memory PNG: red disc on blue background)

The API key is resolved silently (env -> ~/.hermes/.env) and never printed.
Usage: python3 strata_quality_gates.py [--model qwen3.8-flash-next-iq2_xs]
"""
import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

URL = os.environ.get("STRATA_URL", "http://the host:8090/v1/chat/completions")
NEEDLE = "ZEBRA-THIRTEEN-SEVEN"


def resolve_key() -> str:
    key = os.environ.get("STRATA_API_KEY", "").strip()
    if not key:
        env = Path.home() / ".hermes/.env"
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("STRATA_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    return key


def chat(key: str, model: str, content, max_tokens: int, timeout: int = 600):
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    dt = time.perf_counter() - t0
    msg = data["choices"][0]["message"]
    txt = msg.get("content") or ""
    if not txt.strip():
        txt = "(empty content; full message: " + json.dumps(msg)[:300] + ")"
    usage = data.get("usage", {})
    return txt, dt, usage


def haystack() -> str:
    lines = []
    for i in range(1, 401):
        lines.append(f"log line {i:04d}: routine sync of shard {i % 17} ok, "
                     f"latency {(i * 37) % 400} ms, retries 0, checksum pass")
    lines[239] = f"log line 0240: NOTE the coordinator's call sign for this window is {NEEDLE}, do not lose it"
    return "\n".join(lines)


def make_png() -> str:
    w = h = 256
    rows = bytearray()
    for y in range(h):
        rows.append(0)  # filter: none
        for x in range(w):
            if (x - 128) ** 2 + (y - 128) ** 2 <= 70 ** 2:
                rows += bytes((228, 26, 28))    # red disc
            else:
                rows += bytes((31, 78, 180))    # blue background

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("STRATA_MODEL", "qwen3.8-flash-next-iq2_xs"))
    a = ap.parse_args()
    key = resolve_key()
    if not key:
        sys.exit("no API key found")

    print(f"== model: {a.model}")
    ok = {}
    try:
        hs = haystack()
        txt, dt, usage = chat(key, a.model, hs + "\n\nQuestion: what is the coordinator's "
                              "call sign for this window? Answer with just the call sign.",
                              max_tokens=512)
        hit = NEEDLE.lower() in txt.lower()
        ok["needle"] = hit
        print(f"[1] needle (usage in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')} "
              f"cached={usage.get('prompt_tokens_details', {}).get('cached_tokens', 'n/a')}, {dt:.1f}s): "
              f"{'PASS' if hit else 'FAIL'} -> {txt.strip()[:200]!r}")
    except Exception as e:
        ok["needle"] = False
        print(f"[1] needle: ERROR {e}")

    try:
        txt, dt, usage = chat(key, a.model,
                              "Write a Python function is_balanced(s) that returns True if the "
                              "brackets ()[]{} in the string s are correctly balanced and nested. "
                              "Only the function, no prose.", max_tokens=400)
        good = "def is_balanced" in txt and ("return" in txt)
        ok["coding"] = good
        print(f"[2] coding ({dt:.1f}s, out={usage.get('completion_tokens')}): "
              f"{'PASS(plausible)' if good else 'CHECK'} ->\n{txt.strip()[:600]}")
    except Exception as e:
        ok["coding"] = False
        print(f"[2] coding: ERROR {e}")

    try:
        txt, dt, usage = chat(key, a.model, [
            {"type": "text", "text": "In one sentence: what shape and colors do you see?"},
            {"type": "image_url", "image_url": {"url": make_png()}},
        ], max_tokens=512)
        low = txt.lower()
        good = ("red" in low) and ("blue" in low) and any(s in low for s in ("circle", "disc", "dot"))
        ok["vision"] = good
        print(f"[3] vision ({dt:.1f}s): {'PASS' if good else 'CHECK'} -> {txt.strip()[:200]!r}")
    except Exception as e:
        ok["vision"] = False
        print(f"[3] vision: ERROR {e}")

    print("== gates:", " ".join(f"{k}={'PASS' if v else 'FAIL/ERROR'}" for k, v in ok.items()))
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
