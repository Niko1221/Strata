#!/usr/bin/env python3
"""End-to-end L2 rotation benchmark.

Builds three synthetic conversations (two long, one short), rotates
between them to test cache reuse, restarts the engine to test disk
restore, and reports per-request reuse metrics.

Usage:
    python3 l2/bench_rotation.py --url http://127.0.0.1:8080 [--restart-cmd CMD]

Requirements:
    - Strata server with --slot-save-path and STRATA_SESSION_V2=1
    - The reference controller (l2/reference_controller.py) running,
      or use --direct to talk to the engine directly.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

PARAGRAPH = (
    "The layered cache architecture stores attention keys and values "
    "in page-aligned cells while boundary checkpoints preserve message "
    "edges so a diverging conversation can still borrow the deepest "
    "common prefix; the engine streams quantized tensors through the "
    "pinned host pool at memory bandwidth. "
)


def chat(url, messages, tag, timeout=600, api_key=None):
    """Send a chat request and return (latency_s, prompt_tokens, cached_tokens, answer)."""
    body = {
        "messages": messages,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result = json.loads(r.read())
        dt = time.time() - t0
        usage = result.get("usage", {})
        prompt = usage.get("prompt_tokens", 0)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        answer = (result.get("choices", [{}])[0]
                  .get("message", {})
                  .get("content", "")
                  .strip()[:20])
        reuse = cached / prompt if prompt > 1000 else 1.0
        warn = " ⚠LOW-REUSE" if reuse < 0.30 and prompt > 1000 else ""
        print(f"  [{tag}] {dt:.1f}s prompt={prompt} cached={cached} "
              f"reuse={reuse:.0%}{warn} ans={answer!r}")
        return dt, prompt, cached, answer
    except Exception as e:
        dt = time.time() - t0
        print(f"  [{tag}] FAILED {dt:.1f}s: {e}")
        return dt, 0, 0, None


def wait_engine(url, timeout_s=300):
    """Wait for the engine to become healthy."""
    for _ in range(timeout_s // 5):
        try:
            urllib.request.urlopen(url.rstrip("/") + "/health", timeout=3).read()
            return True
        except Exception:
            time.sleep(5)
    return False


def percentile(values, p):
    if not values:
        return 0
    s = sorted(values)
    idx = min(len(s) - 1, int(len(s) * p))
    return s[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080",
                        help="Engine URL (direct) or controller URL")
    parser.add_argument("--api-key", default=os.environ.get("STRATA_TEST_KEY", ""),
                        help="API key if authentication is enabled")
    parser.add_argument("--restart-cmd", default="",
                        help="Command to restart the engine (e.g. 'systemctl restart my-engine')")
    parser.add_argument("--turns", type=int, default=5,
                        help="Number of build turns per long conversation")
    parser.add_argument("--words-per-turn", type=int, default=300,
                        help="Approximate words per user turn")
    args = parser.parse_args()

    results = []

    def track(tag, dt, prompt, cached):
        results.append({"tag": tag, "seconds": dt, "prompt": prompt,
                        "cached": cached,
                        "reuse": cached / prompt if prompt > 0 else 0})

    # === Phase 1: Build three conversations ===
    print("=== Phase 1: Build conversations ===")
    histories = {}
    for name, turns, words in [
        ("LongA", args.turns, args.words_per_turn),
        ("Short", 2, args.words_per_turn // 4),
        ("LongB", args.turns, args.words_per_turn),
    ]:
        system = f"You are assistant {name}. " + PARAGRAPH * 5
        msgs = [{"role": "system", "content": system}]
        for i in range(turns):
            user_text = PARAGRAPH * (words // 25) + f" (turn {i+1}, answer in one word)"
            msgs.append({"role": "user", "content": user_text})
            dt, pt, ct, ans = chat(args.url, msgs, f"{name}-b{i+1}",
                                   api_key=args.api_key)
            track(f"{name}-b{i+1}", dt, pt, ct)
            if ans is None:
                print(f"  !! {name} interrupted at turn {i+1}")
                break
            msgs.append({"role": "assistant", "content": ans})
        histories[name] = msgs

    # === Phase 2: Rotate between conversations ===
    print("\n=== Phase 2: Rotation (cold → warm) ===")
    for round_num in range(2):
        for name in ("LongA", "Short", "LongB"):
            msgs = histories[name]
            msgs.append({"role": "user",
                        "content": f"rotation {round_num+1}: answer one word"})
            dt, pt, ct, ans = chat(args.url, msgs, f"{name}-r{round_num+1}",
                                   api_key=args.api_key)
            track(f"{name}-r{round_num+1}", dt, pt, ct)
            if ans:
                msgs.append({"role": "assistant", "content": ans})

    # === Phase 3: Fork borrow (small conversation shares prefix with LongA) ===
    print("\n=== Phase 3: Fork borrow ===")
    fork = histories["LongA"][:3].copy()  # system + first user/assistant pair
    fork.append({"role": "user",
                "content": "fork question after shared head: answer one word"})
    dt, pt, ct, _ = chat(args.url, fork, "Fork-borrow", api_key=args.api_key)
    track("Fork-borrow", dt, pt, ct)

    # === Phase 4: Restart and restore ===
    if args.restart_cmd:
        print(f"\n=== Phase 4: Engine restart ===")
        print(f"  Running: {args.restart_cmd}")
        subprocess.run(args.restart_cmd, shell=True, check=False)
        if not wait_engine(args.url):
            print("  !! Engine did not recover")
            sys.exit(1)
        print("  Engine recovered")

        print("\n=== Phase 5: Post-restart restore ===")
        for name in ("LongA", "Short", "LongB"):
            msgs = histories[name]
            msgs.append({"role": "user",
                        "content": f"after restart: answer one word"})
            dt, pt, ct, ans = chat(args.url, msgs, f"{name}-restart",
                                   api_key=args.api_key)
            track(f"{name}-restart", dt, pt, ct)
            if ans:
                msgs.append({"role": "assistant", "content": ans})
    else:
        print("\n  (skipping restart phase — no --restart-cmd given)")

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"=== Summary: {len(results)} requests ===")

    low_reuse = [r for r in results if r["reuse"] < 0.30 and r["prompt"] > 1000]
    all_reuse = [r["reuse"] for r in results if r["prompt"] > 1000]
    all_latency = [r["seconds"] for r in results]

    print(f"  Total requests:       {len(results)}")
    print(f"  Low-reuse (<30%):     {len(low_reuse)}")
    if all_reuse:
        print(f"  Reuse median:         {percentile(all_reuse, 0.5):.0%}")
        print(f"  Reuse P90:            {percentile(all_reuse, 0.9):.0%}")
    print(f"  Latency P50:          {percentile(all_latency, 0.5):.1f}s")
    print(f"  Latency P90:          {percentile(all_latency, 0.9):.1f}s")
    print(f"  Failures:             {sum(1 for r in results if r['prompt'] == 0)}")

    print(f"\n{'='*60}")
    print("Per-request details:")
    for r in results:
        print(f"  {r['tag']:25s} {r['seconds']:6.1f}s  "
              f"prompt={r['prompt']:6d}  cached={r['cached']:6d}  "
              f"reuse={r['reuse']:.0%}")


if __name__ == "__main__":
    main()
