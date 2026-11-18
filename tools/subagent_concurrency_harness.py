#!/usr/bin/env python3
"""Does the main conversation keep working while subagent requests run beside it?

    python3 tools/subagent_concurrency_harness.py http://127.0.0.1:8180 --subagents 4

An agent harness (omp and friends) sends the main conversation's requests and its subagents' requests to one server at
the same time.  This tool measures what the server does with them:

  phase 1  one streamed request alone: its time to first token and its mean inter-token latency (the baseline);
  phase 2  the same request fired at the same moment as N subagent-style requests: the same two figures for the
           main request, and the wall time of the whole burst against the sum of the solo times.

Pass (exit 0): the main request starts within (N+1) x its solo start + 2 s (prompt reads are serial on the
main path, so it may queue behind the N subagents' reads but nothing more), keeps its tokens within
--max-itl-factor x its solo spacing, and the burst finishes in under --max-wall-ratio x the summed solo wall time
(closer to the longest single request than to running them one after another).  Fail (exit 1): the main request
queued behind the others or slowed down.  Exit 2: the server never answered (after the reconnect fail-safes), or
the idle preflight probe (a warm-up 1-token request that must answer within --probe-max-s) says another client -
an agent session pointed at this server - is using it: nothing else may talk to the endpoint during a run.

The default factors are the measured M1 Max iq3_xxs batch figures with room: a 5-row window costs 113-127 ms
per row against a 31.6-34 ms solo token (a per-row kernel long tail, largest class 6.6% of the window), so the
main stream runs at 3.6-4.0 x its solo token spacing inside a full batch and the burst wall is 0.79-0.84 x
summed solo.  A serial server lands at a wall ratio near 1.0 and is still caught.

Fail-safes: every phase retries up to --retries times; a dropped stream or a dead server waits for GET /health to
answer again (an engine restart takes a minute or two) before the retry, so a harness run survives the server being
restarted underneath it.  Standard library only.
"""
import argparse, json, statistics, sys, threading, time, urllib.error, urllib.request

SOLO_WORDS = 260          # the essay each request asks for; the token budget is what bounds the decode


def prompt_for(i: int) -> str:
    topics = [
        "the history of lighthouses and how their lenses worked",
        "how a mechanical watch keeps time",
        "why bridges are built as arches and as suspension spans",
        "the water cycle and what a drought does to a river valley",
        "how bread rises, from yeast to crust",
        "the design of a medieval castle and why walls were stepped",
        "how a transistor switches, explained for a curious teenager",
        "the life of a honey bee colony through one year",
    ]
    return (f"[harness {i}] Write a clear, well organised essay of about {SOLO_WORDS} words on {topics[i % len(topics)]}. "
            f"Use plain prose, no lists.")


class Infra(Exception):
    """The server did not answer; the harness cannot say anything about concurrency."""


def wait_healthy(url: str, deadline_s: float) -> bool:
    """GET /health until it answers or the deadline passes: an engine restart is not a harness failure."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(url + "/health", timeout=10) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3.0)
    return False


def stream(url: str, key: str, model: str, i: int, max_tokens: int) -> dict:
    """One streamed chat request: returns ttft (s to the first content token), itl (mean token spacing), tokens, wall."""
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt_for(i)}]}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + key} if key else {})})
    t0 = time.perf_counter()
    stamps, err = [], None
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    j = json.loads(d)
                except ValueError:
                    continue
                for ch in j.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        stamps.append(time.perf_counter())
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        err = f"{type(e).__name__}: {e}"
    if err is None and len(stamps) < 8:
        err = f"the stream carried only {len(stamps)} content tokens"
    if err is not None:
        raise Infra(err)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"ttft": stamps[0] - t0, "itl": statistics.mean(gaps), "tokens": len(stamps),
            "wall": stamps[-1] - t0}


def run_phase(url, key, model, n_sub, max_tokens, retries, label) -> list[dict]:
    """Fire 1 main + n_sub subagent requests at the same moment (n_sub=0: the main request alone), with retries."""
    for attempt in range(1, retries + 1):
        results: list[dict | None] = [None] * (n_sub + 1)
        def worker(k):
            try:
                results[k] = stream(url, key, model, k + 1, max_tokens)
            except Infra as e:
                print(f"[harness] {label} request {k}: {e}", file=sys.stderr, flush=True)
        threads = [threading.Thread(target=worker, args=(k,)) for k in range(n_sub + 1)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.monotonic() - t0
        if all(r is not None for r in results):
            return results, wall
        print(f"[harness] {label}: attempt {attempt}/{retries} failed; waiting for the server to come back ...",
              file=sys.stderr, flush=True)
        if not wait_healthy(url, 240.0):
            raise Infra(f"{label}: the server never came back after attempt {attempt}")
    raise Infra(f"{label}: {retries} attempts all failed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8180")
    ap.add_argument("--key", default="")
    ap.add_argument("--subagents", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8192)   # 2026-10-21: 400 was a token saver; it starves thinking
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--probe-max-s", type=float, default=15.0,
                    help="max seconds for the idle 1-token preflight probe (bail with exit 2 above it)")
    ap.add_argument("--max-itl-factor", type=float, default=4.5)
    ap.add_argument("--max-wall-ratio", type=float, default=0.9)
    ap.add_argument("--skip-baseline", action="store_true",
                    help="reuse the thresholds only (the baseline is measured in the same run by default)")
    a = ap.parse_args()

    if not wait_healthy(a.url, 30.0):
        print(f"[harness] {a.url} is not answering /health", file=sys.stderr)
        return 2
    try:
        with urllib.request.urlopen(a.url + "/v1/models", timeout=15) as r:
            model = json.load(r)["data"][0]["id"]
    except (urllib.error.URLError, OSError, KeyError, IndexError) as e:
        print(f"[harness] cannot read /v1/models: {e}", file=sys.stderr)
        return 2

    # preflight: one 1-token request doubles as warm-up and as a busy-detector - another client
    # (an agent session pointed at this server) queues in front of it, which would poison the baseline.
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(
            a.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps({"model": model, "max_tokens": 1, "stream": False,
                             "messages": [{"role": "user", "content": "Reply with the single word: ok"}]}).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {a.key}"} if a.key else {})})
        with urllib.request.urlopen(req, timeout=120) as r:
            json.load(r)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"[harness] preflight probe failed: {e}", file=sys.stderr)
        return 2
    probe = time.monotonic() - t0
    if probe > a.probe_max_s:
        print(f"[harness] server busy: an idle 1-token probe took {probe:.1f} s > {a.probe_max_s} s. "
              f"Nothing else may talk to {a.url} during the harness (move any agent session off this "
              f"endpoint and let running requests finish), or the measurements are not solo.",
              file=sys.stderr)
        return 2
    print(f"[harness] preflight: idle probe answered in {probe:.1f} s", flush=True)
    try:
        print(f"[harness] baseline: one request alone on {model} ...", flush=True)
        (solo,), solo_wall = run_phase(a.url, a.key, model, 0, a.max_tokens, a.retries, "baseline")
        print(f"[harness]   solo: ttft {solo['ttft']:.2f} s, itl {solo['itl'] * 1000:.1f} ms "
              f"({1 / solo['itl']:.1f} tok/s), {solo['tokens']} tokens, wall {solo_wall:.1f} s", flush=True)

        print(f"[harness] concurrent: 1 main + {a.subagents} subagent requests at once ...", flush=True)
        results, burst_wall = run_phase(a.url, a.key, model, a.subagents, a.max_tokens, a.retries, "concurrent")
        main_r = results[0]
        sum_solo = solo_wall * (a.subagents + 1)
        print(f"[harness]   main under load: ttft {main_r['ttft']:.2f} s, itl {main_r['itl'] * 1000:.1f} ms "
              f"({1 / main_r['itl']:.1f} tok/s), {main_r['tokens']} tokens", flush=True)
        for k, r in enumerate(results[1:], 1):
            print(f"[harness]   subagent {k}: ttft {r['ttft']:.2f} s, itl {r['itl'] * 1000:.1f} ms, "
                  f"{r['tokens']} tokens", flush=True)
        print(f"[harness]   burst wall {burst_wall:.1f} s against summed solo {sum_solo:.1f} s "
              f"(ratio {burst_wall / sum_solo:.2f})", flush=True)
    except Infra as e:
        print(f"[harness] infrastructure failure: {e}", file=sys.stderr)
        return 2

    fails = []
    # prompt reads are serial on the main path (documented engine design): a request fired beside N
    # others starts after at most N+1 reads.  The gate catches queuing BEHIND THAT - e.g. behind the
    # others' whole decodes, which is what a serial server does.
    if main_r["ttft"] > (a.subagents + 1) * solo["ttft"] + 2.0:
        fails.append(f"main time to first token {main_r['ttft']:.1f} s > "
                     f"{a.subagents + 1} x solo {solo['ttft']:.1f} s + 2 (it queued behind more than the "
                     f"others' prompt reads)")
    if main_r["itl"] > a.max_itl_factor * solo["itl"]:
        fails.append(f"main inter-token {main_r['itl'] * 1000:.1f} ms > {a.max_itl_factor} x solo "
                     f"{solo['itl'] * 1000:.1f} ms (its tokens slowed under load)")
    if burst_wall > a.max_wall_ratio * sum_solo:
        fails.append(f"burst wall {burst_wall:.1f} s > {a.max_wall_ratio} x summed solo {sum_solo:.1f} s "
                     f"(the requests ran one after another, not together)")
    if fails:
        print("[harness] FAIL:", file=sys.stderr)
        for f in fails:
            print(f"[harness]   {f}", file=sys.stderr)
        return 1
    print("[harness] PASS: the main request kept its start and its token spacing beside "
          f"{a.subagents} subagent requests, and the burst ran together", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
