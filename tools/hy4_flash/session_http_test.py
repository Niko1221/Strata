"""Exercise two freshly started hy4-flash endpoints with identical model, drafting and sampling settings.

The candidate enables host parking. The reference uses only its active prefix.
Run against dedicated test servers; the test submits chats and disconnects a stream.
"""
import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

@lru_cache
def model_id(base):
    with urllib.request.urlopen(base + "/v1/models", timeout=5) as response:
        return json.load(response)["data"][0]["id"]


def chat(base, messages, max_tokens=16):
    body = dict(model=model_id(base), messages=messages, temperature=0, seed=42,
                max_tokens=max_tokens, reasoning_effort="none", stream=True)
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=300)

def chunks(stream):
    for line in stream:
        if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
            yield json.loads(line[6:])

def complete(base, messages, limit=16):
    text = []
    usage = None
    with chat(base, messages, limit) as stream:
        for chunk in chunks(stream):
            choice = chunk["choices"][0]
            text.append(choice["delta"].get("content", ""))
            if chunk.get("usage"):
                usage = chunk["usage"]
    assert usage and usage["completion_tokens"] > 0, usage
    return "".join(text), usage

def task(key):
    return [{"role": "user", "content": (f"Archive record {key}. " * 32 +
            f"My code word is {key}. Reply with just that word.")}]

def follow(messages, answer):
    return [*messages, {"role": "assistant", "content": answer},
            {"role": "user", "content": "What was my code word? Answer with just the word."}]

def cached(usage):
    return usage["prompt_tokens_details"]["cached_tokens"]

def run(candidate, reference):
    a, b = task("MARIGOLD"), task("COBALT")
    ref_a, _ = complete(reference, a)
    expected, expected_usage = complete(reference, follow(a, ref_a))
    got_a, _ = complete(candidate, a)
    assert got_a == ref_a, (got_a, ref_a)
    got_b, _ = complete(candidate, b)
    a_next, b_next = follow(a, got_a), follow(b, got_b)
    result, usage = complete(candidate, a_next)
    assert result == expected and cached(usage) == cached(expected_usage) > 100, (result, expected, usage, expected_usage)
    print("A/B/A follow-up", json.dumps(usage), repr(result), flush=True)

    # Keep one request active while two unrelated follow-ups enter the queue.
    # Cancel only the active stream, then require both follow-ups to complete.
    long_chat = [{"role": "user", "content": "Write the integers from 1 to 10000, one per line."}]
    with ThreadPoolExecutor(max_workers=2) as pool, chat(candidate, long_chat, 1024) as stream:
        next(c for c in chunks(stream) if c["choices"][0]["delta"].get("content"))
        waiting = [pool.submit(complete, candidate, m) for m in (a_next, b_next)]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with urllib.request.urlopen(candidate + "/status", timeout=5) as r:
                status = json.load(r)
            if status.get("queued", 0) >= 2:
                break
            time.sleep(0.05)
        else:
            raise AssertionError(f"did not observe two queued requests: {status}")
        stream.close()
        outputs = [f.result(timeout=120) for f in waiting]

    for (answer, usage), key in zip(outputs, ("MARIGOLD", "COBALT")):
        assert key in answer.upper() and cached(usage) > 100, (key, answer, usage)
        print("after disconnect", key, json.dumps(usage), repr(answer), flush=True)


def sparse_and_eviction(candidate, reference):
    # For the hy4-flash tokenizer this crosses the 2K sparse-selection threshold.
    prompt = [{"role": "user", "content": "Granite sample. " * 705 +
               "The access word is PERIDOT. Repeat just the access word."}]

    def check(base):
        answer, usage = complete(base, prompt)
        assert usage["prompt_tokens"] > 2048
        if base == candidate:
            complete(base, task("VIOLET"))
        reply, usage = complete(base, follow(prompt, answer))
        return answer, reply, cached(usage)

    with ThreadPoolExecutor(max_workers=2) as pool:
        baseline, switched = pool.map(check, (reference, candidate))
    assert switched == baseline and switched[-1] > 2048, (baseline, switched)
    for key in ("SAPPHIRE", "EMERALD", "RUBY", "OPAL", "JADE"):
        complete(candidate, task(key))
    _, usage = complete(candidate, follow(prompt, switched[0]))
    assert cached(usage) < 100, usage
    print("sparse restore and eviction", switched[-1], cached(usage))


def overlapping_streams(base):
    ready = threading.Barrier(2)

    def request(i):
        prompt = [{"role": "user", "content":
                   f"List every integer from {i * 10000 + 1} to {i * 10000 + 9999}, one per line."}]
        first = usage = None
        events = 0
        with chat(base, prompt, 160) as stream:
            for chunk in chunks(stream):
                if chunk["choices"][0]["delta"].get("content"):
                    if first is None:
                        first = time.monotonic()
                        ready.wait(timeout=120)
                    events += 1
                    if i == 0 and events == 12:
                        break
                if chunk.get("usage"):
                    usage = chunk["usage"]
        return first, time.monotonic(), usage

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.map(request, (0, 1))
    assert min(a[1], b[1]) > max(a[0], b[0])
    assert a[2] is None and b[2]["completion_tokens"] == 160, (a, b)
    print("overlapping replica streams; surviving stream", b[2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("reference")
    parser.add_argument("--sparse", action="store_true",
                        help="also test a 2884-token prompt; use context >= 3072, 192 MiB and 3 parked slots")
    parser.add_argument("--replica-url", help="additional endpoint running two one-card replicas")
    args = parser.parse_args()
    candidate, reference = args.candidate.rstrip("/"), args.reference.rstrip("/")
    run(candidate, reference)
    if args.sparse:
        sparse_and_eviction(candidate, reference)
    if args.replica_url:
        overlapping_streams(args.replica_url.rstrip("/"))
