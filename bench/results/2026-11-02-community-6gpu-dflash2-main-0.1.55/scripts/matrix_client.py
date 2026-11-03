"""One benchmark run against a freshly loaded server: fresh prompt reads, decode, the drafter's ring, an agent session, parallel requests.
Every call is one JSON line. Speeds come from the engine's `timings`; ttft and total are the client's clock. Nothing is retried."""
import hashlib, json, os, sys, threading, time, urllib.error, urllib.request

base = "http://127.0.0.1:8181"
key = os.environ["BENCH_KEY"]
head = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
sections = sys.argv[1].split(",")
out = open(sys.argv[2], "a", buffering=1)
full = open(sys.argv[2] + ".text", "a", buffering=1)
lock = threading.Lock()


def get(path):
    with urllib.request.urlopen(urllib.request.Request(base + path, headers=head), timeout=20) as r:
        return json.load(r)


model = get("/v1/models")["data"][0]["id"]


def emit(rec):
    with lock:
        out.write(json.dumps(rec) + "\n")


def call(section, tag, messages, max_tokens=200, cancel=False, thinking=False, sampling=None, **extra):
    body = dict(model=model, messages=messages, temperature=0, max_tokens=max_tokens, stream=True, stream_options={"include_usage": True},
                chat_template_kwargs={"enable_thinking": thinking})
    if sampling:
        body.update(sampling)
    body.update(extra)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=head)
    t = time.monotonic()
    first, txt, reasoning, usage, timings, finish, tool, error = None, "", "", None, None, None, "", None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if not line.startswith(b"data: "):
                    continue
                d = line[6:].strip()
                if d == b"[DONE]":
                    break
                it = json.loads(d)
                if "error" in it:
                    error = str(it["error"])[:300]
                    break
                usage = it.get("usage") or usage
                timings = it.get("timings") or timings
                for c in it.get("choices", []):
                    finish = c.get("finish_reason") or finish
                    delta = c.get("delta", {})
                    piece = delta.get("content") or ""
                    think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    calls = "".join(((x.get("function") or {}).get("arguments") or "") for x in (delta.get("tool_calls") or []))
                    if (piece or think or calls) and first is None:
                        first = time.monotonic()
                    txt += piece
                    reasoning += think
                    tool += calls
                if cancel and (txt or reasoning):
                    break
    except (urllib.error.URLError, OSError, ValueError) as e:
        error = repr(e)[:300]
    done = time.monotonic()
    u = usage or {}
    emit({"section": section, "tag": tag, "total_s": round(done - t, 3), "ttft_s": round(first - t, 3) if first else None, "finish": finish, "cancelled": cancel, "error": error,
          "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
          "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "timings": timings,
          "sha": hashlib.sha256((reasoning + txt + tool).encode()).hexdigest()[:12], "chars": len(reasoning) + len(txt) + len(tool)})
    with lock:
        full.write(json.dumps({"tag": tag, "reasoning": reasoning, "text": txt, "tool": tool}) + "\n")
    return txt


def records(n, word="alpha beta gamma delta epsilon zeta eta theta"):
    return "\n".join(f"record{i:06d}: {word}." for i in range(1, n + 1))


TAIL = "\nSummarize the structure of these records and explain how to look up a record by its ID."
# about 19.4 tokens a record (the standard protocol's fixture: 211 records = 4,093 tokens)
SIZES = {"4k": 211, "32k": 1689, "128k": 6758}

e = get("/metrics")["engine"]
emit({"section": "start", "drafting": e.get("drafting"), "note": e.get("drafting_note"), "version": e.get("version"), "cache_mib": e.get("conversation_cache_mib"),
      "expert_slots": e.get("expert_slots"), "expert_cache_mib": e.get("expert_cache_mib"), "model": e.get("model"), "max_context": e.get("max_context")})
call("warmup", "warmup", [{"role": "user", "content": "Explain how a hash table finds a value."}], 64)

if "prefill" in sections:
    # read order within a repetition: short to long; a unique first line means nothing past the template is reused
    for rep in range(1, 4):
        for size, n in SIZES.items():
            call("prefill", f"prefill-{size}-r{rep}", [{"role": "user", "content": f"[case prefill-{size}-r{rep}]\n" + records(n) + TAIL}], 32)

DECODE = {
    "code": "Write a complete Python module implementing an LRU cache class with get and put, with type hints and docstrings, then a thread-safe variant, then full unit tests for both.",
    "prose": "Explain how a bicycle stays upright, in ten paragraphs of plain English, covering steering geometry, trail, gyroscopic effects and what experiments have shown.",
    "json": "Return a JSON array of 40 objects describing fictional employees, each with id, name, title, department, start_date, salary and a list of three skills. Output only the JSON.",
    "edit": "Here is a function:\n\ndef total(xs):\n    t = 0\n    for x in xs:\n        t = t + x\n    return t\n\nRewrite it in five different ways. For each, show the code, a unified diff against the original, and a short explanation of the tradeoffs.",
    "chat": "I am planning a two week trip to Japan in autumn. Suggest a day by day plan with cities, trains, what to eat and what to skip, and explain each choice.",
}
TOOLS = [{"type": "function", "function": {"name": "write_file", "description": "Write a file to disk.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]
if "decode" in sections:
    for rep in range(1, 4):
        for name, text in DECODE.items():
            call("decode", f"{name}-r{rep}", [{"role": "user", "content": f"[case {name}-r{rep}]\n{text}"}], 512)
        call("decode", f"tool-r{rep}", [{"role": "user", "content": f"[case tool-r{rep}]\nUse write_file to create server.py containing a complete Flask REST API for a todo list with tests in the same file."}], 512,
             tools=TOOLS, tool_choice={"type": "function", "function": {"name": "write_file"}})
        # the model card's sampling with thinking on: the path a person's client uses
        call("decode", f"thinking-r{rep}", [{"role": "user", "content": f"[case thinking-r{rep}]\nA train leaves at 09:12 at 80 km/h and another at 09:47 at 110 km/h on the same track. When does the second catch the first? Then write Python that checks the answer."}], 512,
             thinking=True, sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": rep})

if "ring" in sections:
    # one prompt three times, the drafter's ring in three states, at temperature 0
    long_a = records(1030, "north river stone copper window garden silver")
    fixed = "The records are numbered lines with the same eight words; to find one, take its ID and search for that prefix."
    ask = "Write a Python class for a bank account with deposit, withdraw and a transaction log, then unit tests for it."
    p = [{"role": "user", "content": "[case ring]\n" + long_a + TAIL}, {"role": "assistant", "content": fixed}, {"role": "user", "content": ask}]
    call("ring", "ring-fresh", p, 300)
    call("ring", "ring-in-place", p, 300)
    call("ring", "other-conversation", [{"role": "user", "content": "[case other]\n" + records(1030, "east market paper lantern bridge orchard iron") + TAIL}], 40)
    call("ring", "ring-saved-copy", p, 300)
    call("ring", "other-conversation-2", [{"role": "user", "content": "[case other2]\n" + records(1030, "west harbor glass thread meadow engine salt") + TAIL}], 40)
    call("ring", "ring-saved-copy-2", p, 300)

if "session" in sections:
    conv = [{"role": "user", "content": "[case session]\n" + records(1100, "agent session line of a project file") + "\nSummarize the notes in one sentence."}]
    for turn in range(6):
        r = call("session", f"turn{turn}", conv, 200)
        conv += [{"role": "assistant", "content": r}, {"role": "user", "content": f"Good. Now write a Python function number {turn} that adds two numbers, with a docstring and a test."}]
    call("session", "side-request", [{"role": "user", "content": "Give a three word title."}], 12)
    r = call("session", "turn-after-side", conv, 200)
    conv += [{"role": "assistant", "content": r}, {"role": "user", "content": "Explain that in two sentences."}]
    other = [{"role": "user", "content": "[case session-other]\n" + records(1100, "second conversation about something else") + "\nSummarize the notes in one sentence."}]
    r = call("session", "other-fresh", other, 150)
    other += [{"role": "assistant", "content": r}, {"role": "user", "content": "Write a Python class for a queue."}]
    r = call("session", "first-again", conv, 200)
    conv += [{"role": "assistant", "content": r}, {"role": "user", "content": "Now a Python function that reverses a list."}]
    r = call("session", "other-again", other, 200)
    other += [{"role": "assistant", "content": r}, {"role": "user", "content": "Add a test for it."}]
    for rnd in range(3):
        res = {}
        ts = [threading.Thread(target=lambda c=c, i=i: res.__setitem__(i, call("parallel", f"long{i}-round{rnd}", c, 150))) for i, c in enumerate((conv, other))]
        t = time.monotonic(); [x.start() for x in ts]; [x.join() for x in ts]
        emit({"section": "parallel", "tag": f"long-round{rnd}-wall", "total_s": round(time.monotonic() - t, 3)})
        conv = conv + [{"role": "assistant", "content": res[0]}, {"role": "user", "content": "Shorter, please."}]
        other = other + [{"role": "assistant", "content": res[1]}, {"role": "user", "content": "Shorter, please."}]
        ts = [threading.Thread(target=call, args=("parallel", f"short{i}-round{rnd}", [{"role": "user", "content": f"[case p{rnd}-{i}]\nExplain what a {w} is, in detail."}], 200))
              for i, w in enumerate(("hash map", "binary tree", "bloom filter"))]
        t = time.monotonic(); [x.start() for x in ts]; [x.join() for x in ts]
        emit({"section": "parallel", "tag": f"short3-round{rnd}-wall", "total_s": round(time.monotonic() - t, 3)})
    call("session", "after-parallel0", conv, 120)
    call("session", "after-parallel1", other, 120)
    story = conv + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "Write a long story."}]
    call("session", "cancel", story, 400, cancel=True)
    call("session", "after-cancel", story, 60)
emit({"section": "end"})
print("CLIENT COMPLETE", flush=True)
