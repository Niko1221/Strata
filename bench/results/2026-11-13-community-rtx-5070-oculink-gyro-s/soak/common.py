"""Shared helpers for the unattended Gyro-S tests of 2026-11-13. Standard library only.
The API key is read from the user's environment in the registry and only ever sent as a header; it is never printed."""
import ctypes, json, os, random, socket, time, urllib.error, urllib.request, winreg

BASE = os.environ.get("GYRO_BASE", "http://127.0.0.1:8091")
MODEL = os.environ.get("GYRO_MODEL", "qwen3.8-flash-next-gyro-gyro-s")
HERE = os.path.dirname(os.path.abspath(__file__))

def _key():
    try:
        return winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment"), "STRATA_API_KEY")[0]
    except OSError:
        return ""
_KEY = _key()

def headers(anthropic=False):
    h = {"Content-Type": "application/json"}
    if _KEY:
        if anthropic:
            h["x-api-key"] = _KEY; h["anthropic-version"] = "2023-06-01"
        else:
            h["Authorization"] = "Bearer " + _KEY
    return h

def log(path, rec):
    rec = dict(rec); rec.setdefault("logged", round(time.time(), 1))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

class _Mem(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
def free_ram_gb():
    m = _Mem(); m.dwLength = ctypes.sizeof(_Mem); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return round(m.ullAvailPhys / 2**30, 2)

def port_open(host="127.0.0.1", port=8091, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def post_json(path, body, anthropic=False, timeout=1800):
    """-> (http status, parsed json or text). Connection-level failures raise."""
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), headers(anthropic))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        try: return e.code, json.loads(txt)
        except Exception: return e.code, txt[:500]

def get(path, timeout=10):
    req = urllib.request.Request(BASE + path, headers=headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")[:2000]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:500]

def stream_chat(messages, max_tokens=600, extra=None, stop_after_chunks=None, timeout=1800, deadline=None):
    """Streaming chat completion. -> dict with prompt/cached/out tokens, ttft, tps, wall, text, reasoning length.
    stop_after_chunks: close the connection after that many content/reasoning chunks (a client that walks away).
    deadline: close the connection once this many seconds have passed (checked whenever the server sends a line)."""
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
    if extra: body.update(extra)
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), headers())
    t0 = time.time(); first = None; text = ""; reason = ""; usage = {}; chunks = 0; finish = None; aborted = False
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if deadline and time.time() - t0 > deadline:
                aborted = True; break
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            try: ev = json.loads(line[5:])
            except Exception: continue
            if ev.get("usage"): usage = ev["usage"]
            for c in ev.get("choices", []):
                d = c.get("delta", {})
                piece = d.get("content") or ""; rp = d.get("reasoning_content") or d.get("reasoning") or ""
                if c.get("finish_reason"): finish = c["finish_reason"]
                if piece or rp:
                    chunks += 1
                    if first is None: first = time.time() - t0
                text += piece; reason += rp
            if stop_after_chunks and chunks >= stop_after_chunks:
                aborted = True; break
    total = time.time() - t0
    out = usage.get("completion_tokens") or 0
    tps = out / (total - first) if first is not None and total > first and out else None
    return {"prompt": usage.get("prompt_tokens"), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "out": out, "ttft": round(first, 2) if first is not None else None, "tps": round(tps, 1) if tps else None,
            "wall": round(total, 1), "text": text, "reasoning_chars": len(reason), "finish": finish, "aborted": aborted, "chunks": chunks}

WORDS = ("amber basin cedar delta ember fjord garnet harbor iris juniper kelp lantern meadow nickel orchard pebble quartz river "
         "saffron timber umber valley willow yarrow zinc copper marble thistle canyon glacier").split()
def filler(lines, facts, rng):
    out = []
    for i in range(lines):
        if i in facts: out.append(facts[i])
        out.append(f"Record {i}: the {rng.choice(WORDS)} {rng.choice(WORDS)} moved {rng.randint(10, 999)} units toward the {rng.choice(WORDS)} {rng.choice(WORDS)}.")
    return "\n".join(out)
