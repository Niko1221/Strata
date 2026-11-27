"""Does drafting pay on THIS card and placement? A short measured probe decides (0221-DSPARK).

Measured 2026-11-23: one drafting step costs 0.7 to 5.4 plain decode steps depending on the card and where the experts live, so the same drafter
is 4x faster on a 4060 with SQ-10 (plain decode 2 tok/s) and 2x slower on the same box with UD-Q4_K_XL (plain decode 11 tok/s). The card class alone
cannot decide it, so at the first start of a model (and again whenever the placement changes) the engine is started twice, plain and with chain
drafting, and a code prompt and a chat prompt are decoded for 128 tokens each. Drafting is chosen only when the mixed gain (geometric mean of the
two gains) reaches MARGIN. The numbers go into the run config ("drafting_probe") and into the status note, so the choice can be explained.

Stdlib only; `python inkling_probe.py --exe ENGINE --out RESULT.json [--margin 1.10] -- ENGINE ARGS (--native MODEL --max-context N ...)` runs one
probe by hand. The decision is used on every backend (owner decision 2026-11-23: a slightly different answer is fine, a loss in speed is not)."""
from __future__ import annotations

import sys
if sys.platform == "win32":
    # CPython 3.12's WMI helper (_wmi, behind platform.system() / uname()) takes an access violation in a worker thread when WMI refuses the
    # account (0x80041003) and the caller has already gone (DIAG-WINHOST-0216). Without _wmi, platform uses the registry and the environment.
    sys.modules["_wmi"] = None

import hashlib
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time

try:
    from strata_control.inkling_probe_prompts import CHAT_IDS, CODE_IDS
except ImportError:  # run as a script from this folder
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inkling_probe_prompts import CHAT_IDS, CODE_IDS

MARGIN = 1.10          # the mixed gain drafting has to reach
# how each probe request is decoded: code greedy (coding tools run cold), chat sampled the way a chat client asks (the drafter then takes the
# sampled path, measured 2026-11-23 on an A6000: chat 0.91x sampled where greedy chat was 1.04x with the governor)
GEN_OPTS = {"code": "temperature=0", "chat": "temperature=0.7 top_p=0.95 top_k=64 seed=7"}
TOKENS = 128           # tokens decoded per probe request (64 was too short: the A6000 with SQ-20 measured 1.05 at 64 tokens, 1.14 at 128 against 1.19 in the long runs)
REPS = 2               # requests per prompt and arm; the median is used
READY_TIMEOUT = 900.0
STEP_TIMEOUT = 300.0
ON_ARGS = ["--draft", "dflash2", "--draft-beam", "1", "--draft-adapt"]   # what a drafting run config starts with (inkling_kind.drafter_args): DFlash 2 is the only drafter
_PLACEMENT_FLAGS = ("--native", "--max-context", "--vram-experts-gb", "--ram-experts-gb", "--ssd-experts", "--kv", "--kv-offload",
                    "--vision", "--gpus", "--layer-split", "--dspark")


# ---- the decision ---------------------------------------------------------------------------------------------

def mixed_gain(gains: dict) -> float:
    """Geometric mean of the per-prompt gains (code and chat weigh the same)."""
    vals = [float(v) for v in gains.values() if v and v > 0]
    if not vals:
        return 0.0
    return statistics.geometric_mean(vals)


def decide(gains: dict, margin: float = MARGIN) -> str:
    """"on" when the mixed gain reaches the margin, else "off"."""
    return "on" if mixed_gain(gains) >= margin else "off"


def placement_key(cfg: dict) -> str:
    """What a probe is valid for: the model file, context, expert placement, K/V, vision, GPU choice and backend of the run config.
    A different value means the numbers no longer describe the run, so the probe is made again.
    NOT the slot count (2026-11-23, rc1 regression): the engine drafts only while exactly one request is active, so the probe, which decodes one request, describes
    a lone request at 1 slot and at 32 alike, and what the slots change in memory is already in the flags above. The slot count is a profile override of
    `strata launch`: the probe hook reads the run config FILE (parallel 1) and the real start reads the file plus the override (parallel 4), so a key with
    the slot count never matched at the start and every launch above 1 slot ran with `--draft off`."""
    args = [str(a) for a in (cfg.get("args") or [])]
    picked: dict[str, str] = {}
    for i, a in enumerate(args):
        if a in _PLACEMENT_FLAGS:
            nxt = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else ""
            picked[a] = os.path.basename(nxt) if a in ("--native", "--vision", "--dspark") else nxt
    native = next((args[i + 1] for i, a in enumerate(args) if a == "--native" and i + 1 < len(args)), "")
    picked["dflash2"] = "1" if (native and os.path.isfile(os.path.join(os.path.dirname(native), "dflash2.safetensors"))) else "0"      # the drafter's file: a probe made without it says nothing about a run with it
    picked["backend"] = str(cfg.get("backend") or "cuda")
    picked["gpu"] = json.dumps(cfg.get("gpu"), sort_keys=True)
    return hashlib.sha1(json.dumps(picked, sort_keys=True).encode()).hexdigest()[:16]


def is_stale(cfg: dict) -> bool:
    """True when the run config has no probe, or one made for another placement."""
    p = cfg.get("drafting_probe")
    return not (isinstance(p, dict) and p.get("key") == placement_key(cfg) and p.get("decision") in ("on", "off"))


def decision_of(cfg: dict) -> str | None:
    """The recorded decision ("on" / "off") when it is valid for this placement, else None."""
    p = cfg.get("drafting_probe")
    if isinstance(p, dict) and p.get("key") == placement_key(cfg) and p.get("decision") in ("on", "off"):
        return p["decision"]
    return None


def note(cfg: dict) -> str:
    """The status sentence: why drafting is on or off for this run config ("" when nothing was probed)."""
    p = cfg.get("drafting_probe")
    if not isinstance(p, dict) or p.get("decision") not in ("on", "off"):
        return ""
    g = p.get("gains") or {}
    parts = ", ".join(f"{k} {v:.2f}x" for k, v in g.items())
    plain = p.get("plain_tps") or {}
    speed = ""
    if plain:
        speed = f" (plain decoding {statistics.mean(plain.values()):.0f} tok/s)"
    stale = "" if p.get("key") == placement_key(cfg) else " This was measured for another placement and will be measured again."
    verdict = "on" if p["decision"] == "on" else "off"
    return (f"Drafting {verdict}: a {p.get('tokens', TOKENS)}-token probe on this card measured {mixed_gain(g):.2f}x with drafting ({parts}){speed}; "
            f"it needs {p.get('margin', MARGIN):.2f}x.{stale}")


def record(cfg: dict, result: dict) -> dict:
    """Put a probe result into the run config (returns the same dict)."""
    cfg["drafting_probe"] = result
    return cfg


# ---- the probe ------------------------------------------------------------------------------------------------

class Engine:
    """One `--serve` engine process on its line protocol (stdin lines in, stdout lines out)."""

    def __init__(self, exe: str, args: list[str], env: dict | None = None, cwd: str | None = None):
        try:
            from strata_control.childenv import quiet_crash_env
            env = quiet_crash_env(dict(os.environ if env is None else env))      # no debugger on a user's machine
        except ImportError:                                                      # run by hand as a script outside the package
            env = dict(os.environ if env is None else env)
            env.setdefault("GGML_NO_BACKTRACE", "1")
        self.p = subprocess.Popen([exe, "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  env=env, cwd=cwd, bufsize=0)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        buf = b""
        while True:
            ch = self.p.stdout.read(65536)
            if not ch:
                break
            buf += ch
            while b"\n" in buf:
                ln, buf = buf.split(b"\n", 1)
                self.q.put(ln.decode("utf8", "replace").rstrip("\r"))
        self.q.put(None)

    def send(self, line: str):
        self.p.stdin.write((line + "\n").encode())

    def readline(self, timeout: float):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return "TIMEOUT"

    def close(self):
        try:
            self.send("QUIT")
        except Exception:  # noqa: BLE001
            pass
        try:
            self.p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()


def _strip_draft_args(args: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a in ("--draft", "--draft-beam", "--draft-cost", "--draft-plain"):
            i += 2
        elif a == "--draft-adapt":
            i += 1
        else:
            out.append(a)
            i += 1
    return out


def _arm(open_engine, args: list[str], prompts: dict, tokens: int, reps: int) -> dict | None:
    """Start one engine, warm it, decode every prompt; {name: median tok/s}, or None when it did not come up or a request failed."""
    eng = open_engine(args)
    try:
        t_end = time.time() + READY_TIMEOUT
        while True:
            ln = eng.readline(max(1.0, t_end - time.time()))
            if ln is None or ln == "TIMEOUT":
                return None
            if ln.startswith("READY"):
                break
        speeds: dict[str, list[float]] = {k: [] for k in prompts}
        plan = [("warm", next(iter(prompts.values())), 16)] + [(k, v, tokens) for k in prompts for v in [prompts[k]] for _ in range(reps)]
        for name, ids, n in plan:
            eng.send("GEN %d %s %s" % (n, GEN_OPTS.get(name, "temperature=0"), ",".join(str(x) for x in ids)))
            while True:
                ln = eng.readline(STEP_TIMEOUT)
                if ln is None or ln == "TIMEOUT" or ln.startswith("ERR"):
                    return None
                if ln.startswith("DONE"):
                    f = ln.split()
                    gen, dec_ms = int(f[1]), float(f[4])
                    if name != "warm" and dec_ms > 0 and gen > 0:
                        speeds[name].append(gen / dec_ms * 1000.0)
                    break
        if any(not v for v in speeds.values()):
            return None
        return {k: statistics.median(v) for k, v in speeds.items()}
    finally:
        eng.close()


def probe(open_engine, base_args: list[str], *, margin: float = MARGIN, tokens: int = TOKENS, reps: int = REPS,
          on_args: list[str] | None = None, prompts: dict | None = None, key: str = "") -> dict:
    """Run the probe: `open_engine(args)` returns an Engine-like object (send / readline / close). base_args are the run config's engine arguments;
    any drafting flags in them are replaced. Plain decoding is measured first, then drafting (`on_args`, default ON_ARGS). The result:
    {decision, mixed_gain, gains, plain_tps, drafted_tps, margin, tokens, key, at} or {decision: "off", error: ...} when an engine failed."""
    prompts = prompts or {"code": CODE_IDS, "chat": CHAT_IDS}
    base = _strip_draft_args(list(base_args))
    plain = _arm(open_engine, base + ["--draft", "off"], prompts, tokens, reps)
    res: dict = {"decision": "off", "margin": margin, "tokens": tokens, "key": key, "at": int(time.time())}
    if plain is None:
        res["error"] = "the plain engine start for the probe failed"
        return res
    drafted = _arm(open_engine, base + list(on_args or ON_ARGS), prompts, tokens, reps)
    if drafted is None:
        res["error"] = "the drafting engine start for the probe failed"
        res["plain_tps"] = {k: round(v, 2) for k, v in plain.items()}
        return res
    gains = {k: drafted[k] / plain[k] for k in prompts}
    res.update({"decision": decide(gains, margin), "mixed_gain": round(mixed_gain(gains), 3), "gains": {k: round(v, 3) for k, v in gains.items()},
                "plain_tps": {k: round(v, 2) for k, v in plain.items()}, "drafted_tps": {k: round(v, 2) for k, v in drafted.items()}})
    return res


def probe_run_config(cfg: dict, exe: str, args: list[str], env: dict | None = None, cwd: str | None = None, **kw) -> dict:
    """Probe the real engine for a run config and record the result in it (the caller saves the config)."""
    result = probe(lambda a: Engine(exe, a, env, cwd), args, key=placement_key(cfg), **kw)
    return record(cfg, result)


def main(argv: list[str]) -> int:
    if "--" not in argv:
        print(__doc__)
        return 2
    cut = argv.index("--")
    opts, args = argv[:cut], argv[cut + 1:]
    exe = out = None
    margin = MARGIN
    tokens = TOKENS
    on_args = None
    i = 0
    while i < len(opts):
        if opts[i] == "--exe":
            exe = opts[i + 1]
            i += 2
        elif opts[i] == "--out":
            out = opts[i + 1]
            i += 2
        elif opts[i] == "--tokens":
            tokens = int(opts[i + 1])
            i += 2
        elif opts[i] == "--no-adapt":
            on_args = [a for a in ON_ARGS if a != "--draft-adapt"]
            i += 1
        elif opts[i] == "--margin":
            margin = float(opts[i + 1])
            i += 2
        else:
            print("unknown option", opts[i])
            return 2
    if not exe:
        print("--exe is required")
        return 2
    result = probe(lambda a: Engine(exe, a), args, margin=margin, tokens=tokens, on_args=on_args, key="")
    text = json.dumps(result, indent=1)
    print(text)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
