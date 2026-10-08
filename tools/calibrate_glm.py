"""Tune Strata's GLM CPU lane for this PC (setup.py --family glm --calibrate).

On one or two GPUs most of a token's experts are not in VRAM; the CPU LANE computes the ones in the RAM tier while the
GPU runs the rest.  Two of its settings depend on the PC more than on the model:
  STRATA_GLM_PCIE_SHARE  the share of a route's RAM-tier experts copied over PCIe and run on the GPU instead of on the
                         CPU (0: the CPU takes every one).  A fast PCIe link and a slow CPU want more; an x8 link beside
                         a many-core CPU wants none.  The engine starts with a guess from timing one expert each way.
  STRATA_GLM_CPU_LANE    the CPU threads that compute them.  Every core is not always fastest: the experts are read
                         from RAM, and past the memory's bandwidth more threads only wait (or, on a hybrid CPU, the
                         efficiency cores make the rest wait for them).
Both are measured through ONE running engine - per-request `strata_tune` keys (pcie_frac, cpu_threads) - because an
engine start reads the whole expert set from disk (minutes).  Decode speed only: a prompt balances its CPU and PCIe
shares itself, layer by layer.

A setting is kept only when it beats the engine's own by more than MIN_GAIN in an interleaved re-measurement - the
expert tiers follow the text and the OS adds noise, so single measurements differ by a few percent.

    python tools/calibrate_glm.py strata-glm-<model>.json  # measure and print; setup --calibrate saves it
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

MIN_GAIN = 0.03                          # a setting must beat the engine's own by this much to be kept
PCIE_SHARES = (0.0, 0.1, 0.25, 0.5, 0.75)
# a sweep stops once a setting falls this far behind its best so far: the next ones go further the same way, and a
# share far past the best runs at a crawl (one 3090 on PCIe 3.0 x8: 20.9 tok/s at none, 8.1 at 0.10, 2.1 at 0.75 -
# its last three shares took ~8 of the tuning's 19 minutes)
STOP_BELOW = 0.7
MAX_NEW = 128
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List twelve European capitals with one sentence about each.",
)
SHARE_ENV, THREADS_ENV = "STRATA_GLM_PCIE_SHARE", "STRATA_GLM_CPU_LANE"
KEYS = (SHARE_ENV, THREADS_ENV)          # the environment settings a calibration owns (apply() sets or clears them)


def prompt_ids(cfg: dict) -> list[list[int]]:
    """PROMPTS as the server sends them: the model's chat template, thinking off (decode speed, not reasoning)."""
    import strata_tokenizer as ST
    from serve.frontend import ChatTemplate, effort_kwargs
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    meta = json.loads((tpath / "tokenizer.json").read_text(encoding="utf-8"))
    tok = ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").splitlines(),
                       json.loads((tpath / "token_type.json").read_text()),
                       pre=meta.get("pre", "glm5"), special_ids=meta.get("special_ids"))
    tpl_path = tpath / "chat_template.jinja"
    tpl = ChatTemplate(tpl_path if tpl_path.exists() else ROOT / "serve" / "chat_template.jinja")
    return [tok.encode(tpl.render([{"role": "user", "content": p}], **effort_kwargs("none")), parse_special=True)
            for p in PROMPTS]


def thread_candidates(default: int, extra=()) -> list[int]:
    """The engine's own count, and fewer: three quarters, two thirds, a half and a quarter (at least 2), plus `extra`
    (the performance cores less one on a hybrid CPU, one socket's cores less one on a 2-socket PC), no repeats."""
    c = [default]
    for w in (round(default * 3 / 4), round(default * 2 / 3), round(default / 2), round(default / 4), *extra):
        if 2 <= w < default and w not in c:
            c.append(w)
    return c


def cpu_list(text: str) -> list[int]:
    """A sysfs CPU list ("0-3,8,10-11") -> its CPU numbers."""
    out = []
    for part in text.strip().split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def host_thread_extras(sys_cpu: Path = Path("/sys/devices/system/cpu"),
                       hybrid_cores: Path = Path("/sys/devices/cpu_core/cpus")) -> list[int]:
    """From the CPU topology: one socket's CPUs less one (several sockets), the performance cores' CPUs less one (a
    hybrid CPU).  [] when it cannot be read."""
    try:
        online = cpu_list((sys_cpu / "online").read_text())
        pkg = {c: (sys_cpu / f"cpu{c}" / "topology" / "physical_package_id").read_text().strip() for c in online}
    except (OSError, ValueError):
        return []
    out = []
    sockets = sorted(set(pkg.values()))
    if len(sockets) >= 2:
        out.append(min(sum(1 for p in pkg.values() if p == s) for s in sockets) - 1)
    try:
        p_cpus = [c for c in cpu_list(hybrid_cores.read_text()) if c in pkg]
        if 0 < len(p_cpus) < len(online):
            out.append(len(p_cpus) - 1)
    except (OSError, ValueError):
        pass
    return [w for w in out if w >= 2]


def pick(measured: dict, default_key, min_gain: float = MIN_GAIN):
    """The key with the best median tok/s, or `default_key` unless the best beats it by more than min_gain."""
    med = {k: statistics.median(v) for k, v in measured.items() if v}
    if not med or default_key not in med:
        return default_key
    best = max(med, key=med.get)
    return best if med[best] > med[default_key] * (1.0 + min_gain) else default_key


def tune_keys(share, threads) -> dict:
    """The request's `strata_tune`: None = what the engine started with."""
    t = {}
    if share is not None:
        t["pcie_frac"] = float(share)
    if threads is not None:
        t["cpu_threads"] = int(threads)
    return t


class Session:
    """One running engine: decode tok/s for a setting (the median of the prompts' rates)."""

    def __init__(self, engine, ids_list):
        self.engine = engine
        self.ids_list = ids_list

    def rate(self, tune: dict | None = None) -> float:
        rates = []
        for ids in self.ids_list:
            sampling = {"temperature": 0}
            if tune:
                sampling["strata_tune"] = tune
            n = sum(1 for t in self.engine.generate(ids, MAX_NEW, sampling, threading.Event()) if t is not None)
            ms = (self.engine.last or {}).get("decode_ms") or 0.0
            if n > 8 and ms > 0:
                rates.append(n / (ms / 1000.0))
        return statistics.median(rates) if rates else 0.0

    def warm_up(self, rounds: int = 2):
        for _ in range(rounds):
            self.rate()


def run(cfg: dict, say=print, start_engine=None) -> dict:
    """Measure on the engine `cfg` describes; returns {"settings": {env: value}, "report": {...}}.  The engine runs
    with the config's environment less the settings a calibration owns, so the start's own choice is the baseline.
    `start_engine(cfg)` returns a started engine (serve.server.StrataEngine, or a stand-in in tests)."""
    base = {**cfg, "env": apply(cfg.get("env") or {}, {})}
    if start_engine is None:
        from serve.server import StrataEngine, child_env, engine_args

        def start_engine(c):
            env = child_env(c)
            for key in KEYS:
                env.pop(key, None)
            return StrataEngine(c["exe"], engine_args(c), cwd=c.get("cwd"), log=c.get("log"), env=env)
    return measure(base, prompt_ids(cfg), start_engine, say, host_thread_extras())


def measure(cfg: dict, ids_list, start_engine, say=print, extra_threads=()) -> dict:
    t0 = time.time()
    report: dict = {}
    settings: dict = {}
    say("  Loading the model for the measurements (it reads the experts into RAM and VRAM first) ...")
    eng = start_engine(cfg)
    try:
        info = dict(getattr(eng, "info", {}) or {})
        d_threads = int(info.get("cpu_threads") or 0)
        d_share = float(info.get("pcie_share", 1.0))
        report["default"] = {"pcie_share": round(d_share, 2), "cpu_threads": d_threads}
        if d_threads <= 0:
            say("    no CPU lane on this PC (every expert the GPU lacks is copied over PCIe): nothing to tune")
            report["seconds"] = round(time.time() - t0)
            return {"settings": {}, "report": report}
        s = Session(eng, ids_list)
        s.warm_up()
        # 1. the PCIe share (None: the engine's own split), every thread
        by_share = {None: [s.rate()]}
        say(f"    the engine's own split (PCIe share {d_share:.2f}): {by_share[None][0]:.1f} tok/s")
        for f in PCIE_SHARES:
            by_share[f] = [s.rate(tune_keys(f, None))]
            say(f"    PCIe share {f:.2f}: {by_share[f][0]:.1f} tok/s")
            if by_share[f][0] < STOP_BELOW * max(v[0] for v in by_share.values()):
                break
        best_share = max(by_share, key=lambda k: by_share[k][0])
        # 2. the CPU threads, at that share
        by_threads = {}
        for w in sorted(thread_candidates(d_threads, extra_threads), reverse=True):
            by_threads[w] = [s.rate(tune_keys(best_share, None if w == d_threads else w))]
            say(f"    {w} CPU threads: {by_threads[w][0]:.1f} tok/s")
            if by_threads[w][0] < STOP_BELOW * max(v[0] for v in by_threads.values()):
                break
        best_threads = max(by_threads, key=lambda k: by_threads[k][0])
        # 3. the winner against the engine's own, interleaved, three times each
        dflt, cand = (None, d_threads), (best_share, best_threads)
        confirm = {dflt: [], cand: []}
        if cand != dflt:
            say("    confirming against the engine's own settings ...")
            for _ in range(3):
                for k in (dflt, cand):
                    confirm[k].append(s.rate(tune_keys(k[0], None if k[1] == d_threads else k[1])))
        chosen = pick(confirm, dflt) if cand != dflt else dflt
        report.update(share_sweep={("own" if k is None else f"{k:.2f}"): v for k, v in by_share.items()},
                      thread_sweep={str(k): v for k, v in by_threads.items()},
                      confirm={f"{'own' if k[0] is None else f'{k[0]:.2f}'}/{k[1]}": v for k, v in confirm.items()})
    finally:
        close(eng)
    if chosen[0] is not None:
        settings[SHARE_ENV] = f"{chosen[0]:.2f}"
    if chosen[1] != d_threads:
        settings[THREADS_ENV] = str(chosen[1])
    rates = confirm.get(chosen) or (by_share.get(None) if chosen == dflt else None)
    report["tok_s"] = round(statistics.median(rates), 1) if rates else None
    report["seconds"] = round(time.time() - t0)
    return {"settings": settings, "report": report}


def engine_error(log: str | None, since: int = 0) -> str | None:
    """The engine's own reason for a failed start: the last line of its log written after byte `since` that is its own
    ("glm ...", "strata ..." or "ERR ..."), else the last line there; None without a log or a new line."""
    if not log:
        return None
    try:
        with open(log, "rb") as f:
            f.seek(since)
            lines = [x.strip() for x in f.read()[-16384:].decode("utf-8", "replace").splitlines() if x.strip()]
    except OSError:
        return None
    return next((x for x in reversed(lines) if x.startswith(("glm", "strata", "ERR"))), lines[-1] if lines else None)


def close(eng):
    proc = getattr(eng, "proc", None)
    if proc is None:
        return
    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(120)
    except Exception:
        proc.kill()


def apply(env: dict, settings: dict) -> dict:
    """`env` (a config's "env" block) with the calibrated settings; one the calibration did not set is removed, so the
    engine chooses it again and an older calibration's value never lingers."""
    out = {k: v for k, v in (env or {}).items() if k not in KEYS}
    for k in KEYS:
        if settings.get(k) is not None:
            out[k] = str(settings[k])
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: calibrate_glm.py <strata-glm-*.json>")
    res = run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    print(json.dumps(res, indent=1))
