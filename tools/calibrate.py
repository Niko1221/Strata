"""Tune the engine's hardware-dependent settings on this PC (setup's --calibrate).

Three settings depend on the machine more than on the model, and the defaults are right for the PC they were
measured on (a Ryzen 5 7600 + RTX 5070 on PCIe 5):
  --pcie-frac     the share of the experts missing from VRAM that are copied over PCIe and run on the GPU instead of
                  on the CPU.  A fast PCIe link and a slow CPU want more; a laptop's x8 link or a fast CPU want less.
  --spec-min-p    how sure the draft layer must be to extend a verify window by another guess.  A slower CPU pays more
                  per extra window row (more experts per window), so it wants a higher floor.
  --pool-workers  the CPU threads that compute experts.  Every physical core is not always best: on hybrid CPUs the
                  efficiency cores can make the whole window wait for them.
  --adapt-every / --adapt-swaps / --adapt-decay
                  how the VRAM expert tier follows the conversation: how often, how many experts it swaps in, how fast
                  its use counts fade.  A PC whose CPU reads the missed experts slowly (DDR3, ~24 GB/s) gains from
                  swapping more: 160 swaps every window measured +7.9% over the default tier on a Xeon E5-2673 v3 (DDR3) with an
                  RTX 4060 Ti on PCIe 3.0 x8 (Q2_0; with the swaps taking effect a window later, #764).
  STRATA_PREFILL_CPU_SHARE_MAX (an engine environment variable, in the config's "env")
                  how large a prompt chunk still hands the idle CPU pool some of the experts the GPU would stream:
                  1,024 tokens by default, 3,072 where the CPU keeps up with the link.  Measured both ways: a 2,048-token
                  chunk lost 2% on a Ryzen 5 7600 + RTX 5070; on a Ryzen 7 5800X3D + RTX 5070 Ti (Swift 1.5 IQ3_S)
                  1,537 tokens read 24% faster, 2,142 tokens 14%, 2,848 tokens 3%, 600 and 4,600 unchanged.
The first two are measured through one engine (per-request `strata_tune` keys); the worker count, the adaptive
tier and the share's range need a restart per value.  The first four are decode speed only: the prompt path streams
every expert whatever they say.  The share's range is the read speed of a fresh ~2K-token prompt.

A setting is kept only when it beats the default by more than MIN_GAIN in a re-measurement (interleaved for the
per-request keys, a restart per value for the others) - the adaptive expert tier and the OS make single measurements
noisy by a few percent.

    python tools/calibrate.py strata-q2_0.json        # measure and print; setup.py --calibrate also saves it
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

MIN_GAIN = 0.03                    # a setting must beat the default by this much to be kept
PCIE_FRACS = (0.0, 0.2, 0.35, 0.55, 0.75, 0.9, 1.0)   # 1.0: every miss over PCIe, the CPU pool gets no expert
SPEC_MIN_PS = (0.3, 0.5, 0.7)
SWEEP_ROUNDS = 3                   # how many times the sweep visits each value (see `sweep`)
# the adaptive tier's candidates (every, swaps, decay) against the engine's own (None): swapping more and remembering
# longer, the rest of the set as the engine has it
ADAPT_CANDIDATES = (None, ("1", "80", "0.97"), ("1", "160", "0.97"))
ADAPT_FLAGS = ("--adapt-every", "--adapt-swaps", "--adapt-decay")
MAX_NEW = 128
# the expert tier follows a text over some windows: 128-token answers end before it shows (on a Xeon E5-2673 v3 with
# DDR3, every 1 / 160 / 0.97 measured +0.7% with 128 tokens and +7.9% with 512-token answers), so its step uses these
TIER_MAX_NEW = 512
# the CPU prefill share's range: the engine's default (STRATA_PREFILL_CPU_SHARE_MAX unset, 1,024 tokens) against 3,072,
# a restart each, measured on fresh reads of one ~2K-token prompt (no checkpoint, so every read is from token 0)
SHARE_ENV = "STRATA_PREFILL_CPU_SHARE_MAX"
SHARE_RANGE = "3072"
SHARE_PROMPT_TOKENS = 2048
SHARE_ROUNDS = 3
ENV_DEFAULTS = {SHARE_ENV: None}   # None: the engine's own default (the variable absent)
SHARE_SENTENCES = (
    "The committee reviewed the budget for the coming year and asked for a shorter report next time.",
    "A refrigerator moves heat from the inside to the outside by compressing and expanding a working fluid.",
    "The function merges two sorted lists in linear time and returns a new list without changing its inputs.",
    "The lighthouse keeper logged the weather every evening and the ships that passed before dark.",
    "Twelve capitals were listed with one sentence each, from Lisbon in the west to Helsinki in the north.",
)
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List twelve European capitals with one sentence about each.",
)


def share_prompt_ids(tok, target: int = SHARE_PROMPT_TOKENS) -> list[int]:
    """A chat prompt of about `target` tokens: numbered sentences of plain prose (what a tool result or a pasted file
    costs to read), the same every run."""
    lines = []
    ids: list[int] = []
    text = lambda: "Read these notes and reply with the single word OK.\n\n" + "\n".join(lines)   # noqa: E731
    while len(ids) < target and len(lines) < target:   # a line is ~20 tokens; the line cap ends it on any tokenizer
        lines.append(f"Note {len(lines) + 1}: {SHARE_SENTENCES[len(lines) % len(SHARE_SENTENCES)]}")
        if len(lines) % 16 == 0:   # tokenize every 16 lines
            ids = chat_ids(tok, text())
    return ids if len(lines) % 16 == 0 else chat_ids(tok, text())


def prompt_rates(eng, ids: list[int], rounds: int = SHARE_ROUNDS) -> list[float]:
    """Fresh reads of `ids` (a one-shot request: no checkpoint is kept, so every round reads from token 0): the prompt
    tokens per second of each.  A read the engine reused from a checkpoint is not counted."""
    out = []
    for _ in range(rounds):
        for _ in eng.generate(ids, 1, {"temperature": 0, "strata_checkpoint": False}, threading.Event()):
            pass
        last = eng.last or {}
        n = int(last.get("prompt_tokens") or 0)
        ms = float(last.get("prompt_ms") or 0.0)
        if n > 0 and ms > 0 and not last.get("reused"):
            out.append(n / (ms / 1000.0))
    return out


def share_step_applies(cfg: dict) -> str | None:
    """Why the CPU share's range cannot change anything for this config (None: it can): the share is on by default on
    CUDA builds with one GPU and no --batch slots, and a set STRATA_PREFILL_CPU_SHARE already has the 3,072 range."""
    args = engine_args(cfg)
    if "--batch" in args:
        return "batch slots keep the CPU prefill share off"
    if "--layer-split" in args:
        return "a layer split keeps the CPU prefill share off by default"
    if "STRATA_PREFILL_CPU_SHARE" in (cfg.get("env") or {}):
        return "STRATA_PREFILL_CPU_SHARE is set: the share's range is already 3,072"
    try:
        build = json.loads((Path(cfg["exe"]).parent / "BUILD.json").read_text(encoding="utf-8"))
        if build.get("backend") == "hip":
            return "the HIP build keeps the CPU prefill share off by default"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def engine_env(cfg: dict, env: dict | None = None, base: dict | None = None) -> dict:
    """The environment a calibration start runs the engine with: the server's (`child_env`, or `base`) without the
    variables the calibration measures (ENV_DEFAULTS: an earlier calibration's value must not be in the baseline), then
    `env` for this start - a value of None takes the variable away."""
    if base is None:
        from serve.server import child_env
        base = child_env(cfg)
    e = dict(base)
    for k in ENV_DEFAULTS:
        e.pop(k, None)
    for k, v in (env or {}).items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = v
    return e


def chat_ids(tok, text: str) -> list[int]:
    return tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      parse_special=True)


def arg_value(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def with_arg(args: list[str], flag: str, value: str | None) -> list[str]:
    """`args` with `flag value` set (replaced if present), or removed when value is None."""
    out = list(args)
    if flag in out:
        i = out.index(flag)
        del out[i:i + 2]
    if value is not None:
        out += [flag, value]
    return out


def worker_candidates(default: int, extra=()) -> list[int]:
    """The engine's own count, and fewer: two thirds, a half and a quarter (at least 2), plus `extra` (the P-cores
    minus one on a hybrid CPU, one socket's cores minus one on a 2-socket PC), without repeats.  Bench #780 #815
    #707 #674: 4 beat 15 by 28.7% on 8P + 16E; 17-18 beat 35 by ~20% on 2-socket Xeons."""
    c = [default]
    for w in (round(default * 2 / 3), round(default / 2), round(default / 4), *extra):
        if 2 <= w < default and w not in c:
            c.append(w)
    return c


def host_worker_extras() -> list[int]:
    """P-cores - 1 (hybrid CPU) and one socket's cores - 1 (several sockets) of this PC: more counts to try."""
    try:
        import setup
        out = []
        cores, sockets = setup.cpu_cores(), setup.cpu_sockets()
        if cores:
            out.append(cores[0] - 1)
        if sockets and sockets[0] >= 2:
            out.append(sockets[1] - 1)
        return out
    except Exception:
        return []


def pick(measured: dict, default_key, min_gain: float = MIN_GAIN):
    """The key with the best median tok/s, or `default_key` unless the best beats it by more than min_gain."""
    med = {k: statistics.median(v) for k, v in measured.items() if v}
    if not med or default_key not in med:
        return default_key
    best = max(med, key=med.get)
    return best if med[best] > med[default_key] * (1.0 + min_gain) else default_key


class Session:
    """One running engine: measure decode tok/s for a setting (the median of the prompts' rates)."""

    def __init__(self, engine, ids_list):
        self.engine = engine
        self.ids_list = ids_list

    def rate(self, tune: dict | None = None, max_new: int | None = None) -> float:
        rates = []
        for ids in self.ids_list:
            sampling = {"temperature": 0}
            if tune:
                sampling["strata_tune"] = tune
            n = sum(1 for t in self.engine.generate(ids, max_new or MAX_NEW, sampling, threading.Event())
                    if t is not None)
            ms = (self.engine.last or {}).get("decode_ms") or 0.0
            if n > 8 and ms > 0:
                rates.append(n / (ms / 1000.0))
        return statistics.median(rates) if rates else 0.0

    def warm_up(self, rounds: int = 2):
        for _ in range(rounds):
            self.rate()


def run(cfg: dict, say=print, start_engine=None) -> dict:
    """Measure on the engine `cfg` describes; returns {"settings": {flag: value}, "env": {variable: value},
    "report": {...}}.  `start_engine(args, env=None)` returns a started engine (serve.server.StrataEngine or a stand-in
    in tests); `env` is given for step 6 only."""
    if start_engine is None:
        from serve.server import StrataEngine

        def start_engine(args, env=None):
            return StrataEngine(cfg["exe"], args, cwd=cfg.get("cwd"), log=cfg.get("log"), env=engine_env(cfg, env))
    import strata_tokenizer as ST
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tpath / "token_type.json").read_text()))
    ids_list = [chat_ids(tok, p) for p in PROMPTS]
    skip = share_step_applies(cfg)
    if skip:
        say(f"  The CPU prefill share's range is not measured: {skip}.")
    return measure(engine_args(cfg), ids_list, start_engine, say, host_worker_extras(),
                   share_ids=None if skip else share_prompt_ids(tok))


def engine_args(cfg: dict) -> list[str]:
    """The arguments the server starts this config's engine with (serve.server.engine_args), so the tuning measures
    the engine as it runs.  #447: this used to read any "gpu" list as a layer split and add --layer-split auto, which
    broke a one-card config with a helper card for the expert tier ("gpu": [0] + --expert-cache-device1: the engine
    refuses a split that leaves no card without a stage), missed the "0,2" spelling, and ignored split_skip_if_fits."""
    from serve.server import engine_args as server_args
    return server_args(cfg)


def engine_error(log: str | None, since: int = 0) -> str | None:
    """#447: the engine's own reason for a failed start - the last line of its log written after byte `since` that is
    its own ("strata ..." or "ERR ..."), else the last line there; None without a log or a new line."""
    if not log:
        return None
    try:
        with open(log, "rb") as f:
            f.seek(since)
            lines = [x.strip() for x in f.read()[-16384:].decode("utf-8", "replace").splitlines() if x.strip()]
    except OSError:
        return None
    return next((x for x in reversed(lines) if x.startswith(("strata", "ERR"))), lines[-1] if lines else None)


def sweep(s, values: list, base_tune: dict, key: str, label: str, say) -> dict:
    """Every value, `SWEEP_ROUNDS` times, one round visiting all of them and every other round in reverse order.

    A single rate per value cannot order neighbours.  #1332 has 0.75 to 0.95 inside each other's noise at 72.7,
    69.0, 71.0 and 71.9, and that box moved 8-28% between engine starts at an unchanged setting; a 5080/Xeon box
    reports +9.57 tok/s between two blocks at an unchanged setting (p = 0.00004), larger than the effect being
    hunted.  Rotating the values inside one engine start puts that drift on every value the same number of times,
    and the median over the rounds keeps a single bad sample from deciding.  The caller takes the winner from the
    medians, so every value keeps the same chance the others had.
    """
    out = {v: [] for v in values}
    for r in range(SWEEP_ROUNDS):
        for v in (values if r % 2 == 0 else values[::-1]):
            out[v].append(s.rate({**base_tune, key: v}))
        say(f"    round {r + 1}/{SWEEP_ROUNDS} {label}: " +
            "  ".join(f"{v:g} {out[v][-1]:.1f}" for v in values))
    med = {v: statistics.median(out[v]) for v in values}
    say(f"    {label}, median of {SWEEP_ROUNDS}: " +
        ", ".join(f"{v:g}: {med[v]:.1f} ({min(out[v]):.1f}-{max(out[v]):.1f})" for v in values) +
        f" -> best {max(med, key=lambda k: med[k]):g}")
    return out


def measure(base_args: list[str], ids_list, start_engine, say=print, extra_workers=(), share_ids=None) -> dict:
    t0 = time.time()
    report: dict = {}
    say("  Loading the model for the measurements ...")
    base_args = apply(base_args, {})                   # the product defaults: what the measurements must beat
    eng = start_engine(base_args)
    try:
        info = dict(getattr(eng, "info", {}) or {})
        d_pcie = float(info.get("pcie_frac", 0.55))
        d_minp = float(info.get("spec_min_p", 0.5))
        d_workers = int(info.get("pool_workers", 0)) or None
        s = Session(eng, ids_list)
        s.warm_up()
        # 1. the PCIe share, at the default draft floor
        by_pcie = sweep(s, sorted(set(PCIE_FRACS) | {round(d_pcie, 2)}), {"spec_min_p": d_minp}, "pcie_frac",
                        "PCIe share", say)
        best_pcie = max(by_pcie, key=lambda k: statistics.median(by_pcie[k]))
        # 2. the draft floor, at that share
        by_minp = sweep(s, sorted(set(SPEC_MIN_PS) | {round(d_minp, 2)}), {"pcie_frac": best_pcie}, "spec_min_p",
                        "draft floor", say)
        best_minp = max(by_minp, key=lambda k: statistics.median(by_minp[k]))
        # 3. the winner against the default, interleaved, the same number of rounds
        dflt, cand = (round(d_pcie, 2), round(d_minp, 2)), (best_pcie, best_minp)
        confirm = {dflt: [], cand: []}
        if cand != dflt:
            for _ in range(SWEEP_ROUNDS):
                for k in (dflt, cand):
                    confirm[k].append(s.rate({"pcie_frac": k[0], "spec_min_p": k[1]}))
        chosen = pick(confirm, dflt) if cand != dflt else dflt
        report.update(default={"pcie_frac": dflt[0], "spec_min_p": dflt[1], "pool_workers": d_workers},
                      pcie_sweep={str(k): v for k, v in by_pcie.items()},
                      min_p_sweep={str(k): v for k, v in by_minp.items()},
                      confirm={f"{k[0]}/{k[1]}": v for k, v in confirm.items()})
    finally:
        close(eng)
    settings = {}
    if chosen != dflt:
        settings["--pcie-frac"] = f"{chosen[0]:.2f}"
        settings["--spec-min-p"] = f"{chosen[1]:.2f}"
    base_rate = statistics.median(confirm[chosen]) if confirm.get(chosen) else None
    # 4. fewer CPU workers (a restart each), with the chosen settings
    if d_workers and len(worker_candidates(d_workers, extra_workers)) > 1:
        tuned = with_arg(with_arg(base_args, "--pcie-frac", f"{chosen[0]:.2f}"), "--spec-min-p", f"{chosen[1]:.2f}")
        by_workers = {}
        for w in worker_candidates(d_workers, extra_workers):
            say(f"  Measuring with {w} CPU workers (restarts the engine) ...")
            e = restart(start_engine, with_arg(tuned, "--pool-workers", None if w == d_workers else str(w)), f"{w} workers",
                        say, report)
            if e is None:
                continue
            try:
                sw = Session(e, ids_list)
                sw.warm_up(1)
                by_workers[w] = [sw.rate(), sw.rate()]
                say(f"    {w} workers: {statistics.median(by_workers[w]):.1f} tok/s")
            finally:
                close(e)
        report["workers"] = {str(k): v for k, v in by_workers.items()}
        w_best = pick(by_workers, d_workers) if by_workers else d_workers
        if w_best != d_workers:
            settings["--pool-workers"] = str(w_best)
            base_rate = statistics.median(by_workers[w_best])
        elif by_workers.get(d_workers):
            base_rate = statistics.median(by_workers[d_workers])
    # 5. the adaptive expert tier (a restart each), with everything chosen so far
    tuned = apply(base_args, settings)
    by_adapt = {}
    for cand in ADAPT_CANDIDATES:
        key = "default" if cand is None else "/".join(cand)
        args = tuned
        for flag, v in zip(ADAPT_FLAGS, cand or (None,) * len(ADAPT_FLAGS)):
            args = with_arg(args, flag, v)
        say(f"  Measuring the expert tier {'(the engine default)' if cand is None else 'every ' + cand[0] + ', ' + cand[1] + ' swaps, decay ' + cand[2]} (restarts the engine) ...")
        e = restart(start_engine, args, "the expert tier " + key, say, report)
        if e is None:
            continue
        try:
            sa = Session(e, ids_list)
            sa.warm_up(1)
            by_adapt[key] = [sa.rate(max_new=TIER_MAX_NEW), sa.rate(max_new=TIER_MAX_NEW)]
            say(f"    {statistics.median(by_adapt[key]):.1f} tok/s")
        finally:
            close(e)
    a_best = pick(by_adapt, "default") if by_adapt else "default"
    report["adapt"] = by_adapt
    if a_best != "default":
        for flag, v in zip(ADAPT_FLAGS, a_best.split("/")):
            settings[flag] = v
        base_rate = statistics.median(by_adapt[a_best])
    elif by_adapt.get("default"):
        base_rate = statistics.median(by_adapt["default"])
    # 6. the CPU prefill share's range (a restart each, with everything chosen so far): the read speed of a fresh
    # ~2K-token prompt with the engine's default range and with 3,072.  `start_engine(args, env)`: the variable set
    # for the candidate, taken away for the default (an earlier calibration may have written it into the config).
    env_settings: dict = {}
    if share_ids:
        tuned = apply(base_args, settings)
        by_share = {}
        for val in (None, SHARE_RANGE):
            key = "default" if val is None else val
            say(f"  Measuring the prompt speed with the CPU share up to {'1,024 tokens (the engine default)' if val is None else val + ' tokens'} (restarts the engine) ...")
            e = restart(start_engine, tuned, "the CPU share range " + key, say, report, env={SHARE_ENV: val})
            if e is None:
                continue
            try:
                rates = prompt_rates(e, share_ids)
                if rates:
                    by_share[key] = rates
                    say(f"    {statistics.median(rates):.0f} prompt tok/s ({min(rates):.0f}-{max(rates):.0f})")
                else:
                    say("    nothing measured (every read reused a checkpoint): this candidate is dropped")
            except Exception as ex:  # noqa: BLE001 - an ERR or a dead engine on a read: this arm loses, the rest stays
                why = str(ex).strip().splitlines()[0][:300] if str(ex).strip() else type(ex).__name__
                say(f"    the read failed for the CPU share range {key}: {why} - this candidate is dropped")
                report.setdefault("failed_starts", {})["the CPU share range " + key] = why
            finally:
                close(e)
        report["share"] = by_share
        if by_share and pick(by_share, "default") == SHARE_RANGE:
            env_settings[SHARE_ENV] = SHARE_RANGE
    report["seconds"] = round(time.time() - t0)
    report["tok_s"] = round(base_rate, 1) if base_rate else None
    return {"settings": settings, "env": env_settings, "report": report}


def restart(start_engine, args, what, say, report, env=None):
    """#1337: a later step's engine start that fails is a candidate that loses, not a failed run: what the earlier
    steps measured (and the settings they chose) must survive it.  Returns the engine, or None after saying why and
    noting it in report["failed_starts"].  (The first start, for steps 1-3, is not covered: with no engine there is
    nothing measured.)  `env`: engine environment variables for this start (step 6), passed on only when given, so a
    `start_engine(args)` of one argument still serves the other steps."""
    try:
        return start_engine(args, env) if env is not None else start_engine(args)
    except Exception as e:  # noqa: BLE001 - whatever the start raised (RuntimeError, EngineDied, OSError ...)
        why = str(e).strip().splitlines()[0][:300] if str(e).strip() else type(e).__name__
        say(f"    the engine did not start for {what}: {why} - this candidate is dropped, the earlier results are kept")
        report.setdefault("failed_starts", {})[what] = why
        return None


def close(eng):
    proc = getattr(eng, "proc", None)
    if proc is None:
        return
    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(60)
    except Exception:
        proc.kill()


DEFAULTS = {"--pcie-frac": None, "--spec-min-p": "0.5", "--pool-workers": None,   # None: the engine's own choice
            "--adapt-every": None, "--adapt-swaps": None, "--adapt-decay": None}


def apply_env(env: dict | None, settings_env: dict | None) -> dict:
    """The config's "env" with the calibrated variables: one the calibration did not choose goes back to the engine's
    own default (absent), as `apply` does for the arguments, so an older calibration's value never lingers - a value
    set by hand is replaced the same way, as a hand-set --pcie-frac is.  Other variables in `env` are kept as they are."""
    out = dict(env or {})
    for var, default in ENV_DEFAULTS.items():
        v = (settings_env or {}).get(var, default)
        if v is None:
            out.pop(var, None)
        else:
            out[var] = v
    return out


def apply(args: list[str], settings: dict) -> list[str]:
    """`args` with the calibrated settings; a setting the calibration did not change goes back to the product
    default (setup's --spec-min-p 0.5, the engine's own PCIe share and worker count), so an older calibration's
    values never linger."""
    out = list(args)
    for flag, default in DEFAULTS.items():
        out = with_arg(out, flag, settings.get(flag, default))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: calibrate.py <strata-*.json>")
    res = run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig")))
    print(json.dumps(res, indent=1))
