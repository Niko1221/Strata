"""Tune the engine's hardware-dependent settings on this PC (setup's --calibrate).

Three settings depend on the machine more than on the model, and the defaults are right for the PC they were
measured on (a Ryzen 5 7600 + RTX 5070 on PCIe 5):
  --pcie-frac     the share of the experts missing from VRAM that are copied over PCIe and run on the GPU instead of
                  on the CPU.  A fast PCIe link and a slow CPU want more; a laptop's x8 link or a fast CPU want less.
  --spec-min-p    how sure the draft layer must be to extend a verify window by another guess.  A slower CPU pays more
                  per extra window row (more experts per window), so it wants a higher floor.
  --pool-workers  the CPU threads that compute experts.  Every physical core is not always best: on hybrid CPUs the
                  efficiency cores can make the whole window wait for them.
The first two are measured through one engine (per-request `strata_tune` keys); the worker count needs a restart
per value.  Decode speed only: the prompt path streams every expert whatever these settings say.

AMD (a config with "backend": "hip"): the PCIe share is left as the config has it (the HIP expert paths - mmap,
resident CPU experts, the staging buffer - were measured with the share they run with; `sweep_pcie=True` sweeps it
anyway), and the draft floor is swept wider: every extra draft token pulls more VRAM-missed experts onto the CPU.

Drafting (`draft=True`, setup's --calibrate-draft): two more settings, a restart per value -
  --spec          the MTP's longest window (with prompt lookup on, the engine allows lookup windows 2 tokens longer)
  --suffix-draft  prompt lookup's shortest match (0 = MTP only)
None of them changes an answer: verification keeps exactly the tokens plain greedy decoding would produce; they
only decide which guesses are checked.  The built-in prompts include one that repeats its input (an edit), where
prompt lookup pays; pass your own (`prompts=`, e.g. requests an agent really sends) for a measurement of YOUR mix.

A setting is kept only when it beats the default by more than MIN_GAIN in an interleaved re-measurement - the
adaptive expert tier and the OS make single measurements noisy by a few percent.

    python tools/calibrate.py strata-q2_0.json        # measure and print; setup.py --calibrate also saves it
    python tools/calibrate.py strata-iq3_xxs.json --draft --prompts my-agent-prompts.json
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
PCIE_FRACS = (0.0, 0.2, 0.35, 0.55, 0.75)
SPEC_MIN_PS = (0.3, 0.5, 0.7)
SPEC_MIN_PS_HIP = (0.3, 0.5, 0.6, 0.7, 0.8)
SPECS = (3, 4, 5)                  # the MTP's longest window (setup's default 4)
SUFFIX_DRAFTS = (0, 3, 6)          # prompt lookup's shortest match (the engine's default 3; 0 = off)
MAX_NEW = 128
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List twelve European capitals with one sentence about each.",
)
# an edit: most of the answer repeats the input, which is what prompt lookup drafts (used with draft=True)
EDIT_PROMPT = (
    "Rename the variable `total` to `running_sum` in this function and return the whole function, nothing else:\n\n"
    "def summarize(values, scale=1.0):\n"
    "    total = 0.0\n"
    "    count = 0\n"
    "    for v in values:\n"
    "        if v is None:\n"
    "            continue\n"
    "        total += v * scale\n"
    "        count += 1\n"
    "    mean = total / count if count else 0.0\n"
    "    return {\"total\": total, \"count\": count, \"mean\": mean}\n"
)


def load_prompts(path) -> list[str]:
    """Prompts from a file: a JSON list of strings, or plain text with prompts separated by a line `---`."""
    text = Path(path).read_text(encoding="utf-8-sig")
    if Path(path).suffix.lower() == ".json" or text.lstrip()[:1] in ("[", "{"):
        items = json.loads(text)
        if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
            raise ValueError(f"{path}: expected a JSON list of strings")
        out = [x for x in items if x.strip()]
    else:
        out = [b.strip() for b in text.replace("\r\n", "\n").split("\n---\n") if b.strip()]
    if not out:
        raise ValueError(f"{path}: no prompts")
    return out


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


def run(cfg: dict, say=print, start_engine=None, prompts=None, draft: bool = False, sweep_pcie=None) -> dict:
    """Measure on the engine `cfg` describes; returns {"settings": {flag: value}, "report": {...}}.
    `start_engine(args)` returns a started engine (serve.server.StrataEngine or a stand-in in tests).
    `prompts`: your own prompt texts (default: the built-in ones, plus an edit when `draft`)."""
    if start_engine is None:
        from serve.server import StrataEngine, child_env

        def start_engine(args):
            return StrataEngine(cfg["exe"], args, cwd=cfg.get("cwd"), log=cfg.get("log"), env=child_env(cfg))
    import strata_tokenizer as ST
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tpath / "token_type.json").read_text()))
    texts = list(prompts) if prompts else list(PROMPTS) + ([EDIT_PROMPT] if draft else [])
    ids_list = [chat_ids(tok, p) for p in texts]
    return measure(engine_args(cfg), ids_list, start_engine, say, host_worker_extras(),
                   backend=cfg.get("backend"), draft=draft, sweep_pcie=sweep_pcie)


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


def measure(base_args: list[str], ids_list, start_engine, say=print, extra_workers=(), backend=None,
            draft: bool = False, sweep_pcie=None) -> dict:
    t0 = time.time()
    hip = backend == "hip"
    if sweep_pcie is None:
        sweep_pcie = not hip
    report: dict = {"backend": backend or "cuda", "draft": bool(draft), "pcie_swept": bool(sweep_pcie)}
    say("  Loading the model for the measurements ...")
    base_args = apply(base_args, {}, backend=backend, draft=draft)   # the product defaults the measurements must beat
    eng = start_engine(base_args)
    try:
        info = dict(getattr(eng, "info", {}) or {})
        d_pcie = float(info.get("pcie_frac", 0.55))
        d_minp = float(info.get("spec_min_p", 0.5))
        d_workers = int(info.get("pool_workers", 0)) or None
        s = Session(eng, ids_list)
        s.warm_up()

        def tune(pcie, minp):
            return {"pcie_frac": pcie, "spec_min_p": minp} if sweep_pcie else {"spec_min_p": minp}
        # 1. the PCIe share, at the default draft floor (NVIDIA; on AMD the config's own share stays)
        by_pcie = {}
        best_pcie = round(d_pcie, 2)
        if sweep_pcie:
            for f in sorted(set(PCIE_FRACS) | {round(d_pcie, 2)}):
                by_pcie[f] = [s.rate(tune(f, d_minp))]
                say(f"    PCIe share {f:.2f}: {by_pcie[f][0]:.1f} tok/s")
            best_pcie = max(by_pcie, key=lambda k: by_pcie[k][0])
        # 2. the draft floor, at that share
        by_minp = {}
        for p in sorted(set(SPEC_MIN_PS_HIP if hip else SPEC_MIN_PS) | {round(d_minp, 2)}):
            by_minp[p] = [s.rate(tune(best_pcie, p))]
            say(f"    draft floor {p:.2f}: {by_minp[p][0]:.1f} tok/s")
        best_minp = max(by_minp, key=lambda k: by_minp[k][0])
        # 3. the winner against the default, interleaved, three times each
        dflt, cand = (round(d_pcie, 2), round(d_minp, 2)), (best_pcie, best_minp)
        confirm = {dflt: [], cand: []}
        if cand != dflt:
            for _ in range(3):
                for k in (dflt, cand):
                    confirm[k].append(s.rate(tune(k[0], k[1])))
        chosen = pick(confirm, dflt) if cand != dflt else dflt
        report.update(default={"pcie_frac": dflt[0], "spec_min_p": dflt[1], "pool_workers": d_workers},
                      pcie_sweep={str(k): v for k, v in by_pcie.items()},
                      min_p_sweep={str(k): v for k, v in by_minp.items()},
                      confirm={f"{k[0]}/{k[1]}": v for k, v in confirm.items()})
    finally:
        close(eng)
    settings = {}
    if chosen != dflt:                                 # (on AMD only the floor can differ: the share is fixed)
        if sweep_pcie:
            settings["--pcie-frac"] = f"{chosen[0]:.2f}"
        settings["--spec-min-p"] = f"{chosen[1]:.2f}"
    base_rate = statistics.median(confirm[chosen]) if confirm.get(chosen) else None
    tuned = with_arg(base_args, "--spec-min-p", f"{chosen[1]:.2f}")
    if sweep_pcie:
        tuned = with_arg(tuned, "--pcie-frac", f"{chosen[0]:.2f}")

    def restarts(label, variants, default_key, measure_args):
        """One engine start per variant: {key: [tok/s, tok/s]}, and the pick against `default_key`."""
        out = {}
        for key in variants:
            say(f"  Measuring with {label} {key} (restarts the engine) ...")
            e = start_engine(measure_args(key))
            try:
                sw = Session(e, ids_list)
                sw.warm_up(1)
                out[key] = [sw.rate(), sw.rate()]
                say(f"    {label} {key}: {statistics.median(out[key]):.1f} tok/s")
            finally:
                close(e)
        return out, pick(out, default_key)

    # 4. drafting (opt-in): the MTP's window, then prompt lookup's match length
    if draft:
        d_spec = int(arg_value(tuned, "--spec") or 4)
        specs = sorted(set(SPECS) | {d_spec})
        by_spec, best_spec = restarts("MTP window", specs, d_spec, lambda v: with_arg(tuned, "--spec", str(v)))
        report["spec"] = {str(k): v for k, v in by_spec.items()}
        if best_spec != d_spec:
            settings["--spec"] = str(best_spec)
            tuned = with_arg(tuned, "--spec", str(best_spec))
        d_sfx = int(arg_value(tuned, "--suffix-draft") or 3)
        sfx = sorted(set(SUFFIX_DRAFTS) | {d_sfx})
        by_sfx, best_sfx = restarts("prompt-lookup match", sfx, d_sfx,
                                    lambda v: with_arg(tuned, "--suffix-draft", None if v == 3 else str(v)))
        report["suffix_draft"] = {str(k): v for k, v in by_sfx.items()}
        if best_sfx != d_sfx:
            settings["--suffix-draft"] = str(best_sfx)
            tuned = with_arg(tuned, "--suffix-draft", None if best_sfx == 3 else str(best_sfx))
        base_rate = statistics.median(by_sfx[best_sfx])
    # 5. fewer CPU workers (a restart each), with the chosen settings
    if d_workers and len(worker_candidates(d_workers, extra_workers)) > 1:
        by_workers, w_best = restarts(
            "CPU workers", worker_candidates(d_workers, extra_workers), d_workers,
            lambda w: with_arg(tuned, "--pool-workers", None if w == d_workers else str(w)))
        report["workers"] = {str(k): v for k, v in by_workers.items()}
        if w_best != d_workers:
            settings["--pool-workers"] = str(w_best)
            base_rate = statistics.median(by_workers[w_best])
        elif by_workers.get(d_workers):
            base_rate = statistics.median(by_workers[d_workers])
    report["seconds"] = round(time.time() - t0)
    report["tok_s"] = round(base_rate, 1) if base_rate else None
    return {"settings": settings, "report": report}


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


DEFAULTS = {"--pcie-frac": None, "--spec-min-p": "0.5", "--pool-workers": None}   # None: the engine's own choice
HIP_DEFAULTS = {"--spec-min-p": "0.5", "--pool-workers": None}                    # the PCIe share is the config's
DRAFT_DEFAULTS = {"--spec": "4", "--suffix-draft": None}                           # setup's --spec, the engine's 3


def apply(args: list[str], settings: dict, backend=None, draft: bool = False) -> list[str]:
    """`args` with the calibrated settings; a setting the calibration did not change goes back to the product
    default (setup's --spec-min-p 0.5, the engine's own PCIe share and worker count), so an older calibration's
    values never linger.  AMD (`backend="hip"`) leaves --pcie-frac as the config has it, unless the settings name
    one; the drafting flags are reset only by a calibration that measured them (`draft`) or that names them."""
    out = list(args)
    flags = dict(HIP_DEFAULTS if backend == "hip" else DEFAULTS)
    if backend == "hip" and "--pcie-frac" in settings:
        flags["--pcie-frac"] = None
    if draft or any(k in settings for k in DRAFT_DEFAULTS):
        flags.update(DRAFT_DEFAULTS)
    for flag, default in flags.items():
        out = with_arg(out, flag, settings.get(flag, default))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="measure Strata's hardware-dependent engine settings on this PC")
    ap.add_argument("config", help="a strata-*.json run config")
    ap.add_argument("--draft", action="store_true", help="also measure the drafting settings (--spec, --suffix-draft)")
    ap.add_argument("--prompts", help="your own prompts: a JSON list of strings, or text blocks separated by ---")
    ap.add_argument("--sweep-pcie", action="store_true", help="AMD: sweep the PCIe share too")
    a = ap.parse_args()
    res = run(json.loads(Path(a.config).read_text(encoding="utf-8-sig")), draft=a.draft,
              prompts=load_prompts(a.prompts) if a.prompts else None, sweep_pcie=True if a.sweep_pcie else None)
    print(json.dumps(res, indent=1))
