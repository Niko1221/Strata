"""On-device autotuning for the AMD (gfx1100) engine: measure, confirm end to end, and save into the run config.

Two layers, both measured on this PC rather than guessed (docs/AMD_HIP_AUTOTUNE.md):

  1. kernels   - tune_decode times every block shape of the decode pass's GPU expert kernels at THIS model's types
                 and shapes, keeps a variant only if it is bitwise equal to the default and faster twice in a row,
                 and writes a table.  Then the engine is started with and without the table (interleaved) and the
                 table is kept only if the generated tokens are identical and decode is not slower.  The table goes
                 into the config's "env" (STRATA_DECODE_TUNING), like the hipBLASLt table.
  2. settings  - (--settings / --draft) tools/calibrate.py on top: the draft floor and CPU workers, and with --draft
                 the MTP window and prompt lookup.  Saved through setup's calibration record, so updates keep them.

Stop the Strata server first: the kernel timings need the GPU idle and ~1 GB of free VRAM.

    python tools/hip/autotune.py strata-iq3_xxs.json                 # kernels
    python tools/hip/autotune.py strata-iq3_xxs.json --settings      # kernels, then the engine settings
    python tools/hip/autotune.py strata-iq3_xxs.json --draft --prompts my-agent-prompts.json
    python tools/hip/autotune.py strata-iq3_xxs.json --off           # remove the kernel table from the config
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

ENV_KEY = "STRATA_DECODE_TUNING"
NATIVE_DOWN = (20, 23, 42)              # the down formats native_expert_grouped runs (IQ4_NL, IQ4_XS, Q2_0)
IQ_MMVQ = (16, 17, 18, 21, 22, 29)      # the dense formats native_mmvq hands to iq_mmvq
MAX_NEW = 128
MAX_SLOWDOWN = 0.02                     # end to end, the table may not be slower than this (it is bitwise exact)


def say(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- the model's shapes
def shapes_from_tensors(tensors) -> dict:
    """The decode kernels' shapes in a model: {"experts": [(gu, down)], "n_embd", "n_ff", "mmvq": [(t, n_in, n_out)]}.
    `tensors`: objects with .name, .shape (GGUF order: n_in first) and .type_id."""
    gate, down = {}, {}
    mmvq = set()
    n_embd = n_ff = None
    for t in tensors:
        n = t.name
        if n.endswith("ffn_gate_exps.weight"):
            gate[n.split(".")[1]] = t.type_id
            n_embd, n_ff = int(t.shape[0]), int(t.shape[1])
        elif n.endswith("ffn_down_exps.weight"):
            down[n.split(".")[1]] = t.type_id
        elif len(t.shape) == 2 and t.type_id in IQ_MMVQ and "token_embd" not in n:
            mmvq.add((t.type_id, int(t.shape[0]), int(t.shape[1])))
    pairs = sorted({(gate[b], down[b]) for b in gate if b in down and down[b] in NATIVE_DOWN})
    return {"experts": pairs, "n_embd": n_embd, "n_ff": n_ff, "mmvq": sorted(mmvq)}


def model_shapes(cfg: dict) -> dict:
    from gguf_reader import GGUFFile
    args = cfg["args"]
    if "--native" not in args:
        raise SystemExit("autotune: the config has no --native model (the AMD engine runs native packs only)")
    first = Path(args[args.index("--native") + 1])
    files = sorted(first.parent.glob(first.name.replace("00001-of", "*-of"))) if "00001-of" in first.name else [first]
    tensors = []
    for f in files:
        tensors += GGUFFile(f).tensors
    sh = shapes_from_tensors(tensors)
    if not sh["experts"]:
        raise SystemExit(f"autotune: no native expert tensors found in {first}")
    return sh


# ---------------------------------------------------------------- the kernel tuner
def find_tuner(cfg: dict, build: bool = True) -> Path | None:
    exe = Path(cfg["exe"])
    for p in (exe.parent / "tune_decode", ROOT / "build-hip" / "tune_decode", exe.parent.parent / "build-hip" / "tune_decode"):
        if p.exists():
            return p
    cache = ROOT / "build-hip" / "CMakeCache.txt"
    if build and cache.exists():
        say("  Building the kernel tuner (tune_decode) in build-hip ...")
        r = subprocess.run(["cmake", "--build", str(ROOT / "build-hip"), "--target", "tune_decode", "-j",
                            str(max(1, (os.cpu_count() or 2) // 2))])
        if r.returncode == 0 and (ROOT / "build-hip" / "tune_decode").exists():
            return ROOT / "build-hip" / "tune_decode"
    return None


def tuner_args(sh: dict, out: Path, quick: bool) -> list[str]:
    a = ["--n-embd", str(sh["n_embd"]), "--n-ff", str(sh["n_ff"]), "--out", str(out)]
    for gu, dn in sh["experts"]:
        a += ["--expert", f"{gu}:{dn}"]
    for t, ni, no in sh["mmvq"]:
        a += ["--mmvq", f"{t}:{ni}:{no}"]
    if quick:
        a += ["--rounds", "5", "--reps", "20"]
    return a


def parse_results(stdout: str) -> list[dict]:
    """The tuner's RESULT lines as dicts (`changed`, `default_us`, `chosen_us`, ...)."""
    out = []
    for line in stdout.splitlines():
        if not line.startswith("RESULT "):
            continue
        f = line.split()
        r = {"what": " ".join(x for x in f[1:] if "=" not in x)}
        for kv in f[1:]:
            if "=" in kv:
                k, _, v = kv.partition("=")
                r[k] = v if k == "chosen" else float(v)
        r["changed"] = bool(r.get("changed"))
        out.append(r)
    return out


def table_shapes(path: Path) -> int:
    """How many shapes a table changes (lines that are neither comments nor the header)."""
    n = 0
    for line in path.read_text().splitlines():
        s = line.split("#", 1)[0].strip()
        if s and not s.startswith("STRATA_DECODE_TUNING_V1"):
            n += 1
    return n


# ---------------------------------------------------------------- end to end
def engine_starter(cfg: dict):
    from serve.server import StrataEngine, child_env

    def start(table: str | None):
        c = json.loads(json.dumps(cfg))
        env = dict(c.get("env") or {})
        env.pop(ENV_KEY, None)
        if table:
            env[ENV_KEY] = table
        c["env"] = env
        e = child_env(c)
        if not table:
            e.pop(ENV_KEY, None)
        return StrataEngine(c["exe"], c["args"], cwd=c.get("cwd"), log=c.get("log"), env=e)
    return start


def decode_run(eng, ids_list) -> tuple[list[float], list[list[int]]]:
    """Greedy decode of every prompt: (tok/s per prompt, the tokens).  One warm-up request first."""
    def one(ids):
        toks = [t for t in eng.generate(ids, MAX_NEW, {"temperature": 0}, threading.Event()) if t is not None]
        ms = (eng.last or {}).get("decode_ms") or 0.0
        return (len(toks) / (ms / 1000.0) if len(toks) > 8 and ms > 0 else 0.0), toks
    one(ids_list[0])
    rates, toks = [], []
    for ids in ids_list:
        r, t = one(ids)
        rates.append(r)
        toks.append(t)
    return rates, toks


def end_to_end(start, table: str, ids_list, pairs: int = 2, say=say, close=None) -> dict:
    """Start the engine without and with the table, `pairs` times each, interleaved; compare tokens and speed."""
    if close is None:
        from calibrate import close
    runs = {"off": [], "on": []}
    tokens = {"off": [], "on": []}
    for i in range(pairs):
        for arm in ("off", "on"):
            say(f"  End-to-end check {i + 1}/{pairs}: {'with' if arm == 'on' else 'without'} the table (restarts the engine) ...")
            eng = start(table if arm == "on" else None)
            try:
                rates, toks = decode_run(eng, ids_list)
            finally:
                close(eng)
            runs[arm].append(statistics.median(rates))
            tokens[arm].append(toks)
            say(f"    {arm}: {runs[arm][-1]:.1f} tok/s")
    base_det = all(t == tokens["off"][0] for t in tokens["off"])
    same = all(t == tokens["off"][0] for t in tokens["on"])
    off, on = statistics.median(runs["off"]), statistics.median(runs["on"])
    if same:
        verdict, keep = "identical tokens", on >= off * (1.0 - MAX_SLOWDOWN)
    elif pairs >= 2 and not base_det:
        # two runs WITHOUT the table already differ (an adaptive expert tier, for one): tokens cannot judge it
        verdict, keep = "not checkable (the default itself varies between runs)", on >= off * (1.0 - MAX_SLOWDOWN)
    else:
        verdict, keep = "TOKENS DIFFER", False
    return {"off": runs["off"], "on": runs["on"], "off_tok_s": round(off, 1), "on_tok_s": round(on, 1),
            "gain": round(on / off - 1.0, 4) if off > 0 else None, "tokens": verdict, "keep": keep}


def server_running(cfg: dict) -> bool:
    port = int(cfg.get("port") or 8080)
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def save_config(cfg_path: Path, cfg: dict, table: str | None):
    bak = cfg_path.with_name(cfg_path.name + ".bak-autotune")
    if not bak.exists():
        shutil.copy2(cfg_path, bak)
    env = dict(cfg.get("env") or {})
    if table:
        env[ENV_KEY] = table
    else:
        env.pop(ENV_KEY, None)
    if env:
        cfg["env"] = env
    else:
        cfg.pop("env", None)
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="on-device autotuning for Strata's AMD (gfx1100) engine")
    ap.add_argument("config", help="the strata-*.json run config")
    ap.add_argument("--settings", action="store_true", help="also tune the engine settings (draft floor, CPU workers)")
    ap.add_argument("--draft", action="store_true", help="--settings, plus the MTP window and prompt lookup")
    ap.add_argument("--prompts", help="your own prompts for the end-to-end measurements (JSON list or --- blocks)")
    ap.add_argument("--quick", action="store_true", help="fewer rounds, one end-to-end pair (less certain)")
    ap.add_argument("--skip-kernels", action="store_true", help="only the settings (with --settings / --draft)")
    ap.add_argument("--off", action="store_true", help="remove the kernel table from the config and stop")
    ap.add_argument("--force", action="store_true", help="run even if something answers on the server's port")
    a = ap.parse_args(argv)
    cfg_path = Path(a.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    if a.off:
        save_config(cfg_path, cfg, None)
        say(f"  The kernel table is off for {cfg_path.name} (the default shapes; a backup is beside it).")
        return 0
    if cfg.get("backend") != "hip":
        say("  Note: this config is not marked \"backend\": \"hip\"; the kernel table applies to HIP builds only.")
    if server_running(cfg) and not a.force:
        say(f"  Something answers on port {cfg.get('port') or 8080}: stop the Strata server first (the measurements need "
            "the GPU to themselves), or pass --force.")
        return 2
    import calibrate as CAL
    report = {"config": str(cfg_path), "date": time.strftime("%Y-%m-%d %H:%M")}

    if not a.skip_kernels:
        sh = model_shapes(cfg)
        say(f"  Model: experts {', '.join(f'{g}:{d}' for g, d in sh['experts'])} (n_embd {sh['n_embd']}, n_ff {sh['n_ff']})"
            + (f", {len(sh['mmvq'])} i-quant dense shapes" if sh["mmvq"] else ""))
        tuner = find_tuner(cfg)
        if tuner is None:
            say("  The kernel tuner is not built: cmake --build build-hip --target tune_decode  (see docs/AMD_HIP_AUTOTUNE.md)")
            return 1
        out = cfg_path.with_name(cfg_path.stem + ".decode-tuning.txt")
        from serve.server import child_env
        env = child_env(cfg)
        env.pop(ENV_KEY, None)
        say(f"  Timing the decode kernels on this GPU ({tuner}) ...")
        r = subprocess.run([str(tuner), *tuner_args(sh, out, a.quick)], capture_output=True, text=True, env=env)
        sys.stderr.write(r.stderr)
        if r.returncode == 3:
            say("  Not enough free VRAM for the timings: stop the Strata server (and other GPU programs) first.")
            return 2
        if r.returncode == 4:
            say("  The tuner found a variant that is NOT bitwise equal to the default - nothing is saved. Please report it.")
            return 4
        if r.returncode != 0 or not out.exists():
            say(f"  The kernel tuner failed (exit {r.returncode}); the config is unchanged.")
            return 1
        results = parse_results(r.stdout)
        report["kernels"] = results
        n = table_shapes(out)
        if n == 0:
            say("  The default kernel shapes are already the fastest here: nothing to save.")
            save_config(cfg_path, cfg, None)
        else:
            texts = CAL.load_prompts(a.prompts) if a.prompts else list(CAL.PROMPTS) + [CAL.EDIT_PROMPT]
            tok = tokenizer(cfg)
            ids_list = [CAL.chat_ids(tok, t) for t in texts]
            e2e = end_to_end(engine_starter(cfg), str(out), ids_list, pairs=1 if a.quick else 2)
            report["end_to_end"] = e2e
            gain = f"{e2e['gain'] * 100:+.1f}%" if e2e["gain"] is not None else "?"
            say(f"  End to end: {e2e['off_tok_s']} -> {e2e['on_tok_s']} tok/s ({gain}); tokens: {e2e['tokens']}")
            if e2e["keep"]:
                save_config(cfg_path, cfg, str(out))
                say(f"  Kept: {n} kernel shape(s) from {out.name} are in {cfg_path.name}'s env ({ENV_KEY}).")
            else:
                save_config(cfg_path, cfg, None)
                say("  Not kept: " + ("the tokens changed" if e2e["tokens"] == "TOKENS DIFFER"
                                      else "decode was slower with it") + " - the config keeps the default shapes.")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))

    if a.settings or a.draft:
        import setup as S
        ok = S.calibrate_config(cfg_path, draft=a.draft, prompts=a.prompts)
        report["settings"] = "saved" if ok else "unchanged"
    log = cfg_path.with_name(cfg_path.stem + ".autotune.json")
    log.write_text(json.dumps(report, indent=1), encoding="utf-8")
    say(f"  Report: {log.name}")
    return 0


def tokenizer(cfg: dict):
    import strata_tokenizer as ST
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((tpath / "token_type.json").read_text()))


if __name__ == "__main__":
    sys.exit(main())
