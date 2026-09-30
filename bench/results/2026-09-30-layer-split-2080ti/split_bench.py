"""The harness behind this folder's numbers: start serve/server.py with a variant of an installed config, send decode
and needle requests, and read the engine log (per-request speed, decode hit rate, STRATA_SPLIT_TIMING's stages).

    python split_bench.py NAME --gpu 0,1 [--config strata-iq2_xs.json] [--exe build-dev/strata] [--layer-split auto|K]
                          [--arg="--pcie-frac 0.4" ...] [--env K=V ...] [--decode-reps 2] [--lengths 2k,8k,16k,30k]
                          [--no-decode] [--no-prompts] [--gpu none]
    python split_bench.py NAME --gpu 0,1 --lend-check

--gpu none leaves every card visible without a split (the helper-GPU caches).  --lend-check runs with a fixed cache
(no swaps, no PCIe share, no suffix drafts): a story and a code answer before and after a 16K prompt, which lends the
top slots of every card's cache and refills them, must be byte-identical.  Results go to runs/NAME/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import needle_bench as nb  # noqa: E402

DECODE = [
    ("story", "Write a short story (about 400 words) about a lighthouse keeper who finds a message in a bottle."),
    ("code", "Implement an LRU cache in Python with O(1) get and put, with a short explanation and a usage example."),
    ("explain", "Explain how TCP congestion control works: slow start, congestion avoidance, fast retransmit and "
                "fast recovery."),
]
PROMPT_RE = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                       r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
HIT_RE = re.compile(r"decode expert cache hit rate: ([\d.]+)%")
STAGE_RE = re.compile(r"stage (\d+): (\d+) windows; per window: wait for the GPU ([\d.]+) ms, pool \+ plan ([\d.]+) ms, "
                      r"host staging ([\d.]+) ms, commit ([\d.]+) ms")


def chat(url, content, max_tokens, timeout=1800):
    body = {"model": "strata", "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"].get("content") or "", out["usage"], time.time() - t0


def smi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return str(e)


def needle(n_tokens, tag, word):
    text = nb.haystack(int(n_tokens * 0.98 * nb.CHARS_PER_TOKEN))
    cut = len(text) // 2
    return (f"[{tag}]\n" + text[:cut] + f"\n\nThe secret code word is: {word}.\n\n" + text[cut:] +
            "\n\nWhat is the secret code word hidden in the text above? Answer with the word only.")


def parse_log(lines):
    reqs = []
    for line in lines:
        if m := PROMPT_RE.search(line):
            reqs.append({"n": int(m[1]), "reused": int(m[2]), "read": int(m[3]), "pp_ms": int(m[4]),
                         "pp_tps": float(m[5]), "gen": int(m[6]), "dec_ms": int(m[7]), "dec_tps": float(m[8]),
                         "acc": int(m[9]), "off": int(m[10])})
        elif (m := HIT_RE.search(line)) and reqs:
            reqs[-1]["hit"] = float(m[1])
        elif (m := STAGE_RE.search(line)) and reqs:
            reqs[-1].setdefault("stages", []).append(
                {"st": int(m[1]), "win": int(m[2]), "wait": float(m[3]), "pool": float(m[4]), "host": float(m[5]),
                 "commit": float(m[6])})
    return reqs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name")
    ap.add_argument("--config", default=str(ROOT / "strata-iq2_xs.json"))
    ap.add_argument("--gpu", default="0,1")
    ap.add_argument("--exe", default=None, help="engine binary (default: the config's)")
    ap.add_argument("--layer-split", default="auto")
    ap.add_argument("--arg", action="append", default=[], help="extra engine argument(s), e.g. --arg=\"--pcie-frac 0.4\"")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--port", type=int, default=8181)
    ap.add_argument("--decode-reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=450)
    ap.add_argument("--lengths", default="2k,8k,16k,30k")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--no-prompts", action="store_true")
    ap.add_argument("--lend-check", action="store_true")
    a = ap.parse_args()

    out_dir = (Path("runs") / a.name).resolve()   # the server runs from the repository root
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    if a.exe:
        cfg["exe"] = str(Path(a.exe).resolve())
    cfg["log"] = str((out_dir / "engine.log").resolve())
    if a.gpu == "none":
        cfg.pop("gpu", None)
    else:
        gpus = [int(x) for x in a.gpu.split(",")]
        cfg["gpu"] = gpus if len(gpus) > 1 else gpus[0]
    cfg["layer_split"] = a.layer_split
    extra = [w for x in a.arg for w in x.split()]
    if a.lend_check:
        extra += ["--adapt-every", "0", "--pcie-frac", "0", "--suffix-draft", "0"]
    cfg["args"] = list(cfg["args"]) + extra
    cfg.pop("port", None)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    Path(cfg["log"]).unlink(missing_ok=True)
    env = dict(os.environ, STRATA_SPLIT_TIMING="1")
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    url = f"http://127.0.0.1:{a.port}"
    t_start = time.time()
    p = subprocess.Popen([sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata",
                          "--config", str(out_dir / "config.json"), "--port", str(a.port)],
                         cwd=str(ROOT), env=env, stdout=open(out_dir / "server.out", "w"), stderr=subprocess.STDOUT)
    res = {"name": a.name, "gpu": a.gpu, "layer_split": a.layer_split, "extra": extra, "env": a.env}
    try:
        while True:
            if p.poll() is not None:
                raise SystemExit(f"the server exited ({p.returncode}); see {out_dir}")
            try:
                with urllib.request.urlopen(url + "/v1/models", timeout=5):
                    break
            except OSError:
                time.sleep(3)
        res["load_s"] = round(time.time() - t_start, 1)
        res["smi_loaded"] = smi()
        if a.lend_check:
            story, code = DECODE[0][1], DECODE[1][1]
            s1, c1 = chat(url, story, 200)[0], chat(url, code, 200)[0]
            ans, usage, _ = chat(url, needle(16 * 1024, a.name, "amber-7"), 12)
            s2, c2 = chat(url, story, 200)[0], chat(url, code, 200)[0]
            res.update({"story_same": s1 == s2, "code_same": c1 == c2, "needle_found": "amber-7" in ans,
                        "prompt_tokens": usage.get("prompt_tokens"), "outputs": [s1, s2, c1, c2]})
            print(json.dumps({k: res[k] for k in ("story_same", "code_same", "needle_found", "prompt_tokens")}))
        else:
            chat(url, "Say hello.", 16)   # warm-up: the first windows capture their graphs
            runs = []
            for rep in range(0 if a.no_decode else a.decode_reps):
                for tag, text in DECODE:
                    txt, usage, dt = chat(url, f"[{a.name} r{rep}] {text}", a.max_tokens)
                    runs.append({"kind": "decode", "tag": tag, "rep": rep, "completion": usage.get("completion_tokens"),
                                 "wall_s": round(dt, 2), "head": txt[:80]})
                    print(f"[{a.name}] decode {tag} r{rep}: {usage.get('completion_tokens')} tok in {dt:.1f} s", flush=True)
            for L in ([] if a.no_prompts else [x for x in a.lengths.split(",") if x]):
                n = int(float(L[:-1]) * 1024) if L[-1] in "kK" else int(L)
                word = nb.WORDS[n % len(nb.WORDS)] + "-" + str(n % 997)
                txt, usage, dt = chat(url, needle(n, a.name, word), 12)
                runs.append({"kind": "prompt", "len": L, "prompt_tokens": usage.get("prompt_tokens"),
                             "found": word in txt, "wall_s": round(dt, 2)})
                print(f"[{a.name}] prompt {L}: {usage.get('prompt_tokens')} tok in {dt:.1f} s, found={word in txt}",
                      flush=True)
            res["runs"] = runs
        res["smi_end"] = smi()
    finally:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
    log = Path(cfg["log"]).read_text(errors="replace").splitlines()
    reqs = parse_log(log)
    res["requests"] = reqs
    res["startup"] = [line for line in log if any(k in line for k in (
        "layer split", "expert cache", "prompt path", "prompt chunk", "VRAM free", "expert arena", "pcie_frac"))]
    dec = [r for r in reqs if r["gen"] >= 100]
    if dec:
        res["decode_mean_tps"] = round(sum(r["dec_tps"] for r in dec) / len(dec), 2)
        res["decode_ms_per_window"] = round(sum(r["dec_ms"] for r in dec) / max(sum(r["gen"] - r["acc"] for r in dec), 1), 2)
        res["decode_mean_hit"] = round(sum(r.get("hit", 0) for r in dec) / len(dec), 2)
    res["prompt_tps"] = {r["read"]: r["pp_tps"] for r in reqs if r["read"] >= 1000}
    (out_dir / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res.get(k) for k in ("name", "load_s", "decode_mean_tps", "decode_ms_per_window",
                                              "decode_mean_hit", "prompt_tps", "smi_loaded")}, indent=1))


if __name__ == "__main__":
    main()
