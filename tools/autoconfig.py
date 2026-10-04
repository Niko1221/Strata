#!/usr/bin/env python3
"""Batching settings from the machine: reads the GPUs (count, VRAM, PCIe link), the RAM and the model, and writes a
copy of a setup.py config with the layer split, --batch / --batch-groups, --trim-stage-weights, the context, the
VRAM reserve and the parking cache chosen for it (docs/BATCHING.md, "Choosing the settings").

  python3 tools/autoconfig.py --config strata-<model>.json                      # rules only: prints, writes .auto.json
  python3 tools/autoconfig.py --config strata-<model>.json --calibrate          # + measures the candidates, keeps the best

The rules come from measurements on a few machines; --calibrate starts the engine with each candidate (a few
minutes each) and measures 1, 2, 4 and all slots, so on other hardware it is the one to trust.  Nothing is changed
in the given config: the result goes to --out (default: the config's name with .auto.json).
"""
import argparse, json, os, shutil, struct, subprocess, sys, time
from pathlib import Path

GIB = 1 << 30
CONTEXTS = (262144, 131072, 65536, 32768)
# measured (IQ3_S, int8 K/V): a sequence's K/V is ~263 bytes per layer per token of context, held in pinned host
# memory with KV streaming (0.77 GiB per 12-layer stage and per sequence at 262,144 tokens)
KV_BYTES_PER_LAYER_TOKEN = 263
# measured: a slot's MTP drafter (--batch-spec) needs ~175 MiB of VRAM per slot at 262,144 tokens of context
DRAFTER_MIB_PER_SLOT_262K = 175
BASE_RESERVE_MIB = 700
# the arguments this tool decides: name -> how many values follow it
MANAGED = {"--batch": 1, "--batch-groups": 1, "--trim-stage-weights": 0, "--vram-reserve-mib": 1, "--max-context": 1,
           "--conversation-cache-mib": 1, "--conversation-cache-slots": 1, "--batch-spec": 1,
           "--batch-spec-max-active": 1, "--layer-split": 1}


def gpus():
    q = "index,name,memory.total,pcie.link.gen.max,pcie.link.width.max,pcie.link.width.current"
    try:
        s = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    found = []
    for line in s.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        num = lambda x: int(x) if x.isdigit() else 0  # noqa: E731
        found.append({"index": num(f[0]), "name": f[1], "mib": num(f[2]), "gen": num(f[3]), "width_max": num(f[4]),
                      "width": num(f[5])})
    return found


def meminfo():
    m = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            m[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return m.get("MemTotal", 0), m.get("MemAvailable", 0)


def gguf_block_count(path):
    """The model's layer count from a GGUF's metadata (<arch>.block_count); None if it cannot be read."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            version, = struct.unpack("<I", f.read(4))
            if version < 2:
                return None
            _, n_kv = struct.unpack("<QQ", f.read(16))
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

            def string():
                n, = struct.unpack("<Q", f.read(8))
                return f.read(n)

            def skip(t):
                if t == 8:
                    string()
                elif t == 9:
                    et, n = struct.unpack("<IQ", f.read(12))
                    for _ in range(n):
                        skip(et)
                else:
                    f.read(sizes[t])

            for _ in range(n_kv):
                key = string().decode("utf-8", "replace")
                t, = struct.unpack("<I", f.read(4))
                if key.endswith(".block_count") and t in (4, 5):
                    return struct.unpack("<I", f.read(4))[0]
                skip(t)
    except (OSError, KeyError, struct.error):
        return None
    return None


def arg(args, name, default=None):
    return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else default


def strip_managed(args):
    out, i = [], 0
    while i < len(args):
        n = MANAGED.get(args[i])
        if n is None:
            out.append(args[i])
            i += 1
        else:
            i += 1 + n
    return out


def split_points(layers, vram):
    """Stage boundaries for the cards in order, each stage's share of the layers proportional to its VRAM."""
    total, acc, pts = sum(vram), 0, []
    for v in vram[:-1]:
        acc += v
        pts.append(round(layers * acc / total))
    return pts


def plan(cfg, cards, ram_total, layers, context=None, spec=0):
    """The rules.  Returns (settings, notes)."""
    notes = []
    n = len(cards)
    s = {}
    if n > 1:
        s["layer_split"] = ",".join(map(str, split_points(layers, [c["mib"] for c in cards])))
        s["trim"] = True
        s["batch"] = min(8, 2 * n)
        s["groups"] = n if s["batch"] % n == 0 else 1   # one group per card: card k runs a group while k+1 runs the next
    else:
        s["layer_split"] = None
        s["trim"] = False
        s["batch"] = 4
        s["groups"] = 1
    for c in cards:
        if c["width"] and c["width_max"] and c["width"] < c["width_max"]:
            notes.append(f"GPU {c['index']} runs its link at x{c['width']} of x{c['width_max']} (a riser, a shared slot, "
                         "or a card to reseat): every hand-off of the split crosses it")
    # context: every sequence (the solo session and each slot) holds its K/V in pinned host memory with KV streaming;
    # keep that under a quarter of the RAM, the rest goes to the experts the GPUs do not hold and the page cache
    seqs = 1 + s["batch"]
    budget = ram_total / 4
    want = context or int(arg(cfg["args"], "--max-context", CONTEXTS[0]))
    ctx = next((c for c in CONTEXTS if c <= want and seqs * layers * c * KV_BYTES_PER_LAYER_TOKEN <= budget), None)
    while ctx is None and s["batch"] > 2:
        s["batch"] -= n if n > 1 else 1
        s["groups"] = n if n > 1 and s["batch"] % n == 0 else 1
        seqs = 1 + s["batch"]
        ctx = next((c for c in CONTEXTS if c <= want and seqs * layers * c * KV_BYTES_PER_LAYER_TOKEN <= budget), None)
    s["context"] = ctx or CONTEXTS[-1]
    if s["context"] < want:
        notes.append(f"context {s['context']:,} instead of {want:,}: {seqs} sequences x {layers} layers of K/V must stay "
                     f"under a quarter of the RAM ({ram_total / GIB:.0f} GiB)")
    s["kv_gib"] = seqs * layers * s["context"] * KV_BYTES_PER_LAYER_TOKEN / GIB
    s["spec"] = spec
    reserve = BASE_RESERVE_MIB
    if spec > 1:
        reserve += -(-DRAFTER_MIB_PER_SLOT_262K * s["batch"] * s["context"] // CONTEXTS[0] // 100) * 100
    s["reserve"] = reserve
    s["parking_mib"] = int(min(8192, ram_total * 0.08 / (1 << 20)) // 512 * 512)
    return s, notes


def apply(cfg, s):
    c = json.loads(json.dumps(cfg))
    a = strip_managed(c["args"])
    a += ["--max-context", str(s["context"]), "--vram-reserve-mib", str(s["reserve"]),
          "--batch", str(s["batch"])]
    if s["groups"] > 1:
        a += ["--batch-groups", str(s["groups"])]
    if s["trim"]:
        a += ["--trim-stage-weights"]
    if s["spec"] > 1:
        a += ["--batch-spec", str(s["spec"])]
    if s["parking_mib"] >= 1024:
        a += ["--conversation-cache-mib", str(s["parking_mib"]), "--conversation-cache-slots", "4"]
    c["args"] = a
    if s["layer_split"]:
        c["layer_split"] = s["layer_split"]   # the server and the tools pass it as --layer-split (an explicit one:
    else:                                     # --trim-stage-weights needs it)
        c.pop("layer_split", None)
    return c


def calibrate(exe, cfg, candidates, max_new):
    """Starts the engine with each candidate config and measures the aggregate rate at 1, 2, 4 and all slots."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from batch_test import Engine, QUESTIONS, tokenizer  # noqa: E402
    tok = tokenizer(cfg["tokenizer"])
    prompts = [tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                          parse_special=True) for q in QUESTIONS]
    results = []
    for name, c in candidates:
        print(f"-- {name}: starting the engine", flush=True)
        batch = int(arg(c["args"], "--batch", 1))
        try:
            eng = Engine(exe, {**c, "args": [x for x in c["args"]]}, 0, {}, [])
        except SystemExit as e:
            print(f"   did not start ({e}); skipped", flush=True)
            results.append((name, c, {}))
            continue
        out, rates = eng.lines(), {}
        eng.send(f"GEN 16 {','.join(map(str, prompts[0]))}")   # warm-up
        for line in out:
            if line.startswith(("DONE", "ERR")):
                break
        for k in sorted({1, 2, 4, batch}):
            if k > batch:
                continue
            t0, total, done = time.time(), 0, set()
            if k == 1:   # one request: the server's solo path (GEN, with drafts)
                eng.send(f"GEN {max_new} {','.join(map(str, prompts[0]))}")
                for line in out:
                    if line.startswith("T "):
                        total += 1
                    elif line.startswith(("DONE", "ERR")):
                        break
                rates[1] = total / max(time.time() - t0, 1e-9)
                print(f"   1 request (solo): {rates[1]:6.1f} tok/s", flush=True)
                continue
            for i in range(k):
                ids = prompts[i % len(prompts)]
                eng.send(f"BGEN {i} {max_new} {','.join(map(str, ids))}")
                for line in out:
                    if line.startswith(("T ", "BT ")):
                        total += 1
                    elif line.startswith("BDONE "):
                        done.add(int(line.split()[1]))
                    elif line.startswith("BADM "):
                        if line.split()[2] == "0":
                            done.add(i)
                        break
                    elif line.startswith("ERR"):
                        break
            for line in out:
                if line.startswith("BT "):
                    total += 1
                elif line.startswith("BDONE "):
                    done.add(int(line.split()[1]))
                    if len(done) >= k:
                        break
                elif line.startswith("ERR"):
                    break
            rates[k] = total / max(time.time() - t0, 1e-9)
            print(f"   {k} slot(s): {rates[k]:6.1f} tok/s", flush=True)
        eng.send("QUIT")
        eng.p.wait(timeout=180)
        results.append((name, c, rates))
    # the best: the highest mean over the measured loads, each relative to the best candidate at that load
    loads = sorted({k for _, _, r in results for k in r})
    best = {k: max((r.get(k, 0) for _, _, r in results), default=0) or 1 for k in loads}
    score = lambda r: sum(r.get(k, 0) / best[k] for k in loads) / max(len(loads), 1)  # noqa: E731
    return max(results, key=lambda x: score(x[2])), results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="the setup.py config (strata-<model>.json)")
    ap.add_argument("--out", help="where to write the result (default: <config>.auto.json)")
    ap.add_argument("--context", type=int, help="the largest context wanted (default: the config's --max-context)")
    ap.add_argument("--calibrate", action="store_true", help="measure the candidates with the engine and keep the best")
    ap.add_argument("--exe", help="the engine (default: the config's exe)")
    ap.add_argument("--max-new", type=int, default=96, help="tokens per request while calibrating")
    a = ap.parse_args()
    path = Path(a.config)
    cfg = json.loads(path.read_text())
    every = gpus()
    if not every:
        raise SystemExit("autoconfig: nvidia-smi found no GPU")
    chosen = [g for g in every if g["index"] in (cfg.get("gpu") or [g["index"] for g in every])]
    ram_total, ram_avail = meminfo()
    native = arg(cfg["args"], "--native")
    layers = (gguf_block_count(native) if native else None) or 48
    print(f"GPUs: " + ", ".join(f"{g['index']}: {g['name']} {g['mib'] / 1024:.0f} GiB PCIe Gen{g['gen']} "
                                f"x{g['width']}" for g in chosen))
    print(f"RAM: {ram_total / GIB:.0f} GiB ({ram_avail / GIB:.0f} available now); model: {layers} layers")
    s, notes = plan(cfg, chosen, ram_total, layers, a.context)
    best = apply(cfg, s)
    if a.calibrate:
        exe = a.exe or cfg.get("exe")
        if not exe or not shutil.which(exe) and not Path(exe).exists():
            raise SystemExit("autoconfig: --calibrate needs the engine (--exe)")
        cands = [("rules", best)]
        s2, _ = plan(cfg, chosen, ram_total, layers, a.context, spec=2)
        cands.append(("rules + --batch-spec 2", apply(cfg, s2)))
        if s["groups"] > 1:
            cands.append(("rules, no pipeline groups", apply(cfg, {**s, "groups": 1})))
        (name, best, _), results = calibrate(exe, cfg, cands, a.max_new)
        print(f"best: {name}")
    else:
        print("rules only (--calibrate measures them):")
    for k in ("layer_split", "batch", "groups", "trim", "context", "reserve", "parking_mib"):
        print(f"  {k:12s} {s[k]}")
    print(f"  pinned K/V   ~{s['kv_gib']:.1f} GiB for {1 + s['batch']} sequences")
    for nline in notes:
        print("  note:", nline)
    out = Path(a.out) if a.out else path.with_name(path.name.replace(".json", "") + ".auto.json")
    out.write_text(json.dumps(best, indent=1) + "\n")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
