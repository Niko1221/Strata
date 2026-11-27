import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

here, build, llama, llama_build, work = map(Path, sys.argv[1:6])
model_arg, pack_arg = sys.argv[6:]
if not model_arg:
    config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    preset = config / "strata/launch-presets/hy4-flash.json"
    if not preset.is_file():
        sys.exit("Pass the first Hy4-Flash GGUF shard and pack directory (see --help).")
    settings = json.loads(preset.read_text())
    model_arg = settings.get("model_path")
    pack_arg = pack_arg or settings.get("pack_path")
if not model_arg or not Path(model_arg).is_file():
    sys.exit("Expected the first shard of a Hy4-Flash GGUF model.")
model = Path(model_arg).absolute()
terminal = here.parents[1] / "terminal"
sys.path.insert(0, str(terminal))
from strata_control.hy4_flash import missing_pack

pack = Path(pack_arg or model.parent / "pack").resolve()
cache_root = Path(os.environ.get("REFERENCE_DIR") or model.parent / "cache")

def available_gpus():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    info = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.free",
                                    "--format=csv,noheader,nounits"], text=True)
    return [gpu.strip() for gpu, free in csv.reader(info.splitlines()) if int(free) >= 46*1024]

def model_shards(first):
    split = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", first.name)
    if not split:
        return [first]
    stem, index, count = split.groups()
    if index != "00001" or int(count) < 1:
        sys.exit("Pass the first GGUF shard (-00001-of-...).")
    return [first.with_name(f"{stem}-{i:05d}-of-{count}.gguf") for i in range(1, int(count) + 1)]

def shard_state(shards):
    stats = [p.stat() for p in shards]
    return [[s.st_dev, s.st_ino, s.st_size, s.st_ctime_ns, s.st_mtime_ns] for s in stats]

def ensure_pack(target):
    marker = target / ".verify-fp16-source.json"
    def ready():
        return not missing_pack(target) and (
            not marker.exists() or json.loads(marker.read_text()) == initial_state)
    if ready():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / ("." + target.name + ".pack.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ready():
            return
        print(f"Preparing {target.name}/ from {model.name}")
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            staged = Path(tmp) / "pack"
            subprocess.run([sys.executable, terminal / "tools/iq_pack.py", "--gguf", model,
                            "--out", staged], check=True)
            missing = missing_pack(staged)
            if missing or shard_state(shards) != initial_state:
                sys.exit(f"GGUF changed or pack is incomplete: {missing}")
            (staged / marker.name).write_text(json.dumps(initial_state))
            if not target.exists():
                staged.rename(target)
            else:
                # Keep an interrupted repair distinguishable from a complete pack.
                invalid = staged.parent / "incomplete.json"
                invalid.write_text("null")
                invalid.replace(marker)
                for item in sorted(staged.rglob("*"), key=lambda p: p.name == marker.name):
                    dest = target / item.relative_to(staged)
                    if item.is_file():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        item.replace(dest)

shards = model_shards(model)
for shard in shards:
    if not shard.is_file():
        sys.exit(f"Missing GGUF shard: {shard}")
initial_state = shard_state(shards)
# Prefer the conventional pack when it belongs to this model. Multiple GGUFs
# in one directory can each keep their own pack.
if not pack_arg:
    index = pack / "index.txt"
    if index.is_file():
        with index.open() as f:
            source = f.readline().strip().partition(" from ")[2]
        if source and Path(source).name != model.name:
            pack = model.with_suffix(".pack")
    elif any(p != model and not re.search(r"-(?!00001)\d{5}-of-\d{5}\.gguf$", p.name)
             for p in model.parent.glob("*.gguf")):
        pack = model.with_suffix(".pack")
ensure_pack(pack)
protocol = hashlib.sha256((here / "parity_cases.tsv").read_bytes() +
                          (here / "parity_reference.cpp").read_bytes()).hexdigest()
cache = cache_root / model.name / "v1.1.0-fp16"
cache.mkdir(parents=True, exist_ok=True)

with (here / "parity_cases.tsv").open() as f:
    cases = list(csv.DictReader(f, delimiter="\t"))
if len({c["case"] for c in cases}) != 15 or sum(int(c["rows"]) for c in cases) != 1263 or len(cases) != 15:
    sys.exit("Incomplete fixed case matrix")
for c in cases:
    name = c["case"]
    if len(c["token_ids"].split(",")) != int(c["prefix"]) + int(c["rows"]):
        sys.exit(f"Invalid token count in {name}")
    (work / (name + ".ids")).write_text(c["token_ids"])

gpu = os.environ.get("GPU") or next(iter(available_gpus()), None)
if gpu is None:
    sys.exit("Auto-selection needs a GPU with at least 46 GiB free. Set GPU to select a card explicitly.")

def prepare_references():
    manifest = cache / "manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.is_file() else {}
    reuse = previous.get("shards") == initial_state and previous.get("protocol") == protocol
    files = previous["files"] if reuse else {}
    refs = {}
    missing = []
    for c in cases:
        name = c["case"]
        target = cache / (name + ".f32")
        refs[name] = target
        if target.is_file():
            stat = target.stat()
            if stat.st_size == int(c["rows"]) * 120832 * 4 and files.get(name) == stat.st_mtime_ns:
                continue
        missing.append(c)
    if not missing:
        print(f"Using {len(refs)} cached references for this model")
        return refs
    ref_gpus = os.environ.get("REFERENCE_GPUS") or ",".join(available_gpus()[:5])
    if not ref_gpus:
        sys.exit("Set REFERENCE_GPUS to GPU(s) with room for the full reference model.")
    print(f"Generating {len(missing)} golden files with v1.1.0", flush=True)
    lib = llama_build / "bin"
    exe = work / "llama-reference"
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", here/"parity_reference.cpp",
                    "-I"+str(llama/"include"), "-I"+str(llama/"ggml/include"), "-L"+str(lib),
                    "-Wl,-rpath,"+str(lib), "-lllama", "-lggml", "-lggml-base", "-o", exe], check=True)
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        tmp = Path(tmp)
        lines = []
        for c in missing:
            name = c["case"]
            (tmp/(name+".ids")).symlink_to(work/(name+".ids"))
            lines.append(f"{c['prefix']} {c['rows']} {name}.ids {name}.f32\n")
        (tmp/"cases").write_text("".join(lines))
        with (work/"reference.log").open("w") as log:
            proc = subprocess.run([exe, model, "cases"], cwd=tmp, stdout=log, stderr=log,
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=ref_gpus))
        if proc.returncode:
            sys.exit((work/"reference.log").read_text()[-4000:] + "\nReference generation failed.")
        if shard_state(shards) != initial_state:
            sys.exit("GGUF changed during reference generation. Retry with stable files.")
        for c in missing:
            name = c["case"]
            captured = tmp/(name+".f32")
            if captured.stat().st_size != int(c["rows"]) * 120832 * 4:
                sys.exit(f"{name}: reference generation returned an unexpected byte count")
            captured.replace(refs[name])
            files[name] = refs[name].stat().st_mtime_ns
        new_manifest = tmp/"manifest.json"
        new_manifest.write_text(json.dumps({"shards": initial_state, "protocol": protocol,
                                            "files": files}) + "\n")
        new_manifest.replace(manifest)
    return refs

with (cache / ".lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    # Keep the selected files even if another run replaces an entry in a shared cache.
    references = {name: path.open("rb") for name, path in prepare_references().items()}

# Isolate the profile from settings used to launch other services.
env = {k:v for k,v in os.environ.items() if not k.startswith(("STRATA_", "HY4_FLASH_"))}
for setting in """
INDEXER=1 DENSE_ARENA=1 DEVICE_EPOCH=1 COMPACT_PAIR_GU=1
RESIDENT_PAIR_GU=1 PIN_EXPERT_MMAP=1 GRAPHS=0 TOKEN_GRAPH=0
ASYNC_MOE=1 HOT_CACHE=1 ASYNC_COMPACT_MISS=1 ASYNC_DEVICE_PLAN=1
ASYNC_ROUTER_PUBLISH=1 ASYNC_SHARED_STREAM=1 CPU_CACHE_FIRST=1 CPU_DIRECT_PARTS=1
PREFILL_PIPELINE=1 PREFILL_RESIDENT_MMQ=1 PREFILL_ROW_PLAN=1 PREFILL_ROW_SCATTER=1
PREFILL_QUANT_REUSE=1 PREFILL_HOST_ASYNC=1 PREFILL_DIRECT_DMA=1 CPU_NATIVE_Q8K=0
PREFILL_LAYER_MAJOR=1 ASYNC_MAP_SYNC_ONCE=0 ASYNC_SHARED_EARLY=0 ASYNC_SINGLE_PHASE=0
HOT_CPU_COLD_FIRST=0 PREFILL_BORROW=1 CPU_SPLIT=auto CPU_MISS_MAX=16
CPU_WORKERS=32 PREFILL_OFFLOAD_MMQ=64 PREFILL_GROUP_ROWS=1024 PREFILL_CHUNK=4096
PREFILL_SUPERCHUNK=8192 KV=f16 GPU_SAMPLE=1 PREFILL_LM_R=vram
ATTN_MMA_PREC=1 WINDOW=0 WINDOW_GRAPH=0 WINDOW_COMPACT_ROWS=0
MTP_GRAPH=0 MTP_CACHE_ONLY=0 WINDOW_SKIP_EMPTY=0 REFERENCE_MIX=1
REFERENCE_ENGRAM=1 REFERENCE_ROUTER=1 CPU_Q81_ALIGNED=1
""".split():
    name, value = setting.split("=")
    env["STRATA_HY4_FLASH_" + name] = value
env.update(
    STRATA_MMQ_KQ="1",
    STRATA_CPU_Q81_VECTOR_QUANT="1",
    STRATA_CPU_Q81_PAIR_GU="1",
    HY4_FLASH_ASYNC_PARALLEL_PIPELINE="0",
    HY4_FLASH_ASYNC_STAGE_PIPELINE="0",
    STRATA_CPU_Q81_PAIR_VNNI="0",
    HY4_FLASH_ASYNC_STAGE_MAX_MISSES="2",
    HY4_FLASH_DIRECT_CONTEXT="8192",
    CUDA_VISIBLE_DEVICES=gpu, STRATA_HY4_FLASH_REFERENCE_OPS_LIBRARY=str(work/"reference.so"),
)

def metrics(ref_path, actual_path, n):
    ref, got = [np.memmap(path, dtype="<f4", mode="r", shape=(n, 120832)) for path in (ref_path, actual_path)]
    matches, total_kl, max_abs = 0, 0., 0.
    for i in range(0, n, 8):
        a, b = ref[i:i+8].astype(np.float64), got[i:i+8].astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("non-finite logits")
        matches += int((a.argmax(1) == b.argmax(1)).sum())
        max_abs = max(max_abs, float(np.max(np.abs(a-b))))
        for x in (a,b):
            x -= x.max(1, keepdims=True)
            x -= np.log(np.exp(x).sum(1, keepdims=True))
        delta = b-a
        total_kl += float((np.exp(a) * (np.expm1(delta)-delta)).sum())
    return matches, total_kl/n, max_abs

print(f"{model.name} | FP16 KV | Q8_1 CPU split | MTP off")
print(f"{'case':<23} {'top-1':>8}   {'mean KL':>10}   {'max |delta|':>11}   result")
failed, total_hits = [], 0
for c in cases:
    name, n = c["case"], int(c["rows"])
    output = work/"actual.f32"
    output.unlink(missing_ok=True)
    cmd = [build/"strata", "--native", model, "--pack", pack, "--expert-vram", "37",
           "--tokens-file", work/(name+".ids"), "--prefix-tokens", c["prefix"], "--dump-logits", output]
    log_path = work/"strata.log"
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=log)
    log = log_path.read_text(errors="replace")
    if proc.returncode or not output.is_file() or output.stat().st_size != n*120832*4:
        sys.exit(f"{name} failed to capture all logits (exit {proc.returncode})\n{log[-3000:]}")
    experts = sum(map(int, re.findall(r"HY4_FLASH_CPU_STATS experts=(\d+)", log)))
    if not experts or "HY4_FLASH_CPU_Q81_ALIGNED active=1" not in log or "HY4_FLASH_CPU_SPLIT mode=auto" not in log:
        sys.exit(f"{name}: aligned Q8_1 CPU split was not exercised")
    hits, kl, diff = metrics(references[name], output, n)
    ok = hits == n and kl <= 1e-10 and diff <= 1e-5
    total_hits += hits
    print(f"{name:<23} {f'{hits}/{n}':>8}   {kl:10.3g}   {diff:11.3g}   {'PASS' if ok else 'FAIL'}")
    if not ok:
        failed.append(name)
if shard_state(shards) != initial_state:
    sys.exit("GGUF changed during validation. Retry with stable files.")
print(f"{len(cases)-len(failed)}/15 cases passed | Top-1 {total_hits}/1263")
if failed:
    sys.exit("Failed: " + ", ".join(failed))
