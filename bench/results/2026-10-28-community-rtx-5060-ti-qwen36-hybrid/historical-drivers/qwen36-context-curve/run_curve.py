"""Phase B: 24 fresh-prompt measurements on the unchanged native Windows baseline.

One synthetic mixed English/Chinese/Python ledger workload; four prompt lengths,
three paired repetitions, MTP 0/2, fixed 32K context, fixed 64 expert slots.
Uses the engine protocol, not HTTP. No profiler or kernel verification is attached.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.request

from windows_telemetry import Telemetry, thread_affinities

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
ROOT = REPO / "build-native-baseline"
REVISION = "83e6b05cd7b97ee194f330f0247faa76deb73f1b"
SOURCE = ROOT / ("src-" + REVISION)
BUILD = ROOT / ("build-" + REVISION + "-cuda132-sm120")
EXE = BUILD / "strata-q36.exe"
CUDA = ROOT / "toolchains/cuda-13.2.0"
MODEL = Path(os.environ["STRATA_REPORT_MODEL"])
EXPECTED_EXE = "7abadfdbac3888b520cb00be77a5f2e70887acc95c8b98ab45316726f78a728b"
LENGTH_ORDERS = [[1024, 4096, 8192, 16384], [16384, 8192, 4096, 1024], [4096, 16384, 1024, 8192]]
KEYS = "temperature=0 top_p=1 top_k=20 min_p=0 penalty_present=0 penalty_freq=0 penalty_repeat=1 penalty_last_n=0 seed=0"
ENV = {"STRATA_Q36_FIXED_SLOTS": "1", "STRATA_MMQ_KQ": "1", "STRATA_Q8_PACKED": "1",
       "CUDA_VISIBLE_DEVICES": "0", "CUDA_MODULE_LOADING": "EAGER"}
URL = "http://127.0.0.1:8080"


def save(name, data):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def tokens_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def ledger(seed):
    rng = random.Random(seed)
    operations = ["evict", "admit", "restore", "prefetch", "merge", "route", "schedule"]
    regions = ["alpha", "delta", "east", "north", "omega", "south", "west"]
    lines = ["Synthetic performance ledger: all values below are fictional test inputs, not measured Strata results.\n"]
    for i in range(650):
        a, b, c = (rng.randrange(100, 100000) for _ in range(3))
        op, region = rng.choice(operations), rng.choice(regions)
        lines.append(f"Record {i:04d}: region={region}, operation={op}, key={a}, budget={b}, cost={c}.\n"
                     f"说明：这是一条合成记录，编号{i}；需要检查预算、缓存逐出、状态恢复与请求隔离。\n"
                     f"def {op}_{i:04d}(queue, limit={b}):\n"
                     f"    selected = [x for x in queue if (x + {a}) % 17 < 9]\n"
                     f"    return selected[:limit], sum(selected) % {c}\n"
                     "Review the bounds and account for the memory cost of retained entries.\n\n")
    return "".join(lines)


def prepare_workloads():
    sys.path.insert(0, str(SOURCE / "tools"))
    from strata_tokenizer import Tokenizer
    tok = Tokenizer.from_gguf(MODEL)
    workloads = {}
    for rep in range(3):
        body = tok.encode(ledger(20261022 + rep))
        for target in (1024, 4096, 8192, 16384):
            case = f"r{rep}-l{target}"
            prefix = f"BENCHMARK_CASE_{case}\nRead this synthetic ledger and review its proposed cache design.\n\n"
            suffix = ("\n\nEnd of excerpt. Write a detailed technical review in English, explaining algorithmic complexity, "
                      "memory ownership, cancellation, fairness and correctness risks. Include pseudocode and concrete examples. "
                      "Aim for at least 400 words; do not just list the record identifiers.")
            template = lambda text: f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            overhead = len(tok.encode(template(prefix + suffix), parse_special=True))
            excerpt = body[:target - overhead]
            while "\ufffd" in tok.decode(excerpt):
                excerpt = excerpt[:-1]
            prompt = template(prefix + tok.decode(excerpt) + suffix)
            ids = tok.encode(prompt, parse_special=True)
            if abs(len(ids) - target) > 8:
                raise RuntimeError(f"unexpected token length: {case}: {len(ids)}")
            if tok.decode(ids) != prompt:
                raise RuntimeError(f"tokenizer round trip failed: {case}")
            (OUT / (case + ".prompt.txt")).write_text(prompt, encoding="utf-8")
            save(case + ".ids.json", ids)
            workloads[case] = {"case": case, "repetition": rep, "target_tokens": target,
                               "prompt_tokens": len(ids), "prompt_ids_sha256": tokens_hash(ids),
                               "prompt_text_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    save("workloads.json", workloads)
    return workloads


def request(path, post=False):
    req = urllib.request.Request(URL + path, data=b"{}" if post else None,
                                 headers={"Content-Type": "application/json", "Origin": URL})
    with urllib.request.urlopen(req, timeout=600 if post else 10) as response:
        return json.load(response)


def measured_request(engine, ids, cap=256):
    start = time.perf_counter()
    engine.send(f"GEN {cap} {KEYS} " + ",".join(map(str, ids)))
    output, pp, ttft = [], [], None
    while True:
        line = engine.line(timeout=300)
        if line is None or line.startswith("ERR"):
            raise RuntimeError(f"engine failed: {line}")
        if line.startswith("T "):
            if ttft is None:
                ttft = time.perf_counter() - start
            output.append(int(line.split()[1]))
        elif line.startswith("PP "):
            pp.append({"client_elapsed_s": time.perf_counter() - start, "line": line})
        elif line.startswith("DONE "):
            wall = time.perf_counter() - start
            f = line.split()
            produced, count = int(f[1]), int(f[2])
            prompt_ms, decode_ms = float(f[3]), float(f[4])
            reused, fresh = int(f[8]), int(f[14])
            if len(output) != produced or count != len(ids) or fresh + reused != count:
                raise RuntimeError(f"protocol count mismatch: {line}")
            if not output or ttft is None or decode_ms <= 0:
                raise RuntimeError(f"empty output/timing: {line}")
            return {"done": line, "prompt_tokens": count, "reused_tokens": reused, "fresh_tokens": fresh,
                    "generated_tokens": produced, "prompt_ms": prompt_ms, "decode_ms": decode_ms,
                    "decode_ms_per_token": decode_ms / produced, "decode_tok_s": produced * 1000 / decode_ms,
                    "prompt_tok_s": fresh * 1000 / prompt_ms if prompt_ms else None,
                    "finish": f[5], "draft_accepted": int(f[6]), "draft_offered": int(f[7]),
                    "client_ttft_s": ttft, "client_total_s": wall,
                    "client_tpot_ms": (wall - ttft) * 1000 / (produced - 1) if produced > 1 else None,
                    "output_ids": output, "output_sha256": tokens_hash(output), "prompt_progress": pp}


def run_matrix(workloads, report):
    sys.path.insert(0, str(SOURCE / "tools/qwen35"))
    from q36_harness import Eng
    env = {key: value for key, value in os.environ.items() if not key.startswith("STRATA_")}
    env.update(ENV)
    env["PATH"] = os.pathsep.join([str(CUDA / "bin/x64"), str(CUDA / "bin"), str(BUILD), env.get("PATH", "")])
    common = ["--native", str(MODEL), "--max-context", "32768", "--prefill", "512",
              "--pool-workers", "7", "--expert-vram", "10.0", "--expert-slots", "64"]
    for rep, lengths in enumerate(LENGTH_ORDERS):
        for mtp in ([0, 2] if rep % 2 == 0 else [2, 0]):
            arm = f"r{rep}-mtp{mtp}"
            print("START", arm, flush=True)
            engine, telemetry = None, None
            arm_info = {"arm": arm, "repetition": rep, "mtp": mtp, "args": common + ["--mtp", str(mtp)]}
            report["arms"].append(arm_info)
            try:
                engine = Eng(str(EXE), arm_info["args"], env=env, log=str(OUT / (arm + ".engine.log")))
                telemetry = Telemetry(engine.p.pid)
                warm_ids = json.loads((OUT / f"r{(rep + 1) % 3}-l1024.ids.json").read_text())
                # Each new engine gets the same 64-output-token warm-up shape, excluded from measured rows.
                arm_info["warmup"] = measured_request(engine, warm_ids, cap=64)
                arm_info["load_s"] = engine.load_s
                arm_info["process_affinity"] = telemetry.process.cpu_affinity()
                arm_info["thread_affinities"] = thread_affinities(engine.p.pid)
                for target in lengths:
                    case = f"r{rep}-l{target}"
                    ids = json.loads((OUT / (case + ".ids.json")).read_text())
                    row = {**workloads[case], "arm": arm, "mtp": mtp, "status": "running"}
                    report["rows"].append(row)
                    telemetry.begin()
                    try:
                        row.update(measured_request(engine, ids))
                        row["status"] = "ok"
                        if row["reused_tokens"] != 0:
                            raise RuntimeError(f"fresh-prompt test unexpectedly reused tokens: {case}")
                        if mtp == 0 and row["draft_offered"] != 0:
                            raise RuntimeError("MTP 0 offered drafts")
                        if mtp == 2 and row["draft_offered"] == 0:
                            raise RuntimeError("MTP 2 did not execute")
                    except BaseException as exc:
                        row["status"], row["error"] = "failed", repr(exc)
                        raise
                    finally:
                        row["resources"] = telemetry.end()
                        save("results.json", report)
                    print(json.dumps({k: row[k] for k in ("arm", "case", "prompt_tokens", "generated_tokens",
                                                         "decode_tok_s", "client_ttft_s", "draft_offered")}), flush=True)
            finally:
                if telemetry:
                    telemetry.close()
                if engine:
                    engine.close()
                    if engine.p.poll() is None:
                        subprocess.run(["taskkill", "/PID", str(engine.p.pid), "/T", "/F"], capture_output=True)
                save("results.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if digest(EXE) != EXPECTED_EXE:
        raise RuntimeError("native baseline binary hash changed; re-establish provenance first")
    if (OUT / "results.json").exists() and not args.prepare_only:
        raise RuntimeError("results.json already exists; use a new archive folder, do not overwrite measurements")
    workloads = prepare_workloads()
    probe = Telemetry(os.getpid())
    try:
        time.sleep(0.6)
        save("resource-preflight.json", probe.sample())
    finally:
        probe.close()
    if args.prepare_only:
        print("Prepared", len(workloads), "workloads and resource counter preflight", flush=True)
        return
    report = {"source_revision": REVISION, "workspace_revision": subprocess.check_output(
                  ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
              "exe": str(EXE), "exe_sha256": EXPECTED_EXE, "model": str(MODEL),
              "model_bytes": MODEL.stat().st_size, "max_context": 32768, "prefill": 512,
              "sampling_protocol": KEYS, "environment_overrides": ENV, "profiler": False,
              "method": __doc__, "arms": [], "rows": []}
    state = {"health_before": request("/health")}
    unloaded = False
    try:
        state["unload"] = request("/unload?wait=30", post=True)
        if state["unload"].get("status") not in ("unloaded", "not loaded"):
            raise RuntimeError(f"service did not release model: {state['unload']}")
        unloaded = True
        run_matrix(workloads, report)
        report["complete"] = len(report["rows"]) == 24 and all(row["status"] == "ok" for row in report["rows"])
    finally:
        save("results.json", report)
        try:
            if unloaded and state["health_before"].get("loaded"):
                print("Restoring original Flash-Next service", flush=True)
                state["load"] = request("/load", post=True)
                state["health_after"] = request("/health")
        except Exception as exc:
            state["restore_error"] = repr(exc)
            raise
        finally:
            save("service-transition.json", state)
    if state["health_before"].get("loaded") and not state.get("health_after", {}).get("loaded"):
        raise RuntimeError("original service did not restore")


if __name__ == "__main__":
    main()
