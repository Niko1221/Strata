"""Serial local native qualification; run inside guard.py, never against user servers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

from guard import ROOT, OUT, NODE, CUDA, environment

sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from native_model_resource_guard import available_memory
from perf131k.windows_job import process_memory_reader

BUILD = OUT / "native-build"
COMPONENT = OUT / "component/0.5.2-windows-x86_64-cuda-cuda13"
MODEL = Path("H:/Strata/models/models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
PACK = Path(str(MODEL) + ".q36epk")
CTEST = "C:/Program Files/CMake/bin/ctest.exe"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gpu_free() -> int:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True, timeout=10)
    return int(text.splitlines()[0].strip()) * 1024**2


def admit(gpu_bytes: int = 1024**3, ram_bytes: int = 3 * 1024**3) -> dict:
    memory = available_memory()
    free = gpu_free()
    assert free >= gpu_bytes, "GPU admission refused; coordinate with parent, do not stop user processes"
    assert memory["available_physical_bytes"] >= ram_bytes, "RAM admission refused"
    assert memory["available_commit_bytes"] >= 4 * 1024**3, "commit floor"
    assert shutil.disk_usage("H:/").free >= 10 * 1024**3, "H disk floor"
    assert shutil.disk_usage("C:/").free >= 5 * 1024**3, "C disk floor"
    return dict(memory, gpu_free_bytes=free)


def save(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def stage() -> None:
    assert json.loads((OUT / "receipts/native-bind-r3.json").read_text())["passed"]
    identity = json.loads((BUILD / "Release/native-build.json").read_text())
    assert identity["version"] == "0.5.2" and identity["sha256"] == sha(BUILD / "Release/strata.exe")
    # The project helper checks the full native source membership and bytes.
    subprocess.run([str(NODE), "--input-type=module", "-e",
        "import {nativeSources} from './apps/host/scripts/release-integrity.mjs';"
        "import {readFileSync} from 'node:fs';"
        "const r=JSON.parse(readFileSync(process.argv[1],'utf8'));"
        "if(JSON.stringify(r.sources)!==JSON.stringify(nativeSources(process.cwd())))throw Error('source drift');",
        str(BUILD / "Release/native-build.json")], check=True, cwd=ROOT, timeout=60)
    source = json.loads((OUT / "build-source-r2.json").read_text())
    old = Path("H:/Strata/releases/strata-0.5.1/component-r1/0.5.1-windows-x86_64-cuda-cuda13")
    from terminal.strata_control import engine_component as ec
    old_manifest = ec.read_manifest(old)
    assert old_manifest.version == "0.5.1"
    assert ec.verify_signature(old_manifest, environ={}) == "verified"
    stage_dir = OUT / "engine-input-r2"
    assert not stage_dir.exists() and not COMPONENT.exists()
    stage_dir.mkdir()
    copied = []
    for name, expected, size in old_manifest.files:
        if not name.startswith(("local-backends/", "notices/")):
            continue
        path = old / name
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == size and sha(path) == expected
        dest = stage_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        assert sha(dest) == expected
        copied.append({"path": name, "sha256": expected, "bytes": size})
    assert any(row["path"] == "local-backends/prism/llama-server.exe" for row in copied)
    assert any(row["path"] == "local-backends/turboquant/llama-server.exe" for row in copied)
    for path in [BUILD / "Release" / name for name in ("strata.exe", "strata-q36.exe", "native-build.json")] + list((BUILD / "Release").glob("*.dll")):
        shutil.copy2(path, stage_dir / path.name)
        assert sha(path) == sha(stage_dir / path.name)
    native_component = OUT / "native-component/0.5.2-windows-x86_64-cuda-cuda13"
    subprocess.run([sys.executable, "-B", str(ROOT / "packaging/package_engine.py"),
                    "--src", str(BUILD / "Release"), "--backend", "cuda", "--version", "0.5.2",
                    "--platform", "windows-x86_64", "--variant", "cuda13", "--arch", "sm_120",
                    "--min-driver", "nvidia=580.65", "--build-id", source["commit"],
                    "--include", "native-build.json", "--include", "strata-q36.exe",
                    "--search", str(CUDA / "bin/x64"), "--search", str(CUDA / "bin"),
                    "--out", str(native_component.parent)], cwd=ROOT, check=True, timeout=180)
    subprocess.run([sys.executable, "-B", str(ROOT / "packaging/package_engine.py"),
                    "--verify", str(native_component)], check=True, cwd=ROOT, timeout=180)
    # The project's composite recipe verifies each independently loaded runtime
    # in its own directory; a global DLL search path would mix Prism and Turbo.
    sys.path.insert(0, str(ROOT / "packaging"))
    import binary_deps
    for backend in ("prism", "turboquant"):
        assert not binary_deps.uncovered(stage_dir / "local-backends" / backend)
    subprocess.run([str(NODE), str(Path(__file__).with_name("compose.mjs")),
                    str(native_component), str(stage_dir), str(COMPONENT)],
                   check=True, cwd=ROOT, timeout=180)
    manifest = json.loads((COMPONENT / "engine-component.json").read_text())
    assert manifest["signature"] is None
    save(OUT / "package-receipt.json", {"passed": True, "sourceCommit": source["commit"],
         "nativeSourcesCount": len(identity["sources"]), "component": str(COMPONENT),
         "manifestSHA256": sha(COMPONENT / "engine-component.json"), "signed": False,
         "reusedSignedManifestSHA256": sha(old / "engine-component.json"),
         "reusedSignatureVerified": True, "reusedFiles": copied,
         "files": manifest["files"]})


def suite() -> None:
    assert COMPONENT.is_dir()
    env = environment()
    env["PATH"] = str(COMPONENT) + ";" + env["PATH"]
    result = {"cases": [], "passed": False}
    path = OUT / "suite.json"
    assert not path.exists()
    save(path, result)

    def run(name: str, command: list[str], timeout: int = 120, gpu: bool = False) -> str:
        admission = admit() if gpu else None
        begin = time.monotonic()
        with (OUT / (name + ".txt")).open("x", encoding="utf-8") as output:
            done = subprocess.run(command, cwd=OUT, env=env, stdout=output,
                                  stderr=subprocess.STDOUT, timeout=timeout,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        text = (OUT / (name + ".txt")).read_text(encoding="utf-8", errors="replace")
        row = {"name": name, "exit": done.returncode, "seconds": time.monotonic() - begin,
               "admission": admission, "command": command,
               "passed": done.returncode == 0 and "***Skipped" not in text and "Not Run" not in text}
        result["cases"].append(row)
        save(path, result)
        print(json.dumps(row), flush=True)
        return text

    for exe in ("strata.exe", "strata-q36.exe"):
        version = run(exe + "-version", [str(COMPONENT / exe), "--version"])
        result["cases"][-1]["reports052"] = bool(re.search(r"\b0\.5\.2\b", version))
        help_text = run(exe + "-help", [str(COMPONENT / exe), "--help"])
        result["cases"][-1]["memoryMarker"] = "Q36_MEMORY_CAPABILITIES" in help_text and "bounded-experts-v1" in help_text
        result["cases"][-1]["kvMarker"] = "Q36_KV_CAPABILITIES" in help_text
        save(path, result)
    cpu = ["q36_bounded_experts_test", "q36_expert_store_test", "q36_transfer_lifetime_test",
           "q36_bounded_cpu_parity_test", "q36_kv_layout_test"]
    for name in cpu:
        run(name, [CTEST, "--test-dir", str(BUILD), "-C", "Release", "--parallel", "1",
                   "--no-tests=error", "--output-on-failure", "--timeout", "90", "-R", "^" + name + "$"], 110)
    spin8 = BUILD / "spin8/Release/spin8_attention_v2_cuda_test.exe"
    run("spin8-b2-first-use", [str(spin8), "--first-use-only"], gpu=True)
    run("spin8-b2-full", [str(spin8)], 180, gpu=True)
    for name in ("q36_kv_lifecycle_test", "q36_kv_cache_test", "native_cache_bridge", "native_cache_owner"):
        run(name, [CTEST, "--test-dir", str(BUILD), "-C", "Release", "--parallel", "1",
                   "--no-tests=error", "--output-on-failure", "--timeout", "90", "-R", "^" + name + "$"], 110, gpu=True)
    sanitizer = CUDA / "compute-sanitizer/compute-sanitizer.exe"
    run("compute-sanitizer-version", [str(sanitizer), "--version"], 30)
    run("q36-kv-memcheck", [str(sanitizer), "--tool", "memcheck", "--error-exitcode", "1",
                           str(BUILD / "Release/q36_kv_lifecycle_test.exe")], 120, gpu=True)
    result["passed"] = all(row["passed"] and row.get("reports052", True)
                           and row.get("memoryMarker", True) and row.get("kvMarker", True)
                           for row in result["cases"])
    save(path, result)
    assert result["passed"], "One or more native qualification cases failed"


def model(mode: str, label: str, short: bool = False) -> None:
    from strata_tokenizer import Tokenizer
    assert mode in ("bounded-experts-v1", "legacy")
    assert label.replace("-", "").isalnum()
    path = OUT / (label + ".json")
    assert not path.exists()
    # Existing model and verified pack only. No packer or receipt priming is invoked.
    protected = (MODEL, PACK, Path(str(PACK) + ".verified"))
    before = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in protected}
    tk = Tokenizer.from_gguf(MODEL)
    text = "The capital of France is"
    ids = tk.encode(text)
    assert tk.decode(ids) == text and 1 <= len(ids) < 128
    args = ["--serve", "--native", str(MODEL), "--q36-expert-store", "mapped-pack",
            "--q36-memory", mode, "--max-context", "4096", "--prefill", "512",
            "--expert-slots", "5", "--mtp", "2", "--no-warmup", "--greedy",
            "--prompt-cache", "0", "--conversation-cache-mib", "0",
            "--private-conversation-cache", "--cache-type-k", "f32", "--cache-type-v", "f32"]
    result = {"passed": False, "mode": mode, "capacity": 4096, "fullInputQuality": False,
              "inputText": text, "inputIds": ids, "inputCount": len(ids), "args": args,
              "modelFilesBefore": before, "exeSHA256": sha(COMPONENT / "strata.exe"),
              "requests": [], "info": [], "effective": [], "memorySamples": 0,
              "maxWorkingSetBytes": 0, "maxPrivateBytes": 0}
    save(path, result)
    proc = None
    begin = time.monotonic()
    env = environment()
    env.update(STRATA_Q36_PIN_PIPELINE="1", STRATA_Q36_STREAM_RESIDENT="1",
               STRATA_Q36_FILE_STAGING="1", STRATA_Q36_STARTUP_DETAIL="1",
               STRATA_Q36_STARTUP_GENERATION=label)
    env["PATH"] = str(COMPONENT) + ";" + env["PATH"]
    messages: queue.Queue = queue.Queue(maxsize=256)
    reader_error: list[str] = []
    query = process_memory_reader()
    last_sample = 0.0

    def sample_native() -> None:
        nonlocal last_sample
        now = time.monotonic()
        if now - last_sample < .2:
            return
        last_sample = now
        counters = query(int(proc._handle))
        memory = available_memory()
        result["memorySamples"] += 1
        for source, target in (("working_set_bytes", "maxWorkingSetBytes"), ("private_bytes", "maxPrivateBytes")):
            assert counters[source] is not None, "native memory measurement failed"
            result[target] = max(result[target], counters[source])
        for source, target in (("available_physical_bytes", "minAvailableRamBytes"), ("available_commit_bytes", "minCommitHeadroomBytes")):
            result[target] = min(result.get(target, memory[source]), memory[source])
        assert memory["available_physical_bytes"] >= 3 * 1024**3, "RAM floor"
        assert memory["available_commit_bytes"] >= 4 * 1024**3, "commit floor"

    def reader(stream, channel: str) -> None:
        try:
            with (OUT / (label + "-" + channel + ".txt")).open("x", encoding="utf-8") as log:
                total = 0
                while True:
                    line = stream.readline(65537)
                    if not line:
                        break
                    total += len(line)
                    if len(line) > 65536 or total > 8 * 1024**2:
                        raise RuntimeError("bounded native output exceeded")
                    log.write(line); log.flush()
                    messages.put((channel, line.rstrip()), timeout=3)
            messages.put((channel, None), timeout=3)
        except Exception as error:
            reader_error.append(str(error))

    def nextline(deadline: float) -> str:
        while time.monotonic() < min(deadline, begin + 1100):
            sample_native()
            assert not reader_error, reader_error
            try:
                channel, line = messages.get(timeout=.1)
            except queue.Empty:
                continue
            if line is None:
                if channel == "stdout":
                    raise RuntimeError("Native stdout closed, exit=" + str(proc.poll()))
                continue
            if channel == "stderr":
                if line.startswith(("Q36_EFFECTIVE ", "Q36_MEMORY_EFFECTIVE ", "Q36_PHYSICAL_MEMORY ")):
                    result["effective"].append(line)
                continue
            return line
        raise TimeoutError("native qualification deadline")

    def request(name: str, prompt: list[int], scope: str | None, limit: int = 1,
                reuse: int = 0, cancel: bool = False) -> list[int]:
        start = time.monotonic()
        proc.stdin.write("GEN " + str(limit) + (" cache_scope=" + scope if scope else "")
                         + " " + ",".join(map(str, prompt)) + "\n")
        proc.stdin.flush()
        generated: list[int] = []
        progress, stopped = [], False
        while True:
            line = nextline(start + 150)
            if line.startswith("T "):
                generated.append(int(line.split()[1]))
                if cancel and not stopped:
                    proc.stdin.write("STOP\n"); proc.stdin.flush(); stopped = True
            elif line.startswith("PP "):
                progress.append(line)
            elif line.startswith("ERR "):
                raise RuntimeError(line)
            elif line.startswith("DONE "):
                break
        fields = line.split()
        row = {"name": name, "submittedCount": len(prompt), "nativePromptCount": int(fields[2]),
               "nativeFreshReadCount": int(fields[14]), "nativeReuseCount": int(fields[8]),
               "generatedCount": int(fields[1]), "generatedIds": generated,
               "generatedText": tk.decode(generated), "finish": fields[5], "done": line,
               "progress": progress, "requestedMaxNew": limit, "seconds": time.monotonic() - start}
        result["requests"].append(row)
        save(path, result)
        assert row["nativePromptCount"] == len(prompt)
        assert row["nativeReuseCount"] == reuse
        assert row["nativeFreshReadCount"] + reuse == len(prompt)
        assert row["generatedCount"] == len(generated) and generated and row["generatedText"].strip()
        assert row["finish"] == "cancel" if cancel else row["finish"] in ("length", "stop")
        print(json.dumps({k: row[k] for k in ("name", "nativePromptCount", "nativeFreshReadCount", "nativeReuseCount", "generatedCount", "finish")}), flush=True)
        return generated

    try:
        result["admission"] = admit(14 * 1024**3, 8 * 1024**3)
        proc = subprocess.Popen([str(COMPONENT / "strata.exe"), *args], cwd=OUT,
                                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
        result["ownedPid"] = proc.pid
        threads = [threading.Thread(target=reader, args=(stream, channel), daemon=True)
                   for stream, channel in ((proc.stdout, "stdout"), (proc.stderr, "stderr"))]
        for thread in threads:
            thread.start()
        while True:
            line = nextline(begin + 360)
            if line.startswith("INFO "):
                result["info"].append(line)
            if line.startswith("READY "):
                result["ready"] = line
                break
        result["readySeconds"] = time.monotonic() - begin
        assert result["ready"] == "READY 4096 stop"
        assert any("q36_private_cache=2" in line for line in result["info"])
        assert any("kv_cache_k=f32 kv_cache_v=f32" in line for line in result["info"])
        first = request("first-owner-A", ids, "a" * 64)
        if not short:
            request("same-owner-A", ids + first, "a" * 64, reuse=len(ids))
            request("different-owner-B", ids, "b" * 64)
            request("missing-owner", ids, None)
            request("cancel-decode", ids, "a" * 64, limit=128, cancel=True)
            request("after-cancel-clears", ids, "a" * 64)
        request("short-generation", ids, "b" * 64, limit=8)
        proc.stdin.close()
        proc.wait(timeout=15)
        for thread in threads:
            thread.join(timeout=3)
        while not messages.empty():
            channel, line = messages.get_nowait()
            if channel == "stderr" and line and line.startswith(("Q36_EFFECTIVE ", "Q36_MEMORY_EFFECTIVE ")):
                result["effective"].append(line)
        assert proc.returncode == 0
        if mode == "bounded-experts-v1":
            assert any("Q36_MEMORY_EFFECTIVE " in line for line in result["effective"])
        result["passed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if proc:
            if proc.poll() is None:
                proc.terminate(); proc.wait(timeout=10)
            result["exitCode"] = proc.returncode
            result["ownedProcessReaped"] = proc.poll() is not None
        after = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in protected}
        result["modelFilesAfter"] = after
        result["modelFilesPreserved"] = before == after
        result["passed"] = result["passed"] and before == after
        result["seconds"] = time.monotonic() - begin
        save(path, result)
        print(json.dumps({k: result.get(k) for k in ("passed", "mode", "readySeconds", "error", "maxWorkingSetBytes", "maxPrivateBytes", "minAvailableRamBytes")}), flush=True)
    assert result["passed"], result.get("error", "Native qualification failed")


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    if action in ("--help", "help"):
        print("stage | suite | model MODE LABEL [short]; invoke inside the task-local guard")
    elif action == "stage":
        stage()
    elif action == "suite":
        suite()
    elif action == "model":
        model(args[0], args[1], short=len(args) == 3 and args[2] == "short")
    else:
        raise SystemExit("stage | suite | model MODE LABEL [short]")
