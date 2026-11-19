"""Opt-in actual-input QA. Default is a CPU-only plan; no app/profile control."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace

from contracts import Exchange, exact_answer, ids_sha, require, validate_input
from activity import ActivityGuard
from fixtures import MODEL, ROOT, TOKENIZER, extension, load, metadata, prepare

BASE = Path("H:/Strata/validation/q36-longinput-20261109")
EXE_SHA = "13fc3b96fe71afcb98990d828d2c743f85e8b59b16174a1f7da3eb4ef9b4f8e3"
SOURCE = "4a171bdeb33675963edcca51f423519863fb7127"
GIB = 1024**3
CASES = (
    "first-owner-A", "same-owner-A-extended", "different-owner-B", "missing-identity-unscoped",
    "cancel-owner-A", "post-cancel-owner-A", "independent-after-restart",
)
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from native_model_resource_guard import available_memory


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resources() -> dict:
    mem = available_memory()
    return dict(
        available_ram=mem["available_physical_bytes"],
        commit_headroom=mem["available_commit_bytes"],
        C_free=shutil.disk_usage("C:/").free,
        H_free=shutil.disk_usage("H:/").free,
    )


def floor_error(r: dict) -> str | None:
    for key, minimum in (
        ("available_ram", 3 * GIB),
        ("commit_headroom", 4 * GIB),
        ("C_free", 5 * GIB),
        ("H_free", 10 * GIB),
    ):
        require(type(r.get(key)) is int and r[key] >= 0, "resource telemetry unavailable")
        if r[key] < minimum:
            return key + " floor"
    return None


def cpu_admit() -> dict:
    r = resources()
    require(floor_error(r) is None, str(floor_error(r)))
    return r


def require_cases(rows: list[dict]) -> None:
    require(tuple(row["name"] for row in rows) == CASES, "mandatory case sequence incomplete or reordered")
    require(all(row.get("protocolPassed") is True for row in rows), "mandatory protocol case failed")
    unscoped = rows[3]
    require(unscoped.get("scopeCarrierPresent") is False and unscoped["nativeReuseCount"] == 0
            and unscoped["nativeFreshReadCount"] == unscoped["submittedCount"], "unscoped carrier/accounting failed")
    require(all(row.get("scopeCarrierPresent") is True for row in rows if row is not unscoped),
            "mandatory scoped carrier missing")


class GuardCallbacks:
    """The production preflight/sample/tick callbacks, injectable for CPU emergency fixtures."""

    def __init__(self, activity: ActivityGuard, *, resources: Callable = resources,
                 gpu: Callable = None, clock: Callable = time.monotonic,
                 abort: Callable, sample_log: Callable):
        self.activity, self.resources, self.gpu = activity, resources, gpu
        self.clock, self.abort, self.sample_log = clock, abort, sample_log
        self.started = clock()
        self.abort_at: float | None = None
        self.last_activity = self.last_gpu = float("-inf")
        self.last_resources: dict = {}

    def preflight(self) -> dict:
        r = self.resources()
        require(floor_error(r) is None, str(floor_error(r)))
        activity = self.activity.preflight()
        self.last_resources = r
        return dict(resources=r, activity=activity)

    def signal_abort(self, reason: str) -> None:
        if self.abort_at is None:
            self.abort_at = self.clock()
            self.abort(dict(reason=reason, normalStopGraceSeconds=15))

    def sample(self, _: Path | None) -> dict:
        now = self.clock()
        try:
            r = self.resources()
            reason = floor_error(r)
            self.last_resources = r
            if reason:
                self.signal_abort(reason)
        except Exception:
            self.signal_abort("resource telemetry unavailable")
            r = dict(self.last_resources, resourceTelemetryUnavailable=True, resourceValues="last-known")
        if self.abort_at is None and now - self.last_activity >= 0.5:
            self.last_activity = now
            try:
                self.activity.preflight()
            except Exception:
                self.signal_abort("activity refused or telemetry unavailable")
        if now - self.last_gpu >= 1:
            self.last_gpu = now
            try:
                g = self.gpu()
            except Exception:
                g = dict(unavailable=True)
                self.signal_abort("whole GPU sampling failed")
            self.sample_log(dict(elapsedSeconds=now - self.started, **r, gpu=g,
                                 activity=self.activity.summary if hasattr(self.activity, "summary") else {}))
        return r

    def tick(self) -> str | None:
        if self.abort_at is not None and self.clock() - self.abort_at >= 15:
            return "latched abort after normal STOP grace; reap only held owned Job"
        return None


def gpu() -> dict:
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    ).strip()
    name, total, used, free = [s.strip() for s in text.split(",")]
    return dict(
        name=name,
        totalMiB=int(total),
        usedMiB=int(used),
        freeMiB=int(free),
        scope="whole GPU 0, includes unrelated GUI/desktop occupancy; not owned-process VRAM",
    )


def fresh(path: Path | None) -> Path:
    require(path is not None, "--report must name a fresh attempt directory")
    base = BASE.resolve()
    require(base.parent.is_dir(), "validation parent missing")
    if not base.exists():
        base.mkdir()
    require(not base.is_symlink() and not base.is_junction(), "linked evidence base")
    path = path.absolute()
    require(
        path.parent.resolve() == base and path.name not in ("", ".", ".."),
        "report must be an immediate child of " + str(base),
    )
    require(not path.exists(), "fresh output root required; retain all earlier attempts")
    path.mkdir()
    from terminal.strata_control.private_files import restrict_to_current_user

    restrict_to_current_user(path, directory=True)
    (path / "temp").mkdir()
    return path


def environment(out: Path) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().startswith(("STRATA_", "DSH_", "PYTHON", "CUDA_", "CMAKE_"))
    }
    env.update(
        TMP=str(out / "temp"),
        TEMP=str(out / "temp"),
        TMPDIR=str(out / "temp"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        CUDA_CACHE_PATH=str(out / "temp"),
        UV_CACHE_DIR=str(out / "temp"),
        HF_HOME=str(out / "temp"),
        XDG_CACHE_HOME=str(out / "temp"),
        STRATA_Q36_PIN_PIPELINE="1",
        STRATA_Q36_STREAM_RESIDENT="1",
        STRATA_Q36_FILE_STAGING="1",
        STRATA_Q36_STARTUP_DETAIL="1",
    )
    return env


def verify_component(component: Path) -> dict:
    """Deferred until execute: built-in signature + full component hashes, never model hashes."""
    from terminal.strata_control import engine_component as ec

    manifest = ec.read_manifest(component)
    require(manifest.version == "0.5.2", "wrong engine version")
    require(ec.verify_signature(manifest, environ={"STRATA_REQUIRE_SIGNED_COMPONENTS": "1"}) == "verified", "signature")
    ec.verify(manifest, full=True)
    require(sha(component / "strata.exe") == EXE_SHA, "not the exact signed r3 engine")
    build = json.loads((component / "native-build.json").read_text())
    require(build["sha256"] == EXE_SHA and build["version"] == "0.5.2", "native build identity")
    return dict(
        exeSHA256=EXE_SHA,
        manifestSHA256=sha(component / "engine-component.json"),
        sourceCheckpointFromPriorQualification=SOURCE,
        nativeBuildSHA256=sha(component / "native-build.json"),
        signature="verified",
        sourceRebuilt=False,
    )


def native_args() -> list[str]:
    return [
        "--serve",
        "--native",
        str(MODEL),
        "--q36-expert-store",
        "mapped-pack",
        "--q36-memory",
        "bounded-experts-v1",
        "--max-context",
        "4096",
        "--prefill",
        "512",
        "--expert-slots",
        "5",
        "--mtp",
        "2",
        "--no-warmup",
        "--greedy",
        "--prompt-cache",
        "0",
        "--conversation-cache-mib",
        "0",
        "--private-conversation-cache",
        "--cache-type-k",
        "f32",
        "--cache-type-v",
        "f32",
    ]


class Native:
    """One newly owned, serial native child, inheriting the verified non-breakaway Job."""

    def __init__(self, component: Path, out: Path, label: str, end: float):
        cpu_admit()
        require(time.monotonic() < end and not (out / "abort.json").exists(), "abort before native creation")
        self.out, self.label, self.end = out, label, end
        self.messages: queue.Queue = queue.Queue(maxsize=4096)
        self.errors: list[str] = []
        self.info: list[str] = []
        self.threads: list[threading.Thread] = []
        self.last_sample = 0.0
        self.closed = False
        self.record: dict = {"label": label, "args": native_args(), "normalStopAttempted": False}
        env = environment(out)
        env["PATH"] = str(component) + ";" + env["PATH"]
        ActivityGuard().preflight()  # fresh read immediately before initial/restart native child creation
        require(not (out / "abort.json").exists(), "controller aborted before native creation")
        self.proc = subprocess.Popen(
            [str(component / "strata.exe"), *native_args()],
            cwd=out,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.record["ownedPid"] = self.proc.pid
        from perf131k.windows_job import process_memory_reader

        self.memory = process_memory_reader()
        self.samples = (out / (label + "-process-memory.jsonl")).open("x", encoding="utf-8")
        self.sent = (out / (label + "-stdin.txt")).open("x", encoding="utf-8")
        for channel, stream in (("stdout", self.proc.stdout), ("stderr", self.proc.stderr)):
            thread = threading.Thread(target=self.read, args=(channel, stream), daemon=True)
            self.threads.append(thread)
            thread.start()

    def read(self, channel: str, stream: object) -> None:
        try:
            with (self.out / (self.label + "-" + channel + ".txt")).open("x", encoding="utf-8") as log:
                total = 0
                while True:
                    line = stream.readline(65537)
                    if not line:
                        break
                    log.write(line)
                    log.flush()
                    total += len(line.encode("utf-8"))
                    require(len(line) <= 65536 and total <= 32 * 1024**2, "transcript bound exceeded; attempt fails")
                    if channel == "stdout":
                        self.messages.put(line.rstrip("\r\n"), timeout=2)
            if channel == "stdout":
                self.messages.put(None, timeout=2)
        except Exception as error:
            self.errors.append(str(error))

    def send(self, line: str) -> None:
        self.sent.write(line + "\n")
        self.sent.flush()
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def tick(self) -> None:
        require(
            time.monotonic() < self.end and not (self.out / "abort.json").exists(), "global deadline/resource abort"
        )
        if time.monotonic() - self.last_sample >= 0.25:
            self.last_sample = time.monotonic()
            r = resources()
            m = self.memory(int(self.proc._handle))
            self.samples.write(json.dumps(dict(monotonic=self.last_sample, **r, **m)) + "\n")
            self.samples.flush()
            require(floor_error(r) is None, str(floor_error(r)))
            require(
                m["private_bytes"] is not None and m["working_set_bytes"] is not None,
                "native RSS/private sample failed",
            )
        require(not self.errors, str(self.errors))

    def line(self) -> str:
        while True:
            self.tick()
            try:
                line = self.messages.get(timeout=0.1)
            except queue.Empty:
                continue
            require(line is not None, "native stdout closed before expected protocol")
            return line

    def ready(self) -> None:
        while True:
            line = self.line()
            require(not line.startswith("ERR"), line)
            if line.startswith("INFO "):
                self.info.append(line)
            elif line.startswith("READY "):
                require(line == "READY 4096 stop", "wrong context/control capability")
                require(any("q36_private_cache=2" in s for s in self.info), "auth-owner carrier missing")
                require(any("kv_cache_k=f32 kv_cache_v=f32" in s for s in self.info), "KV not F32/F32")
                require(any("q36_memory_mode=bounded-experts-v1" in s for s in self.info), "wrong memory mode")
                receipts = [s for s in self.info if s.startswith("INFO q36_memory_mode=")]
                require(len(receipts) == 1, "one pre-READY actual-allocation receipt required")
                fields = [p.split("=", 1) for p in receipts[0].split()[1:]]
                expected = dict(
                    q36_memory_mode="bounded-experts-v1",
                    q36_memory_scope="expert-working-storage",
                    q36_memory_total_process_bound="0",
                    q36_memory_cpu_stage_bytes="18512896",
                    q36_memory_h2d_stage_bytes="67108864",
                    q36_memory_io_scratch_bytes="1048576",
                )
                require(len(fields) == 6 and dict(fields) == expected, "pre-READY memory allocation contract changed")
                self.record.update(info=self.info, ready=line)
                return

    def request(
        self,
        name: str,
        prompt: dict,
        owner: str | None,
        tok: object,
        reuse_min: int = 0,
        reuse_max: int = 0,
        cancel: bool = False,
    ) -> dict:
        ids = prompt["ids"]
        validate_input(ids, prompt["actualInputCount"])
        require(ids_sha(ids) == prompt["orderedIdsSHA256"], "submitted ID hash drift")
        row = dict(
            name=name,
            input=prompt,
            expectedReuseRange=[reuse_min, reuse_max],
            requestedMaxNew=128,
            cancelAfter="first generated T" if cancel else None,
            status="IN_PROGRESS",
            scopeCarrierPresent=owner is not None,
        )
        path = self.out / (self.label + "-" + name + ".json")
        save(path, row)  # commit expected input and oracle before sending GEN
        start = time.monotonic()
        exchange = Exchange(len(ids), reuse_min, reuse_max, cancel)
        stopped = False
        try:
            carrier = " cache_scope=" + owner if owner is not None else ""
            self.send("GEN 128" + carrier + " " + ",".join(map(str, ids)))
            while exchange.done is None:
                exchange.feed(self.line())
                if cancel and exchange.generated and not stopped:
                    self.send("STOP")
                    stopped = True
            row.update(exchange.finish())
            text = tok.decode(exchange.generated, errors="strict")
            require(bool(text.strip()), "generated text is empty")
            eos = tok.tokens[tok.special_ids["tokenizer.ggml.eos_token_id"]]
            row.update(
                generatedText=text,
                protocolPassed=True,
                retrievalPassed=None if cancel else exact_answer(text, eos),
                status="CANCEL_PASS" if cancel else "PASS" if exact_answer(text, eos) else "RETRIEVAL_FAIL",
            )
        except Exception as error:
            row.update(
                status="FAIL",
                error=f"{type(error).__name__}: {error}",
                generatedIds=exchange.generated,
                progress=exchange.progress,
                observedDone=exchange.done,
            )
            raise
        finally:
            row.update(seconds=time.monotonic() - start, stopSent=stopped)
            save(path, row)
        return row

    def close(self) -> dict:
        """Protocol STOP then EOF before owned-handle fallback; outer Job reaps descendants."""
        if self.closed:
            return self.record
        self.closed = True
        if self.proc.poll() is None:
            self.record["normalStopAttempted"] = True
            try:
                self.send("STOP")
                self.proc.stdin.close()
            except (OSError, ValueError) as error:
                self.record["stopError"] = str(error)
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.record["ownedHandleTerminationFallback"] = True
                self.proc.terminate()
                self.proc.wait(timeout=3)
        for thread in self.threads:
            thread.join(timeout=2)
        self.record.update(
            exitCode=self.proc.returncode,
            ownedProcessReaped=self.proc.poll() is not None,
            transcriptReaderErrors=self.errors,
            transcriptReadersReaped=all(not t.is_alive() for t in self.threads),
        )
        self.samples.close()
        self.sent.close()
        save(self.out / (self.label + "-lifecycle.json"), self.record)
        return self.record


def worker(component: Path, out: Path) -> int:
    from perf131k.windows_job import api, require as win_require
    from ctypes import wintypes

    kernel = api()
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    inside = wintypes.BOOL()
    win_require(
        kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(inside)) and inside.value,
        "worker must inherit owned Job before GO",
    )
    require(sys.stdin.readline().strip() == "GO", "owned guard GO required")
    end = time.monotonic() + 1170  # reserve normal STOP/reap time inside outer 1200-second deadline
    result: dict = dict(
        passed=False,
        requests=[],
        lifecycles=[],
        cryptographicConsumedInputProof=False,
        fullWindowQuality=False,
        userAppRestore="parent responsibility; Swift must be restored",
    )
    before = metadata()
    active = None
    try:
        result["identity"] = verify_component(component)
        cpu_admit()
        admission = gpu()
        require(admission["freeMiB"] >= 14 * 1024, "GPU admission <14GiB; never stop unrelated processes")
        result["gpuBefore"] = admission
        tok, template = load()
        fixture = prepare(tok, template, cpu_admit)
        save(out / "fixture.json", fixture)
        # Random per-attempt native owner carrier, equivalent to the authenticated private-scope bridge.
        owner_a, owner_b = os.urandom(32).hex(), os.urandom(32).hex()
        active = Native(component, out, "initial", end)
        active.ready()
        first = active.request("first-owner-A", fixture, owner_a, tok)
        result["requests"].append(first)
        extended = extension(tok, template, fixture, first["generatedIds"])
        second = active.request(
            "same-owner-A-extended",
            extended,
            owner_a,
            tok,
            len(fixture["ids"]),
            len(fixture["ids"]) + len(first["generatedIds"]),
        )
        result["requests"].append(second)
        different = active.request("different-owner-B", extended, owner_b, tok)
        result["requests"].append(different)
        # B has just completed a non-cancel scoped request and populated private state.
        # Extend that exact B prompt/output so a broken missing-owner gate could otherwise reuse it.
        unscoped = extension(tok, template, extended, different["generatedIds"])
        result["requests"].append(active.request("missing-identity-unscoped", unscoped, None, tok))
        cancelled = active.request("cancel-owner-A", fixture, owner_a, tok, cancel=True)
        result["requests"].append(cancelled)
        post_cancel = extension(tok, template, fixture, cancelled["generatedIds"])
        result["requests"].append(active.request("post-cancel-owner-A", post_cancel, owner_a, tok))
        lifecycle = active.close()
        result["lifecycles"].append(lifecycle)
        active = None
        require(lifecycle["exitCode"] == 0 and lifecycle["ownedProcessReaped"], "normal initial shutdown failed")
        cpu_admit()
        require(gpu()["freeMiB"] >= 14 * 1024, "restart GPU admission refused")
        active = Native(component, out, "restart", end)
        active.ready()
        result["requests"].append(active.request("independent-after-restart", fixture, owner_a, tok))
        result["lifecycles"].append(active.close())
        active = None
        require_cases(result["requests"])
        result["retrievalPassed"] = all(
            r["retrievalPassed"] for r in result["requests"] if r["retrievalPassed"] is not None
        )
        result["passed"] = result["retrievalPassed"] and all(
            r["exitCode"] == 0
            and r["ownedProcessReaped"]
            and r["transcriptReadersReaped"]
            and not r["transcriptReaderErrors"]
            for r in result["lifecycles"]
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            if active is not None:
                result["lifecycles"].append(active.close())
            after = metadata()
            result.update(modelBefore=before, modelAfter=after, modelMetadataPreserved=before == after)
            result["passed"] = result["passed"] and before == after
        except Exception as error:
            result.update(passed=False, cleanupError=f"{type(error).__name__}: {error}")
        save(out / "result.json", result)
    return 0 if result["passed"] else 1


def execute(component: Path, out: Path) -> int:
    """Outer containment before worker GO; continuous floors and whole-GPU samples."""
    from perf131k.windows_job import run_owned

    start = time.monotonic()
    report: dict = dict(
        passed=False, deadlineSeconds=1200, admission=cpu_admit(), parentMustRestoreSwift=True, actualInputCount=None
    )
    save(out / "guard.json", report)
    activity = ActivityGuard()
    try:
        callbacks = GuardCallbacks(activity, resources=resources, gpu=gpu,
                                   abort=lambda row: save(out / "abort.json", row), sample_log=lambda row: None)
        report["activityPreflight"] = callbacks.preflight()
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", refusedBeforeChildCreation=True,
                      seconds=time.monotonic() - start)
        save(out / "guard.json", report)
        return 1
    (out / "go.txt").write_text("GO\n", encoding="ascii")
    with (out / "system-memory-gpu.jsonl").open("x", encoding="utf-8") as samples:
        def sample_log(row: dict) -> None:
            samples.write(json.dumps(row) + "\n")
            samples.flush()

        callbacks.sample_log = sample_log

        # Floors are enforced above, with a protocol-stop grace before the held Job is reaped.
        # Zero lower-layer thresholds prevent run_owned racing that normal STOP on the same pressure sample.
        budget = SimpleNamespace(
            job_commit_bytes=None,
            max_processes=8,
            min_available_ram=0,
            min_commit_headroom=0,
            min_H_free=0,
            timeout_s=max(1, 1200 - (time.monotonic() - start)),
            output_bytes=32 * 1024**2,
        )
        try:
            run = run_owned(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--execute",
                    "--worker",
                    "--component",
                    str(component),
                    "--report",
                    str(out),
                ],
                out,
                environment(out),
                out / "worker-transcript.txt",
                budget,
                callbacks.sample,
                stdin_path=out / "go.txt",
                on_tick=callbacks.tick,
                memory_diagnostics=True,
            )
            report["run"] = run
            report["passed"] = (
                run["status"] == "FINISHED"
                and run["native_exit_code"] == 0
                and run["contained_before_resume"]
                and run["termination_confirmed"]
                and callbacks.abort_at is None
            )
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
    report["seconds"] = time.monotonic() - start
    save(out / "guard.json", report)
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute", action="store_true", help="parent admits exclusive runtime window; starts NEW owned Q36 processes"
    )
    mode.add_argument(
        "--prepare", action="store_true", help="CPU tokenizer/formatter + GGUF header only; no tensor/pack reads"
    )
    mode.add_argument("--test", action="store_true", help="small CPU behavior fixtures only")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--component", type=Path, help="exact installed signed 0.5.2 component directory, required for execute"
    )
    parser.add_argument("--report", type=Path, help="fresh immediate child of " + str(BASE))
    args = parser.parse_args()
    if args.worker:
        require(args.execute and not args.prepare and not args.test, "internal worker also requires explicit --execute")
        require(args.component is not None and args.report is not None, "worker arguments")
        return worker(args.component.resolve(), args.report.resolve())
    if args.test:
        cpu_admit()
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        import pytest

        code = pytest.main(
            [
                str(Path(__file__).with_name("test_contracts.py")),
                "-q",
                "-p",
                "no:cacheprovider",
                "--noconftest",
                "-o",
                "addopts=",
            ]
        )
        cpu_admit()
        return code
    if not args.execute and not args.prepare:
        print(
            json.dumps(
                dict(
                    status="PLAN_ONLY",
                    model=str(MODEL),
                    tokenizer=str(TOKENIZER),
                    targetActualInput="3072 +/-24; must be encoded and measured",
                    capacity=4096,
                    maxNew=128,
                    mtpHeadroom=3,
                    args=native_args(),
                    cases=CASES,
                    deadlineSeconds=1200,
                    runtimeAdmission="parent --execute after CUDA compiler window; restore Swift afterwards",
                    proof="PP+DONE protocol count receipt, not cryptographic consumed-input proof",
                ),
                indent=2,
            )
        )
        return 0
    cpu_admit()
    require(not args.execute or args.component is not None, "--execute requires --component")
    out = fresh(args.report)
    if args.prepare:
        os.environ.update(environment(out))
        before = metadata()
        tok, template = load()
        fixture = prepare(tok, template, cpu_admit)
        eos = tok.tokens[tok.special_ids["tokenizer.ggml.eos_token_id"]]
        oracle_ids = tok.encode(json.dumps(fixture["expected"], separators=(",", ":")) + eos, parse_special=True)
        oracle_extension = extension(tok, template, fixture, oracle_ids)
        fixture["cpuOracleOnlyFollowup"] = oracle_extension
        fixture["cpuOracleOnlyAnswerTokens"] = len(oracle_ids)
        fixture["tokenizerFilesSHA256"] = {p.name: sha(p) for p in TOKENIZER.iterdir() if p.is_file()}
        fixture["modelMetadata"] = before
        require(before == metadata(), "model metadata changed during CPU prepare")
        save(out / "fixture.json", fixture)
        print(
            json.dumps(
                dict(
                    status="CPU_PREPARED_NOT_EXECUTED",
                    report=str(out),
                    count=fixture["actualInputCount"],
                    orderedIdsSHA256=fixture["orderedIdsSHA256"],
                    offsets=fixture["lookupTokenOffsets"],
                )
            )
        )
        return 0
    return execute(args.component.resolve(), out)


if __name__ == "__main__":
    raise SystemExit(main())
