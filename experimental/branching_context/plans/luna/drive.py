"""Sequential development runner; ticket outcomes never certify a gate.

Only stdlib is used. No inference, acceptance assertion, or tolerance lives here.
Run this on the host executing the checks, including Linux for model checks.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import unittest
import uuid

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
REPO = PACKAGE.parents[1]
STATES = {"PENDING", "ACTIVE", "RED_RECORDED", "IMPLEMENTED", "VALIDATED", "BLOCKED"}
SOURCE_SUFFIXES = {".py", ".cpp", ".hpp", ".h", ".c", ".cu", ".cuh", ".cmake", ".json"}


class Refusal(Exception):
    """The requested execution or evidence transition cannot be justified."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def meaningful_red(report, expected_ids=(), message=None):
    """An executed assertion failure, not an import/runtime error or fake xfail."""
    if not report.get("test_count") or report.get("errors") or report.get("skipped"):
        return False
    if report.get("expected_failures") or report.get("unexpected_successes"):
        return False
    executed = report.get("executed_ids", [])
    return any(f["id"] in executed and "AssertionError" in f.get("traceback", "")
               and (not expected_ids or any(f["id"].endswith(i) for i in expected_ids))
               and (not message or message in f.get("traceback", ""))
               for f in report.get("failures", []))


def successful(report, expected_ids=()):
    if report.get("status") != "PASS" or not report.get("test_count"):
        return False
    for name in ("failures", "errors", "skipped", "expected_failures", "unexpected_successes",
                 "evidence_integrity_errors"):
        if report.get(name):
            return False
    return all(any(i.endswith(wanted) for i in report.get("executed_ids", []))
               for wanted in expected_ids)


def process_identity(pid):
    """Kernel creation identity avoids mistaking a reused PID for our process.

    None means confirmed absent; inaccessible state raises, never means absent.
    """
    if os.name == "nt":
        from ctypes import wintypes as w
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        kernel.OpenProcess.restype = w.HANDLE
        kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
        kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() in (87, 1168):
                return None
            raise Refusal(f"Cannot inspect process {pid}: Windows error {ctypes.get_last_error()}")
        try:
            code = w.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise Refusal("GetExitCodeProcess failed")
            if code.value != 259:
                return None
            creation, exit_time, system, user = (w.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(x) for x in
                                                   (creation, exit_time, system, user))):
                raise Refusal("GetProcessTimes failed")
            started = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
            return {"pid": pid, "started": str(started), "host": socket.gethostname(), "os": os.name}
        finally:
            kernel.CloseHandle(handle)
    if sys.platform != "linux":
        raise Refusal("Kernel process observation is supported on Windows and Linux only")
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise Refusal(f"Cannot inspect process {pid}: {error}") from error
    fields = raw[raw.rfind(")") + 2:].split()
    if fields[0] == "Z":
        return None
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return {"pid": pid, "started": fields[19], "boot": boot,
            "host": socket.gethostname(), "os": os.name}


@contextmanager
def exclusive(path):
    """OS-released lock, held across execution and ledger publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise Refusal("Another driver owns this ledger; inspect its live process") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class WindowsJob:
    """Kill our entire subprocess tree on timeout or driver termination."""
    def __init__(self):
        self.handle = None
        if os.name != "nt":
            return
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", w.DWORD), ("min_working", ctypes.c_size_t),
                        ("max_working", ctypes.c_size_t), ("active", w.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class Io(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", Io), ("process_memory", ctypes.c_size_t),
                        ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t),
                        ("peak_job", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.handle or not self.kernel.SetInformationJobObject(
                self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close()
            raise Refusal("Cannot create an owned Windows process-tree job")

    def assign(self, child):
        if self.handle and not self.kernel.AssignProcessToJobObject(self.handle, int(child._handle)):
            raise Refusal("Cannot assign subprocess to owned Windows job")

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class Driver:
    def __init__(self, queue_path=HERE / "queue.json", ledger=None, bindings=None,
                 package=PACKAGE, repo=REPO):
        self.package, self.repo = Path(package).resolve(), Path(repo).resolve()
        self.queue_path = Path(queue_path).resolve()
        self.queue = read_json(self.queue_path)
        self.ledger = Path(ledger or self.package / "evidence/runs/luna").resolve()
        self.bindings = read_json(bindings) if bindings else {"artifacts": {}}
        self.tickets = {t["id"]: t for t in self.queue["tickets"]}
        self.validate_queue()

    def validate_queue(self):
        if len(self.tickets) != len(self.queue["tickets"]):
            raise Refusal("Duplicate ticket IDs")
        visited = set()
        for ticket in self.queue["tickets"]:
            if ticket["kind"] not in ("implementation", "observation"):
                raise Refusal("Unknown ticket kind")
            if any(d not in visited for d in ticket["depends_on"]):
                raise Refusal("Dependencies must exist earlier in the queue")
            ids = [c["id"] for c in ticket["commands"]]
            if len(ids) != len(set(ids)):
                raise Refusal("Duplicate command IDs")
            visited.add(ticket["id"])

    def events(self):
        path = self.ledger / "events.jsonl"
        if not path.exists():
            return []
        result, previous = [], None
        for line in path.read_bytes().splitlines():
            try:
                event = json.loads(line)
            except ValueError as error:
                raise Refusal("Incomplete/corrupt ledger: preserve it, do not restart blindly") from error
            checksum = event.pop("sha256", None)
            if event.get("previous") != previous or hashlib.sha256(canonical(event)).hexdigest() != checksum:
                raise Refusal("Ledger hash chain is broken")
            event["sha256"] = checksum
            previous = checksum
            result.append(event)
        return result

    def append(self, kind, **values):
        events = self.events()
        event = {"kind": kind, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "host": socket.gethostname(), "previous": events[-1]["sha256"] if events else None,
                 **values}
        event["sha256"] = hashlib.sha256(canonical(event)).hexdigest()
        self.ledger.mkdir(parents=True, exist_ok=True)
        with (self.ledger / "events.jsonl").open("ab") as stream:
            stream.write(canonical(event) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        return event

    def fingerprint(self):
        for raw, expected in self.queue.get("protected_files", {}).items():
            path = self.package / raw
            if not path.is_file() or sha(path) != expected:
                raise Refusal(f"Protected reference/acceptance file changed: {raw}")
        files = {}
        roots = ("src", "include", "sycl", "cmake", "third_party", "thirdparty", "tests", "tools")
        package_roots = ("src", "tests", "scripts", "native", "reference", "spec")
        candidates = [self.repo / "CMakeLists.txt", self.queue_path, Path(__file__).resolve()]
        for root in [self.repo / r for r in roots] + [self.package / r for r in package_roots]:
            if root.exists():
                candidates.extend(p for p in root.rglob("*") if p.is_file() and
                                  (p.suffix in SOURCE_SUFFIXES or p.name == "CMakeLists.txt"))
        for name in ("docs/acceptance.json", "docs/locked-references.json", "plans/luna/test_drive.py"):
            candidates.append(self.package / name)
        for path in sorted(set(p.resolve() for p in candidates if p.is_file())):
            try:
                label = path.relative_to(self.repo).as_posix()
            except ValueError:
                label = str(path)
            files[label] = sha(path)
        artifacts = {}
        for role, raw in self.bindings.get("artifacts", {}).items():
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = self.package / path
            artifacts[role] = {"path": str(path.resolve()), "sha256": sha(path) if path.is_file() else None}
        data = {"files": files, "artifacts": artifacts, "python": sys.version,
                "host": socket.gethostname(), "platform": platform.platform()}
        return {**data, "sha256": hashlib.sha256(canonical(data)).hexdigest()}

    def state(self, ticket_id):
        states = [e for e in self.events() if e.get("ticket") == ticket_id and e["kind"] == "state"]
        return states[-1] if states else {"state": "PENDING"}

    def runs(self, ticket_id):
        return [read_json(e["report"]) for e in self.events()
                if e.get("ticket") == ticket_id and e["kind"] == "run_end"
                and self.verify_artifact(e["report"], e["report_sha256"])]

    @staticmethod
    def verify_artifact(path, expected):
        if not Path(path).is_file() or sha(path) != expected:
            raise Refusal(f"Missing or changed evidence artifact: {path}")
        return True

    def inspect_active(self, recover=False):
        events = self.events()
        completed = {e["run"] for e in events if e["kind"] in ("run_end", "run_interrupted")}
        starts = [e for e in events if e["kind"] == "run_start" and e["run"] not in completed]
        for start in starts:
            children = [e for e in events if e["kind"] == "child" and e["run"] == start["run"]]
            if start["host"] != socket.gethostname():
                raise Refusal(f"Inspect run {start['run']} on {start['host']}; remote state is unknown")
            if not children:
                raise Refusal(f"Run {start['run']} has an ambiguous launch; inspect owned processes before recovery")
            identity = children[-1]["identity"]
            if identity is not None and process_identity(identity["pid"]) == identity:
                raise Refusal(f"Run {start['run']} is still live (PID {identity['pid']}); do not duplicate it")
            if recover:
                self.append("run_interrupted", run=start["run"], ticket=start["ticket"],
                            command_id=start["command_id"], before=start["before"],
                            reason="Recorded child kernel identity is no longer live; outcome incomplete")
            else:
                raise Refusal(f"Run {start['run']} is interrupted; use check to record the incomplete outcome")

    def ready(self, ticket):
        return ticket.get("enabled", True) and all(self.state(d)["state"] in
                 ("IMPLEMENTED", "VALIDATED") for d in ticket["depends_on"])

    def next_ticket(self):
        self.inspect_active()
        for ticket in self.queue["tickets"]:
            if (self.ready(ticket) and self.state(ticket["id"])["state"] in
                    ("PENDING", "ACTIVE", "RED_RECORDED", "IMPLEMENTED")):
                return ticket
        return None

    def check(self, ticket_id=None):
        self.inspect_active(recover=True)
        current = self.fingerprint()["sha256"]
        result = []
        for ticket in self.queue["tickets"]:
            if ticket_id and ticket_id != ticket["id"]:
                continue
            state = self.state(ticket["id"])
            validated = state["state"] == "VALIDATED"
            result.append({"ticket": ticket["id"], "state": state["state"],
                           "enabled": ticket.get("enabled", True), "ready": self.ready(ticket),
                           "validation_current": validated and state.get("fingerprint") == current,
                           "scope": "implementation ticket only; no gate certification",
                           "note": state.get("note", "")})
            if validated:
                try:
                    self.validate_green(ticket)
                except Refusal as error:
                    result[-1]["validation_current"] = False
                    result[-1]["validation_issue"] = str(error)
        return result

    def validate_green(self, ticket):
        current = self.fingerprint()["sha256"]
        runs = self.runs(ticket["id"])
        required = [c for c in ticket["commands"] if c["phase"] in ("green", "regression", "observe")
                    and c.get("required", True)]
        if not required:
            raise Refusal("No executable validation requirements defined")
        for command in required:
            matches = [r for r in runs if r["command_id"] == command["id"]]
            if not matches or matches[-1]["outcome"] != "PASS" or matches[-1]["before"]["sha256"] != current:
                raise Refusal(f"Missing, failing, or stale validation: {ticket['id']}/{command['id']}")
            self.verify_artifact(matches[-1]["stdout"], matches[-1]["stdout_sha256"])
            self.verify_artifact(matches[-1]["stderr"], matches[-1]["stderr_sha256"])
            for artifact in matches[-1]["artifacts"]:
                self.verify_artifact(artifact["path"], artifact["sha256"])

    def record(self, ticket_id, state, note, evidence=None):
        if state not in STATES or not note.strip():
            raise Refusal("A valid state and factual note are required")
        self.inspect_active()
        ticket = self.tickets[ticket_id]
        old = self.state(ticket_id)["state"]
        if not ticket.get("enabled", True):
            raise Refusal("Long campaign is disabled; a record cannot enable it")
        if state == "ACTIVE":
            if old not in ("PENDING", "BLOCKED") or not self.ready(ticket):
                raise Refusal("Ticket cannot become active from this state or with blocked dependencies")
            self.assert_single_ticket(ticket_id)
        elif state == "RED_RECORDED":
            raise Refusal("Red state is produced by an executed assertion failure, not a manual record")
        elif state == "IMPLEMENTED":
            if old != "RED_RECORDED" or ticket["kind"] != "implementation":
                raise Refusal("Implementation requires an executed meaningful red test")
            self.verify_red_tests(ticket)
        elif state == "VALIDATED":
            if old not in ("IMPLEMENTED", "VALIDATED") and not (ticket["kind"] == "observation" and old == "ACTIVE"):
                raise Refusal("Complete implementation or activate the observation first")
            if ticket["kind"] == "implementation":
                self.verify_red_tests(ticket)
            self.validate_green(ticket)
        elif state == "BLOCKED":
            if not evidence or not Path(evidence).is_file():
                raise Refusal("A blocker needs an existing evidence artifact")
        else:
            raise Refusal("PENDING is initial only; ledger history cannot be reset")
        artifact = {"path": str(Path(evidence).resolve()), "sha256": sha(evidence)} if evidence else None
        return self.append("state", ticket=ticket_id, state=state, note=note,
                           evidence=artifact, fingerprint=self.fingerprint()["sha256"])

    def assert_single_ticket(self, ticket_id):
        for other in self.tickets:
            if other != ticket_id and self.state(other)["state"] in ("ACTIVE", "RED_RECORDED", "IMPLEMENTED"):
                raise Refusal(f"Finish or block active ticket {other} first")

    def verify_red_tests(self, ticket):
        red = [r for r in self.runs(ticket["id"]) if r["outcome"] == "RED"]
        if not red:
            raise Refusal("No executed red evidence")
        current = self.fingerprint()["files"]
        for path, value in red[-1]["before"]["files"].items():
            if ("/tests/" in path or path.startswith("tests/") or path.endswith("test_drive.py")) and current.get(path) != value:
                raise Refusal(f"Tests changed after red: {path}; execute a new red before implementing")

    def run(self, ticket_id, command_id, retry_note=None):
        self.inspect_active(recover=True)
        ticket = self.tickets[ticket_id]
        if not self.ready(ticket):
            raise Refusal("Ticket disabled or its implementation dependencies are incomplete/blocked")
        self.assert_single_ticket(ticket_id)
        command = next((c for c in ticket["commands"] if c["id"] == command_id), None)
        if command is None:
            raise Refusal("Unknown command ID; do not execute an unregistered command")
        if command.get("platform") and command["platform"] != os.name:
            raise Refusal(f"Run this command on platform {command['platform']}; current platform is {os.name}")
        state = self.state(ticket_id)["state"]
        if state == "BLOCKED":
            raise Refusal("Record ACTIVE with a resolved-blocker note before retrying")
        if ticket["kind"] == "implementation":
            if command["phase"] == "red" and state not in ("PENDING", "ACTIVE", "RED_RECORDED"):
                raise Refusal("Red must precede implementation")
            if command["phase"] != "red" and state not in ("IMPLEMENTED", "VALIDATED"):
                raise Refusal("Record IMPLEMENTED after red before running green/regression")
            if command["phase"] != "red":
                self.verify_red_tests(ticket)
        runs = self.runs(ticket_id)
        before = self.fingerprint()
        interrupted = [e for e in self.events() if e["kind"] == "run_interrupted"
                       and e.get("ticket") == ticket_id and e.get("command_id") == command_id]
        if interrupted and interrupted[-1]["before"]["sha256"] == before["sha256"]:
            raise Refusal("Interrupted command cannot be restarted unchanged; inspect and resolve the cause")
        previous = [r for r in runs if r["command_id"] == command_id]
        if previous and previous[-1]["before"]["sha256"] == before["sha256"]:
            if previous[-1]["outcome"] in ("PASS", "RED"):
                raise Refusal("This command already has current evidence; check it instead of duplicating")
            if previous[-1]["outcome"] in ("TIMEOUT", "INTERRUPTED"):
                raise Refusal("Timed-out/interrupted command cannot be restarted unchanged")
            if not retry_note:
                raise Refusal("Unchanged failed command requires --retry-note explaining the new observation")
        needed = command.get("requires_artifacts", [])
        if any(not before["artifacts"].get(role, {}).get("sha256") for role in needed):
            raise Refusal("Missing hashed runtime bindings: " + ", ".join(needed))
        remaining = ticket.get("validation_seconds", 7200) - sum(r["elapsed_seconds"] for r in runs)
        timeout = min(command.get("timeout_seconds", 1800), remaining)
        if timeout <= 0:
            raise Refusal("Ticket validation budget exhausted; preserve evidence and report deferred work")
        if state == "PENDING":
            self.append("state", ticket=ticket_id, state="ACTIVE", note="First registered command started")
        run_id = ticket_id + "-" + uuid.uuid4().hex
        directory = self.ledger / run_id
        directory.mkdir(parents=True)
        result_path = directory / "tests.json"
        substitutions = {"python": sys.executable, "package": str(self.package), "repo": str(self.repo),
                         "run_id": run_id, "run_dir": str(directory), "result": str(result_path),
                         **{role: artifact["path"] for role, artifact in before["artifacts"].items()}}
        if command["kind"] == "unittest":
            spec_path = directory / "worker.json"
            write_json(spec_path, {"package": str(self.package), "spec": command["tests"]})
            argv = [sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path), str(result_path)]
        else:
            try:
                argv = [arg.format_map(substitutions) for arg in command["argv"]]
            except KeyError as error:
                raise Refusal(f"Missing runtime binding {error}") from error
        cwd = Path(command.get("cwd", "{package}").format_map(substitutions)).resolve()
        env = dict(os.environ)
        env.update({k: v.format_map(substitutions) for k, v in command.get("env", {}).items()})
        env["PYTHONUTF8"] = "1"
        if "profile" in needed:
            env["CTT_SHARING_MEASUREMENTS"] = str(directory / "measurements.jsonl")
            env["CTT_NATIVE_TRACE"] = str(directory / "native.jsonl")
            env["CTT_ORACLE_TRACE"] = str(directory / "oracle.jsonl")
        started = time.monotonic()
        out, err = directory / "stdout.log", directory / "stderr.log"
        job = WindowsJob()
        self.append("run_start", run=run_id, ticket=ticket_id, command_id=command_id,
                    argv=argv, cwd=str(cwd), timeout_seconds=timeout, before=before,
                    driver=process_identity(os.getpid()), retry_note=retry_note)
        child = None
        terminal = "EXITED"
        try:
            with out.open("xb") as stdout, err.open("xb") as stderr:
                child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                         start_new_session=os.name != "nt")
                try:
                    job.assign(child)
                    identity = process_identity(child.pid)
                    self.append("child", run=run_id, ticket=ticket_id, identity=identity)
                    child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    terminal = "TIMEOUT"
                except KeyboardInterrupt:
                    terminal = "INTERRUPTED"
                finally:
                    if terminal != "EXITED" or child.poll() is None:
                        if os.name != "nt":
                            try:
                                os.killpg(child.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        else:
                            job.close()
                            if child.poll() is None:
                                child.kill()
                        child.wait(timeout=10)
        except (OSError, Refusal) as error:
            terminal = "LAUNCH_ERROR"
            if child is not None and child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            if not err.exists():
                err.write_text(str(error), encoding="utf-8")
            else:
                with err.open("a", encoding="utf-8") as stream:
                    stream.write("\n" + str(error))
            if not out.exists():
                out.write_bytes(b"")
        finally:
            job.close()
        report_path = result_path
        if command["kind"] == "ticket":
            report_path = self.package / "evidence/runs" / (run_id + ".json")
            matches = list((self.package / "evidence/runs").glob("*-" + run_id + ".json"))
            if len(matches) == 1:
                report_path = matches[0]
        elif command["kind"] == "gate":
            evidence_lines = [line.split("Evidence: ", 1)[1] for line in out.read_text(
                encoding="utf-8", errors="replace").splitlines() if line.startswith("Evidence: ")]
            if len(evidence_lines) == 1:
                report_path = (self.package / evidence_lines[0]).resolve()
        try:
            report = read_json(report_path) if report_path.is_file() else None
        except (ValueError, OSError) as error:
            report = {"status": "FAIL", "report_error": str(error)}
        after = self.fingerprint()
        clean = before == after
        exit_code = child.returncode if child else None
        outcome = "FAIL"
        if terminal != "EXITED":
            outcome = terminal
        elif clean:
            if command["phase"] == "red":
                if exit_code != 0 and report and meaningful_red(
                        report, command.get("expected_ids", ()), command.get("assertion_text")):
                    outcome = "RED"
            elif exit_code == 0:
                if command["kind"] == "command":
                    outcome = "PASS"  # command completion only, never a unittest or numerical claim
                elif command["kind"] == "gate" and report and report.get("status") == "PASS":
                    outcome = "PASS"
                elif report and successful(report, command.get("expected_ids", ())):
                    outcome = "PASS"
        artifacts = []
        if report_path.is_file():
            artifacts.append({"path": str(report_path), "sha256": sha(report_path)})
        for raw in command.get("artifacts", []):
            path = Path(raw.format_map(substitutions))
            if not path.is_absolute():
                path = cwd / path
            if not path.is_file():
                outcome = "MISSING_ARTIFACT"
            else:
                artifacts.append({"path": str(path.resolve()), "sha256": sha(path)})
        executed = report.get("executed_ids", []) if report else []
        if command["kind"] == "gate":
            executed = [m.group(2) if m.group(2).endswith("." + m.group(1)) else
                        f"{m.group(2)}.{m.group(1)}" for m in re.finditer(
                r"^(test_\S+) \(([^)]+)\) \.\.\.", out.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)]
        value = {"schema": 1, "ticket": ticket_id, "run": run_id, "command_id": command_id,
                 "phase": command["phase"], "outcome": outcome, "terminal": terminal,
                 "scope": "ticket only; no cumulative gate certification", "argv": argv,
                 "cwd": str(cwd), "exit_code": exit_code, "elapsed_seconds": time.monotonic() - started,
                 "executed_ids": executed,
                 "before": before, "after": after, "source_unchanged": clean,
                 "stdout": str(out), "stderr": str(err), "stdout_sha256": sha(out),
                 "stderr_sha256": sha(err), "test_report": report, "artifacts": artifacts}
        destination = directory / "run.json"
        write_json(destination, value)
        self.append("run_end", run=run_id, ticket=ticket_id, report=str(destination), report_sha256=sha(destination))
        if outcome == "RED":
            self.append("state", ticket=ticket_id, state="RED_RECORDED",
                        note="Registered assertion failed; no import errors/skips/xfails", run=run_id)
        return value


def worker(spec_path, result_path):
    """Use the existing runner's result collector; do not edit acceptance tests."""
    setup = read_json(spec_path)
    package = Path(setup["package"])
    sys.path.insert(0, str(package / "src"))
    module_spec = importlib.util.spec_from_file_location("luna_ticket_result", package / "scripts/run_ticket.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = setup["spec"]
    start = Path(spec["directory"])
    if not start.is_absolute():
        start = package / start
    sys.path.insert(0, str(start))
    loader = unittest.TestLoader()
    suite = (loader.loadTestsFromNames(spec["names"]) if spec.get("names") else
             loader.discover(str(start), pattern=spec.get("pattern", "test_*.py")))
    result = unittest.TextTestRunner(verbosity=2, resultclass=module.RecordedResult).run(suite)
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "test_count": result.testsRun,
              "executed_ids": result.executed,
              "failures": [{"id": t.id(), "traceback": tb} for t, tb in result.failures],
              "errors": [{"id": t.id(), "traceback": tb} for t, tb in result.errors],
              "skipped": [{"id": t.id(), "reason": reason} for t, reason in result.skipped],
              "expected_failures": [t.id() for t, _ in result.expectedFailures],
              "unexpected_successes": [t.id() for t in result.unexpectedSuccesses]}
    write_json(result_path, report)
    return 0 if successful(report) else 1


def inventory(package):
    registry = read_json(package / "docs/acceptance.json")
    definitions = {}
    for path in sorted((package / "tests/gates").rglob("test_*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.setdefault(node.name, []).append(path.relative_to(package).as_posix())
    gates = {}
    for gate in ("G1", "G2", "G3", "G4", "G5"):
        cases = [case for case in registry["cases"] if case["gate"] == gate]
        found = [case["id"] for case in cases if case["test"] in definitions]
        missing = [case["id"] for case in cases if case["test"] not in definitions]
        gates[gate] = {"registered": len(cases), "methods_present": len(found),
                       "present_ids": found, "missing_ids": missing, "executed_here": []}
    locked = read_json(package / "docs/locked-references.json")
    unchanged = all(sha(package / name) == value for name, value in locked["sha256"].items())
    return {"schema": 1, "scope": "static implementation inventory; no tests executed",
            "case_count": registry["case_count"], "registry_consistent": registry["case_count"] == len(registry["cases"]),
            "locked_references_unchanged": unchanged, "gates": gates,
            "method_locations": definitions, "host": socket.gethostname(), "platform": platform.platform()}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return worker(sys.argv[2], sys.argv[3])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default=str(HERE / "queue.json"))
    parser.add_argument("--ledger")
    parser.add_argument("--bindings", help="Local JSON containing artifacts: {role: absolute_path}")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("status")
    sub.add_parser("next")
    static = sub.add_parser("inventory")
    static.add_argument("--output", required=True)
    check = sub.add_parser("check")
    check.add_argument("ticket", nargs="?")
    run = sub.add_parser("run")
    run.add_argument("ticket")
    run.add_argument("command")
    run.add_argument("--retry-note")
    record = sub.add_parser("record")
    record.add_argument("ticket")
    record.add_argument("state", choices=sorted(STATES))
    record.add_argument("--note", required=True)
    record.add_argument("--evidence")
    args = parser.parse_args()
    try:
        driver = Driver(args.queue, args.ledger, args.bindings)
        if args.action == "inventory":
            value = inventory(driver.package)
            write_json(args.output, value)
            print(json.dumps(value, indent=2))
            return 0 if value["registry_consistent"] and value["locked_references_unchanged"] else 1
        with exclusive(driver.ledger / "driver.lock"):
            if args.action in ("status", "check"):
                value = driver.check(getattr(args, "ticket", None))
            elif args.action == "next":
                value = driver.next_ticket()
            elif args.action == "run":
                value = driver.run(args.ticket, args.command, args.retry_note)
                print(json.dumps({k: value[k] for k in ("ticket", "run", "outcome", "exit_code",
                                                        "elapsed_seconds", "scope")}, indent=2))
                return 0 if value["outcome"] in ("PASS", "RED") else 1
            else:
                value = driver.record(args.ticket, args.state, args.note, args.evidence)
            print(json.dumps(value, indent=2))
            return 0
    except (Refusal, KeyError, ValueError, OSError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
