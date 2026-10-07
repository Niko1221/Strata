"""tools/remote-vision/strata_vision_remote.py - run the image encoder on another machine.

WHY THIS EXISTS
---------------
Strata starts the image encoder as a child process and speaks a small line protocol to it:

    server  -> "ENC <image> <out>"      absolute paths inside a local temp dir
    encoder -> "READY <n_embd>"         once, at startup
    encoder -> "OK <tokens>" | "ERR <message>"     one reply per ENC

The encoder is a llama.cpp/mtmd program and is not tied to a GPU model: it reads the picture and
the mmproj file.  Running it on a second PC takes its VRAM - measured 1.19 GB with the encoder
resident - off the machine that runs the language model, and the engine sizes its expert cache
from the VRAM that is free at startup.  That VRAM becomes resident experts, which is the point.

WHAT THIS PROCESS IS
--------------------
A man in the middle for a protocol that passes PATHS, not data, while the server's temp dir only
exists on this machine:

    server --stdio--> wrapper ==(one multiplexed ssh)==> remote encoder
                        |                                     ^
                        +-- picture and embeddings cross as base64 over that same connection

Two small things cross the network per picture: the image, and the embeddings (~4.9 MB for 1024
image tokens at n_embd 2560, bf16).  The remote encoder stays RESIDENT - mmproj is loaded once,
not per picture.

The remote side needs no Strata install:
    <remote_root>/build/bin/strata-vision
    <remote_root>/mmproj-*.gguf
    <remote_root>/models/text-vocab-stub.gguf      metadata+vocabulary only; see make_vocab_stub.py

CONFIGURATION (environment variables)
    STRATA_VISION_SSH            ssh target - REQUIRED, there is no default (an ssh_config
                                 alias, or user@host).  The encoder's host is yours, not this
                                 project's, so it is never guessed.
    STRATA_VISION_ROOT           remote directory           (default ~/Developer/strata-vision)
    STRATA_VISION_SSH_OPTS       extra ssh options
    STRATA_VISION_LIVENESS_PORT  the port the remote watchdog probes back on THIS machine:
                                 the Strata server's own port (default 1918)
    STRATA_VISION_PROBE_FAILS    probe failures in a row before the encoder is reaped (default 3)
    STRATA_VISION_TIMEOUT        seconds to wait for one ENC reply (default: a fifth under the
                                 server's STRATA_VISION_ENCODE_S - see the TIMEOUTS note)
    STRATA_VISION_START_TIMEOUT  seconds to wait for the READY line (default: a fifth under the
                                 server's STRATA_VISION_READY_S)
    STRATA_VISION_SSH_TIMEOUT    seconds for one helper ssh (preflight, put, get)
    STRATA_VISION_RESIDENT_MIB   Windows: working set kept resident for this wrapper (default 64,
                                 0: off) - see the RESIDENCY note below
    STRATA_VISION_HELPER_MIB     Windows: the same for every ssh helper it spawns (default 24, 0: off)

TIMEOUTS
--------
The two waits here are for the same two lines the server waits for, and the server KILLS this wrapper
when its own wait runs out.  So the wrapper's default is derived from the server's, a fifth shorter:
the host being slow is then reported as an ERR the server can read, instead of looking like an encoder
that died.  Set the STRATA_VISION_* knobs above to override either one outright.

LIFECYCLE
---------
The encoder lives exactly as long as the model SERVER on this machine: the remote side runs under
a watchdog that TCP-probes the server's port back, over the address the ssh session came from
($SSH_CONNECTION).  Port open = the service is running; port closed for a few probes in a row =
the service is gone, reap the encoder.  That covers a hard-killed server, a sleeping PC and a
dropped link - cases where no signal reaches the remote side at all.  A hard kill of THIS
process closes its ssh child's socket, so that session gets SIGHUP and the encoder goes with it;
the probe is the backstop for half-open connections.  This process still kills the remote encoder
on QUIT and on stdin EOF (what a dead parent looks like).  The earlier design - this process
touching a heartbeat file over ssh every 20 s - died in production when a single heartbeat ssh
failed: the watchdog reaped a healthy encoder 600 s later while the server kept running fine.

RESIDENCY (Windows)
-------------------
The model server runs this machine close to its memory ceiling (a 47 GiB model in RAM), and under
that pressure Windows trims the working sets of small background processes first.  Measured in
production: a 5 MB `ssh.exe` helper authenticated, opened its session and then sat for 10 minutes
without delivering its exec request, and one image encode took 1131 s - while the vision host was
idle and healthy.  So the wrapper asks for a working-set FLOOR for itself and for every ssh it
spawns (a hard floor via SetProcessWorkingSetSizeEx, plus a normal memory priority and an
above-normal priority class; a job-object floor is not usable - that limit needs a privilege this
process does not hold).  That does not create memory: it keeps ~100 MiB of already-resident pages
from being trimmed, which is the difference between a 2 s encode and a 19 min one on a loaded
machine.  Set STRATA_VISION_RESIDENT_MIB / STRATA_VISION_HELPER_MIB to 0 to switch it off.  All of
it is best effort - a refusal is logged and the wrapper runs exactly as before.
"""
from __future__ import annotations

import base64
import os
import queue
import subprocess
import sys
import threading
import time

# There is no default host on purpose: the encoder's host is the operator's, not this project's, and a
# name baked in here would either be wrong or would quietly ssh somewhere unexpected on someone else's
# machine.  STRATA_VISION_SSH must be set; preflight() says so.
DEFAULT_SSH = ""
DEFAULT_ROOT = "~/Developer/strata-vision"
# The remote watchdog probes the Strata server's port on this machine once per 10 s and reaps the
# encoder after this many consecutive failures.  The port belongs to the service the encoder
# exists to serve, so "port open" is exactly "the service is still running" - and a dead server
# takes its encoder down in ~30 s instead of 10-30 minutes.
DEFAULT_PORT = "1918"
DEFAULT_PROBE_FAILS = 3


def _under(server_env: str, fallback: int) -> int:
    """Wait a fifth less than the server will for the same line.

    serve/server.py reads STRATA_VISION_ENCODE_S (one ENC reply) and STRATA_VISION_READY_S (the READY
    line) and kills this wrapper when its wait runs out - 0 there means "wait for ever".  Waiting
    slightly less here means a slow host is reported as an ERR line the server can read, instead of a
    killed encoder that looks like a crash."""
    try:
        server_s = float(os.environ.get(server_env, "") or 0)
    except ValueError:
        server_s = 0.0                                     # the server would refuse it too; use our own ceiling
    return fallback if server_s <= 0 else max(15, int(server_s * 0.8))


DEFAULT_TIMEOUT = _under("STRATA_VISION_ENCODE_S", 300)      # one ENC reply; a starved host can stretch a 2 s encode to minutes
DEFAULT_START_TIMEOUT = _under("STRATA_VISION_READY_S", 180)  # the READY line: the encoder loads mmproj, ~10-30 s when healthy
DEFAULT_SSH_TIMEOUT = 120        # every helper ssh (preflight, put, get)
DEFAULT_RESIDENT_MIB = 64        # Windows: this wrapper's working-set floor (0: off)
DEFAULT_HELPER_MIB = 24          # Windows: every ssh helper's floor (0: off)
RESIDENT_MIB = int(os.environ.get("STRATA_VISION_RESIDENT_MIB", DEFAULT_RESIDENT_MIB) or 0)
HELPER_MIB = int(os.environ.get("STRATA_VISION_HELPER_MIB", DEFAULT_HELPER_MIB) or 0)


def log(msg: str) -> None:
    print(f"[vision-remote] {msg}", file=sys.stderr, flush=True)


def shq(s: str) -> str:
    return "'" + str(s).replace("'", "'\\''") + "'"


class LineReader:
    """Read the encoder's replies in a thread so every wait has a deadline.

    `stdout.readline()` blocks for good when the channel goes half-open or when this process is
    starved of CPU by memory pressure, and the server holds its request FIFO across an encode: one
    such stall froze a whole API.  A thread plus a queue turns "forever" into a timeout the
    caller's retry/recovery path can handle.
    """

    def __init__(self, stream) -> None:
        self.stream = stream
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="vision-reader").start()

    def _run(self) -> None:
        try:
            for line in self.stream:
                self.lines.put(line)
        except Exception:
            pass
        self.lines.put(None)                       # EOF: wake every waiter

    def read(self, timeout: float, what: str) -> str:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"no {what} within {timeout:.0f} s: the remote encoder or its "
                               f"channel is stuck") from None
        if line is None:
            raise RuntimeError(f"the remote encoder closed the channel (no {what})")
        return line.strip()


# -- residency: keep the vision path out of the working-set trimmer (Windows) ------------------
# Everything below is best effort and Windows-only: a failure only logs.  See RESIDENCY in the
# module docstring for the production measurements that motivated it.
_JOB = None                                       # the shared job object (None: not tried yet)
_JOB_FAILED = object()                            # sentinel: do not retry a failed creation

_JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x00000020
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
_QUOTA_LIMITS_HARDWS_MIN_ENABLE = 0x00000001
_MEMORY_PRIORITY_NORMAL = 5


def _k32():
    import ctypes
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _residency_job():
    """One job object for the ssh helpers: priority class + kill-on-close. -> handle or None.

    Deliberately WITHOUT JOB_OBJECT_LIMIT_WORKINGSET: that flag needs a privilege this process
    does not hold (measured: SetInformationJobObject -> 1314, ERROR_PRIVILEGE_NOT_HELD).  The
    working-set floor is put on each child handle instead - see keep_child_resident.
    """
    global _JOB
    if os.name != "nt":
        return None
    if _JOB is not None:
        return None if _JOB is _JOB_FAILED else _JOB
    try:
        import ctypes
        from ctypes import wintypes
        k32 = _k32()
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW")

        class _Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                        ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _Io(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_ulonglong), ("WriteOperationCount", ctypes.c_ulonglong),
                        ("OtherOperationCount", ctypes.c_ulonglong), ("ReadTransferCount", ctypes.c_ulonglong),
                        ("WriteTransferCount", ctypes.c_ulonglong), ("OtherTransferCount", ctypes.c_ulonglong)]

        class _Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _Basic), ("IoInfo", _Io),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        info = _Extended()
        info.BasicLimitInformation.PriorityClass = _ABOVE_NORMAL_PRIORITY_CLASS
        for flags in (_JOB_OBJECT_LIMIT_PRIORITY_CLASS | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                      _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE):
            info.BasicLimitInformation.LimitFlags = flags
            if k32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                           ctypes.byref(info), ctypes.sizeof(info)):
                _JOB = job
                log(f"residency: {RESIDENT_MIB} MiB for this wrapper, {HELPER_MIB} MiB per ssh helper")
                return job
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject")
    except Exception as e:
        log(f"residency job unavailable ({e}); helpers get the working-set floor only")
        _JOB = _JOB_FAILED
        return None


def keep_child_resident(proc) -> None:
    """Give a freshly spawned ssh a working-set floor, and put it in the residency job."""
    if os.name != "nt" or HELPER_MIB <= 0:
        return
    try:                                    # the floor: the child handle we just created has
        import ctypes                       # PROCESS_SET_QUOTA, so no extra privilege is needed
        from ctypes import wintypes
        k32 = _k32()
        floor = HELPER_MIB * 1024 * 1024
        k32.SetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_size_t,
                                                   ctypes.c_size_t, wintypes.DWORD]
        k32.SetProcessWorkingSetSizeEx.restype = ctypes.c_int
        if not k32.SetProcessWorkingSetSizeEx(wintypes.HANDLE(int(proc._handle)), floor,
                                              max(floor * 8, 256 * 1024 * 1024),
                                              _QUOTA_LIMITS_HARDWS_MIN_ENABLE):
            raise OSError(ctypes.get_last_error(), "SetProcessWorkingSetSizeEx")
    except Exception as e:
        log(f"could not pin a helper process ({e})")
    job = _residency_job()
    if not job:
        return
    try:
        import ctypes
        k32 = _k32()
        k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        k32.AssignProcessToJobObject.restype = ctypes.c_int
        k32.AssignProcessToJobObject(job, ctypes.c_void_p(int(proc._handle)))
    except Exception as e:
        log(f"could not add a helper to the residency job ({e})")


def keep_self_resident() -> None:
    """Ask Windows for a working-set floor on this process and a normal memory priority."""
    if os.name != "nt" or RESIDENT_MIB <= 0:
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = _k32()
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        me = k32.GetCurrentProcess()
        floor = RESIDENT_MIB * 1024 * 1024
        k32.SetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_size_t,
                                                   ctypes.c_size_t, wintypes.DWORD]
        k32.SetProcessWorkingSetSizeEx.restype = ctypes.c_int
        if not k32.SetProcessWorkingSetSizeEx(me, floor, max(floor * 8, 256 * 1024 * 1024),
                                              _QUOTA_LIMITS_HARDWS_MIN_ENABLE):
            raise OSError(ctypes.get_last_error(), "SetProcessWorkingSetSizeEx")

        class _MemoryPriority(ctypes.Structure):
            _fields_ = [("MemoryPriority", wintypes.ULONG)]

        priority = _MemoryPriority(_MEMORY_PRIORITY_NORMAL)
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation(me, 0, ctypes.byref(priority), ctypes.sizeof(priority))
    except Exception as e:
        log(f"could not pin this process ({e})")


class RemoteVision:
    def __init__(self, mmproj_name: str, gpu: bool, threads: int, max_tokens: int) -> None:
        self.mmproj_name = mmproj_name
        self.gpu, self.threads, self.max_tokens = gpu, threads, max_tokens

        self.target = os.environ.get("STRATA_VISION_SSH", DEFAULT_SSH)
        self.root = os.environ.get("STRATA_VISION_ROOT", DEFAULT_ROOT)
        self.extra = os.environ.get("STRATA_VISION_SSH_OPTS", "").split()
        self.port = os.environ.get("STRATA_VISION_LIVENESS_PORT", DEFAULT_PORT)
        self.probe_fails = int(os.environ.get("STRATA_VISION_PROBE_FAILS", str(DEFAULT_PROBE_FAILS)))
        self.timeout = int(os.environ.get("STRATA_VISION_TIMEOUT", DEFAULT_TIMEOUT))
        self.start_timeout = int(os.environ.get("STRATA_VISION_START_TIMEOUT", DEFAULT_START_TIMEOUT))
        self.ssh_timeout = int(os.environ.get("STRATA_VISION_SSH_TIMEOUT", DEFAULT_SSH_TIMEOUT))

        tmp = os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp"
        self.socket = os.path.join(tmp, f"sv-{os.getpid()}")
        # Connection multiplexing would save the per-command handshake, but Windows OpenSSH
        # cannot use a ControlPath socket ("getsockname failed: Not a socket"), so it is OFF by
        # default here and can be switched on where it works: STRATA_VISION_MUX=1.
        base = [
            "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
        ]
        if os.environ.get("STRATA_VISION_MUX", "") not in ("", "0", "no"):
            base += ["-o", f"ControlPath={self.socket}", "-o", "ControlMaster=auto",
                     "-o", "ControlPersist=1h"]
        self._base = [*base, *self.extra]
        self.proc: subprocess.Popen | None = None
        self.reader: LineReader | None = None
        self.stopping = threading.Event()
        # per-session state: two wrappers (the production server and a test) must never clean up
        # each other's files or kill each other's encoder.  The tag is this process's id unless
        # the caller names the session (test-wrapper does, so it can find the pidfile).
        self.session = os.environ.get("STRATA_VISION_SESSION", "").strip() or str(os.getpid())
        self.scratch = ""                          # set in preflight, once ~ is resolved

    # -- plumbing --------------------------------------------------------------
    def _no_window(self) -> dict:
        if os.name == "nt":
            return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
        return {}

    def ssh(self, command: str, **kw) -> subprocess.CompletedProcess:
        # encoding: text=True alone decodes with the Windows locale (GBK on a Chinese Windows);
        # a remote message in UTF-8 crashed the reader thread and emptied the error we report.
        # stdin: a child ssh forwards ITS stdin to the remote command, so without this it competes
        # for the wrapper's stdin - the ENC pipe from the server - and can swallow a picture
        # request, which then waits for a reply that never comes (a production freeze).
        # timeout: no helper call may hang for good; a starved host makes even `mkdir` crawl.
        # Popen (not run): the handle is needed to put the helper in the residency job.
        timeout = kw.pop("timeout", self.ssh_timeout)
        data = kw.pop("input", None)
        if "stdin" not in kw:
            kw["stdin"] = subprocess.PIPE if data is not None else subprocess.DEVNULL
        p = subprocess.Popen([*self._base, self.target, command],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding="utf-8", errors="replace",
                             **kw, **self._no_window())
        keep_child_resident(p)
        try:
            out, err = p.communicate(data, timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            raise TimeoutError(f"ssh gave no answer within {timeout:.0f} s "
                               f"({command[:70]})") from None
        return subprocess.CompletedProcess(p.args, p.returncode, out, err)

    def _resolve_root(self) -> None:
        """`~/x` must not reach the remote shell quoted: single quotes stop tilde expansion."""
        if not self.root.startswith("~"):
            return
        r = self.ssh(f'echo "$HOME"/{self.root.lstrip("~/")}')
        home = r.stdout.strip()
        if not home or " " in home:
            raise RuntimeError(f"could not resolve the remote path {self.root!r}: {r.stderr.strip()[:200]}")
        log(f"remote root resolves to {home}")
        self.root = home

    def preflight(self) -> None:
        """Fail loudly and early: the server reads READY and does not catch a failure here."""
        if not self.target:
            raise RuntimeError(
                "STRATA_VISION_SSH is not set: it names the host that runs the image encoder, as an "
                "ssh_config alias or user@host (with STRATA_VISION_ROOT for the directory on it). "
                "There is no default: that host is yours, and guessing one would be wrong here and "
                "worse on someone else's machine.")
        self._resolve_root()
        self.scratch = f"{self.root}/scratch/{self.session}"      # after the tilde is resolved
        r = self.ssh(
            f"test -x {shq(self.root)}/build/bin/strata-vision && "
            f"test -f {shq(self.root)}/models/text-vocab-stub.gguf && "
            f"test -f {shq(self.root + '/' + self.mmproj_name)} && echo ok")
        if r.stdout.strip() != "ok":
            raise RuntimeError(
                f"the remote vision host is not ready ({self.target}:{self.root}): "
                f"{(r.stderr or r.stdout or 'ssh produced no output').strip()[:300]}. "
                f"Run tools/remote-vision/update-remote-vision.ps1 first.")
        self.ssh(f"mkdir -p {shq(self.scratch)} && rm -f {shq(self.scratch)}/*.img {shq(self.scratch)}/*.sve")
        log(f"host ready: {self.target}:{self.root}")

    # -- the encoder process and the replies -----------------------------------
    def start(self) -> str:
        self.preflight()
        remote_argv = (
            f"cd {shq(self.root)} && "
            f"CUDA_HOME=$(ls -d /usr/local/cuda-13.3 /usr/local/cuda-13.1 /usr/local/cuda 2>/dev/null | head -1); "
            f"export CUDA_HOME; export PATH=\"$CUDA_HOME/bin:$PATH\"; "
            f"exec ./remote-vision-watchdog.sh {shq(self.port)} {self.probe_fails} "
            f"{shq(self.scratch + '/encoder.pid')} "
            f"./build/bin/strata-vision --mmproj {shq(self.root + '/' + self.mmproj_name)} "
            f"--model {shq(self.root + '/models/text-vocab-stub.gguf')}"
            f"{' --gpu' if self.gpu else ''}"
            f"{' --threads ' + str(self.threads) if self.threads else ''}"
            f"{' --max-tokens ' + str(self.max_tokens) if self.max_tokens else ''}")
        self.proc = subprocess.Popen([*self._base, self.target, remote_argv],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, encoding="utf-8", bufsize=1, **self._no_window())
        keep_child_resident(self.proc)                 # the session that carries every picture
        self.reader = LineReader(self.proc.stdout)     # every later wait has a deadline
        line = self.reader.read(self.start_timeout, "READY line")
        if not line.startswith("READY"):
            raise RuntimeError(f"the remote encoder did not start (said {line!r})")
        return line

    def recover(self) -> bool:
        """Restart the remote encoder after it died.  Strata restarts this wrapper when it notices,
        but a mid-request failure is cheaper to absorb here than to hand the user an error."""
        log("the remote encoder is gone; restarting it")
        try:
            if self.proc is not None:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.ssh(f"rm -rf {shq(self.scratch)}; mkdir -p {shq(self.scratch)}")
            line = self.start()
            log(f"recovered: {line}")
            return True
        except Exception as e:
            log(f"recovery failed: {e}")
            return False

    def send(self, line: str) -> str:
        assert self.proc is not None and self.proc.stdin and self.reader is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        # a deadline, and an empty line means the channel closed (the encoder or its watchdog died
        # and the ssh session went with it): BOTH raise, so the caller's recover() restarts the
        # encoder on THIS request instead of failing it or waiting forever.
        return self.reader.read(self.timeout, "reply")

    # -- moving the two files over the same connection -------------------------
    def put(self, local: str, remote: str) -> None:
        data = base64.b64encode(open(local, "rb").read()).decode()
        r = self.ssh(f"base64 -d > {shq(remote)}", input=data, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"could not send the image: {(r.stderr or '').strip()[:200]}")

    def get(self, remote: str, local: str) -> None:
        r = self.ssh(f"base64 -w0 {shq(remote)}", timeout=120)
        if r.returncode != 0 or not r.stdout:
            raise RuntimeError(f"could not read the embeddings back: {(r.stderr or '').strip()[:200]}")
        with open(local, "wb") as f:
            f.write(base64.b64decode(r.stdout))

    def stop(self) -> None:
        self.stopping.set()
        if self.proc is not None:
            try:
                if self.proc.poll() is None and self.proc.stdin:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        # belt and braces: this session's encoder must not be left holding the remote GPU.
        # The kill is scoped to OUR pidfile - never a global pkill: a test session and the
        # production server must not take each other's encoders down.  (Both old forms were
        # traps: `pkill -x strata-vision-r` never matched the real comm "strata-vision", and
        # `pkill -f <path>` matched this very command line and killed its own session.)
        self.ssh(f"p=$(cat {shq(self.scratch + '/encoder.pid')} 2>/dev/null); "
                 f'[ -n "$p" ] && kill -TERM "$p" 2>/dev/null; '
                 f"rm -rf {shq(self.scratch)}; true")
        try:
            os.remove(self.socket)
        except OSError:
            pass
        log("remote encoder stopped")


def serve(rv: RemoteVision) -> int:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line == "QUIT":
            return 0
        if not line.startswith("ENC "):
            print("ERR expected: ENC <image> <output>", flush=True)
            continue
        parts = line.split()
        if len(parts) != 3:
            print("ERR expected: ENC <image> <output>", flush=True)
            continue
        img, out = parts[1], parts[2]
        name = os.path.basename(img)
        rimg = f"{rv.scratch}/{name}"

        started = time.time()

        def attempt() -> str:
            # No ensure_scratch() here.  The scratch dir is made in preflight() and remade by recover(), and an
            # encoder dying does not remove it, so a per-image check only bought a third helper ssh per picture.
            # Measured on the reference machine: one helper ssh is ~240-275 ms end to end (process, handshake,
            # auth, close), so three per image were ~780 ms of the ~800 ms this wrapper adds to an encode.  That
            # is the whole of it - the transfer is ~150 ms for 10 MB, and base64 costs 36 ms on top of raw.
            rv.put(img, rimg)
            reply = rv.send(f"ENC {rimg} {rv.scratch}/{name}.sve")
            if reply.startswith("OK"):
                n = reply.split()
                rv.get(f"{rv.scratch}/{name}.sve", out)
                text = f"OK {n[1] if len(n) > 1 else ''}".strip()
                log(f"{name}: {text} in {time.time() - started:.2f}s "
                    f"({os.path.getsize(out) / 1048576:.2f} MB back)")
                return text
            log(f"{name}: remote said {reply!r}")
            return reply

        try:
            reply = attempt()
        except Exception as e:
            # The encoder can die between two images (it was reaped, its GPU ran out, ssh
            # hiccuped, the vision host rebooted).  Restart it - recover() confirms READY before
            # it returns - and retry, TWO rounds with a short wait between them, before handing
            # the user an error.  One round absorbs a dead encoder; the second covers a vision
            # host that was briefly unreachable or is mid-reboot.
            log(f"{name}: first attempt failed ({e}); restarting the remote encoder")
            reply = f"ERR the remote encoder is gone: {e}"[:300]
            for round in range(1, 3):
                if round > 1:
                    time.sleep(5)
                if rv.recover():
                    try:
                        reply = attempt()
                        break
                    except Exception as e2:
                        log(f"{name}: after restart {round}: {e2}")
                        reply = f"ERR the remote encoder failed after restart {round}: {e2}"[:300]
                else:
                    log(f"{name}: restart {round} failed")
                    reply = f"ERR the remote encoder is gone (restart {round} failed)"[:300]
            log(reply)
        print(reply, flush=True)


def main() -> int:
    argv = sys.argv[1:]
    mmproj = ""
    gpu, threads, max_tokens = False, 0, 0
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--mmproj" and i + 1 < len(argv):
            mmproj = argv[i + 1]; i += 2
        elif a == "--model" and i + 1 < len(argv):
            i += 2                                    # the remote side uses its own stub
        elif a == "--gpu":
            gpu = True; i += 1
        elif a == "--threads" and i + 1 < len(argv):
            threads = int(argv[i + 1]); i += 2
        elif a == "--max-tokens" and i + 1 < len(argv):
            max_tokens = int(argv[i + 1]); i += 2
        else:
            i += 1
    if not mmproj:
        log("usage: strata-vision-remote --mmproj X --model Y [--gpu] [--threads N] [--max-tokens N]")
        return 2

    rv = RemoteVision(os.path.basename(mmproj), gpu, threads, max_tokens)
    keep_self_resident()                               # Windows: do not let the trimmer page us out

    line = ""
    for attempt in range(3):
        try:
            line = rv.start()
            break
        except Exception as e:
            log(f"start attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2.0)
    if not line.startswith("READY"):
        # the server is blocked on this line; a clear error beats a silent hang
        print("ERR the remote vision encoder could not be started; see the server log", flush=True)
        return 1

    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    log(f"encoder up: {line}")

    try:
        return serve(rv)
    finally:
        rv.stop()


if __name__ == "__main__":
    raise SystemExit(main())
