"""Owned media subprocesses, bounded pipes, cancellation and hard deadlines.

No shell, process-name kills or shared Windows-job termination. The media worker
waits for GO on stdin before it can start a decoder; its job is installed first.
"""
from __future__ import annotations

import contextlib
import os
import queue
import signal
import subprocess
import threading
import time

from .video import VideoError


class _PrivateJob:
    def __init__(self, proc, memory_bytes):
        self.handle = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes
        from .winjob import _ExtendedLimits
        k = self.k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        k.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
        k.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        k.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        k.CloseHandle.argtypes = (wintypes.HANDLE,)
        job = k.CreateJobObjectW(None, None)
        if not job:
            raise VideoError("cannot create an owned video decoder job")
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000 | 0x100 | 0x200  # kill-on-close; process/job memory
        info.ProcessMemoryLimit = info.JobMemoryLimit = memory_bytes
        if not k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) or not \
                k.AssignProcessToJobObject(job, int(proc._handle)):
            k.CloseHandle(job)
            raise VideoError("cannot contain the video decoder in its own job")
        self.handle = job

    def close(self):
        if self.handle is not None:
            self.k.TerminateJobObject(self.handle, 1)
            self.k.CloseHandle(self.handle)
            self.handle = None


class OwnedMediaProcess:
    def __init__(self, args, budget, *, max_stdout, max_stderr=65536, worker_gate=False, inherit_group=False,
                 check_exit=True):
        self.args, self.budget = args, budget
        self.max_stdout, self.max_stderr = max_stdout, max_stderr
        self.worker_gate, self.inherit_group, self.check_exit = worker_gate, inherit_group, check_exit
        self.stop = threading.Event()
        self.chunks = queue.Queue(maxsize=4)
        self.errors, self.stderr, self.threads = {}, bytearray(), []
        self.proc = self.job = None

    def __enter__(self):
        self.budget.check()
        try:
            self.proc = subprocess.Popen(self.args, stdin=subprocess.PIPE if self.worker_gate else subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                                         start_new_session=os.name != "nt" and not self.inherit_group,
                                         env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""})
            if not self.inherit_group:
                self.job = _PrivateJob(self.proc, self.budget.policy.decoder_memory_bytes)
            if self.worker_gate:
                self.proc.stdin.write(b"GO\n")
                self.proc.stdin.close()
            self.threads = [threading.Thread(target=self._stdout, daemon=True, name="strata-media-output"),
                            threading.Thread(target=self._stderr, daemon=True, name="strata-media-errors")]
            for thread in self.threads:
                thread.start()
            return self
        except BaseException:
            self.close()
            raise

    def _put(self, data):
        while not self.stop.is_set():
            try:
                self.chunks.put(data, timeout=0.05)
                return
            except queue.Full:
                pass

    def _stdout(self):
        try:
            while not self.stop.is_set():
                data = self.proc.stdout.read(65536)
                if not data:
                    break
                self._put(data)
        except (OSError, ValueError) as e:
            self.errors["stdout"] = str(e)
        finally:
            self._put(None)

    def _stderr(self):
        total = 0
        try:
            while not self.stop.is_set():
                data = self.proc.stderr.read(4096)
                if not data:
                    break
                total += len(data)
                self.stderr.extend(data)
                del self.stderr[:-16384]
                if total > self.max_stderr:
                    self.errors["stderr"] = "video decoder diagnostics exceed the limit"
                    return
        except (OSError, ValueError) as e:
            self.errors["stderr"] = str(e)

    def __iter__(self):
        total = 0
        while True:
            self.budget.check()
            if self.errors:
                raise VideoError(next(iter(self.errors.values())))
            try:
                data = self.chunks.get(timeout=0.05)
            except queue.Empty:
                continue
            if data is None:
                break
            total += len(data)
            if total > self.max_stdout:
                raise VideoError("video decoder output exceeds the configured limit")
            yield data
        while self.proc.poll() is None:
            self.budget.check()
            self.stop.wait(0.05)
        while any(t.is_alive() for t in self.threads):
            self.budget.check()
            self.stop.wait(0.05)
        if self.errors:
            raise VideoError(next(iter(self.errors.values())))
        if self.proc.returncode and self.check_exit:
            # Diagnostics are bounded; never include source contents or an entire ffmpeg log.
            detail = self.stderr.decode("utf-8", "replace")[-1000:].strip()
            raise VideoError("video subprocess failed" + (": " + detail if detail else ""))

    def read(self):
        return b"".join(self)

    def close(self):
        self.stop.set()
        if self.proc is not None:
            if self.proc.poll() is None:
                # Let a worker cancel and reap its direct decoder first. A stuck downloader
                # gets at most two seconds of grace, then the complete owned group/job dies.
                with contextlib.suppress(OSError):
                    self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(OSError):
                        self.proc.kill()
            if os.name != "nt" and not self.inherit_group:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
            if self.job is not None:
                self.job.close()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=5)
            for thread in self.threads:
                thread.join(timeout=2)
            # Kill/reap before closing pipes, so no reader can hold a blocking I/O lock.
            for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                if pipe is not None:
                    with contextlib.suppress(OSError):
                        pipe.close()
        if any(t.is_alive() for t in self.threads):
            raise VideoError("video pipe reader did not terminate")

    def __exit__(self, *exc):
        self.close()
