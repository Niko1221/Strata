"""Observe real native ENOMEM under a limit applied only to our traced child."""
import hashlib
import os
from pathlib import Path
import re
import resource
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HostAllocationProbe:
    def __init__(self, directory, native_command):
        tracer = shutil.which('strace')
        if not tracer:
            raise RuntimeError('Real host allocation evidence requires installed strace')
        self.tracer = Path(tracer).resolve(strict=True)
        self.native = Path(native_command[0]).resolve(strict=True)
        self.fingerprints = {str(path): sha(path) for path in (self.tracer, self.native)}
        self.trace = Path(directory).resolve() / 'native-allocations.strace'
        if self.trace.exists():
            raise FileExistsError(self.trace)
        # The tracer owns the child; killing the tracer also kills that child.
        # No attachment to a pre-existing engine or another user's process.
        self.command = [str(self.tracer), '--kill-on-exit', '-o', str(self.trace),
                        '-e', 'trace=mmap,mremap,brk', '--', *map(str, native_command)]
        self.native_pid = None
        self.start_ticks = None
        self.process = None
        self.limits = None
        self.restored = False
        self.offset = 0

    def _alive(self):
        try:
            fields = Path('/proc', str(self.native_pid), 'stat').read_text().split(') ')[1].split()
        except FileNotFoundError:
            return False
        if fields[19] != self.start_ticks:
            raise RuntimeError('Native PID was reused; refusing to change its limits')
        return fields[0] != 'Z'

    def bind(self, process):
        children = Path('/proc', str(process.pid), 'task', str(process.pid), 'children').read_text().split()
        if len(children) != 1:
            raise RuntimeError('Expected exactly one tracer-owned native child')
        pid = int(children[0])
        if Path('/proc', str(pid), 'exe').resolve(strict=True) != self.native:
            raise RuntimeError('Tracer child executable differs from pinned native command')
        self.native_pid = pid
        self.start_ticks = Path('/proc', str(pid), 'stat').read_text().split(') ')[1].split()[19]
        self.process = process
        if not self._alive():
            raise RuntimeError('Native child exited before fault injection')

    def arm(self):
        if self.limits is not None or self.native_pid is None or not self._alive():
            raise RuntimeError('Allocation probe requires one live, not-yet-limited native child')
        vm = int(Path('/proc', str(self.native_pid), 'statm').read_text().split()[0]) * os.sysconf('SC_PAGE_SIZE')
        original = resource.prlimit(self.native_pid, resource.RLIMIT_AS)
        soft = vm + 1024 * 1024
        if any(value != resource.RLIM_INFINITY and value < soft for value in original):
            raise RuntimeError('Existing native address-space limit is below the probe boundary')
        self.offset = self.trace.stat().st_size
        applied = (soft, original[1])
        resource.prlimit(self.native_pid, resource.RLIMIT_AS, applied)
        self.limits = dict(original=list(original), applied=list(applied),
                           vm_before_bytes=vm, additional_virtual_bytes=1024 * 1024)
        if resource.prlimit(self.native_pid, resource.RLIMIT_AS) != applied:
            raise RuntimeError('Native address-space limit was not applied')

    def restore_limit(self):
        if self.limits is not None and not self.restored:
            if self._alive():
                resource.prlimit(self.native_pid, resource.RLIMIT_AS, tuple(self.limits['original']))
                if resource.prlimit(self.native_pid, resource.RLIMIT_AS) != tuple(self.limits['original']):
                    raise RuntimeError('Native address-space limit was not restored')
                self.restored = True

    def allocation_failures(self):
        raw = self.trace.read_bytes()[self.offset:].decode()
        patterns = [('mmap', re.compile(r'^mmap\([^,]+,\s*(\d+),.*\)\s+=\s+-1 ENOMEM\b')),
                    ('mremap', re.compile(r'^mremap\([^,]+,\s*\d+,\s*(\d+),.*\)\s+=\s+-1 ENOMEM\b'))]
        failed = []
        for line in raw.splitlines():
            for name, pattern in patterns:
                match = pattern.search(line)
                if match:
                    failed.append(dict(syscall=name, requested_bytes=int(match[1]), errno='ENOMEM', raw=line))
        return failed

    def receipt(self):
        for name, expected in self.fingerprints.items():
            if sha(name) != expected:
                raise RuntimeError('Allocation-probe executable changed: ' + name)
        raw = self.trace.read_bytes()
        return dict(native_pid=self.native_pid, native_start_ticks=self.start_ticks,
                    native_process_exited=not self._alive(), tracer_pid=self.process.pid,
                    tracer_returncode=self.process.poll(), command=self.command,
                    executable_sha256=self.fingerprints, limits=self.limits,
                    limit_restored=self.restored, trace_path=str(self.trace),
                    trace_bytes=len(raw), trace_sha256=hashlib.sha256(raw).hexdigest(),
                    failure_window_offset=self.offset, failure_window=raw[self.offset:].decode(),
                    allocation_failures=self.allocation_failures(),
                    scope='Actual failed OS mapping requests in the native child; virtual-address exhaustion, not system-RAM or GPU exhaustion')
