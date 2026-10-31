"""STRATA_PARK_RAMLIKE=1 (Windows, with STRATA_PARK_DIR): the parked conversations kept in RAM over an engine restart.

STRATA_PARK_DIR alone writes the parked conversations to files at every graceful stop, which costs an SSD's write
endurance.  A file created with FILE_ATTRIBUTE_TEMPORARY (even with FILE_FLAG_DELETE_ON_CLOSE) and never flushed is no
way around that: measured on Windows 11, all 4 GiB of such a file reached the disk while it was being written, with
the handle still open.  What keeps the bytes in RAM is a named, pagefile-backed section that some process keeps
mapped, so:

  - The engine (PARKSAVE with STRATA_PARK_RAMLIKE=1) writes each parked conversation's per-stage images to sections
    "Local\\strata-park-<name>-s<stage>" (the session file's bytes) and lists them in <dir>/park-mem.manifest; it holds
    them until it exits.
  - Before the server closes the engine it starts this keeper (outside the server's process tree, so a tree kill of
    the server does not take it), which opens every listed section by name, maps it and touches every page, so the
    pages stay in its working set (not on the modified list, whose writer would flush them to the pagefile), then
    writes <dir>/keeper.json {"pid", "ready": true, ...}.  "ready" does not wait for the whole walk: after TOUCH_READY_S
    the keeper says ready with the sections open (that alone keeps them alive) and finishes the walk afterwards
    ("touched_all" says which), and the server waits for it at most SPAWN_WAIT_S in all.
  - The next engine's PARKLOAD opens the sections by name and reads them; the server then removes keeper.json and the
    keeper exits within a few seconds - the sections go with their last handle.  A keeper also ends by itself when
    STRATA_PARK_MAX_AGE_H has passed or when a newer save replaced the manifest.  Nothing survives a reboot: the next
    start finds no keeper and the conversations are read again.

Cost: the sections charge the commit (their whole size, at creation) and occupy physical RAM like any process's memory
until the next start or STRATA_PARK_MAX_AGE_H; under memory pressure Windows may trim the keeper's working set and then
page them to the pagefile - the only case in which this writes a disk.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time

MANIFEST = "park-mem.manifest"
KEEPER = "keeper.json"
SECTION_HEAD = 64                       # [u64 length] then the bytes (conversation_file.cpp, kSectionHead)
TOUCH_CHUNK = 256 << 20                 # touch() walks a section in steps of this many bytes (a step is its time-check unit)
TOUCH_READY_S = 8.0                     # the keeper says "ready" after at most this long of touching; the rest goes on after
SPAWN_WAIT_S = 30.0                     # spawn(): the whole wait for a ready keeper; STRATA_PARK_KEEPER_WAIT_S overrides

if sys.platform == "win32":
    import ctypes.wintypes as W
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenFileMappingW.restype = W.HANDLE
    k32.OpenFileMappingW.argtypes = [W.DWORD, W.BOOL, W.LPCWSTR]
    k32.CreateFileMappingW.restype = W.HANDLE
    k32.CreateFileMappingW.argtypes = [W.HANDLE, W.LPVOID, W.DWORD, W.DWORD, W.DWORD, W.LPCWSTR]
    k32.MapViewOfFile.restype = W.LPVOID
    k32.MapViewOfFile.argtypes = [W.HANDLE, W.DWORD, W.DWORD, W.DWORD, ctypes.c_size_t]
    k32.UnmapViewOfFile.argtypes = [W.LPCVOID]
    k32.CloseHandle.argtypes = [W.HANDLE]
    k32.OpenProcess.restype = W.HANDLE
    k32.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
    k32.GetExitCodeProcess.argtypes = [W.HANDLE, ctypes.POINTER(W.DWORD)]
    k32.GetCurrentProcess.restype = W.HANDLE
    k32.SetProcessWorkingSetSizeEx.argtypes = [W.HANDLE, ctypes.c_size_t, ctypes.c_size_t, W.DWORD]

    class MBI(ctypes.Structure):
        _fields_ = [("BaseAddress", W.LPVOID), ("AllocationBase", W.LPVOID), ("AllocationProtect", W.DWORD),
                    ("PartitionId", W.WORD), ("RegionSize", ctypes.c_size_t), ("State", W.DWORD),
                    ("Protect", W.DWORD), ("Type", W.DWORD)]
    k32.VirtualQuery.restype = ctypes.c_size_t
    k32.VirtualQuery.argtypes = [W.LPCVOID, ctypes.POINTER(MBI), ctypes.c_size_t]

FILE_MAP_READ, FILE_MAP_WRITE = 0x4, 0x2
PAGE_READWRITE, SEC_COMMIT = 0x04, 0x08000000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
STILL_ACTIVE = 259


def manifest_sections(folder: str) -> tuple[list[str], dict]:
    """The section names a park-mem.manifest lists ([] when there is none or it is malformed) and its header."""
    try:
        words = open(os.path.join(folder, MANIFEST), encoding="ascii").read().split()
    except (OSError, UnicodeDecodeError):
        return [], {}
    if len(words) < 7 or words[0] != "strata-park" or words[2] != "tag" or words[4] != "stages":
        return [], {}
    try:
        stages = int(words[5])
    except ValueError:
        return [], {}
    names, i = [], 6
    while i + 3 < len(words) and words[i] == "conv":
        conv = words[i + 1]
        if any(c in conv for c in "/\\:"):
            return [], {}
        names += [f"Local\\strata-park-{conv}-s{s}" for s in range(stages)]
        i += 4
    if i >= len(words) or words[i] != "end":
        return [], {}
    return names, {"tag": words[3], "stages": stages}


class Section:
    """A mapped view of a named section (read-only when opened, read-write when created)."""

    def __init__(self, name: str, create_bytes: int | None = None):
        self.name, self.handle, self.view, self.size = name, None, None, 0
        self._pos = 0                                   # where a touch() that ran out of time stopped
        if create_bytes is None:
            self.handle = k32.OpenFileMappingW(FILE_MAP_READ, False, name)
            access = FILE_MAP_READ
        else:
            cap = SECTION_HEAD + create_bytes
            self.handle = k32.CreateFileMappingW(INVALID_HANDLE_VALUE, None, PAGE_READWRITE | SEC_COMMIT,
                                                 cap >> 32, cap & 0xFFFFFFFF, name)
            if self.handle and ctypes.get_last_error() == 183:          # ERROR_ALREADY_EXISTS
                k32.CloseHandle(self.handle)
                self.handle = None
                raise OSError(183, f"section {name} exists already")
            access = FILE_MAP_WRITE
        if not self.handle:
            raise OSError(ctypes.get_last_error(), f"section {name}")
        self.view = k32.MapViewOfFile(self.handle, access, 0, 0, 0)
        if not self.view:
            err = ctypes.get_last_error()
            self.close()
            raise OSError(err, f"map {name}")
        mbi = MBI()
        k32.VirtualQuery(self.view, ctypes.byref(mbi), ctypes.sizeof(mbi))
        self.size = mbi.RegionSize

    def length(self) -> int:
        return ctypes.c_uint64.from_address(self.view).value

    def read(self) -> bytes:
        n = self.length()
        if n > self.size - SECTION_HEAD:
            raise ValueError(f"section {self.name}: length past its end")
        return ctypes.string_at(self.view + SECTION_HEAD, n)

    def write(self, data: bytes) -> None:
        ctypes.memmove(self.view + SECTION_HEAD, data, len(data))
        ctypes.c_uint64.from_address(self.view).value = len(data)

    def covered(self) -> int:
        """The bytes of this section that hold data (head included)."""
        return min(self.size, SECTION_HEAD + self.length())

    def touch(self, until: float | None = None) -> bool:
        """Read one byte of every page, so each page is valid in this process's working set.  In steps of TOUCH_CHUNK;
        with `until` (a time.monotonic() value) it stops between steps once that has passed and the next call goes on
        from there.  -> True when the whole section is covered."""
        n = self.covered()
        pos = self._pos
        while pos < n:
            if until is not None and time.monotonic() >= until:
                self._pos = pos
                return False
            end = min(n, pos + TOUCH_CHUNK)
            self._touch_range(pos, end)
            pos = end
        self._pos = 0
        return True

    def _touch_range(self, start: int, end: int) -> None:
        try:
            import numpy as np
            a = np.ctypeslib.as_array((ctypes.c_uint8 * (end - start)).from_address(self.view + start))
            int(a[::4096].sum())
        except ImportError:
            mv = memoryview((ctypes.c_uint8 * (end - start)).from_address(self.view + start))
            sum(mv[::4096])

    def close(self) -> None:
        if self.view:
            k32.UnmapViewOfFile(self.view)
        if self.handle:
            k32.CloseHandle(self.handle)
        self.view = self.handle = None


def _write_json(path: str, doc: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    os.replace(tmp, path)


def pid_alive(pid: int) -> bool:
    h = k32.OpenProcess(0x1000, False, int(pid))         # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = W.DWORD()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == STILL_ACTIVE


def status(folder: str) -> dict | None:
    """keeper.json when its keeper is alive and ready, else None."""
    try:
        doc = json.load(open(os.path.join(folder, KEEPER), encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or not doc.get("ready") or not pid_alive(int(doc.get("pid") or 0)):
        return None
    return doc


def release(folder: str) -> None:
    """The keeper ends within a few seconds (it watches keeper.json); its sections go with it."""
    try:
        os.remove(os.path.join(folder, KEEPER))
    except FileNotFoundError:
        pass


def keep(folder: str, max_age_s: float, poll_s: float = 2.0, retouch_s: float = 30.0) -> int:
    """The keeper itself: hold every section the manifest lists until released."""
    t0 = time.time()
    mpath = os.path.join(folder, MANIFEST)
    try:
        mtime = os.stat(mpath).st_mtime_ns
    except OSError:
        return 2
    names, head = manifest_sections(folder)
    held: list[Section] = []
    missing = []
    for name in names:
        try:
            held.append(Section(name))
        except OSError:
            missing.append(name)
    # Ready means "every section is open and mapped" (that alone keeps them alive after the engine goes), not "every
    # page is in my working set": walking several GB of pages can take minutes when memory is short, and the server
    # waits for this line before it ends the engine.  So touch for at most TOUCH_READY_S, say ready, finish the walk below.
    total = sum(s.covered() for s in held)
    t_ready = time.monotonic() + TOUCH_READY_S
    walked = [s.touch(until=t_ready) for s in held]
    try:   # a soft floor under the working set, so a trim takes these pages last (no hard limit: the OS still may)
        k32.SetProcessWorkingSetSizeEx(k32.GetCurrentProcess(), total + (64 << 20), total + (256 << 20), 0)
    except OSError:
        pass
    _write_json(os.path.join(folder, KEEPER), {
        "pid": os.getpid(), "ready": True, "created_unix": t0, "manifest_mtime_ns": mtime, "tag": head.get("tag", ""),
        "sections": len(held), "missing": missing, "bytes": total, "touch_s": round(time.time() - t0, 2),
        "touched_all": all(walked)})
    last_touch = time.time()
    while True:
        time.sleep(poll_s)
        try:
            doc = json.load(open(os.path.join(folder, KEEPER), encoding="utf-8"))
            mine = int(doc.get("pid") or 0) == os.getpid()
        except (OSError, ValueError):
            mine = False
        try:
            same = os.stat(mpath).st_mtime_ns == mtime
        except OSError:
            same = False
        if not mine or not same or (max_age_s and time.time() - t0 > max_age_s):
            break
        if not all(walked):                                    # the walk the first touch had no time for
            until = time.monotonic() + poll_s
            walked = [w or s.touch(until=until) for w, s in zip(walked, held)]
            last_touch = time.time()
        elif time.time() - last_touch >= retouch_s:            # pages a trim moved out come back while still in RAM
            for s in held:
                s.touch()
            last_touch = time.time()
    for s in held:
        s.close()
    if mine:
        release(folder)
    return 0


# creation flags: no console, its own group, out of the caller's job when the job allows it
DETACHED_PROCESS, CREATE_NEW_PROCESS_GROUP, CREATE_BREAKAWAY_FROM_JOB = 0x8, 0x200, 0x01000000


def _detached(args: list[str]) -> subprocess.Popen:
    kw = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    try:
        return subprocess.Popen(args, creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
                                CREATE_BREAKAWAY_FROM_JOB, **kw)
    except OSError:                                             # the job does not allow a breakaway
        return subprocess.Popen(args, creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, **kw)


def spawn_wait_s() -> float:
    try:
        return max(5.0, float(os.environ.get("STRATA_PARK_KEEPER_WAIT_S", "") or SPAWN_WAIT_S))
    except ValueError:
        return SPAWN_WAIT_S


def _why_not_ready(folder: str, mtime: int) -> str:
    """What the folder shows when no keeper became ready: for the message of the RuntimeError."""
    try:
        doc = json.load(open(os.path.join(folder, KEEPER), encoding="utf-8"))
    except (OSError, ValueError):
        return "no keeper.json: the keeper did not start, or is still opening the sections"
    alive = pid_alive(int(doc.get("pid") or 0)) if isinstance(doc, dict) else False
    return (f"keeper.json: pid {doc.get('pid')} {'alive' if alive else 'gone'}, ready={doc.get('ready')}, "
            f"manifest {'same' if doc.get('manifest_mtime_ns') == mtime else 'changed'}")


def spawn(folder: str, max_age_s: float, timeout_s: float | None = None, python: str | None = None) -> dict:
    """Start a keeper for the manifest in `folder` (through a middle process that exits at once, so the keeper is no
    one's descendant and a tree kill of the server passes it by) and wait until it holds every section.
    -> its keeper.json.  RuntimeError when it is not ready in time: `timeout_s` is the WHOLE wait (the middle process
    included; default SPAWN_WAIT_S, STRATA_PARK_KEEPER_WAIT_S), because the unload that called this holds the engine
    open and whoever asked for the stop has a limit of its own."""
    t0 = time.monotonic()
    timeout_s = spawn_wait_s() if timeout_s is None else timeout_s
    end = t0 + timeout_s
    release(folder)                                             # an older keeper (another save) ends
    mpath = os.path.join(folder, MANIFEST)
    mtime = os.stat(mpath).st_mtime_ns
    exe = python or sys.executable
    middle = _detached([exe, "-I", os.path.abspath(__file__), "--middle", exe, folder, str(max_age_s)])
    try:
        middle.wait(max(1.0, min(10.0, end - time.monotonic())))
    except subprocess.TimeoutExpired:                           # slow to start; the keeper may still come
        pass
    while time.monotonic() < end:
        doc = status(folder)
        if doc is not None:
            try:
                now = os.stat(mpath).st_mtime_ns                # the manifest as it is now
            except OSError:
                now = mtime
            if doc.get("manifest_mtime_ns") in (mtime, now):
                return doc
        time.sleep(0.2)
    raise RuntimeError(f"no keeper ready in {folder} after {time.monotonic() - t0:.0f} s "
                       f"({_why_not_ready(folder, mtime)})")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--middle":
        exe, folder, age = sys.argv[2], sys.argv[3], sys.argv[4]
        _detached([exe, "-I", os.path.abspath(__file__), "--keep", folder, age])
        sys.exit(0)
    if len(sys.argv) >= 4 and sys.argv[1] == "--keep":
        sys.exit(keep(sys.argv[2], float(sys.argv[3])))
    print(__doc__)
    sys.exit(2)
