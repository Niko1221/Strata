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
    writes <dir>/keeper.json {"pid", "ready": true, ...}.
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

    def touch(self) -> int:
        """Read one byte of every page, so each page is valid in this process's working set; -> bytes covered."""
        n = min(self.size, SECTION_HEAD + self.length())
        try:
            import numpy as np
            a = np.ctypeslib.as_array((ctypes.c_uint8 * n).from_address(self.view))
            int(a[::4096].sum())
        except ImportError:
            mv = memoryview((ctypes.c_uint8 * n).from_address(self.view))
            sum(mv[::4096])
        return n

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
    total = sum(s.touch() for s in held)
    try:   # a soft floor under the working set, so a trim takes these pages last (no hard limit: the OS still may)
        k32.SetProcessWorkingSetSizeEx(k32.GetCurrentProcess(), total + (64 << 20), total + (256 << 20), 0)
    except OSError:
        pass
    _write_json(os.path.join(folder, KEEPER), {
        "pid": os.getpid(), "ready": True, "created_unix": t0, "manifest_mtime_ns": mtime, "tag": head.get("tag", ""),
        "sections": len(held), "missing": missing, "bytes": total, "touch_s": round(time.time() - t0, 2)})
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
        if time.time() - last_touch >= retouch_s:              # pages a trim moved out come back while still in RAM
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


def spawn(folder: str, max_age_s: float, timeout_s: float = 120.0, python: str | None = None) -> dict:
    """Start a keeper for the manifest in `folder` (through a middle process that exits at once, so the keeper is no
    one's descendant and a tree kill of the server passes it by) and wait until it holds every section.
    -> its keeper.json.  RuntimeError when it is not ready in time."""
    release(folder)                                             # an older keeper (another save) ends
    mtime = os.stat(os.path.join(folder, MANIFEST)).st_mtime_ns
    exe = python or sys.executable
    _detached([exe, "-I", os.path.abspath(__file__), "--middle", exe, folder, str(max_age_s)]).wait(30)
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        doc = status(folder)
        if doc is not None and doc.get("manifest_mtime_ns") == mtime:
            return doc
        time.sleep(0.2)
    raise RuntimeError(f"no keeper ready in {folder} after {timeout_s:.0f} s")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--middle":
        exe, folder, age = sys.argv[2], sys.argv[3], sys.argv[4]
        _detached([exe, "-I", os.path.abspath(__file__), "--keep", folder, age])
        sys.exit(0)
    if len(sys.argv) >= 4 and sys.argv[1] == "--keep":
        sys.exit(keep(sys.argv[2], float(sys.argv[3])))
    print(__doc__)
    sys.exit(2)
