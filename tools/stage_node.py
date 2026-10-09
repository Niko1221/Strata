"""A remote-stage node agent: starts and stops this PC's Strata stage workers for the layer-split tuner
(tools/stage_tune.py), and reports the PC's GPU, RAM and CPU and the link speed toward other nodes.  Python's
standard library only, so it runs wherever the engine runs (Linux, Windows, WSL).

    python tools/stage_node.py node.json

node.json (paths are this PC's):
    {
      "exe": "/srv/stage/bin/strata",              the engine (strata.exe on Windows)
      "lib_dirs": ["/srv/stage/bin"],              added to LD_LIBRARY_PATH (PATH on Windows)
      "bind": "192.0.2.11",                        the address the workers (and the agent) listen on
      "agent_bind": "0.0.0.0",                     optional: the agent's own (default: bind) - "0.0.0.0" also on a
                                                   second NIC, so the tuner's link test runs over the stage link
      "agent_port": 7840,
      "log_dir": "/srv/stage",
      "args": ["--spec", "4", "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768"],
      "token_file": "/srv/stage/stage-token",      or STRATA_STAGE_TOKEN in the environment
      "data_dir": "/srv/stage/shipped",            where tools/stage_ship.py writes (optional)
      "pack": ".../packs/<pack>", "native": "...-00001-of-0000N.gguf", "ple_gguf": "...-00002-of-0000N.gguf",
      "expert_profile": ".../expert-profile.bin",  the model a /start that names none runs (optional with data_dir)
      "drop_cache": true,                          default: drop the model files from the page cache
      "allow_put_bin": false,                      let /put write executables (bin/, mode): the engine shipped
      "extra_allow": [],                           engine flags a /start may add beyond the default list
      "gpu_clocks": {"graphics": [1500, 1905], "memory": [7001, 7001]}
                                                   optional: lock the GPU's clocks (MHz) while a worker runs - for a
                                                   Windows worker (see Workers.lock_clocks); needs administrator/root
    }

What the token allows: a holder can start and stop this PC's workers with the engine and model files node.json
names, and the flags in EXTRA_ALLOWED (or "extra_allow"); with "data_dir", write files under it; with
"allow_put_bin", also the engine there (that is running code of their choice - for a node you ship the engine to).
The token and the traffic are plain text: a LAN you trust, or a VPN tunnel.

The page cache: a worker reads its layers' experts through the file cache and copies them into pinned memory, and
a shipped file passes through the cache too - left there, a 10-layer worker would take its experts' RAM twice (and
under WSL the VM keeps the cache from Windows).  So the agent drops the files it wrote and the files a worker read
from the cache once the worker listens (posix_fadvise DONTNEED: no root; mapped pages stay).  "drop_cache": false
keeps them (a faster restart with the same layers).

Every request carries the shared token in the X-Stage-Token header (the workers get it as STRATA_STAGE_TOKEN), so
only the tuner and the main process drive this PC.  Endpoints (JSON in and out):
    GET  /ping                 {"ok": true}: a round trip with no work behind it
    GET  /info                 the GPU (nvidia-smi), RAM, CPU and the workers running here
    POST /start                {"begin": K, "end": K2?, "next": "host:port"?, "port": 7841, "prefill": 5888,
                                "cache": "auto"|slots, "extra": [...], "model"?: {"dir": D, "native": F, "pack": P,
                                "profile"?: R, "ple"?: G}}
                               - stops the worker on that port, starts this one and answers when it listens (or with
                               its log's tail when it exits).  "model" runs a model shipped under data_dir
                               (data_dir/models/D/F, data_dir/packs/P), so one agent serves every model shipped to it
    POST /stop                 {"port": 7841} (no port: every worker)
    GET  /log?port=P&lines=N   a worker's log tail
    POST /sink                 reads the body; answers its bytes and the seconds it took (a link test toward here)
    POST /send                 {"url": "http://other:7840", "mib": 64} - sends to another node's /sink
    POST /put?path=P&offset=N&total=T[&mode=755][&sha256=H]
                               writes the body at byte N of data_dir/P (a file of T bytes, sparse where nothing
                               was written) and answers the body's sha256; P.ranges.json records what is written
                               (with sha256=H only when the body hashes to H: a damaged piece is sent again, and
                               a range recorded before under it is dropped).  Several at once to one file (other
                               offsets) are fine, and a /put answered ok (its body read whole) keeps an HTTP/1.1
                               connection open for the next (every other answer closes it)
    GET  /ranges?path=P        the byte ranges of data_dir/P written so far ([[offset, length], ...]) and its size
    GET  /sha256?path=P        the sha256 of data_dir/P (a whole file shipped again only when it changed)
The shipped model files keep every offset of the originals, so a node that holds only its layers' bytes (and
every shard's header) loads them as from the whole file - tools/stage_ship.py sends only those.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# engine flags a /start may add (the tuner's): no flag that names a file to write or read
EXTRA_ALLOWED = {"--max-context", "--stage-bind", "--vram-reserve-mib", "--vram-reserve-later-mib", "--spec",
                 "--kv", "--kv-resident", "--pcie-frac", "--pool-workers", "--adapt-every", "--adapt-swaps",
                 "--adapt-decay", "--spec-min-p"}
PLAIN_NAME = re.compile(r"[A-Za-z0-9._-]+")   # a /start's model names: no separator, no drive, no space
DEVICE_NAME = re.compile(r"(CON|PRN|AUX|NUL|COM\d|LPT\d)(\..*)?", re.I)   # a Windows device in any directory
LISTEN_TIMEOUT_S = 600     # loading a worker's experts: up to ~30 GB from disk on a cold cache
STOP_TIMEOUT_S = 180       # a worker with ~30 GB of pinned experts takes a while to unpin


def load_config(path: str) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    token = os.environ.get("STRATA_STAGE_TOKEN", "")
    if not token and cfg.get("token_file"):
        token = Path(cfg["token_file"]).read_text(encoding="utf-8").strip()
    if not token:
        sys.exit("stage_node: no token (STRATA_STAGE_TOKEN or node.json's token_file)")
    cfg["_token"] = token
    return cfg


def gpu_info() -> list[dict]:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    q = "name,memory.total,memory.free,memory.used,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
    try:
        out = subprocess.run([smi, f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 7:
            continue
        gpus.append({"name": f[0], "vram_mib": int(f[1]), "free_mib": int(f[2]), "used_mib": int(f[3]),
                     "pcie_gen": f[4], "pcie_width": f[5], "pcie_width_max": f[6]})
    return gpus


def ram_info() -> dict:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]
        m = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return {}
        return {"total_mib": m.ullTotalPhys >> 20, "available_mib": m.ullAvailPhys >> 20}
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            mem[k] = int(v.split()[0]) // 1024
        return {"total_mib": mem.get("MemTotal"), "available_mib": mem.get("MemAvailable")}
    except OSError:
        return {}


class Workers:
    """The workers this agent started, by port."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.procs: dict[int, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self.start_lock = threading.Lock()   # one /start at a time (two on a port would orphan the first)
        self.clock_lock = threading.Lock()   # a clock lock or reset with its nvidia-smi calls, one at a time
        self.clocks_locked = False
        self.closing = False                 # the agent is stopping: a /start under way stops its own worker

    def log_path(self, port: int) -> Path:
        return Path(self.cfg.get("log_dir", ".")) / f"stage-worker-{port}.log"

    def stop(self, port: int | None) -> list[int]:
        with self.lock:
            ports = [port] if port is not None else list(self.procs)
            procs = [(p, self.procs.pop(p, None)) for p in ports]
        stopped = []
        for p, proc in procs:   # waited for outside the lock: /info answers meanwhile
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(30)
            stopped.append(p)
        self.reset_clocks()
        return stopped

    def close(self) -> None:
        """The agent is stopping: every worker, and the one a /start under way is starting."""
        with self.lock:
            self.closing = True
        self.stop(None)

    def start(self, req: dict) -> dict:
        with self.start_lock:
            return self._start(req)

    def model(self, req: dict) -> dict:
        """The model files a worker runs: node.json's, or a model shipped under data_dir that /start names -
        {"model": {"dir": D, "native": F, "pack": P, "profile"?: R, "ple"?: G}} is data_dir/models/D/F,
        data_dir/packs/P, data_dir/data/R (default: node.json's expert_profile, else data/expert-profile.bin) and,
        only when named, data_dir/models/D/G for --ple-gguf (a worker never runs the PLE layer; the engine finds the
        shard by tensor name), tools/stage_ship.py's layout.  Names of letters, digits, '.', '_' and '-' only, not
        dots alone and not a Windows device, so every path stays inside data_dir (a Windows "C:.." would replace the
        root); a symlink the PC's owner put under data_dir is followed (/put makes none)."""
        c = self.cfg
        m = req.get("model")
        if m is None:
            missing = [k for k in ("pack", "native", "expert_profile") if not c.get(k)]
            if missing:
                raise ValueError(f"node.json names no {', '.join(missing)}: /start must name a model shipped here")
            return {k: str(c.get(k) or "") for k in ("pack", "native", "ple_gguf", "expert_profile")}
        if not c.get("data_dir"):
            raise ValueError("this node has no data_dir: it runs node.json's model only")
        if not isinstance(m, dict) or not all(isinstance(m.get(k), str) for k in ("dir", "native", "pack")):
            raise ValueError('model: {"dir": ..., "native": ..., "pack": ...} expected')
        for k in ("dir", "native", "pack", "profile", "ple"):
            v = m.get(k)
            if v is not None and (not isinstance(v, str) or not PLAIN_NAME.fullmatch(v) or not v.strip(".")
                                  or DEVICE_NAME.fullmatch(v)):
                raise ValueError(f"model {k}: a plain file or directory name expected, not {v!r}")
        root = Path(c["data_dir"])
        mdir = root / "models" / m["dir"]
        out = {"pack": root / "packs" / m["pack"], "native": mdir / m["native"],
               "ple_gguf": mdir / m["ple"] if m.get("ple") else None,
               "expert_profile": root / "data" / m["profile"] if m.get("profile") else
               Path(c.get("expert_profile") or root / "data" / "expert-profile.bin")}
        for k in ("pack", "native", "expert_profile"):
            if not out[k].exists():
                raise ValueError(f"model {k} is not shipped here: tools/stage_ship.py first")
        return {k: str(p or "") for k, p in out.items()}

    def _start(self, req: dict) -> dict:
        port = int(req.get("port", 7841))
        allowed = EXTRA_ALLOWED | set(self.cfg.get("extra_allow", []))
        extra = [str(a) for a in req.get("extra", [])]
        bad = [a for a in extra if a.startswith("-") and a not in allowed]
        if bad:
            return {"ok": False, "error": f"flags not allowed here: {bad} (node.json extra_allow)"}
        if (req.get("end") is None) != (req.get("next") is None):
            return {"ok": False, "error": "a relay needs both end and next"}
        try:
            m = self.model(req)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        self.stop(port)
        c = self.cfg
        args = [c["exe"], "--serve", "--stage-worker", str(port), "--stage-begin", str(int(req["begin"]))]
        if req.get("end") is not None:
            args += ["--stage-end", str(int(req["end"])), "--stage-next", str(req["next"])]
        args += ["--pack", m["pack"], "--native", m["native"]] + (["--ple-gguf", m["ple_gguf"]] if m["ple_gguf"] else [])
        args += ["--expert-profile", m["expert_profile"], "--expert-cache", str(req.get("cache", "auto")),
                 "--prefill", str(int(req.get("prefill", 2048))), "--stage-bind", c["bind"]]
        args += [str(a) for a in c.get("args", [])] + extra
        env = dict(os.environ)
        env["STRATA_STAGE_TOKEN"] = c["_token"]
        env["STRATA_REMOTE_TIMING"] = "1"
        libs = os.pathsep.join(c.get("lib_dirs", []))
        if libs:
            var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
            env[var] = libs + (os.pathsep + env[var] if env.get(var) else "")
        log = self.log_path(port)
        kw = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        try:
            with open(log, "wb") as fh:
                proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                        **kw)
        except OSError as e:   # a wrong exe or log_dir: an answer, not a dropped connection
            return {"ok": False, "port": port, "error": f"cannot start the worker: {e}"}
        with self.lock:
            closing = self.closing
            if not closing:
                self.procs[port] = proc
        if closing:   # close() has run or is running: it would not see this worker
            proc.kill()
            proc.wait(30)
            return {"ok": False, "port": port, "error": "the agent is stopping"}
        clock_note = self.lock_clocks()   # (the worker reads its experts for seconds before its first window)
        t0 = time.time()
        while time.time() - t0 < LISTEN_TIMEOUT_S:
            time.sleep(1)
            text = log.read_text(encoding="utf-8", errors="replace")
            if "listening on" in text:
                if c.get("drop_cache", True):   # its experts are pinned now: the cached file pages are a second copy
                    for f in model_files(m):
                        drop_cache(f)
                r = {"ok": True, "port": port, "seconds": round(time.time() - t0, 1), "log": tail(text, 25)}
                if clock_note:
                    r["clocks"] = clock_note
                return r
            if proc.poll() is not None:
                with self.lock:
                    self.procs.pop(port, None)
                self.reset_clocks()
                return {"ok": False, "port": port, "exit": proc.returncode, "log": tail(text, 40)}
        self.stop(port)
        return {"ok": False, "port": port, "error": f"not listening after {LISTEN_TIMEOUT_S} s",
                "log": tail(log.read_text(encoding="utf-8", errors="replace"), 40)}

    def lock_clocks(self) -> str:
        """node.json "gpu_clocks": {"graphics": [MIN, MAX], "memory": [MIN, MAX], "gpu"?: INDEX} (MHz) - held with
        nvidia-smi -lgc/-lmc (on GPU INDEX, else every GPU) while a worker runs here, and reset when none does (a /stop,
        a worker that exits, the agent stopped by Ctrl+C, SIGTERM or Ctrl+Break; not a killed agent or a closed console
        window: then `nvidia-smi -rgc` and `-rmc` reset them, as does a reboot).  Why: a stage worker's GPU idles
        between decode windows (the other stages run meanwhile), and the Windows driver lowers its clocks then - an RTX
        3070 spent 94% of a decode run below P2 (P5: 810 MHz memory) and its windows took 12-22 ms instead of 3.7-4.1
        ms locked.  The Linux driver held P2 throughout.  Locking needs an administrator (Windows) or root; otherwise
        the reply says "NOT locked" and the worker runs unlocked.  Returns a note for the /start reply ("" when nothing
        was asked)."""
        gc = self.cfg.get("gpu_clocks")
        if not gc:
            return ""
        smi = shutil.which("nvidia-smi")
        if smi is None:
            return "gpu_clocks: NOT locked (no nvidia-smi)"
        dev = ["-i", str(gc["gpu"])] if gc.get("gpu") is not None else []
        notes = []
        with self.clock_lock:   # a reset under way finishes first; one that follows sees the flag and the worker
            if not self.running():
                return ""        # the worker exited already: nothing to hold the clocks for
            with self.lock:
                self.clocks_locked = True
            for key, flag in (("graphics", "-lgc"), ("memory", "-lmc")):
                if key not in gc:
                    continue
                lo, hi = (gc[key], gc[key]) if isinstance(gc[key], int) else gc[key]
                try:
                    p = subprocess.run([smi, *dev, flag, f"{int(lo)},{int(hi)}"], capture_output=True, text=True,
                                       timeout=30)
                    out = (p.stdout + p.stderr).strip().splitlines()
                    notes.append(f"{key} {lo}-{hi} MHz: " + ("locked" if p.returncode == 0 else
                                                              "NOT locked (" + (out[0] if out else
                                                                                f"exit {p.returncode}") +
                                                              "; the agent needs administrator / root)"))
                except (OSError, subprocess.TimeoutExpired) as e:
                    notes.append(f"{key}: NOT locked ({e})")
        return "; ".join(notes)

    def reset_clocks(self) -> None:
        """Undo lock_clocks once no worker runs here."""
        with self.clock_lock:
            with self.lock:
                if not self.clocks_locked or any(p.poll() is None for p in self.procs.values()):
                    return
                self.clocks_locked = False
            smi = shutil.which("nvidia-smi")
            gc = self.cfg.get("gpu_clocks") or {}
            dev = ["-i", str(gc["gpu"])] if gc.get("gpu") is not None else []
            for flag in ("-rgc", "-rmc"):
                try:
                    subprocess.run([smi, *dev, flag], capture_output=True, timeout=30)
                except (OSError, subprocess.TimeoutExpired, TypeError):
                    pass

    def reap(self) -> None:
        """Forget the workers that exited on their own, and reset the clocks if none runs."""
        with self.lock:
            for p in [p for p, proc in self.procs.items() if proc.poll() is not None]:
                self.procs.pop(p)
        self.reset_clocks()

    def running(self) -> list[int]:
        with self.lock:
            return [p for p, proc in self.procs.items() if proc.poll() is None]


def tail(text: str, n: int) -> list[str]:
    return text.splitlines()[-n:]


def drop_cache(path: Path, offset: int = 0, length: int = 0) -> None:
    """Drop a file's clean pages from the page cache (Linux; elsewhere nothing)."""
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


def new_sparse(p: Path, total: int) -> None:
    """p emptied and grown to total bytes, all of them a hole: the bytes nobody writes take no disk."""
    with open(p, "wb", buffering=0) as f:
        if os.name != "nt":
            f.truncate(total)
            return
        # Windows: a file is not sparse unless marked (NTFS and ReFS write a gap as zeros), and the CRT's
        # truncate grows a file by writing zeros itself; SetEndOfFile grows it as a hole
        import ctypes
        import msvcrt
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = wintypes.HANDLE(msvcrt.get_osfhandle(f.fileno()))
        done = wintypes.DWORD()
        FSCTL_SET_SPARSE = 0x900C4
        # no dense fallback: the first write near a shard's end would then write every byte before it (tens of GB)
        if not k32.DeviceIoControl(h, FSCTL_SET_SPARSE, None, 0, None, 0, ctypes.byref(done), None):
            raise ctypes.WinError(ctypes.get_last_error(),
                                  f"{p}: cannot mark it sparse - put data_dir on NTFS or ReFS (not FAT32/exFAT)")
        f.seek(total)
        if not k32.SetEndOfFile(h):
            raise ctypes.WinError(ctypes.get_last_error(), f"{p}: cannot grow it to {total} bytes")


def model_files(cfg: dict) -> list[Path]:
    """The files a worker reads: the GGUF shards next to "native", the pack's files."""
    out: list[Path] = []
    if cfg.get("native"):
        out += sorted(Path(cfg["native"]).parent.glob("*.gguf"))
    if cfg.get("pack") and Path(cfg["pack"]).is_dir():
        out += [p for p in Path(cfg["pack"]).rglob("*") if p.is_file()]
    return out


def merge_ranges(rs: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for off, n in sorted(rs):
        if out and off <= out[-1][0] + out[-1][1]:
            out[-1][1] = max(out[-1][1], off + n - out[-1][0])
        else:
            out.append([off, n])
    return out


def cut_range(rs: list[list[int]], off: int, n: int) -> list[list[int]]:
    """rs without the bytes [off, off + n)."""
    out: list[list[int]] = []
    for o, k in rs:
        if o < off:
            out.append([o, min(k, off - o)])
        if o + k > off + n:
            s = max(o, off + n)
            out.append([s, o + k - s])
    return out


class Store:
    """data_dir: the shipped files and, next to each, the byte ranges written so far (P.ranges.json)."""

    def __init__(self, cfg: dict):
        self.root = Path(cfg["data_dir"]).resolve() if cfg.get("data_dir") else None
        self.drop = bool(cfg.get("drop_cache", True))
        self.allow_bin = bool(cfg.get("allow_put_bin", False))
        self.lock = threading.Lock()

    def path(self, rel: str) -> Path:
        if self.root is None:
            raise ValueError("this node has no data_dir")
        with self.lock:   # not while a /put creates the file: Windows' realpath can answer "\\?\C:\..." meanwhile
            p = (self.root / rel).resolve()
        if self.root not in p.parents:
            raise ValueError(f"{rel}: outside data_dir")
        return p

    @staticmethod
    def side(p: Path) -> Path:
        return p.with_name(p.name + ".ranges.json")

    def read_side(self, p: Path) -> dict:
        """{"fp": the source's fingerprint, "ranges": [[offset, length], ...]} (an old list: no fingerprint)."""
        s = self.side(p)
        if not s.exists():
            return {"fp": None, "ranges": []}
        d = json.loads(s.read_text())
        return d if isinstance(d, dict) else {"fp": None, "ranges": d}

    def ranges(self, rel: str) -> dict:
        p = self.path(rel)
        with self.lock:   # not a sidecar a /put is rewriting
            d = self.read_side(p)
        return {"ok": True, "path": rel, "size": p.stat().st_size if p.exists() else None, "ranges": d["ranges"],
                "fp": d["fp"]}

    def put(self, rel: str, offset: int, total: int, body, n: int, mode: str | None, fp: str | None,
            want: str | None = None) -> dict:
        """`want`: the body's sha256 as the shipper read it - the range is recorded only when the bytes that arrived
        hash to it (else they stay on disk unrecorded, and the next ship sends them again)."""
        p = self.path(rel)
        if not self.allow_bin and (mode or rel.split("/", 1)[0] == "bin"):
            return {"ok": False, "error": "this node does not take executables (node.json allow_put_bin)"}
        with self.lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            d = self.read_side(p) if p.exists() else {"fp": None, "ranges": []}
            # (a sidecar without a fingerprint, from before they existed: kept, and it takes this one)
            if not p.exists() or p.stat().st_size != total or (fp and d["fp"] is not None and d["fp"] != fp):
                # a new file, or a different source (size or fingerprint): what was written of it is void
                new_sparse(p, total)
                self.side(p).write_text(json.dumps({"fp": fp, "ranges": []}))
        # the body streamed to disk 4 MiB at a time, hashed on the way; a handle of this request's own, so /puts at
        # other offsets of the file (the shipper's pieces in flight) write at once without a lock (Windows has no
        # pwrite: an own handle has its own position); the sidecar's read-merge-write is under the lock
        h = hashlib.sha256()
        with open(p, "r+b") as f:
            f.seek(offset)
            left = n
            while left > 0:
                chunk = body.read(min(left, 4 << 20))
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                left -= len(chunk)
            if self.drop:   # written through: not left in the page cache
                f.flush()
                os.fsync(f.fileno())
        if self.drop:
            drop_cache(p, offset, n)
        bad = (f"the body ended {left} bytes short" if left else
               "sha256 differs: the piece arrived damaged (not recorded)" if want is not None and h.hexdigest() != want
               else None)
        with self.lock:
            d = self.read_side(p)
            if bad:   # these bytes are on disk now: a range an earlier put recorded there is no longer sure
                self.side(p).write_text(json.dumps({"fp": d["fp"], "ranges": cut_range(d["ranges"], offset, n - left)}))
                return {"ok": False, "error": bad, "sha256": h.hexdigest()}
            if p.stat().st_size != total or (fp and d["fp"] is not None and d["fp"] != fp):
                # another ship (a different source) started the file anew meanwhile: these bytes are not its
                return {"ok": False, "error": "the file was started anew by another ship meanwhile (not recorded)"}
            if mode:
                os.chmod(p, int(mode, 8))
            self.side(p).write_text(json.dumps({"fp": d["fp"] or fp,
                                                "ranges": merge_ranges(d["ranges"] + [[offset, n]])}))
        return {"ok": True, "sha256": h.hexdigest(), "bytes": n}


class Body:
    """A request body of n bytes: what is left of it to read."""

    def __init__(self, f, n: int):
        self.f, self.left = f, n

    def read(self, k: int) -> bytes:
        b = self.f.read(min(k, self.left))
        self.left -= len(b)
        return b

    def drain(self) -> None:
        while self.left > 0 and self.read(1 << 20):
            pass


def make_handler(cfg: dict, workers: Workers, store: Store):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.1, but only a /put answered ok (its body read whole) keeps the connection (the shipper's next piece
        # follows on it); every other answer closes it, as HTTP/1.0 did - no handler is left with an unread body before
        # a next request.  An idle kept connection is closed after `timeout` s (its thread freed).
        protocol_version = "HTTP/1.1"
        timeout = 600

        def log_message(self, fmt, *args):   # one line per request on stderr
            sys.stderr.write("stage_node: %s %s\n" % (self.address_string(), fmt % args))

        def reply(self, code: int, obj, keep: bool = False) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if not keep:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def authorized(self) -> bool:
            if hmac.compare_digest(self.headers.get("X-Stage-Token", "").encode(), cfg["_token"].encode()):
                return True
            self.reply(403, {"ok": False, "error": "wrong or missing X-Stage-Token"})
            return False

        def body_json(self) -> dict:
            n = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            if not self.authorized():
                return
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/ping":   # a round trip with no work behind it (/info runs nvidia-smi)
                self.reply(200, {"ok": True})
            elif u.path == "/info":
                workers.reap()
                self.reply(200, {"ok": True, "host": platform.node(), "system": platform.platform(),
                                 "cpus": os.cpu_count(), "gpus": gpu_info(), "ram": ram_info(),
                                 "workers": workers.running(), "bind": cfg["bind"],
                                 "gpu_clocks": cfg.get("gpu_clocks")})
            elif u.path == "/sha256":
                try:
                    p = store.path(q["path"][0])
                    h = hashlib.sha256()
                    with open(p, "rb") as f:
                        for b in iter(lambda: f.read(8 << 20), b""):
                            h.update(b)
                    self.reply(200, {"ok": True, "sha256": h.hexdigest(), "size": p.stat().st_size})
                except (KeyError, ValueError, OSError) as e:
                    self.reply(200, {"ok": False, "error": str(e)})
            elif u.path == "/ranges":
                try:
                    self.reply(200, store.ranges(q["path"][0]))
                except (KeyError, ValueError, OSError) as e:
                    self.reply(400, {"ok": False, "error": str(e)})
            elif u.path == "/log":
                port = int(q.get("port", ["7841"])[0])
                lines = int(q.get("lines", ["200"])[0])
                p = workers.log_path(port)
                text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
                self.reply(200, {"ok": True, "port": port, "log": tail(text, lines)})
            else:
                self.reply(404, {"ok": False, "error": "unknown path"})

        def do_POST(self):
            if not self.authorized():
                return
            u = urlparse(self.path)
            path = u.path
            if path == "/put":   # a shipped file's bytes (tools/stage_ship.py)
                q = parse_qs(u.query)
                body = Body(self.rfile, 0)
                try:
                    body = Body(self.rfile, int(self.headers.get("Content-Length", "0")))
                    r = store.put(q["path"][0], int(q["offset"][0]), int(q["total"][0]), body, body.left,
                                  q.get("mode", [None])[0], q.get("fp", [None])[0], q.get("sha256", [None])[0])
                except (KeyError, ValueError, OSError) as e:
                    r = {"ok": False, "error": str(e)}
                # a refused /put's body read to its end first: closed with bytes unread, the socket would be reset
                # under the sender (on Windows) and the reason lost; kept alive only when the body was read whole
                try:
                    body.drain()
                except OSError:
                    pass
                self.reply(200 if r.get("ok") else 400, r, keep=bool(r.get("ok")) and body.left == 0 and
                           not self.headers.get("Transfer-Encoding"))
                return
            if path == "/sink":   # a link test: read the body as fast as it comes
                n = int(self.headers.get("Content-Length", "0"))
                t0 = time.perf_counter()
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    left -= len(chunk)
                self.reply(200, {"ok": True, "bytes": n - left, "seconds": time.perf_counter() - t0})
                return
            try:
                req = self.body_json()
                if not isinstance(req, dict):
                    raise ValueError("a JSON object expected")
            except (ValueError, json.JSONDecodeError) as e:
                self.reply(400, {"ok": False, "error": f"bad JSON: {e}"})
                return
            if path == "/start":
                if "begin" not in req:
                    self.reply(400, {"ok": False, "error": "start needs begin"})
                    return
                self.reply(200, workers.start(req))
            elif path == "/stop":
                self.reply(200, {"ok": True, "stopped": workers.stop(req.get("port"))})
            elif path == "/send":
                self.reply(200, send_test(req["url"], int(req.get("mib", 64)), cfg["_token"]))
            else:
                self.reply(404, {"ok": False, "error": "unknown path"})

    return Handler


def send_test(url: str, mib: int, token: str) -> dict:
    """POST `mib` MiB to another node's /sink: the link's throughput from here (the wall time of the request)."""
    data = b"\0" * (mib << 20)
    req = urllib.request.Request(url.rstrip("/") + "/sink", data=data, method="POST",
                                 headers={"X-Stage-Token": token, "Content-Type": "application/octet-stream"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        got = json.load(r)
    s = time.perf_counter() - t0
    return {"ok": bool(got.get("ok")), "bytes": len(data), "seconds": s, "mb_s": len(data) / s / 1e6}


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cfg = load_config(sys.argv[1])
    workers = Workers(cfg)
    addr = cfg.get("agent_bind") or cfg["bind"]
    srv = ThreadingHTTPServer((addr, int(cfg.get("agent_port", 7840))), make_handler(cfg, workers, Store(cfg)))
    sys.stderr.write(f"stage_node: listening on {addr}:{cfg.get('agent_port', 7840)}\n")

    def interrupt(signum, frame):   # a service stop (SIGTERM) or Ctrl+Break: stop the workers, reset the clocks
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupt)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):   # a second Ctrl+C must not cut the clean-up short
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), signal.SIG_IGN)
        workers.close()


if __name__ == "__main__":
    main()
