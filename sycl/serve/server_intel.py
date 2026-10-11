"""serve/server.py for an Intel Arc: the same server, with the Intel pieces added from outside.

    python sycl/serve/server_intel.py --engine strata --config strata-<model>.json --port 8080 [...]

Every argument is serve/server.py's.  Nothing in serve/ is edited (the SYCL port keeps out of the shared files so
upstream merges stay clean); this wrapper adds, at run time:

  - the Monitor tab's GPU readings on an Intel Arc (sycl/serve/xe_telemetry.py): on Windows always for a "sycl"
    config (the NVIDIA card that drives the desktop answers NVML there and would be shown instead), elsewhere when no
    NVIDIA card answers;
  - on Windows, the engine's environment for a "sycl" config (sycl_env): the card by its SYCL index
    (ONEAPI_DEVICE_SELECTOR=level_zero:N), the Level Zero v2 adapter, no persistent JIT cache, Sysman, the port's
    device-built verify plan, the stager threads, the VRAM heartbeat, and the engine folder first on the PATH for
    the oneAPI runtime DLLs beside strata.exe.  The config's own "env" wins.  The engine is engine\\strata.exe
    itself (no strata-sycl.sh: that is the Linux container wrapper);
  - a model menu in the web app's header, for a host that swaps models on one GPU: the config names the RPC that
    does the swap as "model_switcher" (it takes {"mode": m} and answers {"mode", "up", "starting", "choices",
    "urls"}); the menu offers the models served on this server's port.  Without that key nothing changes;
  - OpenAI `logprobs` / `top_logprobs` on /v1/chat/completions (streamed or not): the engine (the SYCL port,
    `logprobs=K` on its GEN line) writes an `LP` line after each `T` line, which upstream's server would ignore.
    The values are the model's log-probabilities from the verify window's logits, before sampling, temperature and
    penalties; with thinking on, the thinking tokens are listed too (in order, under `reasoning_content`);
  - the start narration for a config whose args carry --stream-experts (install_narrator): upstream's narrator
    describes the CUDA engine's start (tens of GB locked in RAM, a freeze warning), which this route does not do.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import serve.server as S  # noqa: E402
import serve.telemetry as T  # noqa: E402
from xe_telemetry import xe_gpu  # noqa: E402

WIN = os.name == "nt"

# The engine's environment on Windows (the server sets it; the config's "env" wins): the Level Zero v2 adapter (the
# v1 one leaks events per graph finalize on Windows, intel/llvm#23073; the v2 one ran the Arc Pro B70), no persistent
# JIT cache (it segfaults on Xe2 during the first compile, docs/INTEL.md), relaxed allocation limits and Sysman for
# the free-memory query (what llama.cpp's Windows SYCL users set), the port's device-built verify plan with the host
# waiting for the whole window (strata-sycl.sh sets the same two on Linux), and the VRAM heartbeat every 10 s (Windows
# evicts an idle card's memory to RAM, and a Level Zero queue idle for minutes can return DEVICE_LOST,
# intel/llvm#23209; 0 turns it off).  docs/INTEL_WINDOWS.md has the table.
SYCL_ENV = {"SYCL_UR_USE_LEVEL_ZERO_V2": "1", "SYCL_CACHE_PERSISTENT": "0", "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS": "1",
            "ZES_ENABLE_SYSMAN": "1", "STRATA_VERIFY_DEVICE_PLAN": "1", "STRATA_VERIFY_NO_HOST": "1",
            "STRATA_SYCL_HEARTBEAT_S": "10"}


def run_config(argv) -> dict:
    """The run config named by --config in the server's arguments ({} without one, or one that does not parse)."""
    argv = list(argv)
    if "--config" in argv and argv.index("--config") + 1 < len(argv):
        try:
            return json.loads(Path(argv[argv.index("--config") + 1]).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            pass
    return {}


def install_xe_reader(argv=()):
    """telemetry.gpu_reader -> the Intel reader: on Windows for a "sycl" config always (a reader that cannot read
    leaves the Monitor without a GPU, as upstream's Windows HIP state, never the NVIDIA card's figures), elsewhere
    when NVML has no card and an Intel Arc is there."""
    backend = run_config(argv).get("backend")
    orig = T.gpu_reader

    def gpu_reader(index=0, amd=False):
        if WIN and backend == "sycl":
            return xe_gpu(index)
        g = orig(index, amd)
        if g.ok():
            return g
        xe = xe_gpu(index)
        return xe if xe.ok() else g

    T.gpu_reader = gpu_reader


def physical_cores() -> int:
    """The CPU's physical cores (psutil, which setup installs), else half the logical ones."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
    except (ImportError, OSError, RuntimeError):
        n = None
    return int(n) if n else max(1, (os.cpu_count() or 8) // 2)


def sycl_env(cfg: dict, base: dict | None = None) -> dict:
    """The engine's environment for a "sycl" config on Windows: the OS environment (`base` in tests), SYCL_ENV, the
    card by its SYCL index (ONEAPI_DEVICE_SELECTOR=level_zero:N, so the engine's device 0 is that card and never the
    integrated GPU or the NVIDIA one; every Level Zero GPU when the config names none), the stager threads capped at
    the physical cores, then the config's own "env" over all of it, and its "lib_dirs" (relative to the config's
    "cwd", else the repo) first on the PATH: the oneAPI runtime DLLs live beside strata.exe in engine\\."""
    env = dict(os.environ if base is None else base)
    gl = S.gpu_list(cfg)
    env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:" + (",".join(str(i) for i in gl) if gl else "gpu")
    env.update(SYCL_ENV)
    env["STRATA_STAGER_THREADS"] = str(min(12, physical_cores()))
    for k, v in (cfg.get("env") or {}).items():
        env[str(k)] = str(v)
    here = Path(cfg.get("cwd") or ROOT)
    dirs = []
    for d in cfg.get("lib_dirs") or []:
        p = Path(str(d)) if os.path.isabs(str(d)) else here / str(d)
        if p.is_dir():
            dirs.append(str(p))
    if dirs:
        env["PATH"] = os.pathsep.join(dirs + ([env["PATH"]] if env.get("PATH") else []))
    return env


def install_child_env():
    """serve/server.py's child_env -> sycl_env for a "sycl" config on Windows; everything else as upstream."""
    orig = S.child_env

    def child_env(cfg):
        if WIN and cfg.get("backend") == "sycl":
            return sycl_env(cfg)
        return orig(cfg)

    S.child_env = child_env


def install_free_vram(argv=()):
    """Service.free_vram_mib (the min_free_vram_mib gate) reads the Arc for a "sycl" config on Windows, not NVML's
    first card; None when the Arc cannot be read (then nothing is refused, as upstream).  The config is the one the
    server was started with (--config): Service does not keep the backend itself."""
    cfg = run_config(argv)
    backend = cfg.get("backend")
    gl = S.gpu_list(cfg)
    orig = S.Service.free_vram_mib

    def free_vram_mib(self):
        if WIN and backend == "sycl":
            try:
                g = xe_gpu(int(gl[0]) if gl else 0)
                r = g.read() if g.ok() else {}
                if r.get("mem_total") is None or r.get("mem_used") is None:
                    return None
                return int((int(r["mem_total"]) - int(r["mem_used"])) >> 20)
            except Exception:  # noqa: BLE001 - a reading must never take a request down
                return None
        return orig(self)

    S.Service.free_vram_mib = free_vram_mib


def model_switcher(url: str, port, model, mode: str = "") -> dict:
    """The host's model swapper: the models it serves on THIS port, which one holds the card, whether one is loading.
    With `mode`, asks for that one first; this server is then usually the one stopped, so the web app waits for
    /health to name the new model."""
    try:
        body = json.dumps({"mode": mode} if mode else {}).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            res = json.load(r)
        st = res.get("result", res)
    except (OSError, ValueError) as e:
        return {"enabled": True, "error": f"the model switcher did not answer ({e})"}
    urls = st.get("urls") or {}
    mine = {k: v for k, v in (st.get("choices") or {}).items()
            if not urls or str(urllib.parse.urlsplit(str(urls.get(k, ""))).port) == str(port)}
    return {"enabled": True, "mode": st.get("mode"), "up": st.get("up"), "starting": st.get("starting"),
            "choices": mine, "model": model}


def install_switcher(argv):
    """Wraps make_handler: GET/POST /switcher, the menu's script and style, and the index page that loads them."""
    def arg(name, default=None):
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default

    url = run_config(argv).get("model_switcher")
    if not url:
        return
    port = arg("--port", "8080")
    web = HERE / "web"
    types = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
    make_handler = S.make_handler

    def make(svc):
        base = make_handler(svc)

        class Handler(base):
            def _send(self, body: bytes, ctype: str):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0].rstrip("/")
                if path == "/switcher":
                    if self._authorized():
                        self._json(200, model_switcher(url, port, svc.model))
                    return
                if path.startswith("/sycl-web/"):
                    f = web / path[len("/sycl-web/"):]
                    if f.parent != web or f.suffix not in types or not f.is_file():
                        self._json(404, {"error": {"message": "not found"}})
                        return
                    self._send(f.read_bytes(), types[f.suffix])
                    return
                if path == "":
                    page = (ROOT / "serve" / "web" / "index.html").read_text(encoding="utf-8")
                    page = page.replace("</head>", '<link rel="stylesheet" href="sycl-web/switcher.css">\n</head>', 1)
                    page = page.replace("</body>", '<script src="sycl-web/switcher.js"></script>\n</body>', 1)
                    self._send(page.encode("utf-8"), "text/html; charset=utf-8")
                    return
                super().do_GET()

            def do_POST(self):
                path = self.path.split("?")[0].rstrip("/")
                if path == "/switcher":
                    if not self._authorized():
                        return
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    self._json(200, model_switcher(url, port, svc.model, str(req.get("mode") or "")))
                    return
                super().do_POST()

        return Handler

    S.make_handler = make


def install_logprobs():
    """`LP` lines next to their tokens, the request's `logprobs` to the engine, OpenAI's `logprobs` in the reply."""
    Eng = S.StrataEngine

    def _pump(self):   # serve/server.py's StrataEngine._pump, plus: an LP line goes beside its token, not in the queue
        proc, lines = self.proc, self.lines
        lp = self.__dict__.setdefault("lp_lines", collections.deque())
        last_t = None
        for line in proc.stdout:
            if line.startswith("LP "):
                lp.append((last_t, line))
                continue
            if line.startswith("T "):
                try:
                    last_t = int(line[2:])
                except ValueError:
                    last_t = None
            lines.put(line)
        if self.proc is proc:
            self.ended = True
        lines.put(None)
    Eng._pump = _pump

    keys = Eng.sampling_keys

    def sampling_keys(sampling):
        k = top_logprobs(sampling or {})
        return keys(sampling) + (f" logprobs={k}" if k is not None else "")
    Eng.sampling_keys = staticmethod(sampling_keys)

    generate = Eng.generate

    def gen(self, *a, **kw):
        self.__dict__.setdefault("lp_lines", collections.deque()).clear()   # this request's lines only
        return generate(self, *a, **kw)
    Eng.generate = gen

    def entry(tok, t, line):
        f = line.split()
        if len(f) < 2 or f[1] == "nan":
            return None

        def item(i, lp):
            b = tok.token_bytes(i) if hasattr(tok, "token_bytes") else tok.decode([i]).encode()
            return {"token": b.decode("utf-8", "replace"), "logprob": lp, "bytes": list(b)}
        e = item(t, float(f[1])) if t is not None else {"token": "", "logprob": float(f[1]), "bytes": []}
        e["top_logprobs"] = [item(int(i), float(v)) for i, _, v in (x.partition(":") for x in f[2:])]
        return e

    chunks = S.openai_chunks

    def openai_chunks(svc, req, *a, **kw):
        if top_logprobs(req) is None:
            yield from chunks(svc, req, *a, **kw)
            return
        eng = svc.engine
        for c in chunks(svc, req, *a, **kw):
            lp = getattr(eng, "lp_lines", None)
            if c and c.get("choices") and lp:
                got = []
                while lp:
                    t, line = lp.popleft()
                    if t in svc.stop_ids:   # the stop token ends the reply; OpenAI does not list it
                        continue
                    e = entry(svc.tok, t, line)
                    if e is not None:
                        got.append(e)
                if got:
                    d = c["choices"][0].get("delta") or {}
                    key = "reasoning_content" if d.get("reasoning_content") and not d.get("content") else "content"
                    c["choices"][0]["logprobs"] = {key: got}
            yield c
    S.openai_chunks = openai_chunks

    collect = S.openai_collect

    def openai_collect(cs):
        merged = {"content": [], "reasoning_content": []}

        def tap():
            for c in cs:
                if c and c.get("choices") and c["choices"][0].get("logprobs"):
                    for k, v in c["choices"][0]["logprobs"].items():
                        merged[k].extend(v)
                yield c
        out = collect(tap())
        if merged["content"] or merged["reasoning_content"]:
            out["choices"][0]["logprobs"] = {k: v for k, v in merged.items() if v} | {"content": merged["content"]}
        return out
    S.openai_collect = openai_collect


def install_narrator():
    """The start narrator of serve/server.py says the CUDA engine's thing - tens of GB read into RAM, "YOUR PC CAN BE
    SLOW OR STOP RESPONDING" - at the experts' step. With --stream-experts the SYCL port reads them from the GGUF
    into VRAM (and a pinned mirror), which does not do that: say that instead. Without it (an i915 card) the experts
    do go into a RAM arena and the original text stands."""
    orig = S.narrate_start

    def narrate_start(log_path, offset, args, done, heartbeat=20.0):
        if "--stream-experts" not in args:
            return orig(log_path, offset, args, done, heartbeat)
        t0 = last = time.time()
        said = set()

        def say(key, text):
            nonlocal last
            if key not in said:
                said.add(key)
                last = time.time()
                print(text, flush=True)

        say("weights", "[strata] starting the engine: reading the model's weights ...")
        pos = offset
        while not done.wait(0.5):
            try:
                with open(log_path, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
            except OSError:
                chunk = b""
            if chunk.count(b"\n"):
                cut = chunk.rfind(b"\n") + 1
                pos += cut
                for line in chunk[:cut].decode("utf-8", "replace").splitlines():
                    if "PLE on" in line or "expert arena:" in line:
                        say("arena", "[strata] streaming the experts from the model files into the GPU's memory "
                                     "(and a pinned RAM mirror for what does not fit); the first start takes a few minutes ...")
                    elif " loaded " in line and "GiB at" in line:
                        say("loaded", "[strata] experts loaded: " + line.split(" loaded ", 1)[1].strip() +
                            f" ({time.time() - t0:.0f} s so far)")
                    elif "expert cache " in line and " slots, " in line and "auto" not in line:
                        n = line.split("expert cache ", 1)[1].split(";")[0].replace(" slots,", " experts,").strip()
                        say("cache", f"[strata] filling the GPU's expert cache ({n}) ...")
                    elif "session is up" in line:
                        say("up", "[strata] almost ready ...")
            if time.time() - last > heartbeat:
                last = time.time()
                print(f"[strata] still starting ({time.time() - t0:.0f} s) - please wait ...", flush=True)
    S.narrate_start = narrate_start


def top_logprobs(req: dict):
    """The K for the engine's logprobs=K, or None when the request did not ask (OpenAI: logprobs=true, top_logprobs
    0..20; the legacy completions form, logprobs=N, is taken as N)."""
    lp = req.get("logprobs")
    if lp is True:
        k = req.get("top_logprobs")
        return max(0, min(20, int(k))) if isinstance(k, int) and not isinstance(k, bool) else 0
    if isinstance(lp, int) and not isinstance(lp, bool) and lp >= 0:
        return min(20, lp)
    return None


if __name__ == "__main__":
    install_xe_reader(sys.argv[1:])
    install_child_env()
    install_free_vram(sys.argv[1:])
    install_narrator()
    install_switcher(sys.argv[1:])
    install_logprobs()
    sys.exit(S.main())
