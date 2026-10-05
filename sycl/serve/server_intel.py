"""serve/server.py for an Intel Arc: the same server, with the Intel pieces added from outside.

    python sycl/serve/server_intel.py --engine strata --config strata-<model>.json --port 8080 [...]

Every argument is serve/server.py's.  Nothing in serve/ is edited (the SYCL port keeps out of the shared files so
upstream merges stay clean); this wrapper adds, at run time:

  - the Monitor tab's GPU readings on an Intel Arc (sycl/serve/xe_telemetry.py) when no NVIDIA card answers;
  - a model menu in the web app's header, for a host that swaps models on one GPU: the config names the RPC that
    does the swap as "model_switcher" (it takes {"mode": m} and answers {"mode", "up", "starting", "choices",
    "urls"}); the menu offers the models served on this server's port.  Without that key nothing changes;
  - OpenAI `logprobs` / `top_logprobs` on /v1/chat/completions (streamed or not): the engine (the SYCL port,
    `logprobs=K` on its GEN line) writes an `LP` line after each `T` line, which upstream's server would ignore.
    The values are the model's log-probabilities from the verify window's logits, before sampling, temperature and
    penalties; with thinking on, the thinking tokens are listed too (in order, under `reasoning_content`).
"""
from __future__ import annotations

import collections
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import serve.server as S  # noqa: E402
import serve.telemetry as T  # noqa: E402
from xe_telemetry import _XeGpu  # noqa: E402


def install_xe_reader():
    """telemetry.gpu_reader -> the xe reader when NVML has no card and an Intel Arc is there."""
    orig = T.gpu_reader

    def gpu_reader(index=0, amd=False):
        g = orig(index, amd)
        if g.ok():
            return g
        xe = _XeGpu(index)
        return xe if xe.ok() else g

    T.gpu_reader = gpu_reader


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

    try:
        cfg = json.loads(Path(arg("--config")).read_text(encoding="utf-8-sig")) if arg("--config") else {}
    except (OSError, ValueError):
        cfg = {}
    url = cfg.get("model_switcher")
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
    install_xe_reader()
    install_switcher(sys.argv[1:])
    install_logprobs()
    sys.exit(S.main())
