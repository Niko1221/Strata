"""The remote-stage layer-split tuner: measures this PC (the main process: the first layers, the head and the drafter)
and every node (a PC running tools/stage_node.py with a stage worker) once, then recommends the split points and the
prompt chunk, and writes the configs that run them.

    python3 tools/stage_tune.py tune  cluster.json OUTDIR     measure, search, write OUTDIR/{main.json,nodes.json,report.md}
    python3 tools/stage_tune.py apply cluster.json OUTDIR     start the nodes' workers as OUTDIR/nodes.json says
                                                              (then serve OUTDIR/main.json as any config)

cluster.json:
    {
      "main": {"config": "/opt/strata/strata-iq3_s.json",   the model's server config (args, key, sampling)
               "exe": "/opt/strata/build/strata", "cwd": "/opt/strata",
               "python": "/opt/strata/.venv/bin/python", "port": 8080,
               "api_key_file": "/opt/strata/api-key", "extra": []},
      "nodes": [{"name": "node-b", "agent": "http://192.0.2.11:7840", "host": "192.0.2.11", "port": 7841,
                 "ship": false,                      true: send the node its layers first (tools/stage_ship.py); its
                                                     /start then names the shipped model, so node.json need not
                 "cache": "auto",                    or N: an expert cache of N times the model's largest expert
                                                     blob (a desktop's card, a VRAM budget: ~2.2 MiB a slot for
                                                     IQ3_S, ~3 MiB for UD-Q4_K_XL, plus ~1 GiB for the rest)
                 "max_layers": null}],               the most layers it may hold (its RAM: ~1.1 GiB pinned a layer
                                                     for IQ3_S, ~1.5 GiB for UD-Q4_K_XL)
      "ship": {"model_dir": "/data/models/<model>", "pack_dir": "/data/packs/<pack>",
               "profile": "/opt/strata/data/expert-profile.bin", "bin_dir": null},
      "token_file": "/opt/strata/stage-token",
      "corpus": "/opt/strata/corpus.txt",
      "goal": {
        "measure_tokens": 32000,      the prompt every searched configuration reads ("prompt_tokens" works too)
        "answer_tokens": 500,
        "profile": [[8000, 0.3], [32000, 0.5], [100000, 0.2]],   the workload: prompt lengths and their weights
        "max_context": 131072,        every process's --max-context (262144: the model's limit); the pick is checked
        "verify_max_context": true,   with one prompt near it
        "top_n": 3                    how many of the searched configurations the profile measures
      },
      "chunks": [4096, 5888, 8192],   the prompt chunks to try (the first one is the safe start)
      "max_runs": 10, "init_splits": null    e.g. [26] or [20, 34]; default: by VRAM
    }

The stages run in the order the nodes are listed: main -> node 1 -> node 2 -> ...; every node but the last is a
relay worker (its rows go on to the next node, the next node's reply comes back through it).

The model: its layer count and every layer's bytes come from the GGUF shards next to the main config's --native
(Qwen3.8-Flash-Next: 48 layers; the UD packs have a few heavier layers), so a split is balanced by bytes.

What it measures, per configuration (every process restarted): a prompt of goal.measure_tokens with a 320-token
answer, then three short prompts with 256-token answers; and from the logs (STRATA_REMOTE_TIMING=1) each stage's own
time per prompt chunk and per decode window.  A request's time is prompt / prefill + answer_tokens / decode.

The search: a first split by VRAM; the split that balances the stages' measured prompt cost per byte; the larger
prompt chunks at the better of the two (a larger chunk costs VRAM the decode cache would use - when the engine says
"add --vram-reserve-mib N" the chunk is tried once more with it); then, at the best chunk, each split point two and
then one layer either way while that wins (the best split moves with the chunk).  Then the goal.top_n best read
every prompt length of goal.profile (the score: the weighted mean request time), and the winner reads one prompt
near goal.max_context (a configuration that does not finish it is passed over for the next).  Measured on a busy PC
the numbers move a few percent between runs: the report lists every run.
"""
from __future__ import annotations

import collections
import itertools
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

N_LAYERS_DEFAULT = 48      # Qwen3.8-Flash-Next (every variant Strata runs), when the GGUF cannot be read
MIN_LAYERS = 2             # the fewest layers a stage may hold
CHARS_PER_TOKEN = 3.0      # this tokenizer on English prose (bench_ctx.py; real prompts come out ~10% shorter)
LONG_ANSWER = 320
SHORT_PROMPTS = ("Write a Python function that merges two sorted lists into one sorted list, with a docstring and "
                 "two tests.",
                 "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
                 "List twelve European capitals with one sentence about each.")
SHORT_ANSWER = 256


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def arg_value(args: list, flag: str) -> str | None:
    return str(args[args.index(flag) + 1]) if flag in args and args.index(flag) + 1 < len(args) else None


def with_arg(args: list, flag: str, value: str) -> list:
    out = list(args)
    if flag in out:
        out[out.index(flag) + 1] = value
    else:
        out += [flag, value]
    return out


def model_layers(native: str | None) -> tuple[int, list[int]]:
    """The layer count and each layer's tensor bytes, from the GGUF shards next to --native (uniform without)."""
    if not native or not Path(native).exists():
        return N_LAYERS_DEFAULT, [1] * N_LAYERS_DEFAULT
    from gguf_reader import GGUFFile
    by: collections.Counter = collections.Counter()
    blocks = nextn = None
    for p in sorted(Path(native).parent.glob("*.gguf")):
        g = GGUFFile(p)
        for k, v in g.metadata.items():
            if k.endswith(".block_count") and isinstance(v, int):
                blocks = v
            elif k.endswith(".nextn_predict_layers") and isinstance(v, int):
                nextn = v
        for t in g.tensors:
            m = re.match(r"blk\.(\d+)\.", t.name)
            if m:
                by[int(m.group(1))] += t.expected_bytes() or 0
    n = (blocks - (nextn or 0)) if blocks else (max(by) + 1 if by else N_LAYERS_DEFAULT)
    return n, [by.get(i, 0) or 1 for i in range(n)]


# ---------------------------------------------------------------- nodes (tools/stage_node.py)
class Node:
    def __init__(self, spec: dict, token: str):
        self.name = spec.get("name", spec["host"])
        self.agent = spec["agent"].rstrip("/")
        self.host = spec["host"]
        self.port = int(spec.get("port", 7841))
        self.cache = spec.get("cache", "auto")
        self.extra = list(spec.get("extra", []))
        self.ship = bool(spec.get("ship", False))
        self.max_layers = int(spec["max_layers"]) if spec.get("max_layers") else None
        self.token = token
        self.info: dict = {}

    @property
    def stage_addr(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def stage_agent(self) -> str:
        """The agent at the stage link's address (the same port): reachable there when node.json's agent_bind is."""
        u = urllib.parse.urlparse(self.agent)
        return f"{u.scheme}://{self.host}:{u.port or 7840}"

    def call(self, path: str, body: dict | None = None, timeout: float = 900, data: bytes | None = None,
             base: str | None = None) -> dict:
        headers = {"X-Stage-Token": self.token}
        if data is None and body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        url = (base or self.agent) + path
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"}
        except (urllib.error.URLError, OSError) as e:
            return {"ok": False, "error": f"{base or self.agent}: {e}"}

    def log(self, lines: int = 4000) -> list[str]:
        return self.call(f"/log?port={self.port}&lines={lines}").get("log", [])


# ---------------------------------------------------------------- the main process (serve/server.py)
class Main:
    def __init__(self, spec: dict, token: str, outdir: Path):
        self.spec = spec
        self.token = token
        self.outdir = outdir
        self.base = json.loads(Path(spec["config"]).read_text(encoding="utf-8"))
        self.port = int(spec.get("port", 8080))
        kf = spec.get("api_key_file")
        self.key = Path(kf).read_text().strip() if kf and Path(kf).exists() else os.environ.get("STRATA_API_KEY", "")
        self.proc: subprocess.Popen | None = None
        self.log_path = outdir / "main-engine.log"

    def config(self, splits: list[int], chunk: int, first: Node, timing: bool, max_context: int, extra=()) -> dict:
        args = list(self.base["args"])
        if "--conversation-cache-mib" in args:   # the remote stage turns it off; say so in the config too
            i = args.index("--conversation-cache-mib")
            del args[i:i + 2]
        args = with_arg(args, "--prefill", str(chunk))
        args = with_arg(args, "--max-context", str(max_context))
        args += ["--layer-split", str(splits[0]), "--split-device", "0", "--remote-stage", first.stage_addr]
        args += [str(a) for a in self.spec.get("extra", [])] + [str(a) for a in extra]
        cfg = dict(self.base)
        env = dict(cfg.get("env") or {})
        env["STRATA_STAGE_TOKEN"] = self.token
        if timing:
            env["STRATA_REMOTE_TIMING"] = "1"
        cfg.update(exe=self.spec["exe"], cwd=self.spec["cwd"], args=args, env=env,
                   log=str(self.log_path), model_name=self.base.get("model_name", "strata") + "-stages")
        return cfg

    def start(self, cfg: dict) -> tuple[bool, str]:
        self.stop()
        path = self.outdir / "main-run.json"
        path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        if self.log_path.exists():
            self.log_path.unlink()
        out = open(self.outdir / "main-server.out", "wb")
        self.proc = subprocess.Popen([self.spec["python"], "serve/server.py", "--engine", "strata", "--config",
                                      str(path), "--port", str(self.port)], cwd=self.spec["cwd"],
                                     stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        out.close()
        t0 = time.time()
        while time.time() - t0 < 900:
            time.sleep(3)
            if self.proc.poll() is not None:
                return False, tail(self.outdir / "main-server.out", 30)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                    h = json.load(r)
                if h.get("loaded"):
                    return True, ""
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if self.log_path.exists():   # an engine that exited reports it in the server's output
                srv = (self.outdir / "main-server.out").read_text(encoding="utf-8", errors="replace")
                if "stopped unexpectedly" in srv or "engine exited" in srv:
                    return False, tail(self.outdir / "main-server.out", 30)
        return False, "not loaded after 900 s"

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(30)
        self.proc = None
        exe = self.spec["exe"]
        for _ in range(120):   # the engine (the server's child) frees the GPU before the next start
            if subprocess.run(["pgrep", "-f", exe], capture_output=True).returncode != 0:
                break
            time.sleep(1)

    def chat(self, content: str, max_tokens: int) -> dict:
        body = {"model": "strata", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                "reasoning_effort": "none", "temperature": 0.0}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", json.dumps(body).encode(),
                                     headers)
        try:
            with urllib.request.urlopen(req, timeout=7200) as r:
                resp = json.load(r)
        except urllib.error.HTTPError as e:
            return {"error": e.read().decode(errors="replace")[:300]}
        except (urllib.error.URLError, OSError) as e:
            return {"error": str(e)}
        tm = resp.get("timings") or {}
        return {"prompt_n": tm.get("prompt_n"), "prefill_tps": tm.get("prompt_per_second"),
                "gen_n": tm.get("predicted_n"), "decode_tps": tm.get("predicted_per_second"),
                "text": (resp["choices"][0]["message"].get("content") or "")[:120]}


def tail(path: Path, n: int) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


# ---------------------------------------------------------------- logs -> per-stage times
RE_PF_OWN = re.compile(r"strata prefill: layers (\d+)-(\d+), chunk T=(\d+) at (\d+): own (\d+) ms")
RE_WK_CHUNK = re.compile(r"strata stage worker: chunk T=(\d+) at (\d+): read in (\d+) ms")
RE_WK_WIN = re.compile(r"strata stage worker: windows: own layers ([\d.]+) ms")
RE_MAIN_WIN = re.compile(r"strata remote: main process between windows ([\d.]+) ms")
RE_VRAM = re.compile(r"(\d+) MiB of VRAM free with everything loaded")
RE_RESERVE = re.compile(r"add --vram-reserve-mib (\d+)")


def median_or_none(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def stage_times(lines: list[str], chunk: int, last: bool) -> dict:
    """A stage's own prompt-chunk time (full chunks only) and decode-window time from its log."""
    pf = [float(m.group(5)) for m in map(RE_PF_OWN.search, lines) if m and int(m.group(3)) == chunk]
    if last or not pf:   # the last worker reports the chunk it read (a relay its sending stage's line)
        pf = pf or [float(m.group(3)) for m in map(RE_WK_CHUNK.search, lines) if m and int(m.group(1)) == chunk]
    win = [float(m.group(1)) for m in map(RE_WK_WIN.search, lines) if m]
    win = win or [float(m.group(1)) for m in map(RE_MAIN_WIN.search, lines) if m]
    vram = [int(m.group(1)) for m in map(RE_VRAM.search, lines) if m]
    return {"pf_chunk_ms": median_or_none(pf), "window_ms": median_or_none(win[1:] or win),
            "vram_free_mib": vram[-1] if vram else None}


# ---------------------------------------------------------------- the tuner
class Tuner:
    def __init__(self, cluster: dict, outdir: Path):
        self.cl = cluster
        self.outdir = outdir
        outdir.mkdir(parents=True, exist_ok=True)
        tf = cluster.get("token_file")
        self.token = (Path(tf).read_text().strip() if tf else os.environ.get("STRATA_STAGE_TOKEN", "")).strip()
        if not self.token:
            sys.exit("stage_tune: no token (cluster.json's token_file or STRATA_STAGE_TOKEN)")
        self.nodes = [Node(n, self.token) for n in cluster["nodes"]]
        self.main = Main(cluster["main"], self.token, outdir)
        self.n_layers, self.layer_bytes = model_layers(arg_value(self.main.base["args"], "--native"))
        self.pre = [0] + list(itertools.accumulate(self.layer_bytes))
        goal = cluster.get("goal", {})
        self.measure_tokens = int(goal.get("measure_tokens", goal.get("prompt_tokens", 32000)))
        self.answer_tokens = int(goal.get("answer_tokens", 500))
        self.profile = [(int(n), float(w)) for n, w in goal.get("profile", [])]
        base_ctx = arg_value(self.main.base["args"], "--max-context")
        self.max_context = int(goal.get("max_context") or base_ctx or 131072)
        self.verify_max = bool(goal.get("verify_max_context", True))
        self.top_n = int(goal.get("top_n", 3))
        self.ship_cfg = cluster.get("ship") or {}
        ml = cluster["main"].get("max_layers")
        self.max_layers = [int(ml) if ml else None] + [n.max_layers for n in self.nodes]   # per stage
        self.corpus = Path(cluster["corpus"]).read_text(encoding="utf-8", errors="replace")
        self.cpt = CHARS_PER_TOKEN   # chars per token of this corpus, learned from the prompts read
        self.runs: list[dict] = []
        self.runs_path = outdir / "runs.jsonl"
        self.clock_notes: dict[str, str] = {}   # node -> its agent's "NOT locked" answer (gpu_clocks failed)

    def stages(self, splits: list[int]) -> list[tuple[int, int]]:
        b = [0] + list(splits) + [self.n_layers]
        return [(b[i], b[i + 1]) for i in range(len(b) - 1)]

    def node_extra(self, n: Node) -> list:
        return [str(a) for a in n.extra] + ["--max-context", str(self.max_context)]

    def ship_layers(self, n: Node, lo: int, hi: int) -> tuple[bool, str]:
        """Send the node what layers lo..hi-1 need and it lacks (tools/stage_ship.py)."""
        import stage_ship
        c = self.ship_cfg
        if not c.get("model_dir"):
            return False, "a node with \"ship\": true needs cluster.json's \"ship\" paths"
        items = stage_ship.plan(Path(c["model_dir"]), Path(c["pack_dir"]), Path(c["profile"]),
                                Path(c["bin_dir"]) if c.get("bin_dir") else None, lo, hi)
        try:
            stage_ship.ship(stage_ship.Agent(n.agent, self.token), items, say=lambda m: say("  " + m))
        except (SystemExit, OSError, urllib.error.URLError) as e:
            return False, f"shipping layers {lo}-{hi - 1}: {e}"
        return True, ""

    def start_req(self, i: int, lo: int, hi: int, chunk: int) -> dict:
        """Node i's /start for layers lo..hi-1."""
        n = self.nodes[i]
        req = {"begin": lo, "port": n.port, "prefill": chunk, "cache": n.cache, "extra": self.node_extra(n)}
        if n.ship:   # the model shipped to the node's data_dir: its node.json need not name one
            c = self.ship_cfg
            req["model"] = {"dir": Path(c["model_dir"]).name, "pack": Path(c["pack_dir"]).name,
                            "native": Path(arg_value(self.main.base["args"], "--native")).name,
                            "profile": Path(c["profile"]).name}
        if i + 1 < len(self.nodes):
            req.update(end=hi, next=self.nodes[i + 1].stage_addr)
        return req

    def node_notes(self) -> list[str]:
        """What the nodes' /info and their /start answers say the user should change."""
        notes = [f"{name}: its gpu_clocks were not applied ({why}); the measurements ran with the driver's clocks"
                 for name, why in self.clock_notes.items()]
        for n in self.nodes:
            system = str(n.info.get("system", ""))
            if (system.startswith("Windows") or "microsoft" in system.lower()) and not n.info.get("gpu_clocks"):
                notes.append(f"{n.name} runs under Windows (WDDM, WSL included): its driver lowers the GPU's clocks "
                             "between decode windows. Lock them while the worker runs - node.json \"gpu_clocks\" "
                             "with the agent run as administrator (an RTX 3070: 12-22 ms a window unlocked, 3.7-4.1 "
                             "ms locked; docs/REMOTE_STAGE.md). Under WSL lock them from Windows (nvidia-smi -lgc/-lmc).")
        return notes

    def start_nodes(self, splits: list[int], chunk: int) -> tuple[bool, str]:
        for n in self.nodes:
            n.call("/stop", {})
        ranges = self.stages(splits)[1:]
        for i in reversed(range(len(self.nodes))):   # the last first: a relay connects to the next when it starts
            n, (lo, hi) = self.nodes[i], ranges[i]
            if n.ship:
                ok, why = self.ship_layers(n, lo, hi)
                if not ok:
                    return False, f"{n.name}: {why}"
            r = n.call("/start", self.start_req(i, lo, hi, chunk))
            if not r.get("ok"):
                return False, f"{n.name}: " + (r.get("error") or "") + "\n" + "\n".join(r.get("log", [])[-15:])
            say(f"  {n.name}: layers {lo}-{hi - 1} up in {r.get('seconds')} s"
                + (f"; clocks: {r['clocks']}" if r.get("clocks") else ""))
            if "NOT locked" in str(r.get("clocks", "")):
                self.clock_notes[n.name] = r["clocks"]
            elif r.get("clocks"):
                self.clock_notes.pop(n.name, None)   # locked this time (the agent restarted as administrator)
        return True, ""

    def prompt(self, tokens: int, salt: int) -> str:
        n = int(tokens * self.cpt)
        text = self.corpus * (1 + (n + salt) // max(1, len(self.corpus)))
        start = salt % max(1, len(self.corpus))   # a different slice per length
        return ("Here is a long document. After reading it, write a detailed technical summary of its main ideas in "
                "about 250 words.\n\n<document>\n" + text[start:start + n] + "\n</document>")

    def score(self, by_ctx: dict, weights: list[tuple[int, float]]) -> float | None:
        tot = wsum = 0.0
        for n, w in weights:
            r = by_ctx.get(str(n))
            if not r or not r.get("prefill_tps") or not r.get("decode_tps"):
                return None
            tot += w * ((r["prompt_n"] or n) / r["prefill_tps"] + self.answer_tokens / r["decode_tps"])
            wsum += w
        return tot / wsum if wsum else None

    def measure(self, splits: list[int], chunk: int, extra=(), contexts=None, kind: str = "search") -> dict:
        contexts = list(contexts or [self.measure_tokens])
        key = [list(splits), chunk, list(extra), kind, contexts]
        for r in self.runs:   # measured already
            if r["key"] == key:
                return r
        say(f"measure ({kind}): splits {splits} (stages {self.stages(splits)}), chunk {chunk} {' '.join(extra)}"
            + (f", prompts {contexts}" if kind != "search" else ""))
        run = {"key": key, "kind": kind, "splits": list(splits), "chunk": chunk, "extra": list(extra),
               "contexts": contexts, "stages": self.stages(splits), "ok": False, "by_ctx": {}}
        ok, why = self.start_nodes(splits, chunk)
        if ok:
            ok, why = self.main.start(self.main.config(splits, chunk, self.nodes[0], True, self.max_context, extra))
        if not ok:
            run["error"] = why.strip()[-1500:]
            if self.main.log_path.exists():   # an engine that ran out of VRAM says how much headroom it wants
                res = [int(m.group(1)) for m in map(RE_RESERVE.search, self.main.log_path.read_text(
                    encoding="utf-8", errors="replace").splitlines()) if m]
                if res:
                    run["suggested_reserve_mib"] = res[-1]
            self.main.stop()
            say(f"  does not run: {why.strip().splitlines()[-1] if why.strip() else '?'}")
            self.record(run)
            return run
        failed = None
        for i, n in enumerate(contexts):
            text = self.prompt(n, 7919 * (i + 1))
            r = self.main.chat(text, LONG_ANSWER)
            run["by_ctx"][str(n)] = r
            if r.get("error") or not r.get("prefill_tps") or not r.get("decode_tps"):
                failed = r.get("error", "no prefill or decode timing")
                break
            if r.get("prompt_n"):   # the corpus's real chars per token (code or CJK read denser than prose)
                self.cpt = len(text) / r["prompt_n"]
        if failed is None and kind != "max":
            shorts = [self.main.chat(p, SHORT_ANSWER) for p in SHORT_PROMPTS]
            run["short_decode"] = median_or_none([s["decode_tps"] for s in shorts if s.get("decode_tps")])
        main_lines = self.main.log_path.read_text(encoding="utf-8", errors="replace").splitlines() \
            if self.main.log_path.exists() else []
        st = [stage_times(main_lines, chunk, last=False)]
        for i, nd in enumerate(self.nodes):
            st.append(stage_times(nd.log(), chunk, last=i + 1 == len(self.nodes)))
        for s, (lo, hi) in zip(st, run["stages"]):
            s["layers"] = hi - lo
            s["bytes"] = self.pre[hi] - self.pre[lo]
        run["per_stage"] = st
        res = [int(m.group(1)) for m in map(RE_RESERVE.search, main_lines) if m]
        if res:
            run["suggested_reserve_mib"] = res[-1]
        if failed is not None:
            run["error"] = failed
            say(f"  failed: {failed[:200]}")
        else:
            run["ok"] = True
            first = run["by_ctx"][str(contexts[0])]
            run["prefill_tps"], run["decode_tps"] = first["prefill_tps"], first["decode_tps"]
            weights = self.profile if kind == "profile" else [(n, 1.0) for n in contexts]
            run["request_s"] = self.score(run["by_ctx"], weights)
            say(f"  " + "; ".join(f"{n // 1000}K: prefill {run['by_ctx'][str(n)]['prefill_tps']:.0f}, decode "
                                  f"{run['by_ctx'][str(n)]['decode_tps']:.1f}" for n in contexts)
                + (f" (short {run['short_decode']:.1f})" if run.get("short_decode") else "")
                + f", request {run['request_s']:.1f} s; stages: "
                + ", ".join(f"{s['layers']}L pf {s['pf_chunk_ms'] or 0:.0f} ms win {s['window_ms'] or 0:.1f} ms"
                            for s in st))
        self.main.stop()
        self.record(run)
        return run

    def record(self, run: dict) -> None:
        self.runs.append(run)
        with open(self.runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run) + "\n")

    # ------------------------------------------------------------ the search
    def split_by_cost(self, rate: list[float]) -> list[int]:
        """The split points that make the slowest stage fastest, stage s costing rate[s] per byte of its layers
        (every split of the layers into len(rate) runs of at least MIN_LAYERS is tried)."""
        n, S = self.n_layers, len(rate)
        best = None
        for sp in itertools.combinations(range(MIN_LAYERS, n - MIN_LAYERS + 1), S - 1):
            b = (0,) + sp + (n,)
            if not self.valid(list(sp)):
                continue
            cost = max((self.pre[b[i + 1]] - self.pre[b[i]]) * rate[i] for i in range(S))
            if best is None or cost < best[0]:
                best = (cost, list(sp))
        if best is None:
            sys.exit("stage_tune: no split fits every stage's max_layers")
        return best[1]

    def initial_splits(self) -> list[int]:
        if self.cl.get("init_splits"):
            return [int(x) for x in self.cl["init_splits"]]
        # by VRAM: each stage's share of the bytes follows its card's free memory (the main process keeps ~1.5 GB
        # for the head, the drafter and the windows)
        vram = [self.main_vram_mib() - 1500]
        for n in self.nodes:
            g = (n.info.get("gpus") or [{}])[0]
            vram.append(max(1000, g.get("free_mib", g.get("vram_mib", 8000)) - 500))
        return self.split_by_cost([1.0 / v for v in vram])

    def main_vram_mib(self) -> int:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20).stdout
            return int(out.split()[0])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return 10000

    def balanced(self, run: dict) -> list[int] | None:
        """The split that gives every stage the same prompt time, from the run's cost per byte of each stage."""
        rate = []
        for s in run.get("per_stage", []):
            if not s.get("pf_chunk_ms") or not s.get("bytes"):
                return None
            rate.append(s["pf_chunk_ms"] / s["bytes"])
        return self.split_by_cost(rate)

    def valid(self, splits: list[int]) -> bool:
        b = [0] + splits + [self.n_layers]
        caps = getattr(self, "max_layers", None) or [None] * (len(b) - 1)
        return all(b[i + 1] - b[i] >= MIN_LAYERS and (caps[i] is None or b[i + 1] - b[i] <= caps[i])
                   for i in range(len(b) - 1))

    def best(self, chunk: int | None = None, kind: str = "search") -> dict | None:
        ok = [r for r in self.runs if r["ok"] and r["kind"] == kind and (chunk is None or r["chunk"] == chunk)]
        return min(ok, key=lambda r: r["request_s"]) if ok else None

    def search(self) -> dict | None:
        max_runs = int(self.cl.get("max_runs", 10))
        chunks = [int(c) for c in self.cl.get("chunks", [4096, 5888, 8192])]
        c0 = chunks[0]
        r = self.measure(self.initial_splits(), c0)
        if not r["ok"]:
            say("the first split does not run; trying an even one")
            r = self.measure(self.split_by_cost([1.0] * (len(self.nodes) + 1)), c0)
            if not r["ok"]:
                return None
        # 1. balance the stages' prompt time
        bal = self.balanced(r)
        if bal and bal != r["splits"] and self.valid(bal):
            self.measure(bal, c0)
        # 2. the prompt chunk, at the best split so far
        splits = self.best(c0)["splits"]
        for c in chunks[1:]:
            r = self.measure(splits, c)
            if not r["ok"] and r.get("suggested_reserve_mib"):
                extra = ["--vram-reserve-mib", str(r["suggested_reserve_mib"] + 100)]
                say(f"  the engine asks for more VRAM headroom: once more with {' '.join(extra)}")
                r = self.measure(splits, c, extra)
            if not r["ok"]:
                say(f"  chunk {c} does not run at this split; larger ones are not tried")
                break
        best = self.best()
        chunk, extra = best["chunk"], best["extra"]
        # 3. the split points at that chunk: two, then one layer either way, while that wins
        for step in (2, 1):
            improved = True
            while improved and len(self.runs) < max_runs:
                improved = False
                cur = min((x for x in self.runs if x["ok"] and x["kind"] == "search" and x["chunk"] == chunk
                           and x["extra"] == extra), key=lambda x: x["request_s"])
                for i in range(len(cur["splits"])):
                    for d in (-step, step):
                        cand = list(cur["splits"])
                        cand[i] += d
                        if not self.valid(cand) or len(self.runs) >= max_runs:
                            continue
                        r = self.measure(cand, chunk, extra)
                        if r["ok"] and r["request_s"] < cur["request_s"]:
                            improved = True
                            break
                    if improved:
                        break
        return self.best()

    def tune(self) -> dict | None:
        say(f"model: {self.n_layers} layers, {self.pre[-1] / 2**30:.1f} GiB of layer tensors "
            f"({min(self.layer_bytes) / 2**30:.2f}-{max(self.layer_bytes) / 2**30:.2f} GiB a layer); goal: "
            f"{self.measure_tokens}-token prompts searched, profile {self.profile or 'none'}, max context "
            f"{self.max_context}")
        say("nodes:")
        for n in self.nodes:
            n.info = n.call("/info", timeout=30)
            if not n.info.get("ok"):
                sys.exit(f"stage_tune: {n.name} ({n.agent}) does not answer: {n.info.get('error')}")
            g = (n.info.get("gpus") or [{}])[0]
            say(f"  {n.name}: {g.get('name')} {g.get('vram_mib')} MiB ({g.get('free_mib')} free), PCIe gen "
                f"{g.get('pcie_gen')} x{g.get('pcie_width')}, RAM {n.info.get('ram', {}).get('total_mib')} MiB, "
                f"{n.info.get('cpus')} CPUs")
        for x in self.node_notes():
            say("  note: " + x)
        self.links()
        if self.search() is None:
            return None
        # the best few over the whole workload profile
        seen, cands = set(), []
        for r in sorted((x for x in self.runs if x["ok"] and x["kind"] == "search"), key=lambda x: x["request_s"]):
            k = (tuple(r["splits"]), r["chunk"], tuple(r["extra"]))
            if k not in seen:
                seen.add(k)
                cands.append(r)
        cands = cands[:self.top_n]
        if self.profile and [n for n, _ in self.profile] != [self.measure_tokens]:
            ranked = []
            for c in cands:
                p = self.measure(c["splits"], c["chunk"], c["extra"], [n for n, _ in self.profile], "profile")
                if p["ok"]:
                    ranked.append(p)
            ranked.sort(key=lambda x: x["request_s"])
        else:
            ranked = cands
        if not ranked:
            return None
        if not self.verify_max:
            return ranked[0]
        # the pick must read a prompt near the largest context it is configured for
        near = int(self.max_context * 0.95) - LONG_ANSWER
        for r in ranked:
            v = self.measure(r["splits"], r["chunk"], r["extra"], [near], "max")
            if v["ok"]:
                r["max_check"] = v["by_ctx"][str(near)]
                return r
            say(f"  does not read a {near}-token prompt: the next one")
        return None

    def links(self) -> None:
        """main -> node 1 -> node 2 ...: each hop's throughput (64 MiB through the agents) and round trip - on the
        stage link's address when the agent answers there (node.json agent_bind), else on the agent's own."""
        say("links:")
        prev = None
        for n in self.nodes:
            on_stage = n.stage_agent == n.agent or n.call("/ping", timeout=5, base=n.stage_agent).get("ok")
            base = n.stage_agent if on_stage else n.agent
            t = []
            for _ in range(5):
                t0 = time.perf_counter()
                n.call("/ping", timeout=30, base=base)
                t.append((time.perf_counter() - t0) * 1000)
            if prev is None:
                data = b"\0" * (64 << 20)
                t0 = time.perf_counter()
                r = n.call("/sink", data=data, timeout=300, base=base)
                mbs = len(data) / (time.perf_counter() - t0) / 1e6 if r.get("ok") else None
                frm = "main"
            else:
                r = prev.call("/send", {"url": base, "mib": 64}, timeout=300)
                mbs = r.get("mb_s")
                frm = prev.name
            where = "the stage link" if on_stage else (
                f"the agent's address - the stage link to {n.host} is not measured (node.json agent_bind)")
            say(f"  {frm} -> {n.name}: {mbs or 0:.0f} MB/s, round trip {min(t):.1f} ms ({where})")
            prev = n

    def write(self, best: dict) -> None:
        main_cfg = self.main.config(best["splits"], best["chunk"], self.nodes[0], False, self.max_context,
                                    best["extra"])
        (self.outdir / "main.json").write_text(json.dumps(main_cfg, indent=1), encoding="utf-8")
        ranges = self.stages(best["splits"])[1:]
        nodes = []
        for i, (n, (lo, hi)) in enumerate(zip(self.nodes, ranges)):
            nodes.append({"name": n.name, "agent": n.agent, **self.start_req(i, lo, hi, best["chunk"])})
        (self.outdir / "nodes.json").write_text(json.dumps(nodes, indent=1), encoding="utf-8")
        st = lambda r: " ".join(f"{a}-{b - 1}" for a, b in r["stages"])   # noqa: E731
        lines = ["# Layer-split tuning", "",
                 f"Model: {self.n_layers} layers, {self.pre[-1] / 2**30:.1f} GiB of layer tensors. Max context "
                 f"{self.max_context}. Searched with {self.measure_tokens}-token prompts and "
                 f"{self.answer_tokens}-token answers (score: a request's seconds).", "",
                 "## Search", "",
                 "| stages (layers) | chunk | prefill tok/s | decode tok/s | short decode | request s | per stage: "
                 "prompt chunk ms / window ms |", "|---|---|---|---|---|---|---|"]
        for r in (x for x in self.runs if x["kind"] == "search"):
            if not r["ok"]:
                lines.append(f"| {st(r)} | {r['chunk']} {' '.join(r['extra'])} | - | - | - | does not run | |")
                continue
            per = "; ".join(f"{s['pf_chunk_ms'] or 0:.0f} / {s['window_ms'] or 0:.1f}" for s in r["per_stage"])
            lines.append(f"| {st(r)} | {r['chunk']} {' '.join(r['extra'])} | {r['prefill_tps']:.0f} | "
                         f"{r['decode_tps']:.1f} | {r.get('short_decode') or 0:.1f} | {r['request_s']:.1f} | {per} |")
        prof = [x for x in self.runs if x["kind"] == "profile"]
        if prof:
            ctxs = [n for n, _ in self.profile]
            lines += ["", f"## Workload profile {self.profile}", "",
                      "| stages | chunk | " + " | ".join(f"{n // 1000}K prefill / decode" for n in ctxs)
                      + " | weighted request s |", "|---" * (len(ctxs) + 3) + "|"]
            for r in prof:
                cells = [f"{r['by_ctx'][str(n)]['prefill_tps']:.0f} / {r['by_ctx'][str(n)]['decode_tps']:.1f}"
                         if str(n) in r["by_ctx"] and r["by_ctx"][str(n)].get("prefill_tps") else "-" for n in ctxs]
                lines.append(f"| {st(r)} | {r['chunk']} | " + " | ".join(cells) + " | "
                             + (f"{r['request_s']:.1f} |" if r["ok"] else "does not run |"))
        mx = [x for x in self.runs if x["kind"] == "max"]
        if mx:
            lines += ["", f"## Near the max context ({self.max_context})", ""]
            for r in mx:
                c = next(iter(r["by_ctx"].values()), {})
                lines.append(f"- {st(r)}, chunk {r['chunk']}: " + (
                    f"{c.get('prompt_n')} tokens read at {c.get('prefill_tps'):.0f} tok/s, decode "
                    f"{c.get('decode_tps'):.1f}" if r["ok"] else f"does not run ({(r.get('error') or '')[:120]})"))
        notes = self.node_notes()
        if notes:
            lines += ["", "## Notes", ""] + [f"- {x}" for x in notes]
        lines += ["", f"**Pick**: stages {st(best)}, chunk {best['chunk']} {' '.join(best['extra'])}", "",
                  "Run it: `python3 tools/stage_tune.py apply cluster.json OUTDIR` starts the workers, then serve "
                  "`OUTDIR/main.json` (serve/server.py --config).", ""]
        (self.outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def apply(cluster: dict, outdir: Path) -> None:
    tf = cluster.get("token_file")
    token = (Path(tf).read_text().strip() if tf else os.environ.get("STRATA_STAGE_TOKEN", "")).strip()
    nodes = json.loads((outdir / "nodes.json").read_text(encoding="utf-8"))
    for spec in reversed(nodes):   # the last first: a relay connects to the next when it starts
        n = Node({"name": spec["name"], "agent": spec["agent"], "host": "-", "port": spec["port"]}, token)
        req = {k: spec[k] for k in ("begin", "end", "next", "port", "prefill", "cache", "extra", "model") if k in spec}
        r = n.call("/start", req)
        if not r.get("ok"):
            sys.exit(f"{n.name}: " + (r.get("error") or "") + "\n" + "\n".join(r.get("log", [])[-15:]))
        say(f"{n.name}: layers {spec['begin']}-{spec.get('end', 'last')} up in {r.get('seconds')} s"
            + (f"; clocks: {r['clocks']}" if r.get("clocks") else ""))
    say(f"the workers run; serve {outdir / 'main.json'} (serve/server.py --engine strata --config ...)")


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ("tune", "apply"):
        sys.exit(__doc__)
    cluster = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    outdir = Path(sys.argv[3])
    if sys.argv[1] == "apply":
        apply(cluster, outdir)
        return
    if os.name == "nt":   # it starts and stops serve/server.py and the engine with pgrep / kill
        sys.exit("stage_tune: tune runs on a Linux main PC (the nodes may run Windows); apply runs anywhere")
    t = Tuner(cluster, outdir)
    try:
        best = t.tune()
    finally:
        t.main.stop()
    if best is None:
        sys.exit("stage_tune: no configuration ran; see " + str(t.runs_path))
    t.write(best)
    say(f"pick: stages {t.stages(best['splits'])}, chunk {best['chunk']} {' '.join(best['extra'])}, "
        f"request {best['request_s']:.1f} s -> {outdir}/main.json, nodes.json, report.md")


if __name__ == "__main__":
    main()
