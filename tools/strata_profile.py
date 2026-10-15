"""Where the time goes on this PC: one command that runs a few typical requests through the real HTTP API with the
engine's timing switches on, reads the engine's own breakdowns from its log, and writes a report that says which part
of the machine each kind of request waits for - and which setting moves that part.

    python tools/strata_profile.py                       (the newest strata-*.json of this folder, or --config FILE)

It starts the server with your existing settings plus `STRATA_PREFILL_TIMING=1` and `STRATA_DECODE_TIMING=1` in the
engine's environment (docs/DETAILS.md, "Where a decode window's time goes"), on the config's port, with its log in a
temporary file, and sends:

  1. chat          three short questions, 128 new tokens each: a decode window at a short context
  2. 2K and 16K    fresh prompts of about 2,048 and 16,384 tokens: the prompt path's phases (the 16K read is skipped
                   when the context is smaller)
  3. long          a fresh prompt of about 32,768 tokens (--long N for another size, 0 for none), then 128 new tokens:
                   the decode at a long context, and how much of the KV cache the card reads from VRAM
  4. agent         a coding-agent session (a system prompt, four tools, then five turns that each add about 1,500
                   words - six requests, the same text as tools/strata_bench.py): per turn, what the conversation
                   cache reused, what was read, the decode window and the expert cache hit rate

For every request the report has the engine's figures: the prompt read's GPU time by phase, grouped (experts
streamed over PCIe, their dequant, the expert GEMMs, attention, GDN, the rest), and the decode window's milliseconds
split into the wait for the card, the experts computed on the CPU (the misses of the VRAM expert cache), the draft
and the adaptive tier, with the expert cache hit rate and the PCIe share beside them.  The findings at the end say in
words what each request waited for, with the number, and name the setting on that path (the ones tools/calibrate.py
measures where it does).  The tokens per second in the report were measured with the switches on (a few event marks
per prompt chunk, timestamps per window): quote speeds from tools/strata_bench.py, which runs without them.

The server is started from a temporary copy of the config: the engine's log goes to a temporary file (the user's log
does not get these requests), `--expert-profile-save` is dropped for the run (a few short prompts must not replace a
saved profile, as the bench does), and the Chat's shared settings file (named after the config) is not loaded.  The
copy is deleted when the run ends.  Nothing of the user's is written: the config is read, never changed.

`--url http://host:port` uses a server that is already running.  Its log is read only with `--engine-log FILE` (and
the breakdown lines are in it only when that server was started with the two variables set); without it the report
has the API's figures (tokens per second, reused tokens, the hit rate from /metrics) and no breakdown.
`--verify-profile` also sets `STRATA_VERIFY_PROFILE=1`: the GPU time of each decode stage (it times every stage with
events, so the decode is a little slower - compare the stages, do not quote the tok/s of that run; the Intel A770
build answers such requests with HTTP 400 at the time of writing, docs/A770_SPEED_SPEC.md).

The result goes to bench/results/<date>-profile-<gpu>-<model>/ (README.md and results.json, home folders and names
replaced as the benchmark does), or `--out DIR`.  Standard library only.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_bench as B                           # noqa: E402  the HTTP client, the prompt sizes, the machine facts, the redaction
from ab_engine import PROMPTS                      # noqa: E402

SCHEMA = 1
TIMING_ENV = {"STRATA_PREFILL_TIMING": "1", "STRATA_DECODE_TIMING": "1"}
VERIFY_PROFILE_ENV = {"STRATA_VERIFY_PROFILE": "1"}
CHAT_MAX_TOKENS = 128
READ_MAX_TOKENS = 24
LONG_MAX_TOKENS = 128
LONG_DEFAULT = 32768
AGENT_MAX_TOKENS = 64
LOG_WAIT_S = 3.0                                   # how long to wait for the engine's lines after an answer ended
LOG_SETTLE_S = 0.3                                 # and then for the log to stay still (the KV line has no fixed place)
KV_BLOCK_BYTES = 4224.0                            # a missed KV block read from RAM (generate.cpp: the KV streaming line)

# the prompt path's phases (src/prefill/prefill.cpp, kPfNames) grouped by what they wait for; a phase name the engine
# adds later lands in "other"
PROMPT_GROUPS = (
    ("experts over PCIe", ("wait copy", "gather", "host grouping")),
    ("dequant", ("dequant",)),
    ("expert GEMM", ("gemm gate/up", "gemm down", "combine")),
    ("attention", ("qsa proj", "qsa indexer", "qsa select", "qsa attn", "kv stage")),
    ("GDN", ("gdn", "gdn conv+gates", "gdn recurrence", "gdn out proj")),
)
OTHER = "other"
DECODE_GROUPS = ("GPU", "CPU experts", "draft", "adapt wait", "other")


# ----------------------------------------------------------------------------------------------- the engine's lines
_num = r"(-?\d+(?:\.\d+)?)"


def parse_prefill(line: str) -> dict | None:
    """`strata prefill timing: N tokens, GPU timeline X ms, wall Y ms, host staging Z ms: <phase> <ms> (<pct>%) ...`"""
    m = re.search(r"prefill timing: (\d+) tokens, GPU timeline " + _num + r" ms, wall " + _num + r" ms, host staging "
                  + _num + r" ms:(.*)$", line)
    if not m:
        return None
    phases = {}
    for name, ms, _pct in re.findall(r"\s*([a-z][a-z0-9+/ ]*?) " + _num + r" \(" + _num + r"%\)", m.group(5)):
        phases[name.strip()] = float(ms)
    return {"tokens": int(m.group(1)), "gpu_ms": float(m.group(2)), "wall_ms": float(m.group(3)),
            "host_staging_ms": float(m.group(4)), "phases": phases}


def parse_prefill_host(line: str) -> dict | None:
    """`strata prefill timing: host: chunk setup (...) A ms, waiting for each chunk B ms, after each chunk (...) C ms, PLE D ms`"""
    m = re.search(r"prefill timing: host: chunk setup .*? " + _num + r" ms, waiting for each chunk " + _num
                  + r" ms, after each chunk .*? " + _num + r" ms, PLE " + _num + r" ms", line)
    if not m:
        return None
    return {"chunk_setup_ms": float(m.group(1)), "wait_chunk_ms": float(m.group(2)), "after_chunk_ms": float(m.group(3)),
            "ple_ms": float(m.group(4))}


def parse_decode(line: str) -> dict | None:
    """`strata decode timing: W windows, avg T t, k tokens/window, M ms/window = verify V (GPU-reach wait G + per-layer host
    H [plan p actq a jobs j CPU c] + stage s) + commit/emit e + draft d; per layer-window: CPU experts x (y entries), VRAM
    hits v, PCIe q; adapt wait w join jn, n swaps/window`"""
    m = re.search(r"decode timing: (\d+) windows, avg T " + _num + r", " + _num + r" tokens/window, " + _num
                  + r" ms/window = verify " + _num + r" \(GPU-reach wait " + _num + r" \+ per-layer host " + _num
                  + r" \[plan " + _num + r" actq " + _num + r" jobs " + _num + r" CPU " + _num + r"\] \+ stage " + _num
                  + r"\) \+ commit/emit " + _num + r" \+ draft " + _num, line)
    if not m:
        return None
    f = [float(x) for x in m.groups()]
    d = {"windows": int(f[0]), "avg_T": f[1], "tokens_per_window": f[2], "ms_per_window": f[3], "verify": f[4], "gpu": f[5],
         "host": f[6], "host_plan": f[7], "host_actq": f[8], "host_jobs": f[9], "host_cpu": f[10], "stage": f[11],
         "commit": f[12], "draft": f[13]}
    m2 = re.search(r"CPU experts " + _num + r" \(" + _num + r" entries\), VRAM hits " + _num + r", PCIe " + _num, line)
    if m2:
        d.update(cpu_experts=float(m2.group(1)), cpu_entries=float(m2.group(2)), vram_hits=float(m2.group(3)), pcie=float(m2.group(4)))
    m3 = re.search(r"adapt wait " + _num + r" join " + _num + r", " + _num + r" swaps/window", line)
    if m3:
        d.update(adapt_wait=float(m3.group(1)), adapt_join=float(m3.group(2)), swaps=float(m3.group(3)))
    return d


def parse_gpu_stages(line: str) -> dict | None:
    """`strata decode GPU stages (ms/window): GDN layers: <name> <ms> ... | QSA layers: ... | total T ms/window over W windows`"""
    if "decode GPU stages (ms/window):" not in line:
        return None
    body = line.split("(ms/window):", 1)[1]
    out = {"gdn": {}, "qsa": {}, "total": None, "windows": None}
    for part in body.split("|"):
        part = part.strip()
        if part.startswith("GDN layers:") or part.startswith("QSA layers:"):
            key = "gdn" if part.startswith("GDN") else "qsa"
            for name, ms in re.findall(r"(\S+(?: \S+)*?) " + _num + r"(?= |$)", part.split(":", 1)[1]):
                out[key][name.strip()] = float(ms)
        else:
            m = re.search(r"total " + _num + r" ms/window over (\d+) windows", part)
            if m:
                out["total"], out["windows"] = float(m.group(1)), int(m.group(2))
    return out


def parse_summary(line: str) -> dict | None:
    """`strata serve: prompt N tokens = R reused + X read in A ms (B tok/s), G generated in C ms (D tok/s), drafts accepted
    a of b, c checkpoints`"""
    m = re.search(r"serve: prompt (\d+) tokens = (\d+) reused \+ (\d+)(?: of \d+)? read in " + _num + r" ms \(" + _num
                  + r" tok/s\), (\d+) generated in " + _num + r" ms \(" + _num + r" tok/s\), drafts accepted (\d+) of (\d+)", line)
    if not m:
        return None
    return {"prompt_tokens": int(m.group(1)), "reused": int(m.group(2)), "read": int(m.group(3)), "prompt_ms": float(m.group(4)),
            "prompt_tok_s": float(m.group(5)), "generated": int(m.group(6)), "decode_ms": float(m.group(7)),
            "decode_tok_s": float(m.group(8)), "drafts_accepted": int(m.group(9)), "drafts_offered": int(m.group(10))}


def parse_hit_rate(line: str) -> dict | None:
    """`strata serve: decode expert cache hit rate: P% (H hits / L lookups)[; O more read by the GPU over PCIe or from another
    GPU (Q% of all R routed)]`"""
    m = re.search(r"decode expert cache hit rate: " + _num + r"% \((\d+) hits / (\d+) lookups\)", line)
    if not m:
        return None
    d = {"hit_rate": round(float(m.group(1)) / 100.0, 4), "hits": int(m.group(2)), "lookups": int(m.group(3)), "pcie_share": 0.0,
         "offloaded": 0}
    m2 = re.search(r"; (\d+) more read by the GPU over PCIe or from another GPU \(" + _num + r"% of all (\d+) routed\)", line)
    if m2:
        d.update(offloaded=int(m2.group(1)), pcie_share=round(float(m2.group(2)) / 100.0, 4), routed=int(m2.group(3)))
    return d


def parse_kv(line: str) -> dict | None:
    """`strata serve: KV streaming: P% of N block reads hit VRAM, M MiB read from RAM`"""
    m = re.search(r"KV streaming: " + _num + r"% of (\d+) block reads hit VRAM, " + _num + r" MiB read from RAM", line)
    if not m:
        return None
    return {"kv_hit": round(float(m.group(1)) / 100.0, 4), "block_reads": int(m.group(2)), "ram_mib": float(m.group(3)),
            "misses": int(round(float(m.group(3)) * 1048576.0 / KV_BLOCK_BYTES))}


def kv_request(now: dict | None, prev: dict | None) -> dict | None:
    """The KV streaming line counts since the engine started; this request's own share is the difference from the
    line before it (None without both, or without a read in between)."""
    if not now or not prev:
        return None
    reads = now["block_reads"] - prev["block_reads"]
    misses = now["misses"] - prev["misses"]
    if reads <= 0 or misses < 0:
        return None
    return {"kv_hit": round(1.0 - misses / reads, 4), "block_reads": reads, "ram_mib": round(now["ram_mib"] - prev["ram_mib"], 1)}


PARSERS = (("prefill", parse_prefill), ("prefill_host", parse_prefill_host), ("decode", parse_decode),
           ("gpu_stages", parse_gpu_stages), ("summary", parse_summary), ("hit", parse_hit_rate), ("kv", parse_kv))


def engine_facts(lines) -> dict:
    """The engine's figures for one request from the log lines written during it: the last line of each kind (a
    request reads its prompt once and decodes once; the summary lines follow).  Keys absent when no line matched."""
    out: dict = {}
    for line in lines:
        for key, fn in PARSERS:
            d = fn(line)
            if d is not None:
                out[key] = d
                break
    return out


# ------------------------------------------------------------------------------------------------------ grouping
def prompt_groups(pf: dict | None) -> dict | None:
    """The prompt read's GPU time by group, in ms, and the share of the timeline each has (the phases not in any group
    are "other").  None without a prefill line."""
    if not pf or not pf.get("phases"):
        return None
    phases = dict(pf["phases"])
    ms = {}
    for name, members in PROMPT_GROUPS:
        ms[name] = sum(phases.pop(p, 0.0) for p in members)
    ms[OTHER] = sum(phases.values())
    total = pf.get("gpu_ms") or sum(ms.values()) or 1.0
    return {"ms": ms, "share": {k: v / total for k, v in ms.items()}, "gpu_ms": pf.get("gpu_ms"), "tokens": pf.get("tokens"),
            "top": sorted(pf["phases"].items(), key=lambda kv: -kv[1])[:5]}


def decode_groups(dc: dict | None) -> dict | None:
    """The decode window's milliseconds by what the host waited for: the card (GPU-reach wait), the experts the CPU
    pool computed (the per-layer host time: plan, activation quantization, jobs, CPU), the draft, the adaptive tier's
    wait, and the rest (commit, stage, what the line does not itemize).  None without a decode line."""
    if not dc:
        return None
    gpu, host, draft = dc.get("gpu", 0.0), dc.get("host", 0.0), dc.get("draft", 0.0)
    adapt = dc.get("adapt_wait", 0.0)
    total = dc.get("ms_per_window", 0.0)
    other = max(0.0, total - gpu - host - draft - adapt)
    ms = {"GPU": gpu, "CPU experts": host, "draft": draft, "adapt wait": adapt, "other": other}
    return {"ms": ms, "share": {k: v / total if total else 0.0 for k, v in ms.items()}, "ms_per_window": total,
            "tokens_per_window": dc.get("tokens_per_window"), "windows": dc.get("windows"), "cpu_entries": dc.get("cpu_entries"),
            "cpu_experts": dc.get("cpu_experts"),
            "vram_hits": dc.get("vram_hits"), "pcie": dc.get("pcie"), "swaps": dc.get("swaps")}


def merge_decode(decodes: list[dict]) -> dict | None:
    """Several requests' decode lines as one, weighted by their windows (the chat's three questions)."""
    decodes = [d for d in decodes if d and d.get("windows")]
    if not decodes:
        return None
    w = sum(d["windows"] for d in decodes)
    out = {"windows": w}
    for k in ("avg_T", "tokens_per_window", "ms_per_window", "verify", "gpu", "host", "host_plan", "host_actq", "host_jobs",
              "host_cpu", "stage", "commit", "draft", "cpu_experts", "cpu_entries", "vram_hits", "pcie", "adapt_wait",
              "adapt_join", "swaps"):
        vals = [(d.get(k), d["windows"]) for d in decodes if d.get(k) is not None]
        if vals:
            out[k] = sum(v * n for v, n in vals) / sum(n for _, n in vals)
    return out


def pct(x) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def ms(x, digits=1) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def num(x) -> str:
    return "-" if x is None else f"{x:,.0f}"


# ------------------------------------------------------------------------------------------------------ findings
def findings(results: dict) -> list[str]:
    """What each kind of request waited for, in words with the number, then - once - the setting that moves it.  Only
    what the lines say: a request without its engine lines gets no finding."""
    out, advice = [], []
    for label, r in results.get("prompts", {}).items():
        g = r.get("groups")
        if not g:
            continue
        s = g["share"]
        link = s["experts over PCIe"] + s["dequant"]
        compute = s["expert GEMM"] + s["attention"] + s["GDN"]
        tps = (r.get("summary") or {}).get("prompt_tok_s") or r.get("prompt_tps")
        head = f"The {label} read ({num(r.get('prompt_tokens') or g['tokens'])} tokens, {num(tps)} tok/s)"
        if link >= 0.4:
            out.append(f"{head} waits for the experts streamed over PCIe: {pct(s['experts over PCIe'])} of its GPU time "
                       f"waiting for their copies and {pct(s['dequant'])} dequantizing them.")
            if not advice or not advice[0].startswith("A short read"):
                advice.insert(0, "A short read that waits for expert copies is on the path the CPU prefill share works on "
                                 "(STRATA_PREFILL_CPU_SHARE, its range STRATA_PREFILL_CPU_SHARE_MAX; docs/DETAILS.md, \"Short "
                                 "prompts: the CPU shares the experts\"). Its range stops at 1,024 tokens by default and a wider one "
                                 "is not a sure win (2,048-token chunks measured 2% slower on one PC, docs/DETAILS.md): measure it "
                                 "here before setting it (tools/ab_engine.py A/Bs an environment variable).")
        elif compute >= 0.5:
            out.append(f"{head} is compute-bound: expert GEMM {pct(s['expert GEMM'])}, attention {pct(s['attention'])}, "
                       f"GDN {pct(s['GDN'])} of its GPU time; the experts' copies and dequant {pct(link)}.")
        else:
            out.append(f"{head}: experts' copies and dequant {pct(link)}, compute {pct(compute)}, other {pct(s['other'])} "
                       f"of its GPU time.")
    cpu_bound = False
    for label, r in results.get("decode", {}).items():
        g = r.get("groups")
        if not g:
            continue
        s, m = g["share"], g["ms"]
        hit = (r.get("hit") or {}).get("hit_rate")
        hit_txt = f", expert cache hit rate {pct(hit)}" if hit is not None else ""
        tps = (r.get("decode_tps") or {}).get("median")
        head = f"{label} ({num(r.get('context'))} tokens of context, {num(tps)} tok/s{hit_txt})"
        if m["CPU experts"] > m["GPU"]:
            cpu_bound = True
            out.append(f"{head}: of a {ms(g['ms_per_window'])} ms window the card takes {ms(m['GPU'])} ms and the experts "
                       f"computed on the CPU {ms(m['CPU experts'])} ms ({g.get('cpu_experts') or 0:.1f} experts, "
                       f"{g.get('cpu_entries') or 0:.1f} entries per layer-window): the misses of the VRAM expert cache take "
                       f"the larger share of the window.")
        else:
            out.append(f"{head}: of a {ms(g['ms_per_window'])} ms window the card takes {ms(m['GPU'])} ms, the experts "
                       f"computed on the CPU {ms(m['CPU experts'])} ms, the draft {ms(m['draft'])} ms: the card takes the "
                       f"larger share of the window.")
        if s["draft"] >= 0.15:
            out.append(f"{label}: the draft takes {pct(s['draft'])} of the window ({ms(g['tokens_per_window'], 2)} tokens accepted "
                       f"per window): --spec and the draft floor (--spec-min-p, setup --calibrate) are its settings.")
        kv = r.get("kv")
        if kv and kv["kv_hit"] < 0.9:
            out.append(f"{label}: {pct(kv['kv_hit'])} of the KV block reads hit VRAM ({num(kv['ram_mib'])} MiB read from RAM): "
                       f"--kv-resident sets how much of the KV cache stays on the card.")
    if cpu_bound:
        advice.append("A decode whose larger share is the experts computed on the CPU follows the expert cache's hit rate: "
                      "the settings on that path are the cache's slots (VRAM: --max-context, --kv-resident, --vram-reserve-mib; "
                      "docs/DETAILS.md, \"The expert cache size is a budget\"), the PCIe share and the adaptive tier (setup "
                      "--calibrate measures both on this PC) and a saved expert profile (expert_profile_save).")
    a = results.get("agent")
    if a and a.get("turns"):
        hits = [(t["turn"], t["hit"]["hit_rate"]) for t in a["turns"] if t.get("hit") and t["hit"].get("hit_rate") is not None]
        if len(hits) >= 2:
            lo = min(hits, key=lambda x: x[1])
            hi = max(hits, key=lambda x: x[1])
            out.append(f"Agent turns: the expert cache hit rate per turn runs from {pct(lo[1])} (turn {lo[0]}) to {pct(hi[1])} "
                       f"(turn {hi[0]}).")
        reads = [(t.get("summary") or {}).get("read") for t in a["turns"][1:] if t.get("summary")]
        if reads and all(x is not None for x in reads):
            out.append(f"Agent turns: each follow-up turn read {num(min(reads))}-{num(max(reads))} tokens after what the "
                       f"conversation cache reused.")
    return out + advice


# ----------------------------------------------------------------------------------------------------- the engine log
class EngineLog:
    """The engine's log file: the lines written since a mark.  None path: nothing is read."""

    def __init__(self, path: Path | None):
        self.path = path
        self.pos = self._size()

    def _size(self) -> int:
        try:
            return os.path.getsize(self.path) if self.path else 0
        except OSError:
            return 0

    def mark(self):
        self.pos = self._size()

    def since(self) -> list[str]:
        """The complete lines written since the mark (a line still being written waits for the next call)."""
        if not self.path:
            return []
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
        except OSError:
            return []
        cut = data.rfind(b"\n")
        if cut < 0:
            return []
        return data[:cut].decode("utf-8", "replace").splitlines()

    def facts(self, wait_for=("decode", "summary", "hit"), timeout: float = LOG_WAIT_S, settle: float = LOG_SETTLE_S) -> dict:
        """The request's facts once its lines are in the log: waits up to `timeout` for every kind in `wait_for`,
        then `settle` seconds of a still log (the KV streaming line comes after them), and takes what is there.  The
        lines read are consumed: a later line is the next request's."""
        t0 = time.time()
        lines: list[str] = []
        while self.path:
            lines = self.since()
            f = engine_facts(lines)
            if all(k in f for k in wait_for) or time.time() - t0 > timeout:
                break
            time.sleep(0.1)
        if self.path and settle > 0:
            while True:
                time.sleep(settle)
                more = self.since()
                if len(more) == len(lines) or time.time() - t0 > timeout + 5 * settle:
                    lines = more
                    break
                lines = more
        self.pos += sum(len(x.encode("utf-8", "replace")) + 1 for x in lines) if lines else 0
        return engine_facts(lines)


# ----------------------------------------------------------------------------------------------------- the run
class Profile:
    def __init__(self, base: str, key: str, ctx: int | None, log: EngineLog, say=print, long_tokens: int = LONG_DEFAULT):
        self.base, self.key, self.ctx, self.log, self.say = base, key, ctx, log, say
        self.long_tokens = long_tokens
        self.suite = B.Suite(base, key, ctx, 1, False, log=lambda *a: None)   # its prompt sizes, agent files and tags
        self.results: dict = {"chat": {}, "prompts": {}, "decode": {}, "agent": {}, "skipped": {}, "failures": []}
        self.kv_prev = None                                   # the last KV streaming line (its counts are cumulative)

    def request(self, messages, max_tokens, tools=None) -> tuple[dict, dict]:
        r = B.chat(self.base, self.key, messages, max_tokens, False, tools=tools)
        d = B.digest(r)
        d["text"] = r.get("text") or ""
        facts = self.log.facts() if d["ok"] else self.log.facts(wait_for=())
        if facts.get("kv"):
            facts["kv_request"] = kv_request(facts["kv"], self.kv_prev)
            self.kv_prev = facts["kv"]
        if not d["ok"]:
            self.results["failures"].append({"error": d.get("err") or "no token came back"})
            self.say(f"    FAILED: {d.get('err') or 'no token came back'}")
        if "hit" not in facts and d["ok"]:
            last = self.metrics_last()
            if last:
                facts["hit"] = {"hit_rate": last.get("hit_rate"), "pcie_share": last.get("pcie_share"), "source": "/metrics"}
        return d, facts

    def metrics_last(self) -> dict | None:
        m = B.try_json(self.base, "/metrics", self.key, 30) or {}
        reqs = m.get("requests") or []
        return reqs[0] if reqs and isinstance(reqs[0], dict) and reqs[0].get("hit_rate") is not None else None

    def warm_up(self):
        self.say("warming up (not measured) ...")
        self.suite.warm_up()
        f = self.log.facts(wait_for=())                       # the warm-up's lines are consumed; the KV counts start here
        self.kv_prev = f.get("kv") or self.kv_prev

    def fits(self, tokens: int, new: int) -> bool:
        return not self.ctx or tokens + new + 256 <= self.ctx

    def run_chat(self):
        self.say(f"chat: {len(PROMPTS)} short questions, {CHAT_MAX_TOKENS} new tokens each")
        rows, decodes = [], []
        for i, p in enumerate(PROMPTS):
            d, f = self.request([{"role": "user", "content": f"[{self.suite.tag(i, 'c')}] {p}"}], CHAT_MAX_TOKENS)
            if d["ok"]:
                rows.append({**d, "engine": f})
                decodes.append(f.get("decode"))
                self.say(f"  q{i + 1}: {d['decode_tps']} tok/s, {d['generated_tokens']} tokens" + self.decode_note(f))
        merged = merge_decode(decodes)
        self.results["chat"] = {"requests": rows}
        if rows:
            self.results["decode"]["chat"] = {
                "context": statistics.median(r["prompt_tokens"] or 0 for r in rows), "decode_tps": B.stat(r["decode_tps"] for r in rows),
                "groups": decode_groups(merged), "decode": merged, "hit": self.median_hit(rows),
                "summary": None, "gpu_stages": next((r["engine"].get("gpu_stages") for r in rows[::-1] if r["engine"].get("gpu_stages")), None)}

    @staticmethod
    def median_hit(rows) -> dict | None:
        hs = [r["engine"]["hit"] for r in rows if r.get("engine", {}).get("hit", {}).get("hit_rate") is not None]
        if not hs:
            return None
        return {"hit_rate": statistics.median(h["hit_rate"] for h in hs),
                "pcie_share": statistics.median(h.get("pcie_share") or 0.0 for h in hs)}

    @staticmethod
    def decode_note(f: dict) -> str:
        g = decode_groups(f.get("decode"))
        if not g:
            return ""
        h = f.get("hit", {}).get("hit_rate")
        return (f"; window {ms(g['ms_per_window'])} ms = GPU {ms(g['ms']['GPU'])} + CPU experts {ms(g['ms']['CPU experts'])} "
                f"+ draft {ms(g['ms']['draft'])}" + (f", hit rate {pct(h)}" if h is not None else ""))

    def run_read(self, tokens: int, label: str):
        if not self.fits(tokens, READ_MAX_TOKENS):
            self.results["skipped"][label] = f"the context is {self.ctx} tokens"
            self.say(f"{label} read: skipped (the context is {self.ctx} tokens)")
            return
        self.say(f"{label} read: a fresh prompt of about {tokens} tokens")
        d, f = self.request(self.suite.prompt_for(tokens, 0), READ_MAX_TOKENS)
        if not d["ok"]:
            return
        g = prompt_groups(f.get("prefill"))
        self.results["prompts"][label] = {**d, "engine": f, "groups": g, "summary": f.get("summary"), "prefill_host": f.get("prefill_host")}
        note = ""
        if g:
            note = "; GPU time: " + ", ".join(f"{k} {pct(v)}" for k, v in g["share"].items())
        self.say(f"  {d['prompt_tokens']} tokens, prompt {d['prompt_tps']} tok/s ({d['reused_tokens'] or 0} reused){note}")

    def run_long(self):
        n = self.long_tokens
        if n <= 0:
            return
        if not self.fits(n, LONG_MAX_TOKENS):
            self.results["skipped"]["long"] = f"the context is {self.ctx} tokens"
            self.say(f"long: skipped (the context is {self.ctx} tokens, the read would be {n})")
            return
        self.say(f"long: a fresh prompt of about {n} tokens, then {LONG_MAX_TOKENS} new tokens")
        d, f = self.request(self.suite.prompt_for(n, 1), LONG_MAX_TOKENS)
        if not d["ok"]:
            return
        g = prompt_groups(f.get("prefill"))
        label = f"{n // 1024}K (long)"
        self.results["prompts"][label] = {**d, "engine": f, "groups": g, "summary": f.get("summary"), "prefill_host": f.get("prefill_host")}
        self.results["decode"]["long context"] = {
            "context": d["prompt_tokens"], "decode_tps": B.stat([d["decode_tps"]]), "groups": decode_groups(f.get("decode")),
            "decode": f.get("decode"), "hit": f.get("hit"), "summary": f.get("summary"), "kv": f.get("kv_request"),
            "gpu_stages": f.get("gpu_stages")}
        self.say(f"  {d['prompt_tokens']} tokens, prompt {d['prompt_tps']} tok/s, decode {d['decode_tps']} tok/s" + self.decode_note(f)
                 + (f"; KV reads from VRAM {pct(f['kv_request']['kv_hit'])}" if f.get("kv_request") else ""))

    def run_agent(self):
        if self.ctx and self.ctx < B.AGENT_MIN_CONTEXT:
            self.results["skipped"]["agent"] = f"context {self.ctx} is below ~10K"
            self.say(f"agent: skipped: context {self.ctx} is below ~10K")
            return
        self.say(f"agent: a session of {B.AGENT_TURNS + 1} requests with four tools, {AGENT_MAX_TOKENS} new tokens per turn")
        files = self.suite.agent_files()
        msgs = [{"role": "system", "content": f"{B.SYSTEM}\n(session {self.suite.tag(0, 'a')})"},
                {"role": "user", "content": "I will show you files from a project one at a time. After each file, answer in "
                                            "one or two sentences what it is for. Do not call tools."}]
        turns = []
        for t in range(B.AGENT_TURNS + 1):
            if t > 0:
                rel, body = files[t - 1]
                msgs.append({"role": "user", "content": f"File {rel}:\n{body}\n\nWhat is this file for? Two sentences."})
            d, f = self.request(msgs, AGENT_MAX_TOKENS, tools=B.TOOLS)
            if not d["ok"]:
                break
            row = {**d, "turn": t, "engine": f, "groups": decode_groups(f.get("decode")), "hit": f.get("hit"), "summary": f.get("summary")}
            turns.append(row)
            self.say(f"  turn {t}: prompt {d['prompt_tokens']}, reused {d['reused_tokens']}, read {d['read_tokens']}, "
                     f"decode {d['decode_tps']} tok/s" + self.decode_note(f))
            msgs.append({"role": "assistant", "content": d.get("text") or "Understood."})
        self.results["agent"] = {"turns": turns}
        follow = [t for t in turns if t["turn"] > 0 and t["engine"].get("decode")]
        if follow:
            merged = merge_decode([t["engine"].get("decode") for t in follow])
            self.results["decode"]["agent turns"] = {
                "context": statistics.median(t["prompt_tokens"] or 0 for t in follow), "decode_tps": B.stat(t["decode_tps"] for t in follow),
                "groups": decode_groups(merged), "decode": merged, "hit": self.median_hit(follow), "summary": None,
                "gpu_stages": None}

    def run(self) -> dict:
        self.warm_up()
        self.run_chat()
        self.run_read(2048, "2K")
        self.run_read(16384, "16K")
        self.run_long()
        self.run_agent()
        self.results["findings"] = findings(self.results)
        return self.results


# ----------------------------------------------------------------------------------------------------- the report
def prompt_table(results: dict) -> list[str]:
    names = [n for n, _ in PROMPT_GROUPS] + [OTHER]
    lines = ["| Read | Tokens | Prompt tok/s | GPU ms | " + " | ".join(names) + " | Largest phases |",
             "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in names) + " | --- |"]
    for label, r in results.get("prompts", {}).items():
        g = r.get("groups")
        tps = (r.get("summary") or {}).get("prompt_tok_s") or r.get("prompt_tps")
        if g:
            top = ", ".join(f"{k} {v / (g['gpu_ms'] or 1) * 100:.0f}%" for k, v in g["top"][:3])
            lines.append(f"| {label} fresh | {num(r.get('prompt_tokens'))} | {num(tps)} | {num(g['gpu_ms'])} | "
                         + " | ".join(pct(g["share"][n]) for n in names) + f" | {top} |")
        else:
            lines.append(f"| {label} fresh | {num(r.get('prompt_tokens'))} | {num(tps)} | - | " + " | ".join("-" for _ in names)
                         + " | no prefill timing line |")
    return lines


def decode_table(results: dict) -> list[str]:
    lines = ["| Request | Context | Decode tok/s | Hit rate | PCIe share | ms/window | GPU | CPU experts | draft | adapt wait | "
             "other | tokens/window | CPU experts per layer-window | their entries |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, r in results.get("decode", {}).items():
        g, h = r.get("groups"), r.get("hit") or {}
        tps = r.get("decode_tps") or {}
        tps_txt = f"{tps['median']:.1f}" if tps else "-"
        if g:
            m = g["ms"]
            lines.append(f"| {label} | {num(r.get('context'))} | {tps_txt} | {pct(h.get('hit_rate'))} | {pct(h.get('pcie_share'))} | "
                         f"{ms(g['ms_per_window'])} | {ms(m['GPU'])} | {ms(m['CPU experts'])} | {ms(m['draft'])} | {ms(m['adapt wait'])} | "
                         f"{ms(m['other'])} | {ms(g['tokens_per_window'], 2)} | {ms(g.get('cpu_experts'), 1)} | {ms(g.get('cpu_entries'), 1)} |")
        else:
            lines.append(f"| {label} | {num(r.get('context'))} | {tps_txt} | {pct(h.get('hit_rate'))} | {pct(h.get('pcie_share'))} | "
                         "- | - | - | - | - | - | - | - | - |")
    return lines


def agent_table(results: dict) -> list[str]:
    lines = ["| Turn | Prompt tokens | Reused | Read | Prompt tok/s | Decode tok/s | Hit rate | ms/window | GPU | CPU experts |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for t in (results.get("agent") or {}).get("turns", []):
        g, h = t.get("groups"), t.get("hit") or {}
        lines.append(f"| {t['turn']} | {num(t.get('prompt_tokens'))} | {num(t.get('reused_tokens'))} | {num(t.get('read_tokens'))} | "
                     f"{num(t.get('prompt_tps'))} | {num(t.get('decode_tps'))} | {pct(h.get('hit_rate'))} | "
                     + (f"{ms(g['ms_per_window'])} | {ms(g['ms']['GPU'])} | {ms(g['ms']['CPU experts'])} |" if g else "- | - | - |"))
    return lines


def gpu_stage_lines(results: dict) -> list[str]:
    out = []
    for label, r in results.get("decode", {}).items():
        st = r.get("gpu_stages")
        if not st:
            continue
        items = sorted([(f"{part.upper()} {k}", v) for part in ("gdn", "qsa") for k, v in st[part].items()], key=lambda kv: -kv[1])[:8]
        out.append(f"- {label} (the last request's windows): total {ms(st.get('total'), 2)} ms/window; largest stages: "
                   + ", ".join(f"{k} {v:.2f}" for k, v in items))
    return out


def render(res: dict) -> str:
    mc, out = res["machine"], res["results"]
    gpu = " + ".join(mc.get("gpu") or []) or "CPU only"
    lines = [f"# Where the time goes: {gpu}, {mc.get('model') or '?'}, {res['date']}", "",
             f"Measured by `tools/strata_profile.py` through the HTTP API with `STRATA_PREFILL_TIMING=1` and "
             f"`STRATA_DECODE_TIMING=1` in the engine's environment; the figures are the engine's own lines.", "",
             f"- GPU: {gpu}" + (f", {mc['gpu_vram_mib'] / 1024:.0f} GiB VRAM" if mc.get("gpu_vram_mib") else "")
             + (f", driver {mc['gpu_driver']}" if mc.get("gpu_driver") else ""),
             f"- CPU: {mc.get('cpu') or '?'}" + (f", {mc['cpu_cores']} cores" if mc.get("cpu_cores") else "")
             + (f", {mc['ram_gib']:.0f} GiB RAM" if mc.get("ram_gib") else ""),
             f"- OS: {mc.get('os') or '?'}; Strata {mc.get('strata_version') or '?'}"
             + (f" ({mc['strata_commit']})" if mc.get("strata_commit") else "") + f", engine {mc.get('engine_version') or '?'}",
             f"- Model: {mc.get('model') or '?'}" + (f" ({mc['quant']})" if mc.get("quant") else "")
             + f", context {num(mc.get('context'))}",
             f"- Launch: `{mc['launch']}`" if mc.get("launch") else "- Launch: not read (a server that was already running)",
             ""]
    if res.get("params", {}).get("engine_log") is False:
        lines += ["No engine log was read (a server that was already running, no --engine-log): the breakdown columns are "
                  "empty, the API's figures are there.", ""]
    lines += ["## Prompt reads (the GPU timeline by phase)", "", *prompt_table(out), "",
              "\"experts over PCIe\" is the time the card waited for expert copies (wait copy, gather, host grouping); "
              "\"dequant\" their expansion on the card; \"expert GEMM\" gate/up, down and combine; \"attention\" the QSA "
              "projections, indexer, selection, attention and KV staging; \"GDN\" the linear-attention layers; \"other\" "
              "the hc read, router, PLE and embeddings.", "",
              "## Decode windows (ms per window)", "", *decode_table(out), "",
              "\"GPU\" is the host's wait for the card to reach the end of the window; \"CPU experts\" the experts the CPU "
              "pool computed because they were not in VRAM (plan, activation quantization, jobs and the compute); "
              "\"draft\" the draft layer; \"adapt wait\" the adaptive tier's swaps; \"other\" commit, stage and what the "
              "line does not itemize. \"Hit rate\" is the VRAM share of the experts looked up; \"PCIe share\" the share of "
              "all routed experts the card read over PCIe instead (not in the hit rate). The tokens per second were measured "
              "with the timing switches on: for speeds, run tools/strata_bench.py.", ""]
    kvs = [(label, r["kv"]) for label, r in out.get("decode", {}).items() if r.get("kv")]
    if kvs:
        lines += ["KV streaming (this request's block reads): " + "; ".join(
            f"{label} {pct(k['kv_hit'])} of {num(k['block_reads'])} reads from VRAM, {num(k['ram_mib'])} MiB from RAM" for label, k in kvs), ""]
    st = gpu_stage_lines(out)
    if st:
        lines += ["GPU stages (`STRATA_VERIFY_PROFILE=1`; the profile itself slows the decode a little):", "", *st, ""]
    if (out.get("agent") or {}).get("turns"):
        lines += ["## Agent session, turn by turn", "", *agent_table(out), ""]
    lines += ["## Findings", ""]
    lines += [f"- {f}" for f in out.get("findings", [])] or ["- nothing to say: no engine line matched"]
    if out.get("skipped"):
        lines += ["", "Skipped: " + "; ".join(f"{k} ({v})" for k, v in out["skipped"].items())]
    if out.get("failures"):
        lines += ["", f"{len(out['failures'])} request(s) failed: " + "; ".join(str(f.get('error')) for f in out["failures"][:3])]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------------------- server
def profile_config(cfg: dict, log_path: Path, verify_profile: bool) -> dict:
    """The config the profile starts the server with: the user's settings, the timing switches in the engine's
    environment, the engine's log in `log_path`, and no --expert-profile-save (the bench does the same: a few short
    prompts must not overwrite a saved profile).  The user's file is not touched."""
    out, _ = B.without_profile_save(cfg)
    out = dict(out)
    env = dict(out.get("env") or {})
    env.update(TIMING_ENV)
    if verify_profile:
        env.update(VERIFY_PROFILE_ENV)
    out["env"] = env
    out["log"] = str(log_path)
    return out


def start_server(cfg: dict, port: int, gpu: str | None, server_log: Path, engine_log: Path, verify_profile: bool):
    """The server from a temporary copy of the config (profile_config), readable by this user only: it holds the
    config's API key.  The caller deletes it (remove_config) when the run ends."""
    import subprocess
    tmp = server_log.with_suffix(".config.json")
    tmp.touch(mode=0o600)
    tmp.write_text(json.dumps(profile_config(cfg, engine_log, verify_profile)), encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(tmp), "--port", str(port)]
    if gpu:
        cmd += ["--gpu", gpu]
    logf = open(server_log, "w", encoding="utf-8", errors="replace")
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)


def remove_config(server_log: Path | None):
    if server_log is None:
        return
    try:
        server_log.with_suffix(".config.json").unlink()
    except OSError:
        pass


def results_dir_name(machine: dict, date: str) -> str:
    return f"{date}-profile-{B.gpu_slug(machine)}-{B.slug(B.model_label(machine))}"


def write_results(res: dict, parent: Path, red) -> Path:
    d = B.unique_dir(parent, results_dir_name(res["machine"], res["date"]))
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(B.redact_obj(res, red), indent=1, ensure_ascii=False), encoding="utf-8")
    (d / "README.md").write_text(red(render(res)), encoding="utf-8")
    return d


# --------------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Where the time goes on this PC (see the top of this file).")
    ap.add_argument("--config", help="the model config to run (default: the newest strata-*.json of this folder)")
    ap.add_argument("--url", help="use the server already running at this address instead of starting one")
    ap.add_argument("--engine-log", help="with --url: that server's engine log (the breakdown lines, when it runs with the timing switches)")
    ap.add_argument("--port", type=int, help="the port to start the server on (default: the config's)")
    ap.add_argument("--gpu", help="start the server on this GPU only (as the server's --gpu)")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_KEY", ""), help="the server's API key, if it has one (or STRATA_KEY)")
    ap.add_argument("--long", type=int, default=LONG_DEFAULT, metavar="N", help=f"the long prompt's size in tokens (default {LONG_DEFAULT}; 0: none)")
    ap.add_argument("--verify-profile", action="store_true", help="also STRATA_VERIFY_PROFILE=1: the GPU time of each decode stage (slows the decode a little)")
    ap.add_argument("--out", help="where to write the results folder (default: bench/results)")
    ap.add_argument("--ready-timeout", type=float, default=1800, help="seconds to wait for the model to load")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    say = (lambda *x, **k: None) if a.quiet else (lambda *x, **k: print(*x, **k, flush=True))

    cfg_path, cfg, proc = None, None, None
    if a.config or not a.url:
        if a.config:
            cfg_path = Path(a.config)
            if not cfg_path.is_absolute() and not cfg_path.exists():
                cfg_path = ROOT / a.config
        else:
            found = B.find_configs()
            if found:
                cfg_path = found[0]
                if len(found) > 1:
                    say(f"Several models are installed; using {cfg_path.name} (the one used last). Pick another with --config FILE.")
        if cfg_path is None or not cfg_path.is_file():
            print("No Strata model is set up in this folder: run START-HERE.bat (Windows) or ./setup.sh (Linux) first, "
                  "or give --url of a running server.", file=sys.stderr)
            return 2
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    key = a.api_key or ((cfg or {}).get("api_key") or "")
    base = a.url.rstrip("/") if a.url else f"http://127.0.0.1:{a.port or int(cfg.get('port', 8080))}"
    red = B.redactor(key=key)
    started, engine_log, server_log = False, None, None
    try:
        if B.is_strata(base, key):
            say(f"Using the Strata server already running at {base}.")
            if a.engine_log:
                engine_log = Path(a.engine_log)
            else:
                say("Its engine log is not read (no --engine-log): the report has the API's figures and no breakdown.")
        elif a.url:
            print(f"No Strata server answers at {base}.", file=sys.stderr)
            return 2
        else:
            port = int(base.rsplit(":", 1)[1])
            if B.try_json(base, "/health", key, 3) is not None:
                print(f"Port {port} is used by something that is not Strata: pass --port N.", file=sys.stderr)
                return 2
            server_log = Path(tempfile.gettempdir()) / f"strata-profile-{os.getpid()}.log"
            engine_log = Path(tempfile.gettempdir()) / f"strata-profile-{os.getpid()}.engine.log"
            say(f"Starting {cfg_path.name} on port {port} with the timing switches (loading takes 1-3 minutes) ...")
            proc = start_server(cfg, port, a.gpu, server_log, engine_log, a.verify_profile)
            started = True
            if not B.wait_ready(base, key, proc, a.ready_timeout, say):
                tail = []
                for p in (server_log, engine_log):
                    if p.exists():
                        tail += [f"--- {p.name}"] + p.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
                print("The server did not become ready. Last log lines:\n" + red("\n".join(tail)), file=sys.stderr)
                return 1
        t0 = time.time()
        machine = B.collect_machine(base, key, cfg, a.gpu)
        prof = Profile(base, key, machine.get("context"), EngineLog(engine_log), say=say, long_tokens=a.long)
        say("Running the requests (about 2-5 minutes) ...")
        out = prof.run()
        res = {"schema": SCHEMA, "tool": "tools/strata_profile.py", "date": datetime.date.today().isoformat(),
               "params": {"long": a.long, "verify_profile": a.verify_profile, "started_server": started, "engine_log": engine_log is not None},
               "elapsed_s": time.time() - t0, "machine": machine, "results": out}
        d = write_results(res, Path(a.out) if a.out else ROOT / "bench" / "results", red)
        say("")
        say(red(render(res)))
        try:
            shown = d.relative_to(ROOT)
        except ValueError:
            shown = d
        say(f"Report: {shown / 'README.md'} (home folders, user and host names and IP addresses are replaced; read it once before you share it)")
        return 1 if out["failures"] and not out["decode"] and not out["prompts"] else 0
    finally:
        if started:
            B.stop_server(proc)
        remove_config(server_log)


if __name__ == "__main__":
    sys.exit(main())
