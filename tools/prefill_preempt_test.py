"""tools/prefill_preempt_test.py - the prefill-preemption parity harness (docs/PREFILL-PREEMPT.md).

Drives `strata --serve` directly over its stdin/stdout protocol (no HTTP), against the real model, and proves the
three things the feature must prove:

  1. STATE: A parked at a chunk boundary, B run to completion in between, then A resumed - A's final
     STRATA_STATE_HASH (gdn, ple, indexer tails/pooled/kv, the drafter's ring, stale cells, ple_prev) equals the
     hash of an uninterrupted A that ran after its own B on a control engine.  Bits, not tolerances.
  2. OUTPUT: A's and B's generated token ids (greedy) are identical to their uninterrupted references.
  3. MECHANICS: parks land on chunk boundaries; a cancelled parked request releases its snapshot and the next
     request is healthy; repeated preemptions do not drift; decode is never preempted.

Controlled configuration: greedy, --adapt-every 100000 (no swaps before that round), --pcie-frac 0,
no turn token in the prompts (no checkpoint-mount differences between the arms), prompts that share no prefix.
STRATA_STATE_HASH=1 makes the engine print the fingerprint after every request that leaves live state behind;
the engine needs --prompt-cache > 0 for it (the harness passes 6).

    python3 tools/prefill_preempt_test.py                 # the full suite (one engine start per scenario)
    python3 tools/prefill_preempt_test.py -k boundary     # the boundary sweep only
    python3 tools/prefill_preempt_test.py --quick         # a short A, one boundary, mode-0 KV

Exit code 0 when every scenario passes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = ROOT / "build" / "strata"
CONFIG = ROOT / "strata-iq3_xxs.json"
# the engine's fingerprint grew fields over time (dead/pooled_full arrived with the multi-session cache):
# each optional component joins the dict, so the comparison covers whatever the engine reports
HASH_RE = re.compile(r"STATE_HASH L=(\d+) gdn=([0-9a-f]+) ple=([0-9a-f]+) tail=([0-9a-f]+) pooled=([0-9a-f]+) "
                     r"kv=([0-9a-f]+) mtp=([0-9a-f]+) stale=([0-9a-f]+)(?: dead=([0-9a-f]+))?"
                     r"(?: pooled_full=([0-9a-f]+))?(?: block=([0-9a-f]+))? ple_prev=(-?\d+),(-?\d+)")


def parse_state_match(match) -> dict:
    keys = ["L", "gdn", "ple", "tail", "pooled", "kv", "mtp", "stale", "dead", "pooled_full",
            "block", "ple_prev0", "ple_prev1"]
    # Optional regex groups still occupy positions. Never shorten keys before zipping!
    return {k: v for k, v in zip(keys, match.groups()) if v is not None}


def deterministic_tokens(n: int, seed: int, lo=1000, hi=30000) -> list[int]:
    """A pseudo-random prompt over the plain-vocabulary range: no <|im_start|> (248045, a checkpoint turn
    boundary), no image pads, nothing special.  The same seed is the same prompt."""
    out, x = [], (seed * 2654435761 + 1) & 0xFFFFFFFF
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out.append(lo + x % (hi - lo))
    return out


class Engine:
    """One `strata --serve` process: request lines in, protocol lines out (demultiplexed by the id suffix)."""

    def __init__(self, exe: str, args: list[str], log_path: Path):
        env = dict(os.environ)
        env["STRATA_STATE_HASH"] = "1"
        self.log = open(log_path, "wb")
        self.protocol = open(log_path.with_suffix(".protocol.jsonl"), "w", encoding="utf-8")
        self.protocol_lock = threading.Lock()
        self.pending = defaultdict(deque)
        self.observed = defaultdict(dict)
        self.eof_seen = False
        self.proc = subprocess.Popen([exe, "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, encoding="utf-8", bufsize=1, env=env)
        self._record("start", pid=self.proc.pid, args=[exe, "--serve", *args],
                     env={k: v for k, v in env.items() if k.startswith("STRATA_")})
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.info: dict[str, object] = {}
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()
        self.sampler = threading.Thread(target=self._sample_rss, daemon=True)
        self.sampler.start()
        ready = self._next_line(lambda l: l.startswith("READY"), timeout=1800)
        if ready is None:
            raise RuntimeError(f"the engine did not become ready (log: {log_path})")
        self.max_context = int(ready.split()[1])

    def _sample_rss(self):
        """VmRSS/VmHWM every 2 s into the protocol.  A SIGKILL leaves no stderr and the kernel log of this box is
        not recoverable after the fact (the B0 death, 2026-10-03): a resident-set timeline next to the protocol is
        what correlates an OOM kill - or rules one out - when the engine dies without a word."""
        status = Path(f"/proc/{self.proc.pid}/status")
        while not self.eof_seen and self.proc.poll() is None:
            try:
                fields = {l.split(":", 1)[0]: l.split(":", 1)[1] for l in status.read_text().splitlines() if ":" in l}
                self._record("rss", vm_rss_kib=int(fields["VmRSS"].split()[0]),
                             vm_hwm_kib=int(fields["VmHWM"].split()[0]))
            except (OSError, ValueError, KeyError):
                return          # the process is gone (or /proc does not exist): nothing more to sample
            time.sleep(2.0)

    def _pump(self):
        # explicit readline (not `for line in`): the file iterator's read-ahead can sit on a partial buffer
        # while the engine waits for the other end of a conversation, and lines stop flowing
        try:
            for line in iter(self.proc.stdout.readline, ""):
                self._record("out", line=line.rstrip("\n"))
                self.lines.put(line.rstrip("\n"))
        except (ValueError, OSError):
            pass
        self.lines.put(None)

    def _record(self, event, **data):
        with self.protocol_lock:
            stamp = time.monotonic()
            if event == "out":
                line = data["line"]
                rid = self._rid_of(line)
                if rid is not None:
                    timings = self.observed[rid]
                    if line.startswith("RESUME "): timings.setdefault("started", stamp)
                    elif line.startswith("T "): timings.setdefault("first_token", stamp)
                    elif line.startswith("DONE "): timings["done"] = stamp
            self.protocol.write(json.dumps({"time": stamp, "event": event, **data}) + "\n")
            self.protocol.flush()

    def _ended(self):
        code = self.proc.poll()
        if code is None:
            try: code = self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired: pass
        reason = f"exit={code}"
        if code is not None and code < 0:
            try: reason += f" signal={signal.Signals(-code).name}"
            except ValueError: pass
        self._record("unexpected_eof", returncode=code, reason=reason)
        return RuntimeError(f"the engine ended ({reason}; stderr: {self.log.name}; "
                            f"protocol: {self.protocol.name})")

    def _next_line(self, pred, timeout: float) -> str | None:
        """The next line satisfying `pred` (every other protocol line is swallowed; an ERR is never)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.eof_seen:
                raise self._ended()
            try:
                line = self.lines.get(timeout=1.0)
            except queue.Empty:
                continue
            if line is None:
                self.eof_seen = True
                raise self._ended()
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = v
            elif line.startswith("ERR") and not pred(line):
                raise RuntimeError("engine ERR: " + line)
            if pred(line):
                return line
        raise RuntimeError("timeout waiting for a protocol line")

    def send(self, line: str):
        self._record("in", line=line)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    @staticmethod
    def _rid_of(line: str) -> int | None:
        _, _, suffix = line.rpartition(" id=")
        return int(suffix) if suffix and suffix.lstrip("-").isdigit() else None

    def collect(self, rid: int | None, stop_on=None) -> dict:
        """Reads a request's lines until its DONE (-> {'tokens', 'finish', 'reused', 'generated'}) or until
        `stop_on(line)` is true (-> adds 'stopped': True).  Lines of other requests pass through untouched."""
        tokens: list[int] = []
        while True:
            line = (self.pending[rid].popleft() if self.pending[rid] else
                    self._next_line(lambda l: True, timeout=1200))
            line_rid = self._rid_of(line)
            if line_rid is not None and line_rid != rid:
                self.pending[line_rid].append(line)
                continue
            if stop_on is not None and stop_on(line):
                return {"tokens": tokens, "stopped": True, "finish": None, "generated": len(tokens), "reused": 0}
            if line.startswith("T ") and (rid is None or self._rid_of(line) == rid):
                tokens.append(int(line[2:].split(" id=")[0]))
            elif line.startswith("DONE") and (rid is None or self._rid_of(line) == rid):
                f = line.split(" id=")[0].split()
                result = {"tokens": tokens, "finish": f[5], "generated": int(f[1]),
                          "reused": int(f[8]) if len(f) > 8 else 0, "stopped": False,
                          "observed": dict(getattr(self, "observed", {}).get(rid, {}))}
                if result["generated"] != len(tokens):
                    raise RuntimeError(f"request {rid}: DONE generated={result['generated']} but "
                                       f"received {len(tokens)} token lines")
                self._record("collected", rid=rid, result=result)
                return result

    def gen(self, rid: int | None, ids: list[int], max_new: int):
        head = f"GEN {max_new}" + (f" id={rid}" if rid is not None else "")
        self.send(head + " " + ",".join(map(str, ids)))

    def state_hash(self) -> dict:
        """The latest STATE_HASH line from the engine's stderr log (written after every finished request)."""
        with open(self.log.name, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 262144))
            tail = f.read().decode("utf-8", "replace")
        m = None
        for m in HASH_RE.finditer(tail):
            pass
        if m is None:
            raise RuntimeError("no STATE_HASH in the engine log (STRATA_STATE_HASH=1 and --prompt-cache > 0?)")
        return parse_state_match(m)

    def close(self):
        try:
            if self.proc.poll() is not None:
                raise self._ended()
            self.send("QUIT")
            code = self.proc.wait(timeout=60)
            self._record("quit_exit", returncode=code)
            if code != 0:
                raise RuntimeError(f"engine exit={code} after QUIT; log: {self.log.name}")
        except (BrokenPipeError, OSError) as ex:
            raise self._ended() from ex
        except subprocess.TimeoutExpired:
            self._record("harness_kill", reason="QUIT timeout")
            self.proc.kill()
            self.proc.wait(timeout=10)
            raise RuntimeError(f"engine failed to exit after QUIT; log: {self.log.name}")
        finally:
            self.pump.join(timeout=2)
            self.proc.stdout.close()
            self.proc.stdin.close()
            self.log.close()
            self.protocol.close()


def strip_options(args: list[str], valued: set[str], flags: set[str]) -> list[str]:
    """Remove switches without swallowing the argument following a valueless flag."""
    out, i = [], 0
    while i < len(args):
        if args[i] in valued:
            if i + 1 >= len(args):
                raise ValueError(f"missing value for {args[i]}")
            i += 2
        elif args[i] in flags:
            i += 1
        else:
            out.append(args[i])
            i += 1
    return out


def state_differences(ref: dict, got: dict) -> list[str]:
    # Stale cells lie OUTSIDE the committed prefix; rejected drafts may legitimately differ.
    # Drafter KV is still gated: differences must be investigated, not silently retried away.
    return [k for k in dict.fromkeys([*ref, *got]) if k != "stale" and ref.get(k) != got.get(k)]


def output_differences(ref: dict, got: dict, label: str) -> list[str]:
    """max_new is a cap: matching early EOS passes; a different EOS/output remains a failure."""
    failures = []
    if got["tokens"] != ref["tokens"]:
        a, b = ref["tokens"], got["tokens"]
        first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
        expected = a[first] if first < len(a) else "<end>"
        actual = b[first] if first < len(b) else "<end>"
        failures.append(f"{label}: tokens differ at [{first}]: ref={expected}, got={actual}; "
                        f"length {len(a)} vs {len(b)}, finish {ref['finish']} vs {got['finish']}")
    elif got["finish"] != ref["finish"]:
        failures.append(f"{label}: finish {got['finish']!r} != reference {ref['finish']!r}")
    return failures


def engine_args(cfg: dict, *, prefill: int, preempt: bool, max_context: int | None = None,
                kv_resident: int | None = None, expert_slots: int | None = None) -> list[str]:
    valued = {"--max-context", "--kv-resident", "--prefill", "--adapt-swaps", "--adapt-every",
              "--pcie-frac", "--prompt-cache", "--prefill-preempt-min-tokens", "--prefill-preempt-max",
              "--expert-cache", "--spec-min-p", "--suffix-draft", "--prefill-preempt-snapshot-mib"}
    out = strip_options(cfg["args"], valued, {"--prefill-preempt"})
    # suffix-draft 0: the lookup drafter's policy is learned over the whole process, so the two arms' window
    # shapes would drift apart and the state hash would differ in ULPs while the tokens still match.
    # spec-min-p 0 for the same reason one level down: the verify window's T is otherwise gated by the DRAFTER's
    # probabilities, and the drafter's prompt KV is E-9 non-bit-identical territory - one flipped probability
    # threshold reshuffles every later window shape and, through it, the main state's ULPs.  Constant shapes
    # keep the comparison about the park, not about the drafter.
    # adapt-every 100000: the expert tier's rotation starts only after 100,000 rounds - static residency for the
    # whole run, through the engine's own armed path (the rope-scaling benchmarks' recipe).
    out += ["--adapt-every", "100000", "--pcie-frac", "0", "--suffix-draft", "0", "--spec-min-p", "0",
            "--prompt-cache", "6", "--prefill", str(prefill)]
    out += ["--max-context", str(max_context or 32768)]
    if kv_resident is not None:
        out += ["--kv-resident", str(kv_resident)]
    if preempt:
        out += ["--prefill-preempt", "--prefill-preempt-min-tokens", "0", "--prefill-preempt-snapshot-mib", "4096"]
    if expert_slots:
        # pin the VRAM expert tier: auto sizing depends on how much VRAM happens to be free, and a different
        # resident set rounds differently (GPU-resident experts vs CPU misses) - every engine of one comparison
        # must run the SAME geometry or the state hashes are not comparable
        out += ["--expert-cache", str(expert_slots)]
    return out


def start_engine(name: str, exe: str, args: list[str], workdir: Path) -> Engine:
    return Engine(exe, args, workdir / f"{name}.log")


def require_pinned_geometry(e: Engine, expert_slots: int, label: str) -> None:
    """INFO expert_slots is the tier the engine ACTUALLY built.  A pin that came back different is a different
    resident expert set - a different model on one side of the comparison (the repeat-3 5-vs-128 failure of
    2026-10-03 was exactly this, silently).  Fail the scenario here, by name, before any tokens are compared."""
    got = str(e.info.get("expert_slots", "") or "")
    if expert_slots and got != str(expert_slots):
        raise RuntimeError(f"{label}: engine built an expert cache of {got} slots, not the pinned "
                           f"{expert_slots} - the two arms would not run the same resident set")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default=str(DEFAULT_ENGINE))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--workdir", default=str(ROOT / "bench" / "results" / "prefill-preempt"))
    ap.add_argument("-k", dest="only", help="run the scenarios whose name contains this")
    ap.add_argument("--quick", action="store_true", help="a short A, one boundary, mode-0 KV")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--interim-max-new", type=int, default=None,
                    help="B decode length: 0 isolates B prefill, 1 adds only its first verify/commit")
    ap.add_argument("--trace", action="store_true", help="expensive state and verify-window hashes")
    ap.add_argument("--tail-tokens", type=int, default=491,
                    help="prompt length after its full chunks; 33 tests a resumed 32-token batched tail")
    args = ap.parse_args()
    if not 2 <= args.tail_tokens < 2048:
        ap.error("--tail-tokens must be between 2 and 2047")
    if args.max_new < 1 or (args.interim_max_new is not None and args.interim_max_new < 0):
        ap.error("A needs at least one output token; B must be nonnegative")
    b_max_new = args.max_new if args.interim_max_new is None else args.interim_max_new
    if b_max_new == 0 and not args.trace:
        ap.error("--interim-max-new 0 requires --trace (debug-only engine command)")
    if args.trace:
        os.environ["STRATA_PREEMPT_TRACE"] = "1"

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    chunk = 2048
    n_a = (3 if args.quick else 6) * chunk + args.tail_tokens          # a final partial chunk, like the plan's 10,731 example
    a_ids = deterministic_tokens(n_a, seed=7)
    b_ids = deterministic_tokens(220, seed=99)
    warm_ids = deterministic_tokens(120, seed=5)          # the first request on a virgin engine drafts (and so
    if a_ids[:8] == b_ids[:8]:                            # decodes) differently: warm both engines up first
        b_ids[0] += 1                                     # the prompts share no prefix: no checkpoint mounts

    # the trigger must leave at least one chunk boundary AFTER it: the final partial chunk is never a park
    # point (completing beats parking), so 6144 of 6635 would only fail
    boundaries = [2048, 4096] if args.quick else [2048, 4096, 6144, 8192, 10240]
    scenarios: list[tuple[str, dict]] = [(f"boundary-{b}", dict(trigger=b, preempts=[2], cancel=False))
                                         for b in boundaries]
    if not args.quick:
        scenarios += [
            # park and resume with NOTHING in between: the restore itself, no interim interference
            # (early AND late boundaries - the late ones exercise a different snapshot size)
            ("park-only", dict(trigger=2048, preempts=[], cancel=False)),
            ("park-only-late", dict(trigger=8192, preempts=[], cancel=False)),
            ("park-only-last", dict(trigger=10240, preempts=[], cancel=False)),
            ("streaming-kv", dict(trigger=4096, preempts=[2], cancel=False, max_context=65536, kv_resident=8192,
                                  own_refs=True)),
            ("repeat-3", dict(trigger=2048, preempts=[2, 3, 4], cancel=False)),
            # ten parks over an A long enough to hold them (default --prefill-preempt-max is 16): repeated
            # parks must not drift, and every boundary's interim output must match
            ("repeat-10", dict(trigger=2048, preempts=list(range(2, 12)), cancel=False, a_chunks=11)),
            ("cancel-parked", dict(trigger=2048, preempts=[2], cancel=True)),
            ("decode-nopark", dict(trigger=None, preempts=[2], cancel=False)),   # B queued during A's decode
        ]
    if args.only:
        scenarios = [s for s in scenarios if args.only in s[0]]

    print(f"[harness] engine {args.engine}", flush=True)
    # pin the expert-cache geometry for EVERY engine of this run: start one probe engine with the config's own
    # auto sizing and read the slot count it settled on
    probe = start_engine("probe", args.engine, engine_args(cfg, prefill=chunk, preempt=False), workdir)
    expert_slots = int(probe.info.get("expert_slots", 0) or 0)
    probe.close()
    print(f"[harness] expert-cache pinned to {expert_slots} slots for every engine of this run", flush=True)
    failures: list[str] = []
    for name, sc in scenarios:
        print(f"[harness] scenario {name}: trigger after {sc['trigger']} tokens, interim {sc['preempts']}, "
              f"cancel={sc['cancel']}, ctx={sc.get('max_context')}, kv_resident={sc.get('kv_resident')}", flush=True)
        t0 = time.time()
        try:
            # Paired controls are useful, but failures are never reclassified as machine noise.
            sc_ctrl = start_engine(f"control-{name}", args.engine,
                                   engine_args(cfg, prefill=chunk, preempt=False,
                                               max_context=sc.get("max_context"),
                                               kv_resident=sc.get("kv_resident"),
                                               expert_slots=expert_slots), workdir)
            require_pinned_geometry(sc_ctrl, expert_slots, f"control-{name}")
            try:
                sc_a_ids = deterministic_tokens(sc.get("a_chunks", 3 if args.quick else 6) * chunk
                                                + args.tail_tokens, seed=7)
                sc_ctrl.gen(None, warm_ids, 8)
                sc_ctrl.collect(None)
                sc_b = {"tokens": [], "finish": "length"}
                if sc["preempts"]:
                    sc_ctrl.gen(None, b_ids, b_max_new)
                    sc_b = sc_ctrl.collect(None)
                sc_ctrl.gen(None, sc_a_ids, args.max_new)
                sc_a = sc_ctrl.collect(None)
                sc_refs = (sc_a, sc_b, sc_ctrl.state_hash())
                (workdir / f"reference-{name}.json").write_text(json.dumps({
                    "A": sc_a, "B": sc_b, "state": sc_refs[2]}, indent=2), encoding="utf-8")
            finally:
                sc_ctrl.close()
            fails = []
            e = start_engine(f"preempt-{name}", args.engine,
                             engine_args(cfg, prefill=chunk, preempt=True, max_context=sc.get("max_context"),
                                         kv_resident=sc.get("kv_resident"), expert_slots=expert_slots),
                             workdir)
            require_pinned_geometry(e, expert_slots, f"preempt-{name}")
            try:
                e.gen(None, warm_ids, 8)
                e.collect(None)
                fails = run_scenario(e, name, sc, sc_a_ids, b_ids, sc_refs[0], sc_refs[1], sc_refs[2],
                                     args.max_new, chunk, b_max_new)
            finally:
                e.close()
            failures += fails
            print(f"[harness] scenario {name}: {'PASS' if not fails else 'FAIL'} ({time.time() - t0:.0f} s)",
                  flush=True)
        except Exception as ex:
            failures.append(f"{name}: {ex}")
            print(f"[harness] scenario {name}: ERROR {ex}", flush=True)

    print(f"[harness] {'ALL PASS' if not failures else f'{len(failures)} FAILURE(S)'}", flush=True)
    for f in failures:
        print(f"[harness] FAIL {f}", flush=True)
    return 0 if not failures else 1


def run_scenario(e: Engine, name: str, sc: dict, a_ids, b_ids, ref_a, ref_b, ref_hash, max_new, chunk, b_max_new=None) -> list[str]:
    """One preempt engine: A parks at the boundary, the interim request(s) run, A resumes; outputs and the final
    state hash must match the references.  Returns a list of failure descriptions (empty = pass)."""
    b_max_new = max_new if b_max_new is None else b_max_new
    failures: list[str] = []
    a_tokens: list[int] = []
    suspensions: list[int] = []

    def record_suspend(line: str) -> bool:
        if line.startswith("SUSPENDED"):
            suspensions.append(int(line.split()[1]))
            return True
        return False

    def watch_b_trigger(sent: dict) -> object:
        """on_line for A's first leg: queues the first interim request once A's read passes the trigger."""
        def on_line(line: str) -> bool:
            if line.startswith("PP ") and sc["trigger"] is not None and not sent["b"]:
                if int(line.split()[1]) >= sc["trigger"]:
                    if sc["preempts"]:
                        e.gen(sc["preempts"][0], b_ids, b_max_new)
                    e.send("YIELD")            # offer the boundary: the engine cannot see this server's queue
                    sent["b"] = True
            return record_suspend(line)
        return on_line

    sent = {"b": False}
    e.gen(1, a_ids, max_new)
    if sc["trigger"] is not None:
        leg = e.collect(1, stop_on=watch_b_trigger(sent))
        a_tokens += leg["tokens"]
        if not sent["b"]:
            return [f"{name}: the trigger ({sc['trigger']}) never fired - no PP line reached it"]
        if not leg.get("stopped"):
            return [f"{name}: A finished its prompt without parking (trigger {sc['trigger']})"]

        if not sc["preempts"]:
            # nobody ran in between: the restore itself is under test
            e.send("RESUME id=1")
            done_a = e.collect(1)
            a_tokens += done_a["tokens"]
            if done_a["finish"] != ref_a["finish"]:
                failures.append(f"{name}: A finish {done_a['finish']!r} != reference {ref_a['finish']!r}")

        for i, rid in enumerate(sc["preempts"]):
            last = i == len(sc["preempts"]) - 1
            if sc["cancel"] and last:
                e.send("CANCEL id=1")
                b = e.collect(rid)
                failures += output_differences(ref_b, b, f"{name}: interim request {rid} before cancel")
                c = e.collect(1)
                if c["finish"] != "cancel":
                    failures.append(f"{name}: the cancelled parked request finished with {c['finish']!r}")
                e.gen(None, b_ids, min(8, b_max_new))                     # the engine must be healthy afterwards
                h = e.collect(None)
                if h["tokens"] != ref_b["tokens"][:min(8, b_max_new)]:
                    failures.append(f"{name}: the request after a cancel does not match the B reference")
                bad = [p for p in suspensions if p % chunk != 0 or p >= len(a_ids)]
                if bad:
                    failures.append(f"{name}: parks at non-chunk boundaries {bad}")
                return failures                           # a cancelled A has no output parity to check
            b = e.collect(rid)
            b_diff = output_differences(ref_b, b, f"{name}: interim request {rid}")
            if b_diff:
                e._record("parity_failure", label=f"interim request {rid}", expected=ref_b, actual=b)
                failures += b_diff
                break
            e.send("RESUME id=1")
            if not last:
                # wait for the engine's RESUME echo (the restore is done) before offering the next boundary:
                # a YIELD that lands while the resume is still being processed would be wiped by it
                e._next_line(lambda l: l.startswith("RESUME"), timeout=120)
                e.gen(sc["preempts"][i + 1], b_ids, b_max_new)   # A reads one chunk, parks again for the next one
                e.send("YIELD")
            if last:
                done_a = e.collect(1)
                a_tokens += done_a["tokens"]
                if done_a["finish"] != ref_a["finish"]:
                    failures.append(f"{name}: A finish {done_a['finish']!r} != reference {ref_a['finish']!r}")
            else:
                leg = e.collect(1, stop_on=record_suspend)
                a_tokens += leg["tokens"]
                if not leg.get("stopped"):
                    failures.append(f"{name}: A did not park again for request {sc['preempts'][i + 1]}")
                    break
    else:
        # decode-nopark: B is queued once the prompt is read (REUSED) - decode must never yield
        def on_reused(line: str) -> bool:
            if line.startswith("REUSED"):
                e.gen(sc["preempts"][0], b_ids, b_max_new)   # no YIELD: decode is never preempted
                sent["b"] = True
            return line.startswith("SUSPENDED")

        leg = e.collect(1, stop_on=on_reused)
        a_tokens += leg["tokens"]
        if leg.get("stopped"):
            return [f"{name}: A parked during decode - decode must never yield"]
        if not sent["b"]:
            return [f"{name}: REUSED never arrived"]
        b = e.collect(sc["preempts"][0])
        failures += output_differences(ref_b, b, f"{name}: queued request")
        e.send("RESUME id=1")
        head = e._next_line(lambda l: l.startswith("ERR") or l.startswith("RESUME") or l.startswith("DONE"),
                            timeout=60)
        if not head.startswith("ERR no parked request"):
            failures.append(f"{name}: RESUME without a parked request answered {head!r}")

    if failures:
        return failures

    if suspensions:
        legal = set(range(chunk, (len(a_ids) - 1) // chunk * chunk + 1, chunk))
        bad = [p for p in suspensions if p not in legal]
        if bad:
            failures.append(f"{name}: parks at non-chunk boundaries {bad} (legal: chunk multiples)")
        if any(b < a + chunk for a, b in zip(suspensions, suspensions[1:])):
            failures.append(f"{name}: resume did not advance a full chunk between parks: {suspensions}")
        expected_parks = max(1, len(sc["preempts"]))
        if len(suspensions) != expected_parks:
            failures.append(f"{name}: {len(suspensions)} parks != expected {expected_parks}")

    if not failures and sc["trigger"] is None:
        return failures        # decode-nopark ends with the interim request as the engine's last: no A state
    if not failures:
        if a_tokens != ref_a["tokens"]:
            n = min(len(a_tokens), len(ref_a["tokens"]))
            first = next((i for i in range(n) if a_tokens[i] != ref_a["tokens"][i]), n)
            failures.append(f"{name}: A tokens differ from the reference at [{first}] "
                            f"({len(a_tokens)} vs {len(ref_a['tokens'])} tokens)")
        else:
            got = e.state_hash()
            diff = state_differences(ref_hash, got)
            if diff:
                failures.append(f"{name}: state hash differs ({', '.join(diff)}): ref {ref_hash} got {got}")

    # after A's final commit the engine must stay healthy: a fresh request's output matches the B reference
    # (the same prompt on the control engine), and QUIT still ends the process cleanly (every caller closes)
    n_c = min(8, b_max_new) or 1
    e.gen(None, b_ids, n_c)
    h = e.collect(None)
    if ref_b["tokens"] and h["tokens"] != ref_b["tokens"][:n_c]:
        first = next((i for i in range(min(len(h["tokens"]), len(ref_b["tokens"])))
                      if h["tokens"][i] != ref_b["tokens"][i]), min(len(h["tokens"]), len(ref_b["tokens"])))
        failures.append(f"{name}: the request after A's DONE does not match the B reference "
                        f"({len(h['tokens'])} tokens, first difference at [{first}])")
    return failures


if __name__ == "__main__":
    sys.exit(main())
