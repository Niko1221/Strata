#!/usr/bin/env python3
"""Long-context benchmark for an OpenAI-compatible chat server (Strata, llama.cpp, ...).

One request at a time, streaming, temperature 0, reasoning_effort "none".  Token counts come from the server's
`usage` field.  Per depth: A) cold runs, B) cache-append run, C) resource sampling; plus a needle-in-haystack test.

  python3 bench.py --base-url http://127.0.0.1:8080 --api-key-file ~/strata-src/strata-api-key \
      --model strata --depths 1000,50000,100000 --needle-depths 50000,120000 --label phase1 --out-dir DIR

The API key is read from the file and only ever sent as a Bearer header; it is never printed or written.
Results: runs_<label>.csv, needles_<label>.csv, samples_<label>.csv, summary_<label>.json in --out-dir.
"""
import argparse, csv, glob, hashlib, http.client, json, os, random, re, statistics, subprocess, sys, threading, time
import urllib.parse, uuid

CODE_INSTR = [
    "Using ideas from the material above, write a complete, self-contained Python module (at least 80 lines, with "
    "docstrings) that implements an LRU cache with per-entry TTL, plus a small command-line demo. Output only code.",
    "Using ideas from the material above, write a complete Python program (at least 80 lines) that implements a "
    "thread-safe job queue with worker threads, retries and a summary report. Output only code.",
    "Using ideas from the material above, write a complete Python module that parses a simple INI-like config "
    "format, validates it against a schema and offers a command-line interface. Output only code.",
    "Using ideas from the material above, write a complete Python program that converts a small subset of Markdown "
    "(headings, lists, code blocks, links) to HTML, with tests. Output only code.",
    "Using ideas from the material above, write a complete Python module implementing a trie with autocomplete, "
    "prefix counts and unit tests. Output only code.",
]
PROSE_INSTR = ("Now, ignoring the code request above, write roughly 250 words of flowing prose (no code, no lists) "
               "describing what the most recently added section of material is about and why it might matter.")
INTRO = "Reference material follows (source files and documentation). Read it; a task comes at the end.\n\n"


class PhaseAbort(Exception):
    pass


class SizeError(Exception):
    """The server refused the prompt as longer than its context; carries the prompt's real token count."""
    def __init__(self, tokens):
        super().__init__(f"prompt of {tokens} tokens exceeds the context")
        self.tokens = tokens


# ---------------------------------------------------------------- corpus
def build_pool(seed, secret):
    home = os.path.expanduser("~/strata-src/Strata")
    cats = {
        "py": sorted(glob.glob("/usr/lib/python3.12/**/*.py", recursive=True)),
        "strata_code": [f for d in ("src", "serve", "tools", "tests", "bench")
                        for f in glob.glob(f"{home}/{d}/**/*", recursive=True)
                        if f.endswith((".cpp", ".h", ".cu", ".cuh", ".py", ".hpp", ".c"))],
        "docs": sorted(glob.glob(f"{home}/docs/*.md") + glob.glob(f"{home}/*.md") + glob.glob(f"{home}/docs/*/*.md")
                       + [f for f in glob.glob("/usr/share/doc/**/*", recursive=True)
                          if os.path.isfile(f) and not f.endswith((".gz", "copyright"))]),
        "headers": sorted(glob.glob("/usr/include/**/*.h", recursive=True)),
    }
    rng = random.Random(seed)
    seen, lists = set(), {}
    for cat, files in cats.items():
        files = [f for f in files if os.path.isfile(f) and not os.path.islink(f)]
        rng.shuffle(files)
        out = []
        for f in files:
            try:
                if not 2000 <= os.path.getsize(f) <= 150000:
                    continue
                txt = open(f, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            if secret and secret in txt:
                continue
            lines = txt.splitlines(keepends=True)
            if lines and max(len(l) for l in lines) > 1500:
                continue
            keep, total = [], 0
            for i in range(0, len(lines), 20):                       # drop any 20-line chunk seen before
                chunk = "".join(lines[i:i + 20])
                h = hashlib.md5(re.sub(r"\s+", " ", chunk).strip().encode()).hexdigest()
                total += 1
                if h not in seen and len(chunk.strip()) > 80:
                    seen.add(h)
                    keep.append(chunk)
            if total and len(keep) >= 0.6 * total:
                out.append((os.path.relpath(f, "/") if not f.startswith(home) else "strata/" + os.path.relpath(f, home),
                            "".join(keep)))
        lists[cat] = out
    pool, idx = [], 0                                                 # round-robin across categories
    while any(lists.values()):
        for cat in lists:
            if lists[cat]:
                pool.append(lists[cat].pop())
    return pool


class Corpus:
    def __init__(self, pool):
        self.pool = pool
        self.total_chars = sum(len(t) for _, t in pool)

    def text(self, start, nchars):
        out, n, i = [], 0, start % len(self.pool)
        used = 0
        while n < nchars:
            if used >= len(self.pool):
                raise RuntimeError("corpus exhausted (would duplicate text)")
            name, t = self.pool[i]
            piece = f"### FILE: {name}\n{t}\n\n"
            if n + len(piece) > nchars:                                # cut the last file at a line boundary
                cut = piece.rfind("\n", 0, max(nchars - n, 1))
                piece = piece[:cut + 1] if cut > 0 else piece[:nchars - n]
            out.append(piece)
            n += len(piece)
            i = (i + 1) % len(self.pool)
            used += 1
        return "".join(out)


# ---------------------------------------------------------------- resource sampler
class Sampler(threading.Thread):
    def __init__(self, path, swap_limit_mb=4096, interval=2.0):
        super().__init__(daemon=True)
        self.path, self.interval, self.swap_limit = path, interval, swap_limit_mb
        self.label, self.stop_flag, self.rows, self.swap_abort = "init", False, [], False
        self.base_swap = self.mem()[2]
        self.fh = open(path, "w", newline="")
        self.w = csv.writer(self.fh)
        self.w.writerow(["time", "label", "ram_used_mb", "ram_avail_mb", "swap_used_mb", "vram0_mib", "vram1_mib"])

    @staticmethod
    def mem():
        m = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":")
            m[k] = int(v.split()[0]) // 1024
        used = m["MemTotal"] - m["MemFree"] - m["Buffers"] - m["Cached"] - m.get("SReclaimable", 0)
        return used, m["MemAvailable"], m["SwapTotal"] - m["SwapFree"]

    @staticmethod
    def vram():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10).stdout.split()
            return [int(x) for x in o] + [0, 0]
        except Exception:
            return [0, 0]

    def run(self):
        while not self.stop_flag:
            used, avail, swap = self.mem()
            v = self.vram()
            row = [time.strftime("%H:%M:%S"), self.label, used, avail, swap, v[0], v[1]]
            self.rows.append(row)
            self.w.writerow(row)
            self.fh.flush()
            if swap > self.base_swap + self.swap_limit:
                self.swap_abort = True
            time.sleep(self.interval)

    def peaks(self, label):
        r = [x for x in self.rows if x[1] == label]
        if not r:
            return {}
        return {"peak_ram_used_mb": max(x[2] for x in r), "min_ram_avail_mb": min(x[3] for x in r),
                "peak_swap_mb": max(x[4] for x in r), "peak_vram0_mib": max(x[5] for x in r),
                "peak_vram1_mib": max(x[6] for x in r)}


# ---------------------------------------------------------------- client
class Client:
    def __init__(self, base_url, key, model, sampler, timeout=1200, server_log=None):
        u = urllib.parse.urlparse(base_url)
        self.host, self.port, self.prefix = u.hostname, u.port or 80, u.path.rstrip("/")
        self.key, self.model, self.sampler, self.timeout, self.server_log = key, model, sampler, timeout, server_log

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        return h

    def get(self, path):
        c = http.client.HTTPConnection(self.host, self.port, timeout=30)
        c.request("GET", self.prefix + path, headers=self._headers())
        r = c.getresponse()
        d = r.read()
        c.close()
        return json.loads(d)

    def chat(self, messages, max_tokens):
        """Stream one chat completion.  Returns dict with ttft, decode timing, text, usage, cached tokens."""
        res = {}
        conn_box = []
        log_pos = os.path.getsize(self.server_log) if self.server_log and os.path.exists(self.server_log) else None

        def work():
            try:
                c = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout + 30)
                conn_box.append(c)
                body = json.dumps({"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                   "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                                   "reasoning_effort": "none"})
                t0 = time.time()
                c.request("POST", self.prefix + "/v1/chat/completions", body=body, headers=self._headers())
                r = c.getresponse()
                if r.status != 200:
                    res["error"] = f"HTTP {r.status}: {r.read(300).decode(errors='replace')}"
                    return
                text, t_first, usage, timings, finish = [], None, None, None, None
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ch = json.loads(data)
                    if ch.get("usage"):
                        usage = ch["usage"]
                    if ch.get("timings"):
                        timings = ch["timings"]
                    for choice in ch.get("choices") or []:
                        d = choice.get("delta") or {}
                        piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                        if piece:
                            if t_first is None:
                                t_first = time.time()
                            text.append(piece)
                        finish = choice.get("finish_reason") or finish
                t_end = time.time()
                res.update(t0=t0, t_first=t_first, t_end=t_end, text="".join(text), usage=usage, timings=timings,
                           finish=finish)
            except Exception as e:                                        # noqa
                res.setdefault("error", f"{type(e).__name__}: {e}")

        th = threading.Thread(target=work, daemon=True)
        start = time.time()
        th.start()
        while th.is_alive():
            th.join(1.0)
            if self.sampler.swap_abort or time.time() - start > self.timeout:
                for c in conn_box:
                    try:
                        c.close()
                    except Exception:
                        pass
                if self.sampler.swap_abort:
                    raise PhaseAbort("swap rose more than 4 GB above baseline")
                th.join(5)
                return {"error": f"timeout after {self.timeout}s"}
        if "error" in res:
            return res
        if not res.get("usage") or res.get("t_first") is None:
            return {"error": "no usage field or no tokens in stream"}
        u = res["usage"]
        pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
        det = u.get("prompt_tokens_details") or {}
        cached = det.get("cached_tokens")
        if cached is None and res.get("timings"):
            cached = res["timings"].get("cache_n")
        out = {"prompt_tokens": pt, "completion_tokens": ct, "cached_usage": cached, "text": res["text"],
               "finish": res["finish"], "ttft": res["t_first"] - res["t0"],
               "decode_s": res["t_end"] - res["t_first"]}
        out["prefill_tps"] = pt / out["ttft"] if out["ttft"] > 0 else None
        out["decode_tps"] = (ct - 1) / out["decode_s"] if out["decode_s"] > 0 and ct and ct > 1 else None
        out["cached_log"] = None
        if log_pos is not None:
            time.sleep(0.5)
            try:
                with open(self.server_log, "rb") as f:
                    f.seek(log_pos)
                    tail = f.read().decode("utf-8", "replace")
                m = re.findall(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read", tail)
                if m:
                    out["cached_log"] = int(m[-1][1])
            except OSError:
                pass
        return out


# ---------------------------------------------------------------- benchmark
class Bench:
    def __init__(self, a, client, sampler, corpus, outdir):
        self.a, self.c, self.s, self.corpus, self.outdir = a, client, sampler, corpus, outdir
        self.overhead, self.cpt = 0, 3.6
        self.rows_fh = open(os.path.join(outdir, f"runs_{a.label}.csv"), "w", newline="")
        self.rows_w = csv.writer(self.rows_fh)
        self.rows_w.writerow(["label", "depth", "stage", "run", "status", "structure", "prompt_tokens", "cached_usage",
                              "cached_log", "completion_tokens", "ttft_s", "prefill_tps", "decode_tps", "finish",
                              "note"])
        self.needle_fh = open(os.path.join(outdir, f"needles_{a.label}.csv"), "w", newline="")
        self.needle_w = csv.writer(self.needle_fh)
        self.needle_w.writerow(["label", "depth_target", "position_pct", "prompt_tokens", "ttft_s", "expected", "answer",
                                "pass"])
        self.summary = {"label": a.label, "depths": {}, "needles": [], "events": []}
        self.offset_counter = 0

    def log(self, msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    def event(self, msg):
        self.summary["events"].append(msg)
        self.log("EVENT: " + msg)

    def rowout(self, depth, stage, run, status, structure, r, note=""):
        self.rows_w.writerow([self.a.label, depth, stage, run, status, structure, r.get("prompt_tokens"),
                              r.get("cached_usage"), r.get("cached_log"), r.get("completion_tokens"),
                              round(r["ttft"], 3) if r.get("ttft") else "",
                              round(r["prefill_tps"], 1) if r.get("prefill_tps") else "",
                              round(r["decode_tps"], 1) if r.get("decode_tps") else "", r.get("finish"), note])
        self.rows_fh.flush()

    # one request with the "fails twice -> stop the phase" rule
    def request(self, messages, max_tokens, what):
        for attempt in (1, 2):
            r = self.c.chat(messages, max_tokens)
            if "error" not in r:
                return r
            m = re.search(r"prompt \((\d+) tokens\).*(?:exceeds the context|leaves no room to answer in the context)", r["error"])
            if m:
                raise SizeError(int(m.group(1)))
            self.event(f"{what}: request failed (attempt {attempt}): {r['error']}")
        raise PhaseAbort(f"{what}: request failed twice")

    def prompt(self, filler, instr, uid=None):
        head = f"Session id: {uid or uuid.uuid4()}\n\n"      # unique first line: no cached prefix can match
        return head + INTRO + filler + "\n\n---\n" + instr

    # ---- sizing
    def calibrate(self):
        r = self.request([{"role": "user", "content": self.prompt("", CODE_INSTR[0])}], 1, "calibrate-overhead")
        self.overhead = r["prompt_tokens"]
        txt = self.corpus.text(0, 60000)
        r = self.request([{"role": "user", "content": self.prompt(txt, CODE_INSTR[0])}], 1, "calibrate-cpt")
        self.cpt = len(txt) / max(r["prompt_tokens"] - self.overhead, 1)
        self.log(f"calibration: overhead={self.overhead} tokens, {self.cpt:.2f} chars/token")

    def fit(self, start, target, tol=0.02):
        """Return filler text whose prompt is within tol of target tokens (probing the server for small sizes)."""
        want = target - self.overhead
        chars = int(want * self.cpt)
        if target <= 30000:
            for _ in range(4):
                txt = self.corpus.text(start, chars)
                r = self.request([{"role": "user", "content": self.prompt(txt, CODE_INSTR[0])}], 1, "fit-probe")
                got = r["prompt_tokens"]
                self.cpt = len(txt) / max(got - self.overhead, 1)
                if abs(got - target) <= tol * target:
                    return txt
                chars = int(want * self.cpt)
            return self.corpus.text(start, chars)
        sl = min(chars, 120000)                                   # large: probe a slice, extrapolate
        txt = self.corpus.text(start, sl)
        r = self.request([{"role": "user", "content": self.prompt(txt, CODE_INSTR[0])}], 1, "fit-probe-slice")
        self.cpt = len(txt) / max(r["prompt_tokens"] - self.overhead, 1)
        return self.corpus.text(start, int(want * self.cpt))

    def next_start(self):
        self.offset_counter += 1
        return self.offset_counter * 37

    # ---- procedure A/B/C at one depth
    def run_depth(self, depth, max_context):
        label = f"d{depth}"
        self.s.label = label
        info = {"depth": depth, "cold": [], "append": None}
        self.summary["depths"][str(depth)] = info
        if depth + 2500 + 300 + 100 > max_context:
            self.event(f"depth {depth}: skipped, would exceed max_context {max_context}")
            return
        start = self.next_start()
        filler = self.fit(start, depth)
        colds, last = [], None
        target_runs, i, instr_i = self.a.cold_runs, 0, 0
        redo_budget = 3
        while i < target_runs:
            instr = CODE_INSTR[instr_i % len(CODE_INSTR)]
            content = self.prompt(filler, instr)
            try:
                r = self.request([{"role": "user", "content": content}], 256, f"cold d{depth} run{i + 1}")
            except SizeError as e:
                self.event(f"depth {depth}: prompt of {e.tokens} tokens rejected as over the context; shrinking and redoing")
                self.cpt = len(filler) / max(e.tokens - self.overhead, 1)
                filler = self.corpus.text(start, int((depth - self.overhead) * self.cpt))
                continue
            pt, ct = r["prompt_tokens"], r["completion_tokens"]
            cached = max(r["cached_usage"] or 0, r["cached_log"] or 0)
            status, note = "ok", ""
            if abs(pt - depth) > 0.035 * depth and redo_budget:
                status, note = "redo-size", f"prompt {pt} vs target {depth}"
                self.cpt = len(filler) / max(pt - self.overhead, 1)
                filler = self.corpus.text(start, int((depth - self.overhead) * self.cpt))
            elif cached > max(200, 0.01 * pt) and redo_budget:
                status, note = "redo-cached", f"{cached} cached tokens in a cold run"
            elif ct < 200 and redo_budget:
                status, note = "redo-short", f"only {ct} output tokens ({r['finish']})"
                instr_i += 1
            elif colds and r["prefill_tps"] > 3 * statistics.median(x["prefill_tps"] for x in colds) and depth > 5000 \
                    and redo_budget:
                status, note = "redo-fast", "suspiciously fast cold prefill"
            self.rowout(depth, "cold", i + 1, status, "", r, note)
            self.log(f"d{depth} cold{i + 1} [{status}] prompt={pt} ttft={r['ttft']:.1f}s prefill={r['prefill_tps']:.0f} "
                     f"tok/s decode={r['decode_tps'] and round(r['decode_tps'], 1)} out={ct} cached={cached}")
            if status != "ok":
                redo_budget -= 1
                self.event(f"depth {depth} cold run redone: {status} ({note})")
                continue
            r["content"] = content
            colds.append(r)
            last = r
            i += 1
            instr_i += 1
            if r["ttft"] > 360 and target_runs == self.a.cold_runs:
                target_runs = min(target_runs, 2)
                self.event(f"depth {depth}: a cold prefill took {r['ttft']:.0f}s (>6 min), using {target_runs} cold runs")
        info["cold"] = [{k: x[k] for k in ("prompt_tokens", "ttft", "prefill_tps", "decode_tps", "completion_tokens")}
                        for x in colds]
        med = lambda k: statistics.median(x[k] for x in colds)           # noqa
        info.update(prompt_tokens=int(med("prompt_tokens")), cold_ttft=med("ttft"), cold_prefill_tps=med("prefill_tps"),
                    code_decode_tps=med("decode_tps"))
        # B: cache append
        new_text = self.corpus.text(self.next_start() + 5000, int(2500 * self.cpt))
        append_block = "Additional material follows.\n\n" + new_text + "\n\n---\n" + PROSE_INSTR
        structures = [("append-to-user-message", [{"role": "user", "content": last["content"] + "\n\n" + append_block}]),
                      ("multi-turn", [{"role": "user", "content": last["content"]},
                                      {"role": "assistant", "content": last["text"]},
                                      {"role": "user", "content": append_block}])]
        hit = None
        for name, msgs in structures:
            r = self.request(msgs, 256, f"append d{depth} {name}")
            cached = max(r["cached_usage"] or 0, r["cached_log"] or 0)
            new_tokens = r["prompt_tokens"] - cached if cached else r["prompt_tokens"] - last["prompt_tokens"]
            r["eff_prefill"] = new_tokens / r["ttft"]
            ck = self.a.ckpt_interval
            expect = (last["prompt_tokens"] // ck) * ck if ck else last["prompt_tokens"]
            ok = expect > 0 and cached >= 0.9 * expect and r["ttft"] < 0.5 * info["cold_ttft"]
            self.rowout(depth, "append", 1, "ok", name, r, f"new_tokens={new_tokens} hit={ok}")
            self.log(f"d{depth} append[{name}] prompt={r['prompt_tokens']} cached={cached} ttft={r['ttft']:.1f}s "
                     f"(cold {info['cold_ttft']:.1f}s) eff_prefill={r['eff_prefill']:.0f} tok/s "
                     f"decode={r['decode_tps'] and round(r['decode_tps'], 1)} hit={ok}")
            info["append"] = {"structure": name, "ttft": r["ttft"], "cached": cached, "new_tokens": new_tokens,
                              "eff_prefill_tps": r["eff_prefill"], "prose_decode_tps": r["decode_tps"],
                              "prompt_tokens": r["prompt_tokens"], "hit": ok, "expected_cached": expect,
                              "expected_ttft_2500": 2500 / info["cold_prefill_tps"]}
            if ok:
                hit = name
                break
            self.event(f"depth {depth}: cache miss with structure '{name}' (cached={cached}, ttft={r['ttft']:.1f}s)")
        if hit:
            info["cache_verdict"] = f"WORKING ({hit})"
        elif info["append"]["expected_cached"] == 0:
            info["cache_verdict"] = "NO HIT (prompt below the checkpoint interval; nothing to reuse)"
        else:
            info["cache_verdict"] = "BROKEN"
        info.update(self.s.peaks(label))

    # ---- needle test
    def run_needles(self, depth, max_context):
        label = f"needle{depth}"
        self.s.label = label
        if depth + 600 > max_context:
            self.event(f"needle {depth}: skipped, would exceed max_context {max_context}")
            return
        start = self.next_start() + 900
        filler = self.fit(start, depth - 60, tol=0.03)
        for pct in (10, 50, 90):
            code = "-".join(str(random.randint(1, 9)) for _ in range(6))
            for attempt in (1, 2, 3):
                cut = filler.find("\n", int(len(filler) * pct / 100)) + 1
                needle = f"\nNOTE FROM THE ARCHIVIST: The vault code is {code}. Remember it.\n\n"
                text = filler[:cut] + needle + filler[cut:]
                content = (f"Session id: {uuid.uuid4()}\n\n" + INTRO + text + "\n\n---\n"
                           "Question: what is the vault code mentioned by the archivist in the material above? "
                           "Answer with only the code, in the form d-d-d-d-d-d.")
                try:
                    r = self.request([{"role": "user", "content": content}], 64, f"needle {depth} {pct}%")
                except SizeError as e:
                    self.event(f"needle {depth} {pct}%: prompt of {e.tokens} tokens over the context; shrinking and redoing")
                    self.cpt = len(filler) / max(e.tokens - self.overhead, 1)
                    filler = self.corpus.text(start, int((depth - 60 - self.overhead) * self.cpt))
                    continue
                if abs(r["prompt_tokens"] - depth) > 0.04 * depth and attempt < 3:
                    self.event(f"needle {depth} {pct}%: prompt {r['prompt_tokens']} off target; rescaled and redone")
                    self.cpt = len(filler) / max(r["prompt_tokens"] - self.overhead, 1)
                    filler = self.corpus.text(start, int((depth - 60 - self.overhead) * self.cpt))
                    continue
                break
            digits = re.sub(r"\D", "", r["text"])
            ok = digits == code.replace("-", "") or code in r["text"]
            self.needle_w.writerow([self.a.label, depth, pct, r["prompt_tokens"], round(r["ttft"], 2), code,
                                    r["text"].strip().replace("\n", " ")[:80], "PASS" if ok else "FAIL"])
            self.needle_fh.flush()
            self.summary["needles"].append({"depth_target": depth, "prompt_tokens": r["prompt_tokens"], "pct": pct,
                                            "pass": ok, "answer": r["text"].strip()[:80], "expected": code,
                                            "ttft": r["ttft"]})
            self.log(f"needle depth={depth} pos={pct}% prompt={r['prompt_tokens']} ttft={r['ttft']:.1f}s "
                     f"expected={code} got={r['text'].strip()[:40]!r} -> {'PASS' if ok else 'FAIL'}")
        self.summary["needles_peaks_" + label] = self.s.peaks(label)

    def save(self):
        json.dump(self.summary, open(os.path.join(self.outdir, f"summary_{self.a.label}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--api-key-line", type=int, default=1, help="which non-empty line of the key file to use")
    ap.add_argument("--model", required=True)
    ap.add_argument("--depths", default="1000,50000,100000", help="comma-separated prompt sizes in tokens")
    ap.add_argument("--needle-depths", default="", help="comma-separated; 10/50/90%% positions at each")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-context", type=int, default=0, help="0 = ask /health")
    ap.add_argument("--cold-runs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--server-log", default=None, help="optional: Strata log, parsed for reused-token counts")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ckpt-interval", type=int, default=16384,
                    help="server's prefix-checkpoint spacing in tokens (Strata: 16384); 0 = expect the whole prefix")
    a = ap.parse_args()
    keys = [l.strip() for l in open(os.path.expanduser(a.api_key_file)) if l.strip()]   # a key file may hold several keys
    key = keys[a.api_key_line - 1]
    os.makedirs(a.out_dir, exist_ok=True)
    random.seed(a.seed)
    sampler = Sampler(os.path.join(a.out_dir, f"samples_{a.label}.csv"))
    sampler.start()
    client = Client(a.base_url, key, a.model, sampler, a.timeout, a.server_log)
    mc = a.max_context
    if not mc:
        try:
            mc = client.get("/health").get("max_context") or 0
        except Exception:
            mc = 0
    mc = mc or 131072
    print(f"base-url={a.base_url} model={a.model} max_context={mc} depths={a.depths} needles={a.needle_depths}", flush=True)
    corpus = Corpus(build_pool(a.seed, key))
    print(f"corpus: {len(corpus.pool)} unique files, {corpus.total_chars / 1e6:.1f} MB", flush=True)
    b = Bench(a, client, sampler, corpus, a.out_dir)
    b.summary.update(max_context=mc, base_swap_mb=sampler.base_swap, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                     corpus_files=len(corpus.pool), corpus_mb=round(corpus.total_chars / 1e6, 1))
    try:
        sampler.label = "warmup"
        client_r = b.request([{"role": "user", "content": "Say hello in five words."}], 32, "warmup")
        b.log(f"warm-up done ({client_r['completion_tokens']} tokens, not counted)")
        sampler.label = "calibration"
        b.calibrate()
        for d in [int(x) for x in a.depths.split(",") if x]:
            b.run_depth(d, mc)
            b.save()
        for d in [int(x) for x in a.needle_depths.split(",") if x]:
            b.run_needles(d, mc)
            b.save()
        b.summary["completed"] = True
    except PhaseAbort as e:
        b.event(f"PHASE STOPPED: {e}")
        b.summary["completed"] = False
        b.summary["stopped_reason"] = str(e)
    finally:
        sampler.label = "done"
        b.summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        b.summary["peak_swap_overall_mb"] = max([x[4] for x in sampler.rows] or [0])
        b.save()
        sampler.stop_flag = True
    print("finished; completed =", b.summary.get("completed"), flush=True)


if __name__ == "__main__":
    main()
