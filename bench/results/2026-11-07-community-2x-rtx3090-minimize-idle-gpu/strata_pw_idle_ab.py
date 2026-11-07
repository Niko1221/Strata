#!/usr/bin/env python3
"""Does `--pipeline-windows 2` fill the idle half of a two-card layer split? A/B with a GPU sampler.

Build: KUBECONFIG=~/.kube/config python3 -u Strata-Tools/strata_pw_idle_ab.py \
           --url http://g27:8081 --pod-match strata-l28 --host l28 \
           --arms 'pw=0,pw=2 theta=0.35' --runs 6 --context 4000 --tokens 400 --json out.json

A serial layer split is a pipeline, not a tensor split: with one stream the two cards take turns, so
one of them is always waiting (`bench/results/2026-10-20-0.2.1-idle-gpu` measured ~40-55 % per card,
`Measurements/strata/2026-11-01-l28-2x3090-split-sweep` 40/53 % on l28). `--pipeline-windows 2`
builds the next verify window on the first card while the second one verifies the current, which is
the only fill the engine has. This tool measures both sides of that claim on ONE running server:

  * speed  - the engine's own `generated in ... ms (... tok/s)` line, arms paired per round;
  * idle   - `nvidia-smi -lms` sampled on the serving host, sliced into the decode window of each
             request by the millisecond counts the engine reports (so prefill and idle time before
             the request are not in the average).

The arms switch through `STRATA_PIPELINE_SWITCH=<file>` (`generate.cpp`), which the engine re-reads
per request - no restart, same expert placement, same KV state. Needs the server started with
`--pipeline-windows 2` and `STRATA_PIPELINE_DEBUG=1`; `STRATA_DECODE_TIMING` deliberately stays off,
because that flag costs the serial loop ~8 % and would inflate the difference
(`Measurements/strata/2026-11-05-l28-pipeline-acceptance` § 10.1).

Per card it reports the mean utilisation during decode, the sum of the two (how much of the pair is
working), and the share of samples where one card sits below `--idle` while the other works.
"""
import argparse, json, re, statistics, subprocess, sys, threading, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_paths as sp
import strata_pipeline_ab as ab

SERVE_RE = re.compile(r"read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")


def sh(cmd, timeout=600):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                          timeout=timeout).stdout.decode("utf8", "replace")


class Sampler:
    """`nvidia-smi -lms` on a remote host, stamped with THIS clock as the lines arrive.

    nvidia-smi prints one line per card per sample, in card order, so `ngpu` lines make one row:
    `rows = [(t, [[util, power], [util, power], ...]), ...]`.
    """

    def __init__(self, host, ms, ngpu=2):
        self.host, self.ms, self.ngpu = host, ms, ngpu
        self.rows, self.n_dropped, self.p = [], 0, None
        self.stop = threading.Event()

    def _run(self):
        cmd = (f"stdbuf -oL nvidia-smi --query-gpu=utilization.gpu,power.draw "
               f"--format=csv,noheader,nounits -lms {self.ms}")
        p = self.p = subprocess.Popen(["ssh", "-T", "-o", "BatchMode=yes", self.host, cmd],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        buf = []
        for line in iter(p.stdout.readline, b""):        # readline, not `for line in`: the
            if self.stop.is_set():                        # iterator reads ahead in 8 KiB blocks
                break
            parts = [float(x) for x in line.decode().strip().split(",")]
            if len(parts) != 2:
                self.n_dropped += 1
                continue
            buf.append(parts)
            if len(buf) == self.ngpu:
                self.rows.append((time.time(), buf))
                buf = []

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(2.0)
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.p:
            self.p.kill()

    def slice(self, t0, t1, gpu, col=0):
        return [r[1][gpu][col] for r in self.rows if t0 <= r[0] <= t1]


def ask(url, model, text, tokens):
    """One greedy chat completion; returns (end time on this clock, wall seconds)."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}],
                       "max_tokens": tokens, "temperature": 0, "stream": False}).encode()
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                     headers={"Content-Type": "application/json"}), timeout=1800) as r:
        r.read()
    return time.time(), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://g27:8081")
    ap.add_argument("--ns", default="services")
    ap.add_argument("--pod", default="")
    ap.add_argument("--pod-match", default="strata-l28")
    ap.add_argument("--log", default="/opt/strata/scripts/strata-unsloth-ud-iq4_xs.log")
    ap.add_argument("--switch", default="/opt/strata/scripts/pl_switch.txt")
    ap.add_argument("--host", default="l28", help="ssh host that runs the GPUs (nvidia-smi sampler)")
    ap.add_argument("--arms", default="pw=0,pw=2 theta=0.35")
    ap.add_argument("--kinds", default="story,code")
    ap.add_argument("--context", type=int, default=4000)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--ms", type=int, default=100, help="sampler interval in ms")
    ap.add_argument("--idle", type=float, default=10.0, help="a card below this percent counts as idle")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if not a.pod:
        names = sh(["kubectl", "-n", a.ns, "get", "pods", "-o", "name"]).split()
        a.pod = next((n.split("/")[1] for n in names if a.pod_match in n), "")
        if not a.pod:
            raise SystemExit(f"no pod matching '{a.pod_match}' in {a.ns} - KUBECONFIG set?")

    def kexec(cmd):
        return sh(["kubectl", "-n", a.ns, "exec", a.pod, "--", "sh", "-c", cmd])

    saved = kexec(f"cat {a.switch}").strip()
    model = ((json.loads(sh(["curl", "-sS", "-m", "20", f"{a.url}/v1/models"]) or "{}").get("data")
              or [{}])[0]).get("id", "")
    arms, kinds = a.arms.split(","), a.kinds.split(",")
    sp.print_stamp()
    print(f"server  : {a.url}  model {model}  pod {a.pod}")
    print(f"switch  : {a.switch} = {saved!r} (restored at the end)")
    print(f"arms    : {'  |  '.join(arms)}")
    print(f"prompt  : ~{a.context} filler tokens + instruction, {a.tokens} tokens greedy, "
          f"{a.runs} rounds, round-robin, sampler {a.ms} ms over ssh {a.host}\n")

    res = {(arm, kind): [] for arm in arms for kind in kinds}
    skipped = 0

    def log_size():
        out = kexec(f"stat -c %s {a.log}").strip()
        return int(out) if out.isdigit() else 0

    with Sampler(a.host, a.ms) as s:
        ab.ask(a.url, model, ab.KINDS["story"], 8)          # load / warm, not counted
        for run in range(a.runs):
            for arm in arms:
                kexec(f"printf '%s\\n' {json.dumps(arm)} > {a.switch}")
                for kind in kinds:
                    off = log_size()
                    t_end, wall = ask(a.url, model, ab.prompt(kind, a.context, f"r{run}{arm}{kind}"),
                                      a.tokens)
                    lines = kexec(f"tail -c +{off + 1} {a.log}").splitlines()
                    g = [SERVE_RE.search(l) for l in lines if SERVE_RE.search(l)]
                    p = [ab.PIPE_RE.search(l) for l in lines if ab.PIPE_RE.search(l)]
                    # exactly our own request in the window: a second user on the box moves both
                    # the rate and the cards, so such a round is dropped rather than averaged in
                    if len(g) != 1 or int(g[0].group(3)) != a.tokens:
                        skipped += 1
                        continue
                    read_ms, gen_ms, tok_s = int(g[0].group(1)), int(g[0].group(4)), float(g[0].group(5))
                    d0, d1 = t_end - gen_ms / 1000.0 - 0.03, t_end - 0.03
                    u = [s.slice(d0, d1, i) for i in range(2)]
                    if min(len(x) for x in u) < 5:
                        skipped += 1
                        continue
                    both = list(zip(*u))
                    res[(arm, kind)].append({
                        "run": run, "tok_s": tok_s, "wall": wall, "gen_ms": gen_ms,
                        "n_samples": len(both), "u0": statistics.mean(u[0]), "u1": statistics.mean(u[1]),
                        "idle_share": 100.0 * sum(1 for x, y in both
                                                  if min(x, y) < a.idle <= max(x, y)) / len(both),
                        "busy_sum": statistics.mean([x + y for x, y in both]),
                        "watt": sum(statistics.mean(s.slice(d0, d1, i, 1)) for i in range(2)),
                        "pipe": p[-1].groups() if p else None})
            print(f"  round {run + 1}/{a.runs} done", flush=True)
    kexec(f"printf '%s\\n' {json.dumps(saved)} > {a.switch}")
    if a.json:
        Path(a.json).write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in res.items()}, indent=1))

    med = lambda xs: statistics.median(xs) if xs else 0.0
    head = f"\n{'arm':<20} {'kind':<6} {'tok/s':>7} {'CUDA0':>6} {'CUDA1':>6} {'sum':>6} {'W':>6} {'one idle':>9}"
    if any(r["pipe"] for rows in res.values() for r in rows):   # needs STRATA_DECODE_TIMING=1
        head += f" {'ms/win':>7} {'onpath%':>8} {'spec%':>6}"
    print(head)
    for arm in arms:
        for kind in kinds:
            rows = res[(arm, kind)]
            if not rows:
                print(f"{arm:<20} {kind:<6} no data")
                continue
            line = (f"{arm:<20} {kind:<6} {med([r['tok_s'] for r in rows]):7.1f} "
                    f"{med([r['u0'] for r in rows]):5.0f}% {med([r['u1'] for r in rows]):5.0f}% "
                    f"{med([r['busy_sum'] for r in rows]):5.0f}% {med([r['watt'] for r in rows]):5.0f} "
                    f"{med([r['idle_share'] for r in rows]):8.0f}%")
            pick = lambda i: med([float(r["pipe"][i]) for r in rows if r["pipe"]])
            if any(r["pipe"] for r in rows):
                onpath = (100.0 * pick(4) / pick(0)) if pick(0) else 0.0
                spec = (100.0 * pick(3) / pick(0)) if pick(0) else 0.0
                line += f" {pick(2):7.2f} {onpath:7.0f}% {spec:5.0f}%"
            print(line)
    print("\npaired per round (the same round id on both arms; a round whose baseline or whose "
          "arm was dropped to contention does not appear):")
    for kind in kinds:
        for arm in arms[1:]:
            base = {r["run"]: r for r in res[(arms[0], kind)]}
            pairs = [(base[r["run"]], r) for r in res[(arm, kind)] if r["run"] in base]
            if not pairs:
                continue
            d = [x["tok_s"] / b["tok_s"] - 1 for b, x in pairs]
            print(f"  {kind:6s} {arm:<20} {sum(1 for r in d if r > 0)}/{len(d)} rounds ahead, median "
                  f"{statistics.median(d) * 100:+6.1f} %, per round: "
                  + " ".join(f"{r * 100:+.0f}" for r in d) + " %")
            du = [x["busy_sum"] - b["busy_sum"] for b, x in pairs]
            dw = [x["watt"] - b["watt"] for b, x in pairs]
            di = [x["idle_share"] - b["idle_share"] for b, x in pairs]
            print(f"  {'':6s} {'':20} busy sum {statistics.median(du):+6.1f} pp, "
                  f"one-idle {statistics.median(di):+6.1f} pp, "
                  f"{statistics.median(dw):+5.0f} W (median of the per-round differences)")
    print(f"\nsampler: {len(s.rows)} rows, {s.n_dropped} dropped, {skipped} rounds skipped "
          f"(foreign request or too few samples)"
          + (f", {len(s.rows) / max(1.0, s.rows[-1][0] - s.rows[0][0]):.1f} Hz achieved"
             if len(s.rows) > 2 else ""))


if __name__ == "__main__":
    main()
