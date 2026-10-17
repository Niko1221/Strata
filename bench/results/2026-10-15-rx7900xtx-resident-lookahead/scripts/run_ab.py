#!/usr/bin/env python3
"""Bounded same-binary GEN comparison. Does not stop existing services or edit configuration."""

import argparse, hashlib, json, os, queue, re, shutil, subprocess, threading, time
from pathlib import Path

DIAGNOSTICS = (
    "STRATA_LOOKAHEAD_STATS",
    "STRATA_DECODE_TIMING",
    "STRATA_DRAFT_TIMING",
    "STRATA_VERIFY_PROFILE",
    "STRATA_SNAPSHOT_VERIFY",
    "STRATA_STATE_HASH",
    "STRATA_STATE_HASH_GDN",
)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Engine:
    def __init__(self, cfg, arm, directory, deadline):
        self.deadline = deadline
        self.logpath = directory / "stderr.log"
        self.err = self.logpath.open("w", encoding="utf8")
        self.out = (directory / "stdout.log").open("w", encoding="utf8")
        self.lines = queue.Queue()
        env = dict(os.environ, **cfg.get("env", {}))
        for key in (*DIAGNOSTICS, "STRATA_LOOKAHEAD"):
            env.pop(key, None)
        if arm == "on":
            env["STRATA_LOOKAHEAD"] = "1"
        try:
            self.p = subprocess.Popen(
                [cfg["exe"], "--serve", *cfg["args"]],
                cwd=cfg.get("cwd"),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.err,
                text=True,
                bufsize=1,
            )
        except BaseException:
            self.err.close()
            self.out.close()
            raise

        def collect():
            for line in self.p.stdout:
                self.out.write(line)
                self.out.flush()
                self.lines.put(line.strip())
            self.lines.put(None)

        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        try:
            until = min(deadline, time.monotonic() + cfg.get("startup_seconds", 300))
            while not self.read(until).startswith("READY"):
                pass
            logs = self.logpath.read_text(errors="replace")
            if arm == "auto":
                if "off (complete GPU/RAM coverage during serial decode)" not in logs:
                    raise RuntimeError(
                        "automatic policy did not engage; this is not the target workload"
                    )
            elif not re.search(r"routing-aware prefetch[^\n]*\bon \(", logs):
                raise RuntimeError("forced-on arm did not start lookahead")
        except BaseException:
            self.close()
            raise

    def read(self, until):
        left = min(until, self.deadline) - time.monotonic()
        if left <= 0:
            raise TimeoutError("engine deadline")
        try:
            line = self.lines.get(timeout=left)
        except queue.Empty:
            raise TimeoutError("engine response deadline") from None
        if line is None:
            raise RuntimeError("engine exited before completing the protocol")
        if line.startswith("ERR"):
            raise RuntimeError(line)
        return line

    def generate(self, prompt, limit, seed, seconds):
        begin = time.monotonic()
        until = min(self.deadline, begin + seconds)
        self.p.stdin.write(
            f"GEN {limit} temperature=0 seed={seed} "
            + ",".join(map(str, prompt["ids"]))
            + "\n"
        )
        self.p.stdin.flush()
        ids = []
        first = None
        while True:
            line = self.read(until)
            if line.startswith("T "):
                if first is None:
                    first = time.monotonic() - begin
                ids.append(int(line.split()[1]))
            elif line.startswith("DONE "):
                c = line.split()
                if len(c) < 16:
                    raise RuntimeError("unsupported DONE format: " + line)
                produced = int(c[1])
                if produced != len(ids):
                    raise RuntimeError("token stream and DONE count disagree")
                return dict(
                    name=prompt["name"],
                    output_ids=ids,
                    produced=produced,
                    prompt_ms=float(c[3]),
                    decode_ms=float(c[4]),
                    finish=c[5],
                    draft_accepted=int(c[6]),
                    draft_offered=int(c[7]),
                    reused=int(c[8]),
                    decode_file_blobs=int(c[12]),
                    prompt_read=int(c[14]),
                    ttft_s=first,
                    wall_s=time.monotonic() - begin,
                    raw_done=line,
                )

    def close(self):
        try:
            if self.p.poll() is None:
                try:
                    self.p.stdin.write("STOP\nQUIT\n")
                    self.p.stdin.flush()
                    self.p.wait(timeout=20)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    self.p.kill()
                    self.p.wait(timeout=10)
        finally:
            self.thread.join(timeout=5)
            self.err.close()
            self.out.close()


def compare(a, b, limit):
    reasons = []
    if a["output_ids"] != b["output_ids"]:
        reasons.append("different output tokens")
    if a["decode_file_blobs"] or b["decode_file_blobs"]:
        reasons.append("decode read files")
    if any(x["produced"] != limit or x["finish"] != "length" for x in (a, b)):
        reasons.append("different/short output length")
    if (a["reused"], a["prompt_read"]) != (b["reused"], b["prompt_read"]):
        reasons.append("different prompt reuse")
    if min(a["decode_ms"], b["decode_ms"]) <= 0:
        reasons.append("invalid decode timing")
    return dict(
        name=a["name"],
        eligible=not reasons,
        reasons=reasons,
        on_decode_ms=a["decode_ms"],
        auto_decode_ms=b["decode_ms"],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--warmup-new", type=int, default=150)
    ap.add_argument("--total-seconds", type=int, default=7200)
    a = ap.parse_args()
    if (
        not 1 <= a.pairs <= 10
        or not 1 <= a.max_new <= 2048
        or not 1 <= a.warmup_new <= 512
    ):
        ap.error("bounds: pairs 1..10, max-new 1..2048, warmup-new 1..512")
    cfg = json.loads(Path(a.config).read_text())
    fixtures = json.loads(Path(a.fixtures).read_text())
    prompts = fixtures["prompts"]
    warm = fixtures["warmup"]
    if not 1 <= len(prompts) <= 4:
        ap.error("provide 1..4 measured prompts")
    for p in [warm, *prompts]:
        if not p["ids"] or not all(type(x) is int and x >= 0 for x in p["ids"]):
            ap.error("nonempty token IDs required")
    exe = shutil.which(cfg["exe"]) or cfg["exe"]
    cfg["exe"] = str(Path(exe).resolve())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        engine_sha256=digest(cfg["exe"]),
        config_sha256=digest(a.config),
        fixtures_sha256=digest(a.fixtures),
        limits=vars(a),
        diagnostics_removed=list(DIAGNOSTICS),
        hardware_context=cfg.get("hardware_context"),
        cpu_affinity=sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        arms=[],
        comparisons=[],
        completed=False,
    )

    def save():
        (out / "results.json").write_text(json.dumps(result, indent=2))

    deadline = time.monotonic() + a.total_seconds
    save()
    try:
        for pair in range(a.pairs):
            pair_results = {}
            for arm in ("on", "auto") if pair % 2 == 0 else ("auto", "on"):
                d = out / f"{pair:02d}-{arm}"
                d.mkdir()
                e = Engine(cfg, arm, d, deadline)
                record = dict(pair=pair, arm=arm, requests=[])
                result["arms"].append(record)
                try:
                    record["warmup"] = e.generate(
                        warm, a.warmup_new, 1922, cfg.get("request_seconds", 300)
                    )
                    save()
                    for p in prompts:
                        record["requests"].append(
                            e.generate(
                                p, a.max_new, 1922, cfg.get("request_seconds", 300)
                            )
                        )
                        save()
                finally:
                    e.close()
                pair_results[arm] = record["requests"]
            result["comparisons"].append(
                dict(
                    pair=pair,
                    requests=[
                        compare(x, y, a.max_new)
                        for x, y in zip(pair_results["on"], pair_results["auto"])
                    ],
                )
            )
            save()
        result["completed"] = True
    except BaseException as ex:
        result["error"] = type(ex).__name__ + ": " + str(ex)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
