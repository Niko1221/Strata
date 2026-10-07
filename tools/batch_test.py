#!/usr/bin/env python3
"""Batching test: the same prompts decoded alone (GEN, the usual path with drafts) and together in the batch
windows (BGEN), greedy.  Every slot's tokens must equal its solo tokens; prints the aggregate decode rate.

  python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 4 --n 4 \
      --extra "--layer-split 12,24,36 --trim-stage-weights --pcie-frac 0 --adapt-every 1000000"
"""
import argparse, json, os, shlex, subprocess, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

QUESTIONS = [
    "Explique en detail le fonctionnement d'un B-tree.",
    "Write a Python function that merges overlapping intervals, then explain it.",
    "Compare TCP and QUIC: handshake, congestion control, multiplexing.",
    "Raconte l'histoire du calcul de pi, d'Archimede a nos jours.",
    "What are the main causes of the French Revolution?",
    "Explain how a transformer's attention works, step by step.",
    "Donne une recette de tarte aux pommes, etape par etape.",
    "Describe the life cycle of a star like the Sun.",
]


def tokenizer(path):
    import strata_tokenizer as ST

    t = Path(path)
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for s, i in vocab.items():
        tokens[i] = s
    merges = (t / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((t / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


class Engine:
    def __init__(self, exe, cfg, batch, extra_env, extra_args):
        args = list(cfg["args"])
        if len(cfg.get("gpu") or []) > 1:
            args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
        if batch:
            args += ["--batch", str(batch)]
        args += extra_args
        env = dict(os.environ)
        for name, value in extra_env.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
        if os.name == "nt":
            env["PATH"] = os.pathsep.join(cfg.get("lib_dirs", []) + [env.get("PATH", "")])
        self.log_path = os.environ.get("BATCH_TEST_LOG") or os.path.join(tempfile.gettempdir(), "batch_test_engine.log")
        self.log = open(self.log_path, "w")
        self.p = subprocess.Popen([exe, "--serve", *args], cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env)
        for line in self.p.stdout:
            if line.startswith("READY"):
                return
        raise SystemExit(f"the engine ended before READY - see {self.log_path}")

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def lines(self):
        for line in self.p.stdout:
            yield line.rstrip("\n")

    def close(self):
        try:
            if self.p.poll() is None:
                try:
                    self.send("QUIT")
                except BrokenPipeError:
                    pass
                try:
                    self.p.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    self.p.terminate()
                    try:
                        self.p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.p.kill()
                        self.p.wait()
        finally:
            self.p.stdin.close()
            self.p.stdout.close()
            self.log.close()


class ProtocolError(RuntimeError):
    pass


def next_line(out):
    try:
        return next(out)
    except StopIteration:
        raise ProtocolError("the engine ended before every request completed") from None


def run_batch(eng, out, prompts, max_new, keys, stagger_after=(), promote_after=0, engine_slots=None):
    """Admit at measured anchor-token counts; keep tokens emitted during every admission."""
    got = {i: [] for i in range(len(prompts))}
    done, admitted, admissions = {}, set(), []
    started = time.monotonic()
    first_bt = None
    promotion = None
    engine_slots = list(range(len(prompts))) if engine_slots is None else engine_slots
    logical_slot = {slot: i for i, slot in enumerate(engine_slots)}

    def consume(line, pending=None):
        nonlocal first_bt
        fields = line.split()
        if line.startswith("ERR"):
            raise ProtocolError(line)
        if line.startswith("T "):
            if pending is None:
                raise ProtocolError(f"unrouted admission token: {line}")
            got[pending].append(int(fields[1]))
        elif line.startswith(("BT ", "BDONE ")):
            actual_slot = int(fields[1])
            if actual_slot not in logical_slot:
                raise ProtocolError(f"token/completion for an unrequested slot: {line}")
            slot = logical_slot[actual_slot]
            if slot not in admitted or slot in done:
                raise ProtocolError(f"token/completion for a slot that is not active: {line}")
            if fields[0] == "BT":
                got[slot].append(int(fields[2]))
                first_bt = first_bt or time.monotonic()
            else:
                done[slot] = line
        elif line.startswith("BADM "):
            if pending is None or len(fields) != 3 or int(fields[1]) != engine_slots[pending] or fields[2] not in ("0", "1"):
                raise ProtocolError(f"unexpected admission completion: {line}")
            admitted.add(pending)
            if fields[2] == "0":
                done[pending] = line
            return True
        return False

    if promote_after:
        eng.send(" ".join(x for x in ("GEN", str(max_new), keys, ",".join(map(str, prompts[0]))) if x))
        stopped_at = None
        while True:
            line = next_line(out)
            if line.startswith("DONE"):
                if stopped_at is None or len(got[0]) >= max_new:
                    raise ProtocolError("solo anchor completed before it could be promoted; lower --promote-after")
                fields = line.split()
                if len(fields) < 6 or fields[5] != "cancel":
                    raise ProtocolError(f"solo anchor did not cancel for promotion: {line}")
                promotion = {"stop_sent_at_tokens": stopped_at, "resumed_after_tokens": len(got[0])}
                break
            consume(line, 0)
            if stopped_at is None and len(got[0]) >= promote_after:
                stopped_at = len(got[0])
                eng.send("STOP")

    for i, ids in enumerate(prompts):
        if i and stagger_after:
            target = stagger_after[i - 1]
            while len(got[0]) < target:
                if 0 in done:
                    raise ProtocolError(f"anchor finished before admission {i} at {target} tokens; increase --max-new "
                                        "or lower --stagger-after")
                consume(next_line(out))
            if 0 in done:
                raise ProtocolError(f"anchor already finished at admission {i}; staggered overlap was not exercised")
        active = sorted(admitted - done.keys())
        event = {"slot": engine_slots[i], "anchor_tokens": len(got[0]),
                 "active_slots": [engine_slots[s] for s in active]}
        before = sum(len(got[s]) for s in active)
        prefix = got[0] if i == 0 and promotion else []
        eng.send(" ".join(x for x in ("BGEN", str(engine_slots[i]), str(max_new - len(prefix)), keys,
                                    ",".join(map(str, [*ids, *prefix]))) if x))
        while not consume(next_line(out), i):
            pass
        event["tokens_during_admission"] = sum(len(got[s]) for s in active) - before
        event["active_slots_after_admission"] = [engine_slots[s] for s in sorted(admitted - done.keys())]
        admissions.append(event)
        if i and stagger_after and 0 in done:
            raise ProtocolError(f"anchor finished during admission {i}; increase --max-new so the new and anchor "
                                "requests decode together")
    all_admitted = time.monotonic()
    while len(done) < len(prompts):
        consume(next_line(out))
    ended = time.monotonic()
    return got, {"admissions": admissions, "admission_seconds": all_admitted - started,
                 "total_seconds": ended - started, "decode_seconds": ended - (first_bt or started),
                 "promotion": promotion}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=4, help="prompts (<= --batch)")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--skip-solo", action="store_true")
    ap.add_argument("--keys", default="", help='sampling keys for every request, e.g. "temperature=0.7 top_k=20"')
    ap.add_argument("--extra", default="", help='more engine arguments in one string, e.g. "--adapt-every 1000000"')
    ap.add_argument("--dump", default="", help="write every slot's tokens (solo and batch) as JSON: an A/B of two engine builds")
    ap.add_argument("--long-tokens", type=int, default=0,
                    help="the LAST prompt is read from about this many tokens of filler text first: its chunks run "
                         "beside the earlier slots' windows (the prompt path lends experts then)")
    ap.add_argument("--mt-min", default="1", help="STRATA_IQ_MT_MIN for the engine (1: exact; empty: the default)")
    ap.add_argument("--stagger-after", default="", metavar="N,N,...",
                    help="admit later prompts after slot 0 has emitted these token counts (one per later prompt); "
                         "e.g. --n 4 --max-new 128 --stagger-after 8,16,24")
    ap.add_argument("--promote-after", type=int, default=0,
                    help="start the anchor with GEN, STOP after this many tokens, then resume with BGEN(prompt + "
                         "emitted tokens), as the HTTP server does when another request arrives; use with --stagger-after")
    ap.add_argument("--slots", default="", metavar="N,N,...",
                    help="engine slot IDs in prompt order; e.g. 0,2,4,6 spreads four requests across four groups "
                         "with --batch 8 --batch-groups 4 (default: consecutive slots)")
    a = ap.parse_args()
    if not 1 <= a.n <= min(a.batch, len(QUESTIONS)) or a.max_new < 1 or a.long_tokens < 0:
        ap.error("require 1 <= --n <= min(--batch, 8), positive --max-new and nonnegative --long-tokens")
    try:
        stagger_after = [int(n) for n in a.stagger_after.split(",")] if a.stagger_after else []
        engine_slots = [int(n) for n in a.slots.split(",")] if a.slots else list(range(a.n))
    except ValueError:
        ap.error("--stagger-after and --slots must contain comma-separated integers")
    if len(engine_slots) != a.n or len(set(engine_slots)) != a.n or any(n < 0 or n >= a.batch for n in engine_slots):
        ap.error("--slots must contain --n unique IDs from 0 through --batch minus 1")
    if stagger_after and (len(stagger_after) != a.n - 1 or
                          any(n < 1 or n >= a.max_new for n in stagger_after) or
                          any(b <= c for c, b in zip(stagger_after, stagger_after[1:]))):
        ap.error("--stagger-after needs --n minus one strictly increasing counts between 1 and --max-new minus 1")
    if a.promote_after and (not stagger_after or not 0 < a.promote_after < stagger_after[0]):
        ap.error("--promote-after requires --stagger-after and a positive count below its first admission count")
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    prompts = []
    for i, q in enumerate(QUESTIONS[: a.n]):
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if a.long_tokens and i == a.n - 1:
            para = ("The committee reviewed item %d of the long agenda, noting the budget, the schedule and the open "
                    "risks, and asked the staff to report back next quarter. ")
            per = max(1, len(tok.encode(para % 100)))
            filler = "".join(para % i for i in range(a.long_tokens // per + 1))
            text = f"<|im_start|>user\n{filler}\n\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prompts.append(tok.encode(text, parse_special=True))
    env = {"STRATA_DECODE_TIMING": "1", **({"STRATA_VERIFY_PROFILE": "1"} if os.environ.get("PROF") else {})}
    env["STRATA_IQ_MT_MIN"] = a.mt_min or None
    eng = Engine(a.exe, cfg, a.batch, env, shlex.split(a.extra))
    try:
        out = eng.lines()

        solo = []
        if not a.skip_solo:
            for i, ids in enumerate(prompts):
                eng.send(" ".join(x for x in ("GEN", str(a.max_new), a.keys, ",".join(map(str, ids))) if x))
                got, t0 = [], time.time()
                for line in out:
                    if line.startswith("T "):
                        got.append(int(line.split()[1]))
                    elif line.startswith("ERR"):
                        raise ProtocolError(f"solo {i}: {line}")
                    elif line.startswith("DONE"):
                        print(f"solo {i}: {len(got)} tokens in {time.time() - t0:.1f}s  {line[:60]}", flush=True)
                        break
                else:
                    raise ProtocolError(f"the engine ended before solo {i} completed")
                solo.append(got)

        got, stats = run_batch(eng, out, prompts, a.max_new, a.keys, stagger_after, a.promote_after, engine_slots)
        total = sum(len(v) for v in got.values())
        print(f"batch: {len(prompts)} slots, {total} tokens; admissions {stats['admission_seconds']:.1f}s; "
              f"aggregate {total / max(stats['total_seconds'], 1e-9):.1f} tok/s overall, "
              f"{total / max(stats['decode_seconds'], 1e-9):.1f} tok/s from the first batch token",
              flush=True)
        if stagger_after:
            for event in stats["admissions"]:
                print("admission:", json.dumps(event), flush=True)
        ok = True
        for i in range(len(prompts)):
            b = got[i]
            if solo:
                same = b == solo[i]
                ok &= same
                first_diff = next((k for k in range(min(len(b), len(solo[i]))) if b[k] != solo[i][k]),
                                  min(len(b), len(solo[i])) if len(b) != len(solo[i]) else None)
                print(f"slot {i}: {len(b)} tokens, solo {len(solo[i])}: {'IDENTICAL' if same else f'DIFFERS at {first_diff}'}")
            print("   ", ascii(tok.decode(b)[:160]))
        if a.dump:
            Path(a.dump).write_text(json.dumps({"solo": solo, "batch": [got[i] for i in range(len(prompts))],
                                              "stats": stats, "stagger_after": stagger_after,
                                              "promote_after": a.promote_after,
                                              "slots": engine_slots,
                                              "mt_min": a.mt_min, "extra": shlex.split(a.extra)}), encoding="utf-8")
        return 0 if ok else 2
    except ProtocolError as exc:
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        eng.close()


if __name__ == "__main__":
    sys.exit(main())
