"""Real native G0 qualification. No MockEngine and no model downloads.

Use a private config with exe, args, cwd, tokenizer and optional lib_dirs.
The probe starts only that executable, writes artifacts to --out, and closes
its own process in finally. Run on an otherwise idle, authorized GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from serve.frontend import ChatTemplate
from serve.server import Service, StrataEngine, child_env
from strata_tokenizer import Tokenizer


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def tokenizer(path):
    path = Path(path)
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    merges = (path / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((path / "token_type.json").read_text(encoding="utf-8"))
    return Tokenizer(tokens, merges, types)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mode", choices=("target", "mtp", "suffix"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"mode": args.mode, "result": "running", "scripted_model_output": False,
              "executable_sha256": sha256(cfg["exe"]), "args": cfg["args"], "cases": []}
    engine = None
    log = args.out / "engine.txt"
    if log.exists():
        raise RuntimeError("use a fresh output directory; evidence must not be mixed across attempts")

    def record(name, **data):
        report["cases"].append({"name": name, **data})
        (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(name, json.dumps({k: v for k, v in data.items() if k != "tokens"}), flush=True)

    try:
        env = child_env(cfg)
        env["STRATA_TRACE"] = "1"
        env["STRATA_DECODE_TIMING"] = "1"
        env["STRATA_STATE_HASH"] = "1"
        env["STRATA_SNAPSHOT_VERIFY"] = "1"
        # Explicit test environment; coupled mode gets its own later qualification.
        env.pop("STRATA_SPEC_COUPLED", None)
        if args.mode == "target":
            for flags, message in [
                (["--spec", "1", "--mtp", "not-opened"], "requires no --mtp"),
                (["--spec", "1", "--suffix-draft", "3"], "positive --suffix-draft"),
                (["--spec", "1", "--conversation-cache-mib", "32"], "conversation-cache-mib 0"),
                (["--spec", "4"], "with --mtp DIR"),
            ]:
                c = subprocess.run([cfg["exe"], "--serve", *flags], env=env,
                                   capture_output=True, text=True, timeout=15)
                assert c.returncode == 2 and message in c.stderr, (flags, c.returncode, c.stderr)
                record("invalid mode rejected before load", flags=flags, exit_code=c.returncode)

        tok = tokenizer(cfg["tokenizer"])
        template = ChatTemplate(Path(cfg["tokenizer"]) / "chat_template.jinja")
        report["tokenizer_files"] = {name: sha256(Path(cfg["tokenizer"]) / name)
                                     for name in ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja")}
        engine = StrataEngine(cfg["exe"], cfg["args"], cwd=cfg["cwd"], log=str(log), env=env)
        assert engine.can_stop
        report["engine_info"] = dict(engine.info)
        initial_pid = engine.proc.pid
        if args.mode == "target":
            assert engine.info["decode_mode"] == "target"
            assert engine.info["spec"] == 1 and engine.info["lookup"] == 0
            assert engine.info["requested_lookup"] == 3, "exercise the default suffix setting"
            assert engine.info["mtp_loaded"] == 0 and float(engine.info["mtp_vram_mib"]) == 0
        else:
            assert engine.info["decode_mode"] == "mtp" and engine.info["mtp_loaded"] == 1
            assert engine.info["lookup"] == (3 if args.mode == "suffix" else 0)
        record("native startup", pid=initial_pid, info=engine.info)

        def prompt(text):
            return tok.encode(template.render([{"role": "user", "content": text}],
                                              enable_thinking=False), parse_special=True)

        def generate(name, ids, cap=16, sampling=None):
            started = time.monotonic()
            output = [t for t in engine.generate(ids, cap, sampling or {}, threading.Event()) if t is not None]
            last = dict(engine.last)
            assert 0 < len(output) <= cap and len(output) == last["generated"], last
            assert last["finish"] in ("stop", "length"), last
            if args.mode == "target":
                assert last["drafts_offered"] == 0 and last["drafts_accepted"] == 0, last
            record(name, tokens=output, text=tok.decode(output), last=last,
                   seconds=round(time.monotonic() - started, 3), pid=engine.proc.pid)
            return output, last

        ids = prompt("List the integers from 1 through 20, separated by commas. No explanation.")
        first, _ = generate("greedy first request", ids)
        follow, last = generate("live prefix continuation", ids + first, 8)
        # G5 applies the same retained boundary to ordinary speculation: no
        # invisible tail can replace the live prefix at an output cap.
        assert last["reused"] == len(ids) + len(first) - 1, last
        replay, _ = generate("prompt checkpoint rewind", ids)
        assert replay == first, {"first": first, "replay": replay}
        one, last = generate("one token budget", ids, 1)
        assert len(one) == 1 and last["finish"] == "length"
        sampling = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
                    "repetition_penalty": 1.05, "penalty_last_n": 64, "seed": 123}
        sample, _ = generate("sampled penalized request", ids, 20, sampling)
        repeated, _ = generate("sampled same seed replay", ids, 20, sampling)
        assert sample == repeated, {"sample": sample, "repeated": repeated}

        cancel = threading.Event()
        cancel_ids = prompt("List every integer from 1 through 500, separated by commas. No explanation.")
        gen = engine.generate(cancel_ids, 128, {}, cancel)
        seen = []
        try:
            for t in gen:
                if t is not None:
                    seen.append(t)
                    if len(seen) == 2:
                        cancel.set()
                        break
        finally:
            gen.close()  # existing STOP and output draining, not a new cancellation path
        assert engine.last["finish"] == "cancel", engine.last
        record("decode cancellation drained", received=len(seen), last=dict(engine.last))
        clean, _ = generate("request after cancellation", ids)
        assert clean == first

        if args.mode == "target":
            cancel = threading.Event()
            long_ids = prompt("Read this repeated input and say OK:\n" + "alpha beta gamma delta\n" * 330)
            gen = engine.generate(long_ids, 8, {}, cancel)
            try:
                for t in gen:
                    if t is None and engine.progress and engine.progress[0] < engine.progress[1]:
                        cancel.set()
                        break
            finally:
                gen.close()
            assert cancel.is_set() and engine.last["finish"] == "cancel", engine.last
            assert engine.last["prompt_read"] < len(long_ids), engine.last
            record("prefill cancellation drained", last=dict(engine.last))
            clean, _ = generate("request after prefill cancellation", ids)
            assert clean == first

        repeated_block = "def add(a, b):\n    return a + b\n\n"
        copy_ids = prompt("Repeat the following code block exactly six times. Output only code, no fences.\n" +
                          repeated_block * 6)
        _, last = generate("repeated code lookup exercise", copy_ids, 96)
        if args.mode != "target":
            assert last["drafts_offered"] > 0, last

        svc = Service(engine, tok, template)
        semantic_ids, thinking, cap = svc.prepare([{"role": "user", "content": "Say ready."}], None,
                                                  {"enable_thinking": False}, 16)
        events = list(svc.run(semantic_ids, thinking, None, cap, {}, threading.Event(), lifecycle=True))
        assert events[0][0] == "start" and events[-1][0] == "done"
        assert any(k == "event" and e.kind == "content" for k, e in events)
        assert not svc.status["busy"] and svc.status["queued"] == 0
        record("existing semantic service", kinds=[k if k != "event" else e.kind for k, e in events],
               done=events[-1][1])
        del svc

        if args.mode == "target":
            engine.restart()
            assert engine.proc.pid != initial_pid
            restarted, last = generate("real process restart", ids)
            assert last["reused"] == 0 and restarted == first, last
            record("restart identity", before_pid=initial_pid, after_pid=engine.proc.pid)

        engine.close()
        engine = None
        text = log.read_text(encoding="utf-8")
        cursors = [dict((k, int(v)) for k, v in re.findall(r"(\w+)=(-?\d+)", line))
                   for line in text.splitlines() if "strata trace: CURSOR " in line]
        assert cursors, "native cursor evidence missing"
        for c in cursors:
            if args.mode == "target":
                assert c["window"] == 1
            assert c["consumed"] == c["prompt"] + c["produced"] - 1, c
            assert c["selection_position"] == c["consumed"] - 1, c
        if args.mode == "target":
            assert "DRAFT_PREFILL" not in text and "strata mtp:" not in text
            assert "window up to 1 tokens" in text
        report["cursor_windows_checked"] = len(cursors)
        report["lookup_log"] = [line for line in text.splitlines() if "suffix drafts:" in line]
        if args.mode == "suffix":
            assert any(int(re.search(r"suffix drafts: (\d+) windows", line).group(1)) > 0
                       for line in report["lookup_log"]), "suffix proposals were configured but never exercised"
        report["result"] = "passed"
    except Exception as error:
        report["result"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        if engine is not None:
            engine.close()
        (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("G0", args.mode, "passed", flush=True)


if __name__ == "__main__":
    main()
