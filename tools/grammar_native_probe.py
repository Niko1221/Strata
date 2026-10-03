"""G2 real native selector probe using the existing engine process and STOP/drain.

Only use an idle authorized GPU and a private config. This is a native protocol
probe, not HTTP/SDK evidence. Model output is never scripted. It deliberately
exercises raw frames; grammar_api_probe.py qualifies the G3 shared API argument.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from target_only_probe import sha256, tokenizer
from serve.frontend import ChatTemplate
from serve.server import StrataEngine, child_env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=False)
    report = {"result": "running", "scripted_model_output": False, "protocol": "native GENG1",
              "executable_sha256": sha256(cfg["exe"]), "args": cfg["args"], "cases": []}
    tok = tokenizer(cfg["tokenizer"])
    template = ChatTemplate(Path(cfg["tokenizer"]) / "chat_template.jinja")
    env = child_env(cfg)
    env["STRATA_TRACE"] = "1"
    env.pop("STRATA_SPEC_COUPLED", None)
    engine = None

    def record(name, **data):
        report["cases"].append({"name": name, **data})
        (a.out / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(name, json.dumps({k: v for k, v in data.items() if k != "tokens"}, ensure_ascii=False), flush=True)

    def prompt(text):
        return tok.encode(template.render([{"role": "user", "content": text}], enable_thinking=False), parse_special=True)

    def framed(grammar, ids, cap, sampling=None):
        if grammar is not None:
            raw = grammar.encode("utf-8")
            engine.proc.stdin.buffer.write(f"GENG1 {len(raw)}\n".encode("ascii") + raw + b"\n")
            engine.proc.stdin.buffer.flush()
        # This probe owns one process and issues requests sequentially. Production
        # uses frame + GEN atomically under the existing admission owner.
        return engine.generate(ids, cap, sampling or {}, threading.Event())

    def text(ids):
        return b"".join(tok.token_bytes(t) for t in ids if t not in (248044, 248046)).decode("utf-8", errors="strict")

    def generate(name, grammar, ids, cap=64, sampling=None):
        started = time.monotonic()
        output = [t for t in framed(grammar, ids, cap, sampling) if t is not None]
        last = dict(engine.last)
        assert len(output) == last["generated"] and len(output) <= cap
        assert last["drafts_offered"] == 0 and last["drafts_accepted"] == 0
        record(name, grammar=grammar, tokens=output, raw_bytes_hex=b"".join(tok.token_bytes(t) for t in output
                 if t not in (248044, 248046)).hex(), last=last, seconds=time.monotonic() - started)
        return output, last

    try:
        engine = StrataEngine(cfg["exe"], cfg["args"], cwd=cfg["cwd"], log=str(a.out / "engine.txt"), env=env)
        native_capability = engine.info["grammar"]
        assert native_capability in ("gbnf-v1", "gbnf-v2") and engine.info["decode_mode"] == "target"
        assert engine.info["mtp_loaded"] == 0 and engine.info["lookup"] == 0
        record("native capability", info=engine.info, pid=engine.proc.pid)
        ids = prompt("Output the answer to 2+2 as one digit. No explanation.")
        plain, plain_last = generate("unconstrained before grammar", None, ids, 16)
        literal = "begin: caf\u00e9 \U0001f408; end"
        source = 'root ::= "' + literal + '"'
        forced, last = generate("greedy Unicode literal", source, ids)
        assert text(forced) == literal and last["finish"] == "stop"
        sampling = {"temperature": 0.8, "top_k": 3, "top_p": 0.85, "min_p": 0.05,
                    "penalty_last_n": 64, "repetition_penalty": 1.2, "frequency_penalty": 0.2,
                    "presence_penalty": 0.1, "seed": 434}
        sampled, last = generate("sampled penalized literal", source, ids, sampling=sampling)
        assert text(sampled) == literal and last["finish"] == "stop"
        for cap in range(1, len(forced)):
            partial, last = generate("output budget " + str(cap), source, ids, cap)
            raw = b"".join(tok.token_bytes(t) for t in partial)
            assert literal.encode("utf-8").startswith(raw) and last["finish"] == "length"
            assert all(t not in (248044, 248046) for t in partial)
        for mode, params in (("greedy", {}), ("sampled", sampling)):
            output, last = generate(mode + " recursive balanced", 'root ::= "(" root ")" | "x"',
                                    prompt("Write ((x)). No explanation."), sampling=params)
            assert last["finish"] == "stop", "the fixed recursive test must finish within its budget"
            answer = text(output)
            assert re.fullmatch(r"\(*x\)*", answer) and answer.count("(") == answer.count(")")
        b, last = generate("grammar B after A", 'root ::= "false"', ids)
        assert text(b) == "false" and last["finish"] == "stop"
        again, last = generate("unconstrained after grammar", None, ids, 16)
        assert again == plain and last["finish"] == plain_last["finish"]
        for source_bad in ('root ::= missing', 'root ::= root', 'root ::= "x"\x00', 'root[temperature=0] ::= "x"'):
            try: list(framed(source_bad, ids, 8))
            except ValueError as error: record("invalid grammar before output", source=source_bad, error=str(error))
            else: raise AssertionError("invalid grammar was accepted")
        clean, last = generate("clean after grammar errors", 'root ::= "true"', ids)
        assert text(clean) == "true" and last["finish"] == "stop"
        long_text = "a " * 200
        for prefix in range(1, 5):
            output = []
            gen = framed('root ::= "' + long_text + '"', ids, 512)
            try:
                for token in gen:
                    if token is not None:
                        output.append(token)
                        if len(output) == prefix: break
            finally: gen.close()
            assert engine.last["finish"] == "cancel", engine.last
            assert long_text.encode().startswith(b"".join(tok.token_bytes(t) for t in output))
            record("cancel/drain prefix " + str(prefix), tokens=output, last=dict(engine.last))
            clean, last = generate("clean after cancel " + str(prefix), 'root ::= "true"', ids)
            assert text(clean) == "true" and last["finish"] == "stop"
        long_ids = prompt("Read this repeated input and say OK:\n" + "alpha beta gamma delta\n" * 330)
        gen = framed('root ::= "OK"', long_ids, 8)
        stopped = False
        try:
            for token in gen:
                if token is None and engine.progress and engine.progress[0] < engine.progress[1]:
                    stopped = True
                    break
                assert token is None, "prefill cancellation must precede output"
        finally: gen.close()
        assert stopped and engine.last["finish"] == "cancel" and engine.last["prompt_read"] < len(long_ids)
        record("prefill cancellation drained", last=dict(engine.last))
        clean, last = generate("clean after prefill cancel", 'root ::= "true"', ids)
        assert text(clean) == "true" and last["finish"] == "stop"
        old_pid = engine.proc.pid
        engine.restart()
        assert engine.proc.pid != old_pid and engine.info["grammar"] == native_capability
        restarted, last = generate("fresh matcher after native restart", source, ids)
        assert text(restarted) == literal and last["finish"] == "stop" and last["reused"] == 0
        record("native restart", old_pid=old_pid, new_pid=engine.proc.pid)
        engine.close()
        engine = None
        trace = (a.out / "engine.txt").read_text(encoding="utf-8")
        cursor, cursor_count, grammar_count = None, 0, 0
        for line in trace.splitlines():
            if "strata trace: CURSOR " in line:
                cursor = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)}
                assert cursor["window"] == 1 and cursor["consumed"] == cursor["prompt"] + cursor["produced"] - 1
                assert cursor["selection_position"] == cursor["consumed"] - 1
                cursor_count += 1
            elif "strata trace: GRAMMAR " in line:
                state = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)}
                assert cursor and state["tokens"] == cursor["produced"]
                assert not state["terminal"] or state["accepting"]
                grammar_count += 1
        assert grammar_count and grammar_count <= cursor_count
        record("native cursor and matcher relation", cursor_windows=cursor_count, grammar_windows=grammar_count)
        report["result"] = "pass"
    except Exception:
        report["result"] = "fail"
        report["failure"] = traceback.format_exc()
        raise
    finally:
        if engine is not None: engine.close()
        (a.out / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
