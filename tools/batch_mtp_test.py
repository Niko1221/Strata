#!/usr/bin/env python3
"""GPU regression for shared MTP admission and the last row of a batch slot.

Needs a model config with --mtp and --spec >= 2. Also run batch_test.py and
batch_interleave_test.py with --extra "--batch-mtp ..." for token parity/reuse.
Repeat with --extra "--mtp-q4 all --mtp-hnorm stream" to cover shared projections.
"""
import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

from batch_test import Engine, QUESTIONS, tokenizer
from batch_interleave_test import run


def batch(eng, out, prompts, limits):
    got, done = {}, set()
    for slot, (prompt, limit) in enumerate(zip(prompts, limits)):
        first, active = run(eng, out, f"BGEN {slot} {limit} {','.join(map(str, prompt))}", slot=slot)
        got[slot] = first
        if not active:
            done.add(slot)
    while eng.pending or len(done) < len(prompts):
        line = eng.pending.pop(0) if eng.pending else next(out)
        if line.startswith("BT "):
            _, slot, token = line.split()
            got[int(slot)].append(int(token))
        elif line.startswith("BDONE "):
            fields = line.split()
            slot = int(fields[1])
            assert fields[3] == "length", line
            assert int(fields[2]) == len(got[slot]), line
            done.add(slot)
        elif line.startswith("ERR"):
            raise AssertionError(line)
    return [got[s] for s in range(len(prompts))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    prompts = [tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
                          "<think>\n\n</think>\n\n", parse_special=True) for q in QUESTIONS[:4]]
    context = 512
    extra = shlex.split(a.extra) + ["--batch-mtp", "--max-context", str(context), "--kv-resident", "0",
                                   "--eos-ids", "2147483647", "--pcie-frac", "0", "--adapt-every", "1000000",
                                   "--no-prefill-borrow"]
    eng = Engine(a.exe, cfg, 4, {"STRATA_IQ_MT_MIN": "1"}, extra)
    eng.pending = []
    out = eng.lines()
    try:
        refs = [run(eng, out, f"GEN 64 {','.join(map(str, p))}")[0] for p in prompts]
        got = batch(eng, out, prompts, [64] * 4)
        assert got == refs, "four-slot MTP differs from solo"
        print("four slots: 64 tokens each equal solo", flush=True)

        limits = [1, 2, 3, 4]
        got = batch(eng, out, prompts, limits)
        assert got == [r[:n] for r, n in zip(refs, limits)], "short output limit differs from solo"
        print("output limits 1, 2, 3, 4: equal solo prefixes", flush=True)

        # GEN's admission leaves nine guard tokens; BGEN's continuation reaches the
        # final context cell. An out-of-vocabulary EOS ID prevents an early stop.
        filler = tok.encode("Continue counting: one two three four five six seven eight nine ten. " * 64)
        long = [filler[:context - 10 - s] for s in range(4)]
        got = batch(eng, out, long, [64] * 4)
        assert [len(v) for v in got] == [context - len(p) + 1 for p in long], "context limit not reached"
        print("four slots reach the final context cell without staging past it", flush=True)
    finally:
        if eng.p.poll() is None:
            eng.send("QUIT")
            try:
                eng.p.wait(timeout=180)
            except subprocess.TimeoutExpired:
                eng.p.kill()
                eng.p.wait()
                raise
        eng.log.close()
    assert eng.p.returncode == 0, f"engine exited {eng.p.returncode}"
    log = Path(eng.log_path).read_text(errors="replace")
    accepted = re.findall(r"strata batch MTP: drafts accepted (\d+) of (\d+)", log)
    assert accepted and sum(int(a) for a, _ in accepted) > 0, "no batch MTP proposals accepted"
    print("batch MTP acceptance:", accepted)


if __name__ == "__main__":
    main()
