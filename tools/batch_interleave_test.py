#!/usr/bin/env python3
"""Batch slots, the parts batch_test.py does not reach (#465): a long prompt read WHILE other slots decode (its
chunks interleaved with their windows), and a conversation's next turn continued from the slot that holds it.

  1. solo references (GEN): A, then A's next turn (from the live session), B, C (a long prompt)
  2. the same in the slots: A in slot 0, B in slot 1 next to it, then C in slot 2 while A and B decode - every slot's
     tokens must equal its solo tokens
  3. A's next turn (BGEN into slot 0, which still holds A): read from the slot, its tokens must equal the solo next
     turn, and the reuse (REUSED) must be the FULL resident slot-0 prefix - the L3/disk tier must not override a
     valid slot with a shorter record (a performance bug the token parity cannot see)
 4. a long prompt D gives way (BYIELD) at a chunk boundary while its read is in flight: the BYIELD is sent after
    the first progress PP as the server sends it (a BYIELD before the read starts is dropped), D's part read waits
    in its slot, a short E is then admitted, and D goes on from its slot: both equal their solo tokens
  6. A in a slot, stopped after 50 tokens and continued on the solo path (GEN of A + its tokens, from the slot): the
     whole equals solo A (what the server does with a request left alone in a slot)
  5. (a measurement, also a check) a solo next turn continued from slot 0: the drafts accepted, and the reuse must
     be the FULL resident slot-0 prefix again (the same L3-over-slot regression as step 3)
  7. F, A's history as a client sends it back without the reply's thinking, from slot 1's turn checkpoint: equal to
     F solo (which continues from the same checkpoint of the live chain)

Exact comparisons need the same settings as batch_test.py (this script sets STRATA_IQ_MT_MIN=1):
  python tools/batch_interleave_test.py --exe build/strata --config strata-<model>.json \\
      --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
Without --no-prefill-borrow the slots decoding during C's read see the expert cache without the slots C's prompt
borrowed (those experts run on the CPU, which rounds differently): A's and B's text may then drift from solo.
"""
import argparse, atexit, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import Engine, tokenizer  # noqa: E402

LONG = ("The history of mathematics is long and full of surprising turns. Early civilizations counted with tally marks, "
        "then with symbols, and later with place-value systems that made arithmetic far easier. ")


def run(eng, out, line, slot=None):
    """Send one GEN/BGEN; returns (tokens of the request line, the BADM flag) - BT/BDONE of other slots are kept."""
    eng.send(line)
    got = []
    for l in out:
        if l.startswith("T "):
            got.append(int(l.split()[1]))
        elif l.startswith(("BT ", "BDONE ")):
            eng.pending.append(l)
        elif l.startswith("ERR"):
            raise SystemExit("engine: " + l)
        elif l.startswith("REUSED "):
            eng.reused = int(l.split()[1])   # the cached prompt prefix the read continued from (the harness asserts it)
        elif slot is None and l.startswith("DONE"):
            return got, None
        elif slot is not None and l.startswith("BADM "):
            return got, l.split()[2] == "1"
    raise SystemExit(f"the engine ended during: {line[:40]} (see {eng.log_path})")


def stop_engine(eng):
    """End the engine on every path.  The checks and run() raise SystemExit (an engine error, a failed
    comparison, a preemption that did not happen); without this the engine process was left running on the
    GPUs with no QUIT when a check failed."""
    if eng is None:
        return
    try:
        eng.send("QUIT")
        eng.p.wait(timeout=60)
    except Exception:
        try:
            eng.p.kill()
        except Exception:
            pass
        try:
            eng.p.wait(timeout=10)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--long", type=int, default=5000, help="tokens of the long prompt C (several prompt chunks)")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])

    def chat(q):
        return tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                          parse_special=True)
    body = LONG * (a.long // max(1, len(tok.encode(LONG))) + 1)
    A = chat("Write a Python function that merges overlapping intervals, then explain it.")
    B = chat("Compare TCP and QUIC: handshake, congestion control, multiplexing.")
    C = chat(body + "\nSummarize the text above in three sentences.")
    # Use a different prefix so C's prompt checkpoints cannot reduce D to one fresh chunk.
    # BYIELD must interrupt a multi-chunk read, not an already cached prompt.
    D = chat("Read this separate document:\n" + body + "\nWhat are the three most important ideas in the text above?")
    E = chat("Describe the life cycle of a star like the Sun.")
    print(f"prompts: A {len(A)}, B {len(B)}, C {len(C)} tokens", flush=True)
    eng = Engine(a.exe, cfg, 3, {"STRATA_IQ_MT_MIN": "1"}, a.extra.split())
    eng.pending = []
    out = eng.lines()
    atexit.register(lambda: stop_engine(eng))   # a failed check or an engine error must not leave the engine running
    ids = lambda v: ",".join(map(str, v))
    M = a.max_new

    # 1. solo
    sA, _ = run(eng, out, f"GEN {M} {ids(A)}")
    A2 = A + sA + chat("Now do the same in Rust.")
    sA2, _ = run(eng, out, f"GEN {M} {ids(A2)}")
    # F: A's history as a client sends it back WITHOUT the reply's thinking - it shares A only up to A's last turn
    # boundary (the checkpoint there); solo it continues from that checkpoint of the live chain
    turn_at = max(i for i, t in enumerate(A) if t == A[0])          # A[0] is <|im_start|>
    F = A[:turn_at] + tok.encode("<|im_start|>assistant\nA short answer.<|im_end|>\n", parse_special=True) + \
        chat("Now in Go.")
    sF, _ = run(eng, out, f"GEN {M} {ids(F)}")
    sB, _ = run(eng, out, f"GEN {M} {ids(B)}")
    sC, _ = run(eng, out, f"GEN {M} {ids(C)}")
    sD, _ = run(eng, out, f"GEN {M} {ids(D)}")
    sE, _ = run(eng, out, f"GEN {M} {ids(E)}")
    print(f"solo: A {len(sA)}, A2 {len(sA2)}, B {len(sB)}, C {len(sC)}, D {len(sD)}, E {len(sE)} tokens", flush=True)

    # 2. the slots: A, B, then the long C while they decode
    got = {0: [], 1: [], 2: []}
    done = {}
    t0 = time.time()
    for s, P in ((0, A), (1, B), (2, C)):
        first, cont = run(eng, out, f"BGEN {s} {M} {ids(P)}", slot=s)
        got[s] += first
        if not cont:
            done[s] = True
    t_admit = time.time() - t0
    during_c = sum(1 for l in eng.pending if l.startswith("BT "))   # windows the slots ran while C was admitted

    def drain():
        while eng.pending or len(done) < 3:
            l = eng.pending.pop(0) if eng.pending else next(out)
            if l.startswith("BT "):
                _, s, y = l.split()
                got[int(s)].append(int(y))
            elif l.startswith("BDONE "):
                done[int(l.split()[1])] = True
            elif l.startswith("ERR"):
                raise SystemExit("engine: " + l)
    drain()
    print(f"slots: admissions {t_admit:.1f}s; {during_c} slot tokens arrived while the admissions read their prompts",
          flush=True)
    ok = True
    for s, ref, name in ((0, sA, "A"), (1, sB, "B"), (2, sC, "C")):
        same = got[s] == ref
        ok &= same
        d = next((k for k in range(min(len(got[s]), len(ref))) if got[s][k] != ref[k]), None)
        print(f"slot {s} ({name}): {len(got[s])} tokens, solo {len(ref)}: "
              f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)

    # 3. A's next turn from slot 0 (the engine's log says "slot 0 gave back ...")
    first, cont = run(eng, out, f"BGEN 0 {M} {ids(A2)}", slot=0)
    # A's turn stayed resident in slot 0 (its prompt + its reply without the reply's last token - the live head):
    # the read must continue from that FULL prefix.  The L3/disk tier must never win the metadata selection with a
    # shorter record (REUSED then drops from the slot's own length to the disk record's) - a performance bug the
    # token parity cannot see, and a hard failure here.
    A2_reused = getattr(eng, "reused", -1)
    a2_slot = len(A) + len(sA) - 1
    if A2_reused < a2_slot:
        print(f"A2's admission reused {A2_reused} cached tokens, but slot 0 held the full {a2_slot}-token prefix "
              f"of it (the shorter L3/disk record won the metadata selection): A2's prompt was re-read past the "
              f"resident slot", flush=True)
        raise SystemExit(2)
    print(f"A2's admission: reused {A2_reused} cached tokens (the full slot-0 prefix is {a2_slot})", flush=True)
    got2, done = first, {}
    if cont:
        while True:
            l = eng.pending.pop(0) if eng.pending else next(out)
            if l.startswith("BT 0 "):
                got2.append(int(l.split()[2]))
            elif l.startswith("BDONE 0 "):
                break
    same = got2 == sA2
    ok &= same
    d = next((k for k in range(min(len(got2), len(sA2))) if got2[k] != sA2[k]), None)
    print(f"A's next turn from its slot: {len(got2)} tokens, solo {len(sA2)}: "
          f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)
    # 4. a long prompt D gives way (BYIELD) while its read is in flight, then goes on from its slot.  The server
    # sends BYIELD only after the admission's read has started (serve/server.py): a BYIELD that arrives before the
    # read is dropped by the engine between requests (generate.cpp: "for a prompt read that has ended meanwhile").
    # The BYIELD comes after the first progress PP (the read is in flight, past a chunk boundary): the sweep at the
    # next chunk boundary parks the part read in slot 2 - YIELDED 2 <tokens>, then DONE cancel - D's part read
    # waits in its slot, a short E is then admitted, and D goes on from its slot while E decodes: both equal their
    # solo tokens.  A read that does not give way is a hard failure here (the next BGEN into its slot would
    # otherwise hit an active slot), and the engine is stopped on it.  Until the engine fix that keeps the chunk
    # sweep live when a layer-split read starts with every slot idle (generate.cpp read_part), this step fails on
    # the old engine by design.
    eng.send(f"BGEN 2 {M} {ids(D)}")
    yielded = done_cancel = None
    sent_yield = False
    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            l = eng.pending.pop(0) if eng.pending else next(out)
        except StopIteration:
            raise SystemExit(f"the engine ended during step 4 - see {eng.log_path}")
        if l.startswith("YIELDED 2 "):
            yielded = l
        elif l.startswith("DONE ") and yielded is not None:
            done_cancel = l
        elif not sent_yield and l.startswith("PP "):
            pf = l.split()
            if int(pf[1]) < int(pf[2]):            # the read is in flight, past a chunk boundary
                eng.send("BYIELD 2")
                sent_yield = True
                print(f"  D's read in flight at {pf[1]} of {pf[2]} tokens; BYIELD 2 sent", flush=True)
        # anything else (BT/BDONE/T, REUSED, ...) is consumed right here, never re-queued: step 4 runs D alone
        # (no other slot decodes), and the loop pops eng.pending first, so a re-queued line would spin until the
        # deadline.  D's own tokens and the other slots' BT lines come after BADM, read by run()/drain below.
        elif l.startswith("ERR"):
            raise SystemExit("engine: " + l)
        elif l.startswith("BADM 2 "):
            break
    if yielded is None or done_cancel is None:
        print(f"D gave way: {yielded}", flush=True)
        print("step 4: FAILED - the engine did not preempt D's read (BYIELD 2 was sent after its first progress "
              "PP, while the read was in flight at a chunk boundary; it read D to BADM anyway)", flush=True)
        logp = Path(eng.log_path)
        if logp.exists():
            for l in logp.read_text(errors="replace").splitlines():
                if "gives way" in l or "not taken" in l:
                    print("  log:", l[:200], flush=True)
        raise SystemExit(2)
    print(f"D gave way: {yielded} (DONE {done_cancel.split()[1]} generated, {done_cancel.split()[5]} finish)",
          flush=True)
    got4 = {1: [], 2: []}
    first, cont1 = run(eng, out, f"BGEN 1 {M} {ids(E)}", slot=1)
    got4[1] += first
    first, cont2 = run(eng, out, f"BGEN 2 {M} {ids(D)}", slot=2)
    got4[2] += first
    done4 = {s for s, c in ((1, cont1), (2, cont2)) if not c}
    while len(done4) < 2:
        try:
            l = eng.pending.pop(0) if eng.pending else next(out)
        except StopIteration:
            raise SystemExit(f"the engine ended during step 4 - see {eng.log_path}")
        if l.startswith("BT "):
            _, s, y = l.split()
            if int(s) in got4:
                got4[int(s)].append(int(y))
        elif l.startswith("BDONE "):
            s = int(l.split()[1])
            if s in got4:
                done4.add(s)
    for s, ref, name in ((1, sE, "E (short, admitted while D waited)"), (2, sD, "D (gave way, then went on)")):
        same = got4[s] == ref
        ok &= same
        d = next((k for k in range(min(len(got4[s]), len(ref))) if got4[s][k] != ref[k]), None)
        print(f"slot {s} {name}: {len(got4[s])} tokens, solo {len(ref)}: "
              f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)
    # 6. back to the solo path (what the server does with a request left alone in a slot): A in slot 1, BSTOP after
    # 50 tokens, then GEN of A + what it produced - the engine continues from the slot with MTP drafts again
    first, cont = run(eng, out, f"BGEN 1 {M} {ids(A)}", slot=1)
    got6, sent = list(first), False
    while cont:
        l = eng.pending.pop(0) if eng.pending else next(out)
        if l.startswith("BT 1 "):
            got6.append(int(l.split()[2]))
            if len(got6) >= 50 and not sent:
                eng.send("BSTOP 1")
                sent = True
        elif l.startswith("BDONE 1 "):
            break
    tail6, _ = run(eng, out, f"GEN {M - len(got6)} {ids(A + got6)}") if len(got6) < M else ([], None)
    same = got6 + tail6 == sA
    ok &= same
    both = got6 + tail6
    d = next((k for k in range(min(len(both), len(sA))) if both[k] != sA[k]), None)
    print(f"A in a slot, then solo again after {len(got6)} tokens: {len(both)} tokens, solo {len(sA)}: "
          f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)

    # 5. a SOLO next turn continued from slot 0: its tokens are exact either way, but the draft layer's own K/V was
    # built for another conversation, so fewer drafts may be accepted than in a solo next turn whose drafter read the
    # conversation (A2 in step 1).  It must ALSO reuse the FULL resident slot-0 prefix (A2's turn, without its last
    # token) - the L3/disk tier must not win with a shorter record, the same performance regression as step 3.
    A3 = A2 + got2 + chat("Now in Go.")
    eng.send(f"GEN {M} {ids(A3)}")
    for l in out:
        if l.startswith(("BT ", "BDONE ")):
            eng.pending.append(l)
        elif l.startswith("REUSED "):
            eng.reused = int(l.split()[1])
        elif l.startswith("DONE"):
            f = l.split()
            print(f"solo next turn from a slot: drafts accepted {f[6]} of {f[7]} ({f[1]} tokens in {f[4]} ms)",
                  flush=True)
            break
    A3_reused = getattr(eng, "reused", -1)
    a3_slot = len(A2) + len(got2) - 1
    if A3_reused < a3_slot:
        print(f"the solo next turn (A3) reused {A3_reused} cached tokens, but slot 0 held the full "
              f"{a3_slot}-token prefix of it (the shorter L3/disk record won the metadata selection): A3 was "
              f"re-read past the resident slot", flush=True)
        raise SystemExit(2)
    print(f"solo next turn: reused {A3_reused} cached tokens (the full slot-0 prefix is {a3_slot})", flush=True)
    # 7. F from slot 1's turn checkpoint (slot 1 holds A since step 6): the engine log says "(its turn checkpoint)"
    first, cont = run(eng, out, f"BGEN 2 {M} {ids(F)}", slot=2)
    got7 = list(first)
    while cont:
        l = eng.pending.pop(0) if eng.pending else next(out)
        if l.startswith("BT 2 "):
            got7.append(int(l.split()[2]))
        elif l.startswith("BDONE 2 "):
            break
    same = got7 == sF
    ok &= same
    d = next((k for k in range(min(len(got7), len(sF))) if got7[k] != sF[k]), None)
    print(f"F (A's history without the reply's thinking) from slot 1's turn checkpoint: {len(got7)} tokens, solo "
          f"{len(sF)}: {'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)
    eng.send("QUIT")
    eng.p.wait(timeout=180)
    log = Path(eng.log_path).read_text(errors="replace")
    for l in log.splitlines():
        if "drafts accepted" in l:
            print("  log:", l.split("strata serve: ")[-1][:160], flush=True)
    for key in ("gave back", "its turn checkpoint", "takes", "the prompt was read in", "gives way"):
        print(f"engine log '{key}': {log.count(key)} lines", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
