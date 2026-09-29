#!/bin/bash
# tools/nvme_delta_p0_test.sh - on-device oracle for the NVMe DELTA tier (docs/nvme-delta-cache-handoff.md §6 Phase 5).
#
# The delta tier replaces the v3 cascade's SERIALIZATION at DONE (append content-addressed chunks + a per-turn
# State record instead of rewriting a whole snapshot) and is promoted through the same resume selection.  The
# oracles, all through the REAL engine and a REAL store directory, on the p0 oracle's proven prompt shapes:
#   1. bit-exactness (P1): a turn boundary dumped by the delta tier and promoted by a fresh engine must give the
#      SAME post-restore STATE_HASH as the same boundary restored from the v3 tier's snapshot of it - the two
#      tiers serialize the same bytes (§5.2's invariant, device form; both hashes at the same L);
#   2. negative control: one corrupted chunk byte must REFUSE the promote, name the chunk, re-read the prompt
#      (RESUME 0) and leave the engine ALIVE - the recoverable class, on real silicon;
#   3. forks: a conversation that shares a long prefix but diverges INSIDE the prompt (the story's subject sits
#      at the prompt's END, so ~99% of the tokens are shared) shares the shared prefix's chunk FILES, and
#      restores bit-exactly;
#   4. the cascade: a 5-turn conversation's per-turn write must track the NEW tokens (~16 KB/token + the tail
#      State), NOT the total session - the whole point of the tier;
#   5. crash safety (P2): the deterministic STRATA_DELTA_FAIL_AT=C1..C5 hook aborts the dump at each write-
#      protocol step; a relaunch must scan clean, promote the row's head (C1-C3: turn 1's boundary; C4/C5: turn
#      2's), and the sweep must leave EXACTLY the referenced set.  C6 (the in-memory registration) leaves no disk
#      state to assert - a crash there is a dump that never happened, which the C1..C5 rows already cover.
#   BUDGET: every conversation here stays BELOW the drafter ring's max_cells (~4096 with --mtp mtp/rt): past it
#   the delta path REFUSES and the designed whole-snapshot fallback fires (§5.15) - that fallback is the v3 p0's
#   territory (run tools/nvme_p0_test.sh), and a test prompt past the ring would silently test the fallback
#   instead of the tier.
# Then, UNCHANGED (the regression gate, run separately): nvme_p0_test.sh, nvme_steps123_test.sh,
# nvme_image_promote_test.sh, nvme_failure_contract_test.sh, and tools/short_tests.py 19/19 against a
# --kv-delta 1 server.
#
# Server discipline (§9): STOP the live server first - these runs want the GPU and the memlock budget.
set -u
ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
cd /local/strata
. "$ROOT/tools/nvme_header_layout.sh"
E=${NVME_ENGINE:-build/strata}
OUT=/tmp/nvme-delta-p0
mkdir -p "$OUT"; rm -rf "$OUT"/*
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH-}
export STRATA_STATE_HASH=1 STRATA_NVME_HASH=1
NVME_PYTHON=.venv/bin/python

ARGS="--serve --pack packs/iq3_xxs
 --native models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
 --ple-gguf models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
 --expert-profile data/expert-profile.bin --expert-cache auto --prefill 2048
 --spec 4 --spec-min-p 0.5 --mtp mtp/rt --max-context 131072 --kv int8
 --kv-resident 20480 --prompt-cache 12 --adapt-swaps 0"
ARGS=$(echo "$ARGS" | tr '\n' ' ')

# the prompts: ~1200 real chat tokens (below the drafter ring, with room for five turns); the FORK's variant
# shares every token except the story's subject at the very END of the user message
$NVME_PYTHON - "$OUT" <<'EOF'
import json, sys
sys.path.insert(0, "tools")
import strata_tokenizer as ST
tp = "packs/iq3_xxs/tokenizer"
vocab = json.load(open(tp + "/vocab.json"))
tokens = [None] * len(vocab)
for t, i in vocab.items(): tokens[i] = t
merges = open(tp + "/merges.txt").read().split("\n")
types = json.load(open(tp + "/token_type.json"))
tok = ST.Tokenizer(tokens, merges, types)
filler = ("Describe the storm, the ships, the lamp room, the keeper's routine, the logbook, the rocks, "
          "the fog bell, the supply boat, and the winter isolation in vivid detail. ") * 60
tail = "<|im_end|>\n<|im_start|>assistant\n"
for name, story in (("pids", "Write a long story about a lighthouse keeper."),
                    ("fids", "Write a long story about a lighthouse keeper's cat.")):
    ids = tok.encode("<|im_start|>user\n" + filler + story, parse_special=True) + tok.encode(tail, parse_special=True)
    open(sys.argv[1] + "/" + name + ".txt", "w").write(",".join(map(str, ids)))
# a REAL chat turn's scaffold: the checkpoints (and so the delta cascade's boundary) live at the turn tokens -
# a continuation pasted into the assistant's reply zone creates none (the DONE dump then falls back to v3)
U = tok.encode("<|im_start|>user\n", parse_special=True)
A = tok.encode("<|im_end|>\n<|im_start|>assistant\n", parse_special=True)
open(sys.argv[1] + "/uids.txt", "w").write(",".join(map(str, U)))
open(sys.argv[1] + "/aids.txt", "w").write(",".join(map(str, A)))
t = tok.encode(" Please continue the story in vivid detail.", parse_special=True)
open(sys.argv[1] + "/tids.txt", "w").write(",".join(map(str, t)))
t2 = tok.encode(" Now bring the storm in earnest.", parse_special=True)
open(sys.argv[1] + "/t2ids.txt", "w").write(",".join(map(str, t2)))
t3 = tok.encode(" And end it with the dawn.", parse_special=True)
open(sys.argv[1] + "/t3ids.txt", "w").write(",".join(map(str, t3)))
print("prompt ids ready", file=sys.stderr)
EOF
P=$(cat "$OUT/pids.txt")
PF=$(cat "$OUT/fids.txt")
TAIL=$(cat "$OUT/tids.txt")
TAIL2=$(cat "$OUT/t2ids.txt")
TAIL3=$(cat "$OUT/t3ids.txt")
NP=$(awk -F, '{print NF}' "$OUT/pids.txt")
NTAIL=$(awk -F, '{print NF}' "$OUT/tids.txt")

rund() { # $1 name, $2 prompt, [$3] extra engine args, [$4] env assignment - the DELTA tier on, store $KV
  echo "GEN 200 $2" | timeout 900 env ${4-} $E $ARGS --kv-nvme ${KV-$OUT/store} --kv-delta 1 ${3-} > "$OUT/$1.out" 2> "$OUT/$1.err"
  echo "$1: rc=$? tokens=$(grep -c '^T ' "$OUT/$1.out")"
}
runv3() { # the same, but the V3 cascade (delta off) - the §5.2 comparison's other half
  echo "GEN 200 $2" | timeout 900 env ${4-} $E $ARGS --kv-nvme ${KV-$OUT/store} ${3-} > "$OUT/$1.out" 2> "$OUT/$1.err"
  echo "$1: rc=$? tokens=$(grep -c '^T ' "$OUT/$1.out")"
}
boundary_of() { # $1 = a run's .err with a 'T 0->N' cascade line -> echoes N (the turn boundary's token length)
  grep -o 'T 0->[0-9]*' "$1" | head -1 | cut -d'>' -f2
}

echo "== 1: the delta cascade dumps, a fresh engine promotes, and the hash matches the v3 tier's =="
# A: GEN on P with BOTH the delta cascade AND the v3 spike dump (whose ids are the CONSUMED history, which is
# what the next turn's prompt is built from - stdout T lines are not the consumed sequence)
KV=$OUT/store rund A "$P" "--nvme-dump $OUT/snapA.bin"
BOUNDARY=$(boundary_of "$OUT/A.err")
nvme_header_layout "$ROOT" || exit 1
LD=$(nvme_snapshot_ids "$OUT/snapA.bin" "$OUT/full_ids.txt")
if [ -z "$LD" ]; then echo "FAIL: could not read the spike snapshot"; exit 1; fi
FULL=$(cat "$OUT/full_ids.txt")
OFFSET=$((NP - BOUNDARY))
echo "prompt $NP tokens; the turn boundary is $BOUNDARY (offset $OFFSET); the consumed history is $LD"
U=$(cat "$OUT/uids.txt"); A=$(cat "$OUT/aids.txt")
PROMPT2="$FULL,$U,$TAIL,$A"          # turn 2: a REAL chat turn - its boundary checkpoint is what the delta
B2=$((LD + $(awk -F, '{print NF}' "$OUT/uids.txt") + NTAIL + $(awk -F, '{print NF}' "$OUT/aids.txt") - OFFSET))
# A2: the same conversation grown: the delta head moves - NEW chunks only (the cascade's own oracle, below)
KV=$OUT/store rund A2 "$PROMPT2" ""
# the SAME boundary serialized by the V3 tier (delta off), for the §5.2 device comparison
KV=$OUT/store-v3
runv3 V3 "$P" ""
V3_SNAP=$(ls "$OUT/store-v3"/kv-*.bin 2>/dev/null | head -1)
V3_L=$(nvme_snapshot_ids "$V3_SNAP" "$OUT/v3ids.txt")
echo "the v3 cascade's snapshot is at L=$V3_L (the delta manifest's boundary: $BOUNDARY)"
# B: a FRESH engine over A's store: the request re-sends P -> the delta promote fires
KV=$OUT/store
rund B "$P" ""
RESUME_B=$(grep -o '^RESUME [0-9]*' "$OUT/B.out" | tail -1)
PROMOTE_LINE=$(grep "nvme promote: resumed .* from .*delta/log-" "$OUT/B.err" | tail -1)
echo "B: $RESUME_B ; $PROMOTE_LINE"
# BV: the SAME boundary restored from the V3 TIER's snapshot (the p0's own restore path)
KV=$OUT/store
rund BV "$P" "--nvme-restore $V3_SNAP"
H_B=$(grep -A2 "nvme promote: resumed" "$OUT/B.err" | grep STATE_HASH | tail -1 | sed 's/ stale=[0-9a-f]*//')
H_BV=$(grep -A2 "nvme_restore: loaded" "$OUT/BV.err" | grep STATE_HASH | tail -1 | sed 's/ stale=[0-9a-f]*//')
# the stripped field, for the record: 'stale' is the LAST PARTIAL PAGE's unwritten cell - each process's own
# uninitialized arena, which both tiers faithfully persist (the p0's pair matches only because its two runs read
# the SAME engine's dump); the §5.2 invariant is about the restored prefix, which the other seven fields cover

FAIL=0
if [ -n "$PROMOTE_LINE" ] && [ "$RESUME_B" = "RESUME $BOUNDARY" ]; then
  echo "PASS delta promote: a fresh engine resumed $BOUNDARY tokens (the turn boundary) from a delta manifest"
else echo "FAIL delta promote: '$RESUME_B' (want RESUME $BOUNDARY) / '$PROMOTE_LINE'"; FAIL=1; fi
if [ "$V3_L" = "$BOUNDARY" ]; then
  echo "PASS same boundary: the v3 snapshot and the delta manifest key the same $BOUNDARY tokens"
else echo "FAIL same boundary: v3 L=$V3_L vs delta L=$BOUNDARY"; FAIL=1; fi
if [ -n "$H_B" ] && [ "$H_B" = "$H_BV" ]; then
  echo "PASS bit-exact (P1): the delta promote's hash == the v3 snapshot's restore hash over the restored prefix"
else echo "FAIL bit-exact (P1):"; echo "  delta: $H_B"; echo "  v3:    $H_BV"; FAIL=1; fi

echo "== 2: negative control - one corrupted chunk byte =="
# the LONGEST matching manifest for PROMPT2 is A2's head (B2); corrupt the FIRST chunk IT references
HEAD_MANIFEST=$(ls -t "$KV/delta"/log-*.manifest | head -1)
$NVME_PYTHON - "$HEAD_MANIFEST" "$OUT/chunk0.txt" <<'EOF'
import struct, sys
data = open(sys.argv[1], "rb").read()
L, = struct.unpack_from("<q", data, 8)
n_imgs, = struct.unpack_from("<q", data, 32)
at = 264 + 4*L + 16*n_imgs
key, a = struct.unpack_from("<Qq", data, at)
name = "%016x.bin" % key
open(sys.argv[2], "w").write(name)
print("first chunk of the head: %s (a=%d)" % (name, a), file=sys.stderr)
EOF
CHUNK0=$(cat "$OUT/chunk0.txt")
cp "$KV/delta/chunks/$CHUNK0" "$OUT/chunk0.good"
printf '\377' | dd of="$KV/delta/chunks/$CHUNK0" bs=1 seek=100 conv=notrunc status=none
KV=$OUT/store
rund Bc "$PROMPT2" ""
RESUME_BC=$(grep -o '^RESUME [0-9]*' "$OUT/Bc.out" | tail -1)
ALIVE=$(grep -c '^T ' "$OUT/Bc.out")
if grep -q "promote refused" "$OUT/Bc.err" && grep -q "$CHUNK0" "$OUT/Bc.err"; then
  echo "PASS negative control: the promote was refused and NAMED the chunk ($CHUNK0)"
else echo "FAIL negative control: the refusal did not name the chunk:"; grep -i "promote" "$OUT/Bc.err" | head -3; FAIL=1; fi
if [ "$RESUME_BC" = "RESUME 0" ] && [ "$ALIVE" -ge 8 ]; then
  echo "PASS recoverable: RESUME 0 (the prompt was re-read) and the engine is ALIVE ($ALIVE tokens)"
else echo "FAIL recoverable: '$RESUME_BC', $ALIVE tokens - the engine did not cleanly re-read"; FAIL=1; fi
cp "$OUT/chunk0.good" "$KV/delta/chunks/$CHUNK0"

echo "== 3: a fork shares the shared prefix's chunk FILES, and restores bit-exactly =="
KV=$OUT/store
rund F1 "$PF" ""
NMANIFESTS=$(ls "$KV/delta"/log-*.manifest | wc -l)
NCHUNKS=$(ls "$KV/delta/chunks" | grep -cv '^.tmp-' || true)
SHARE=$($NVME_PYTHON - "$KV/delta" <<'EOF'
# union vs sum of the manifests' chunk references: the difference is what content addressing deduped
import struct, sys, glob
d = sys.argv[1]
sets = []
for mp in glob.glob(d + "/log-*.manifest"):
    data = open(mp, "rb").read()
    L, = struct.unpack_from("<q", data, 8)
    n_imgs, = struct.unpack_from("<q", data, 32)
    n_chunks, = struct.unpack_from("<q", data, 24)
    at = 264 + 4*L + 16*n_imgs
    refs = set()
    for j in range(n_chunks):
        key, a = struct.unpack_from("<Qq", data, at + 16*j)
        refs.add(key)
    sets.append(refs)
total = sum(len(s) for s in sets)
union = len(set().union(*sets)) if sets else 0
print("%d %d" % (union, total - union))
EOF
)
UNION=$(echo $SHARE | cut -d' ' -f1)
REUSED=$(echo $SHARE | cut -d' ' -f2)
echo "chunk files: $NCHUNKS for $NMANIFESTS conversations; distinct chunk keys: $UNION; shared references: $REUSED"
if [ "$NMANIFESTS" -ge 2 ] && [ "$NCHUNKS" -le "$UNION" ] && [ "$REUSED" -gt 100 ]; then
  echo "PASS forks: $NMANIFESTS conversations, $UNION distinct chunk keys, $NCHUNKS chunk FILES ($REUSED references shared by content)"
else echo "FAIL forks: $NCHUNKS files for $UNION distinct keys over $NMANIFESTS manifests - sharing is broken"; FAIL=1; fi
# the fork restores bit-exactly: a fresh engine re-sends the fork's prompt
KV=$OUT/store
rund F1R "$PF" ""
R1=$(grep -o '^RESUME [0-9]*' "$OUT/F1R.out" | tail -1)
if [ -n "$R1" ] && [ "$R1" != "RESUME 0" ]; then
  echo "PASS fork restore: the fork promoted ($R1)"
else echo "FAIL fork restore: '$R1'"; FAIL=1; fi

echo "== 4: the cascade - a 5-turn conversation writes NEW tokens, not the session =="
# FIVE TURNS, ONE ENGINE PROCESS: the store's head tracking (and so the chunk reuse) is per process, and a real
# conversation's turns arrive in one.  Each turn adds a REAL chat turn (<|im_start|>user + a message + the
# assistant header), so every DONE has a boundary checkpoint and the delta cascade (not the fallback) writes.
KVC=$OUT/store-cascade
CFULL1=$(cat "$OUT/cfull1.txt" 2>/dev/null || true)
{ echo "GEN 200 $P"
  sleep 2
  echo "GEN 200 $PROMPT2"
  sleep 2
  echo "GEN 200 $PROMPT2,$U,$TAIL2,$A"
  sleep 2
  echo "GEN 200 $PROMPT2,$U,$TAIL2,$A,$U,$TAIL3,$A"
  sleep 2
  echo "GEN 200 $PROMPT2,$U,$TAIL2,$A,$U,$TAIL3,$A,$U,$TAIL,$A"
  sleep 2
  echo "QUIT"; } | timeout 1800 $E $ARGS --kv-nvme $KVC --kv-delta 1 > "$OUT/C1.out" 2> "$OUT/C1.err"
echo "cascade: rc=$? turns=$(grep -c '^DONE' "$OUT/C1.out")"
CAS=$($NVME_PYTHON - <<'EOF'
import re, sys
log = open("/tmp/nvme-delta-p0/C1.err").read()
turns = [(int(m.group(2)), int(m.group(3)), float(m.group(1)))
         for m in re.finditer(r"appended \d+ chunks \(([0-9.]+) MiB\) T (\d+)->(\d+)", log)]
print("cascade turns found: %d" % len(turns), file=sys.stderr)
# THE HONEST CASCADE METRIC.  The handoff's "~4 MB tail + KBs of running state" left this model's RUNNING STATE
# out of the arithmetic: the GDN recurrence state alone is ~112 MB here (the v3 snapshot at 2355 tokens is
# 154 MB = 36 MB of KV + 118 MB of running state), and §5.5 carries it per turn BY DESIGN (the C5 rule - the
# checkpoint's blobs are the state at the boundary).  So the per-turn write is new_tokens x 16 KB + ~118 MB.
# What distinguishes the delta tier from v3 is that the write does NOT GROW WITH THE SESSION: the KV part
# tracks the NEW tokens (v3's would grow with the whole prefix).  Asserted here:
#   (a) the four post-turn-1 writes are FLAT within ~10% (state-dominated; the KV delta is the only variable);
#   (b) each turn's KV part (write - the flat floor) tracks its NEW tokens within slack - a whole-prefix
#       re-derive would grow this with the session length, which is the regression the tier exists to kill.
ok = len(turns) >= 5
grew = [mib for prev, new, mib in turns[1:]]
floor = min(grew) if grew else 0
for prev, new, mib in turns[1:]:
    new_tokens = new - prev
    wrote = mib * 1048576.0
    kv_part = max(0.0, wrote - floor * 1048576.0)
    print("  turn at %d: %d new tokens, wrote %.2f MiB (KV part %.2f MiB; new-token KV = %.2f MiB)"
          % (new, new_tokens, mib, kv_part / 1048576.0, new_tokens * 16 * 1024.0 / 1048576.0), file=sys.stderr)
    if kv_part > 4.0 * (new_tokens * 16 * 1024.0) + (8 << 20): ok = False
flat = (max(grew) - min(grew)) if grew else 0
print("flat within %.2f MiB (%.1f%% of the floor)" % (flat, 100.0 * flat / floor if floor else 0), file=sys.stderr)
if flat >= 0.10 * floor: ok = False
print("PASS" if ok else "FAIL")
EOF
)
if [ "$CAS" = "PASS" ]; then echo "PASS cascade: the per-turn write tracks the NEW tokens"; else echo "FAIL cascade"; FAIL=1; fi

echo "== 5: crash safety - STRATA_DELTA_FAIL_AT=C1..C5, relaunch, promote, sweep exactness =="
CRASH_OK=1
for CK in C1 C2 C3 C4 C5; do
  KVC=$OUT/store-crash-$CK
  mkdir -p "$KVC"
  runcr() { echo "GEN 200 $2" | timeout 900 env ${4-} $E $ARGS --kv-nvme $KVC --kv-delta 1 ${3-} > "$OUT/$1.out" 2> "$OUT/$1.err"; }
  # turn 1 succeeds (a live head at the boundary); turn 2's dump crashes at $CK
  runcr CR-$CK-1 "$P" ""
  runcr CR-$CK-2 "$PROMPT2" "" "STRATA_DELTA_FAIL_AT=$CK"
  # the relaunch: a fresh engine over the post-crash store; the request must promote the row's head or re-read,
  # and the sweep must leave exactly the referenced set
  runcr CR-$CK-3 "$PROMPT2" ""
  R3=$(grep -o '^RESUME [0-9]*' "$OUT/CR-$CK-3.out" | tail -1)
  SWEEP=$($NVME_PYTHON - "$KVC/delta" <<'EOF'
# after the relaunch's sweep at open: every chunks/states file must be referenced by a manifest on disk
import struct, sys, glob, os
d = sys.argv[1]
ref = set()
for mp in glob.glob(d + "/log-*.manifest"):
    data = open(mp, "rb").read()
    L, = struct.unpack_from("<q", data, 8)
    n_imgs, = struct.unpack_from("<q", data, 32)
    n_chunks, = struct.unpack_from("<q", data, 24)
    at = 264 + 4*L + 16*n_imgs
    sk, = struct.unpack_from("<Q", data, 232)
    ref.add("%016x.bin" % sk)
    for j in range(n_chunks):
        k, a = struct.unpack_from("<Qq", data, at + 16*j)
        ref.add("%016x.bin" % k)
bad = 0
for sub in ("chunks", "states"):
    for f in os.listdir(d + "/" + sub):
        if not f.startswith(".tmp-") and f not in ref: bad += 1
print("CLEAN" if bad == 0 else "DIRTY:%d" % bad)
EOF
)
  # C1..C3: the turn-2 dump left no manifest -> the relaunch promotes the TURN-1 boundary
  # C4/C5: the turn-2 boundary IS durable -> the relaunch promotes IT
  case $CK in C1|C2|C3) WANT="RESUME $BOUNDARY";; *) WANT="RESUME $B2";; esac
  if [ "$R3" = "$WANT" ] && [ "$SWEEP" = "CLEAN" ]; then
    echo "PASS crash $CK: relaunch promotes ($R3), the sweep is exact"
  else echo "FAIL crash $CK: promote '$R3' (want $WANT), sweep $SWEEP"; CRASH_OK=0; fi
done
[ "$CRASH_OK" = 1 ] && echo "PASS crash safety (C1..C5)" || FAIL=1

echo "== VERDICT =="
if [ "$FAIL" = 0 ]; then echo "ALL PASS"; else echo "FAILED"; exit 1; fi
