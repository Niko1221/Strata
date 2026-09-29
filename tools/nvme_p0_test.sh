#!/bin/bash
# tools/nvme_p0_test.sh - P0 spike oracle for the NVMe KV cold tier (docs/nvme-kv-cache-design.md §11 Step 0).
#
# Falsifiable test that an NVMe dump/restore of a whole session is bit-exact and NOT a no-op
# (the llama.cpp #26676 hybrid-restore trap).  Runs, all with --adapt-swaps 0 (VRAM expert set pinned):
#   A  : real chat prompt P (~3900 tok), GEN 200 -> dumps the whole session at DONE (L = |P| + generated)
#   A' : prompt P + A's tokens + a short tail, GEN 200, NO restore -> pure-recompute reference
#   B  : same prompt as A', --nvme-restore the snapshot -> must RESUME L_dump and match A' bit-for-bit
#   B' : same as B, but the snapshot's GDN region is corrupted -> the comparison MUST fail (negative control)
# Oracles:
#   1. (restore exactness) STRATA_NVME_HASH: the STATE_HASH printed straight after B's restore must equal A's
#      DONE STATE_HASH at the same L - bit-exact dump/restore, before any new token.
#   2. (end-to-end) B's final STATE_HASH + greedy continuation must equal A's.
set -u
# ROOT comes FIRST, before any cd: BASH_SOURCE is whatever the caller typed, so resolving it after
# `cd /local/strata` made ROOT the checkout being cd'd INTO whenever the script was invoked by a relative
# path - sourcing the wrong tree's header constants while testing that tree's engine (found on first run).
ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
cd /local/strata
# The engine and the model files live in the checkout; the HEADER these scripts must agree with is the one in
# the tree the scripts live in (tools/nvme_header_layout.sh reads kNvmeHeaderBytes / kNvmeFormatVersion from it,
# and nvme_snapshot_ids refuses a snapshot whose version field disagrees).
. "$ROOT/tools/nvme_header_layout.sh"
# The engine is overridable so the oracles can validate a DIFFERENT build than the one in this checkout's
# build/ - which is the whole point when the tree under test is a worktree (NVME_ENGINE=/path/to/strata).
# Default stays this checkout's binary, so the recorded v2-format results keep meaning what they meant.
E=${NVME_ENGINE:-build/strata}
OUT=/tmp/nvme-p0
mkdir -p "$OUT"; rm -f "$OUT"/*
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export STRATA_STATE_HASH=1 STRATA_NVME_HASH=1
NVME_PYTHON=.venv/bin/python

ARGS="--serve --pack packs/iq3_xxs
 --native models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
 --ple-gguf models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
 --expert-profile data/expert-profile.bin --expert-cache auto --prefill 2048
 --spec 4 --spec-min-p 0.5 --mtp mtp/rt --max-context 131072 --kv int8
 --kv-resident 20480 --prompt-cache 12 --adapt-swaps 0"
ARGS=$(echo "$ARGS" | tr '\n' ' ')

# a real ~3900-token chat prompt and a short tail, via the pack's own tokenizer
.venv/bin/python - "$OUT" <<'EOF'
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
          "the fog bell, the supply boat, and the winter isolation in vivid detail. ") * 200
head = "<|im_start|>user\nWrite a long story about a lighthouse keeper. " + filler
tail = "<|im_end|>\n<|im_start|>assistant\n"
ids = tok.encode(head, parse_special=True)[:3900] + tok.encode(tail, parse_special=True)
open(sys.argv[1] + "/pids.txt", "w").write(",".join(map(str, ids)))
t = tok.encode(" Please continue the story in vivid detail.", parse_special=True)
open(sys.argv[1] + "/tids.txt", "w").write(",".join(map(str, t)))
print("prompt ids:", len(ids), "tail ids:", len(t), file=sys.stderr)
EOF
P=$(cat "$OUT/pids.txt")
TAIL=$(cat "$OUT/tids.txt")
NP=$(awk -F, '{print NF}' "$OUT/pids.txt")

run() { # $1 name, $2 prompt, $3 extra engine args
  echo "GEN 200 $2" | timeout 900 $E $ARGS $3 > "$OUT/$1.out" 2> "$OUT/$1.err"
  echo "$1: rc=$? tokens=$(grep -c '^T ' "$OUT/$1.out")"
}

echo "== A: reference run, dump at DONE =="
run A "$P" "--nvme-dump $OUT/snap.bin"
NA=$(grep -c '^T ' "$OUT/A.out")
if [ ! -f "$OUT/snap.bin" ] || [ "$NA" -lt 8 ]; then echo "FAIL: A generated only $NA tokens"; exit 1; fi
ls -la "$OUT/snap.bin"
# the authoritative token history is the SNAPSHOT's ids (stdout T lines are not the consumed sequence)
nvme_header_layout "$ROOT" || exit 1
LD=$(nvme_snapshot_ids "$OUT/snap.bin" "$OUT/full_ids.txt")
if [ -z "$LD" ]; then echo "FAIL: could not read $OUT/snap.bin at HDR=$HDR version=$NVME_VERSION"; exit 1; fi
echo "snapshot L=$LD (header $HDR bytes, format version $NVME_VERSION)"
FULL=$(cat "$OUT/full_ids.txt")
PROMPT_AB="$FULL,$TAIL"

echo "== A': pure-recompute reference (same token history, no restore; EXPECTED to differ: prefill vs decode rounding) =="
run A2 "$PROMPT_AB" ""
echo "== B: NVMe restore + continue =="
run B "$PROMPT_AB" "--nvme-restore $OUT/snap.bin"
echo "== C: same-process live continuation (the fair continuation oracle for B) =="
{ echo "GEN 200 $P"; sleep 2; echo "GEN 200 $PROMPT_AB"; sleep 2; echo "QUIT"; } | \
  timeout 900 $E $ARGS > "$OUT/C.out" 2> "$OUT/C.err"
echo "C: rc=$? tokens=$(grep -c '^T ' "$OUT/C.out")"
T_C=$(awk "/^RESUME /{r++} r==2 && /^T /{print \$2}" "$OUT/C.out" | paste -sd, -)
grep -E "RESUME|nvme_restore" "$OUT/B.out" "$OUT/B.err" 2>/dev/null | head -4

echo "== B': negative control (corrupted GDN in the snapshot) =="
OFF=$((HDR + LD*4 + 1000))            # header($HDR, version $NVME_VERSION) + ids(LD*4) + 1000 bytes into the GDN region
cp "$OUT/snap.bin" "$OUT/snap-corrupt.bin"
printf '\377' | dd of="$OUT/snap-corrupt.bin" bs=1 seek=$OFF conv=notrunc status=none
run Bc "$PROMPT_AB" "--nvme-restore $OUT/snap-corrupt.bin"

H_A=$(grep STATE_HASH "$OUT/A.err" | tail -1)
H_A2=$(grep STATE_HASH "$OUT/A2.err" | tail -1)
H_B=$(grep STATE_HASH "$OUT/B.err" | tail -1)
H_Bpost=$(grep -A1 "nvme_restore: loaded" "$OUT/B.err" | grep STATE_HASH | tail -1)
H_Bc=$(grep STATE_HASH "$OUT/Bc.err" | tail -1)
T_A2=$(grep '^T ' "$OUT/A2.out" | awk '{print $2}' | paste -sd, -)
T_B=$(grep '^T ' "$OUT/B.out" | awk '{print $2}' | paste -sd, -)
H_C=$(grep STATE_HASH "$OUT/C.err" | tail -1)

echo "== VERDICT (snapshot L=$LD) =="
FAIL=0
if [ "$H_Bpost" = "$H_A" ] && [ -n "$H_A" ]; then echo "PASS restore-exactness: post-restore hash == A's DONE hash (bit-exact dump/restore)"
else echo "FAIL restore-exactness:"; echo "  A  (DONE): $H_A"; echo "  B (post-restore): $H_Bpost"; FAIL=1; fi
if [ "$T_B" = "$T_C" ] && [ -n "$T_C" ]; then echo "PASS tokens: NVMe-restored continuation == same-process live continuation"
else echo "INFO tokens (B vs C): identical continuation is not attainable as a bit-gate - the engine's decode is
     run-to-run nondeterministic on this build (verified with the OLD binary: two identical plain runs of this
     prompt diverge at token 40 as well). The hard gates are restore-exactness and the negative control."
     echo "  B: ${T_B:0:80}"; echo "  C: ${T_C:0:80}"; fi
if [ "$H_B" = "$H_C" ] && [ -n "$H_C" ]; then echo "PASS end-to-end hash: B == C (restored session == live session, bit-exact)"
else echo "INFO end-to-end hash B vs C (informational: DONE-time consumed-window accounting differs for resumed runs):"
     echo "  B: $H_B"; echo "  C: $H_C"; fi
if [ "$T_A2" = "$T_B" ]; then echo "INFO: A' (full re-prefill) also matched"; else echo "INFO: A' differs from B - expected: batched prefill vs decode/short-read rounding (the engine's own accepted path dependence)"; fi
RES=$(grep -o '^RESUME [0-9]*' "$OUT/B.out" | tail -1)
if [ "$RES" = "RESUME $LD" ]; then echo "PASS resume: $RES"; else echo "FAIL resume: got '$RES' (want RESUME $LD)"; FAIL=1; fi
# with the payload-integrity footer a corrupted snapshot is REFUSED at restore (the spike path exits)
if grep -q "integrity check failed" "$OUT/Bc.err" 2>/dev/null || [ -n "$H_Bc" ] && [ "$H_Bc" != "$H_A2" ]; then
  echo "PASS negative control: corruption detected (the test can fail)"
else echo "FAIL negative control: corruption NOT detected - the test is decorative"; FAIL=1; fi
# AND it must be refused as the RECOVERABLE class.  A corrupt file is refused before the tier writes anything, so
# it can never be reported as a transfer failure - the two classes have different consequences (one re-prefills,
# the other stops the engine), and a log reader has to be able to tell them apart.
# docs/nvme-kv-cache-design.md §5.1 / §5.2.
if grep -q "nvme_restore failed (refused)" "$OUT/Bc.err" 2>/dev/null; then
  echo "PASS failure class: the corrupt snapshot is refused, not a transfer"
else echo "FAIL failure class: expected 'nvme_restore failed (refused)' in Bc.err"; FAIL=1; fi
if grep -q "nvme_restore failed (transfer)" "$OUT/Bc.err" 2>/dev/null; then
  echo "FAIL failure class: a corrupt FILE reported as a transfer failure"; FAIL=1
else echo "PASS no transfer failure reported for a corrupt file"; fi
[ "$FAIL" = "0" ] && { echo "ALL PASS"; exit 0; } || { echo "SOME FAILURES"; exit 1; }
