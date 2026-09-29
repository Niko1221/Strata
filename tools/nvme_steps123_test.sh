#!/bin/bash
# tools/nvme_steps123_test.sh - end-to-end test of the automatic NVMe cold tier (design doc Steps 1-3).
#   Step 1: every DONE cascades the consumed session to --kv-nvme DIR synchronously and idempotently
#           (a growing conversation stays ONE file: the previous dump is superseded).
#   Step 2: after a process restart, a request whose prompt starts with a stored session is PROMOTED
#           automatically (no client call): "nvme promote: resumed L tokens" + RESUME L + a fast prompt read.
#   Step 3: --kv-nvme-max caps the store; the least recently stored snapshots are evicted.
set -u
# ROOT comes FIRST, before any cd: BASH_SOURCE is whatever the caller typed, so resolving it after
# `cd /local/strata` made ROOT the checkout being cd'd INTO whenever the script was invoked by a relative
# path - sourcing the wrong tree's header constants while testing that tree's engine (found on first run).
ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
cd /local/strata
# See tools/nvme_header_layout.sh: the snapshot offset and format version come from the header in THIS tree, not
# from a number written into this script by whoever last changed the header.
. "$ROOT/tools/nvme_header_layout.sh"
# The engine is overridable so the oracles can validate a DIFFERENT build than the one in this checkout's
# build/ - which is the whole point when the tree under test is a worktree (NVME_ENGINE=/path/to/strata).
# Default stays this checkout's binary, so the recorded v2-format results keep meaning what they meant.
E=${NVME_ENGINE:-build/strata}
OUT=/tmp/nvme-s123
STORE=$OUT/store
rm -rf "$OUT"; mkdir -p "$OUT"
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH-}   # ${VAR-}: set -u kills an unset LD_LIBRARY_PATH (found on first run)
NVME_PYTHON=.venv/bin/python

ARGS="--serve --pack packs/iq3_xxs
 --native models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
 --ple-gguf models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
 --expert-profile data/expert-profile.bin --expert-cache auto --prefill 2048
 --spec 4 --spec-min-p 0.5 --mtp mtp/rt --max-context 131072 --kv int8
 --kv-resident 20480 --prompt-cache 12 --adapt-swaps 0"
ARGS=$(echo "$ARGS" | tr '\n' ' ')

# six distinct ~3900-token chat prompts + three short tails, via the pack's tokenizer
.venv/bin/python - "$OUT" <<'PYEOF'
import json, sys
sys.path.insert(0, "tools")
import strata_tokenizer as ST
tp = "packs/iq3_xxs/tokenizer"
vocab = json.load(open(tp + "/vocab.json"))
tokens = [None] * len(vocab)
for t, i in vocab.items(): tokens[i] = t
tok = ST.Tokenizer(tokens, open(tp + "/merges.txt").read().split("\n"),
                   json.load(open(tp + "/token_type.json")))
filler = ("Describe the storm, the ships, the lamp room, the keeper's routine, the logbook, the rocks, "
          "the fog bell, the supply boat, and the winter isolation in vivid detail. ") * 200
for i, subj in enumerate(["a lighthouse keeper", "a clockmaker", "a beekeeper",
                          "a cartographer", "a bridge engineer", "a tea farmer"], 1):
    ids = tok.encode(f"<|im_start|>user\nWrite a long story about {subj}. " + filler,
                     parse_special=True)[:3900] + tok.encode("<|im_end|>\n<|im_start|>assistant\n", parse_special=True)
    open(f"{sys.argv[1]}/p{i}.txt", "w").write(",".join(map(str, ids)))
for i, t in enumerate(["<|im_end|>\n<|im_start|>user\nWhat happened next? Continue the story.\n<|im_start|>assistant\n",
                       "<|im_end|>\n<|im_start|>user\nAnd the winter? Continue.\n<|im_start|>assistant\n",
                       "<|im_end|>\n<|im_start|>user\nFinish the tale.\n<|im_start|>assistant\n"], 1):
    open(f"{sys.argv[1]}/t{i}.txt", "w").write(",".join(map(str, tok.encode(t, parse_special=True))))
print("prompts written", file=sys.stderr)
PYEOF

P1=$(cat "$OUT/p1.txt"); T1=$(cat "$OUT/t1.txt"); T2=$(cat "$OUT/t2.txt"); T3=$(cat "$OUT/t3.txt")

wait_done() { # $1 outfile, $2 how many DONE lines
  for _ in $(seq 1 300); do
    local n; n=$(grep -c '^DONE' "$1" 2>/dev/null); n=${n:-0}
    [ "$n" -ge "$2" ] && return 0
    sleep 1
  done
  echo "TIMEOUT waiting for DONE #$2 in $1"; return 1
}
snap_ids() { # $1 snapshot file: prints L, writes the comma ids to $OUT/full.txt
  nvme_snapshot_ids "$1" "$OUT/full.txt"
}
fsize() { stat -c %s "$1" 2>/dev/null || echo 0; }

echo "== process 1: Step 1 (automatic cascade + supersede, same process) =="
nvme_header_layout "$ROOT" || exit 1
echo "reading snapshots at header $HDR bytes, format version $NVME_VERSION"
FIFO=$OUT/in; mkfifo "$FIFO"
$E $ARGS --kv-nvme $STORE < "$FIFO" > "$OUT/proc1.out" 2> "$OUT/proc1.err" &
EPID=$!
exec 3> "$FIFO"
echo "GEN 200 $P1" >&3
wait_done "$OUT/proc1.out" 1 || { kill $EPID; exit 1; }
S1=$(fsize "$STORE"/kv-*.bin)
L1=$(snap_ids "$STORE"/kv-*.bin | tail -1)
FULL1=$(cat "$OUT/full.txt")
echo "snapshot after r1: L=$L1 size=$S1"
echo "GEN 50 $FULL1,$T1" >&3
wait_done "$OUT/proc1.out" 2 || { kill $EPID; exit 1; }
L2=$(snap_ids "$STORE"/kv-*.bin | tail -1)
FULL2=$(cat "$OUT/full.txt")
S2=$(fsize "$STORE"/kv-*.bin)
echo "GEN 50 $FULL2,$T2" >&3
wait_done "$OUT/proc1.out" 3 || { kill $EPID; exit 1; }
L3=$(snap_ids "$STORE"/kv-*.bin | tail -1)
FULL3=$(cat "$OUT/full.txt")
S3=$(fsize "$STORE"/kv-*.bin)
echo "QUIT" >&3
exec 3>&-
wait $EPID
N1=$(ls "$STORE"/kv-*.bin 2>/dev/null | wc -l)
echo "files after process 1: $N1 (want 1); L=$L1->$L2->$L3 sizes=$S1->$S2->$S3 (want growing)"
grep -E "NVMe KV store|kv-nvme dump failed" "$OUT/proc1.err" | head -3

echo "== process 2: Step 2 (promote after restart, no client call) =="
P2=$(cat "$OUT/p2.txt"); P3=$(cat "$OUT/p3.txt")
{ echo "GEN 50 $FULL3,$T3"; sleep 3; echo "GEN 50 $P2"; sleep 3; echo "GEN 50 $P3"; sleep 3; echo "QUIT"; } | \
  timeout 900 $E $ARGS --kv-nvme $STORE > "$OUT/proc2.out" 2> "$OUT/proc2.err"
echo "proc2 rc=$?"
grep -E "NVMe KV store|nvme promote" "$OUT/proc2.err" | head -4
grep "^RESUME" "$OUT/proc2.out" | head -3
PROM_MS=$(grep -oE "reused \+ [0-9]+ read in [0-9]+ ms" "$OUT/proc2.err" | head -1)
echo "promoted request read: $PROM_MS"

echo "== process 3: Step 3 (byte cap evicts the oldest) =="
P4=$(cat "$OUT/p4.txt"); P5=$(cat "$OUT/p5.txt"); P6=$(cat "$OUT/p6.txt")
{ echo "GEN 50 $P4"; sleep 3; echo "GEN 50 $P5"; sleep 3; echo "GEN 50 $P6"; sleep 3; echo "QUIT"; } | \
  timeout 900 $E $ARGS --kv-nvme $STORE --kv-nvme-max 1 > "$OUT/proc3.out" 2> "$OUT/proc3.err"
echo "proc3 rc=$?"
N3=$(ls "$STORE"/kv-*.bin 2>/dev/null | wc -l)
echo "files after the capped process: $N3"
{ echo "GEN 1 $P6"; sleep 3; echo "QUIT"; } | \
  timeout 900 $E $ARGS --kv-nvme $STORE > "$OUT/proc4.out" 2> "$OUT/proc4.err"
grep "NVMe KV store" "$OUT/proc4.err" | head -1

echo "== VERDICT =="
FAIL=0
# THE KV LINE (docs/nvme-kv-cache-web-design.md §3): the store's state straight after READY, then one per request
KVSTART=$(grep -c "^KV start=1 entries=" "$OUT/proc1.out" 2>/dev/null); KVSTART=${KVSTART:-0}
[ "$KVSTART" = "1" ] || { echo "FAIL: no 'KV start=1 entries=' line after READY"; FAIL=1; }
NKV=$(grep -c "^KV src=" "$OUT/proc1.out" 2>/dev/null); NKV=${NKV:-0}
NDONE=$(grep -c "^DONE" "$OUT/proc1.out" 2>/dev/null); NDONE=${NDONE:-0}
[ "$NKV" = "$NDONE" ] || { echo "FAIL: $NKV KV lines for $NDONE DONE lines (one per request, before DONE)"; FAIL=1; }
[ "$N1" = "1" ] || { echo "FAIL step1: expected 1 file after process 1, got $N1"; FAIL=1; }
if [ -n "$S1" ] && [ -n "$S3" ] && [ "$S3" -gt "$S1" ]; then echo "PASS step1 supersede: one growing file ($S1 -> $S3 bytes)"
else echo "FAIL step1 supersede: sizes r1=$S1 r3=$S3"; FAIL=1; fi
grep -q "nvme promote: resumed" "$OUT/proc2.err" || { echo "FAIL step2: no promote happened"; FAIL=1; }
RES=$(grep -o "^RESUME [0-9]*" "$OUT/proc2.out" | head -1)
[ "$RES" = "RESUME $L3" ] || { echo "FAIL step2: first resume '$RES' != snapshot L $L3"; FAIL=1; }
FR=$(echo "$PROM_MS" | sed -E 's/.*reused \+ ([0-9]+) read.*/\1/')
if [ -n "$FR" ] && [ "$FR" -lt 100 ]; then echo "PASS step2 ttft: only $FR fresh tokens read"; else echo "FAIL step2 ttft: fresh read = '$FR'"; FAIL=1; fi
if [ "$N3" -lt 7 ]; then echo "PASS step3 cap: $N3 files remain (evicted)"; else echo "FAIL step3: cap did not evict ($N3 files)"; FAIL=1; fi
# the capped process must SAY it evicted: a KV line with evict >= 1 and the bytes it dropped
KEV=$(grep -cE "^KV src=.* evict=[1-9][0-9]* evict_bytes=[1-9]" "$OUT/proc3.out" 2>/dev/null); KEV=${KEV:-0}
if [ "$KEV" -ge 1 ]; then echo "PASS step3 KV: the cap turn reported $(grep -oE 'evict=[0-9]+ evict_bytes=[0-9]+' "$OUT/proc3.out" | head -1)"
else echo "FAIL step3 KV: no KV line with evict>=1 in the capped process"; FAIL=1; fi
GB=$(grep "NVMe KV store" "$OUT/proc4.err" | grep -oE "[0-9.]+ GiB" | head -1 | cut -d' ' -f1)
if [ -n "$GB" ] && .venv/bin/python -c "import sys; sys.exit(0 if float('$GB') <= 1.05 else 1)"; then
  echo "PASS step3 cap bytes: $GB GiB <= cap"
else
  echo "FAIL step3 cap bytes: '$GB' GiB"; FAIL=1
fi
if grep -h "^ERR" "$OUT"/proc*.out >/dev/null 2>&1; then echo "FAIL: engine printed ERR"; FAIL=1; fi
[ "$FAIL" = "0" ] && echo "ALL PASS" || echo "SOME FAILURES"
