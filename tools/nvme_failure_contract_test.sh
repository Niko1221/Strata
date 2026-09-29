#!/bin/bash
# tools/nvme_failure_contract_test.sh - executes BOTH classes of the failure contract on real silicon
# (docs/nvme-kv-cache-design.md §5; collision C7).
#
# The contract has three classes and, before this oracle, only the first had ever been executed: every recorded
# test corrupts a FILE (a refusal, which never reaches the transfer pass) and none injects a failing COPY - and
# the "clean-reset fallback" this tier documented for years in fact aborted inside its own first step on an
# unconsumed CUDA error.  STRATA_TEST_FAIL_CUDA (test-only, default inert, see kv_nvme.cpp) names one transfer
# to break, so the fatal path can finally be RUN rather than argued:
#
#   invalid (recoverable)     corrupt the payload of a stored snapshot -> the PROMOTE refuses, the entry is
#                             dropped, the engine re-reads the prompt and KEEPS SERVING; the file is unlinked.
#   transfer_failed (fatal)   hook the gdn / spare / sync transfer -> the engine exits 1 with the correct ERR,
#                             the snapshot stays on disk, and a fresh engine process then serves normally:
#                             the new process IS the supervisor-restart recovery, with a new CUDA context.
#
# The hook is asserted inert-by-default implicitly: run 2 of nvme_p0_test.sh restores the same snapshot with no
# env set and must keep passing.
set -u
ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
cd /local/strata
. "$ROOT/tools/nvme_header_layout.sh"
nvme_header_layout "$ROOT" || exit 1
E=${NVME_ENGINE:-build/strata}
OUT=/tmp/nvme-fc
STORE=$OUT/store
STORE2=$OUT/store2
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH-}
export STRATA_STATE_HASH=1 STRATA_NVME_HASH=1
NVME_PYTHON=/local/strata/.venv/bin/python

ARGS="--serve --pack packs/iq3_xxs
 --native models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
 --ple-gguf models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
 --expert-profile data/expert-profile.bin --expert-cache auto --prefill 2048
 --spec 4 --spec-min-p 0.5 --mtp mtp/rt --max-context 131072 --kv int8
 --kv-resident 20480 --prompt-cache 12 --adapt-swaps 0"
ARGS=$(echo "$ARGS" | tr '\n' ' ')
rm -rf "$OUT"; mkdir -p "$STORE"

$NVME_PYTHON - "$OUT" <<'PYEOF'
import json, sys
out = sys.argv[1]
tp = "packs/iq3_xxs/tokenizer"
vocab = json.load(open(tp + "/vocab.json"))
tokens = [None] * len(vocab)
for t, i in vocab.items(): tokens[i] = t
merges = open(tp + "/merges.txt").read().split("\n")
types = json.load(open(tp + "/token_type.json"))
sys.path.insert(0, "tools")
import strata_tokenizer as ST
tok = ST.Tokenizer(tokens, merges, types)
filler = "Describe the harbour, the tide tables, the gulls, the fish market and the winter storms. " * 14
ids = tok.encode("<|im_start|>user\n" + filler + "<|im_end|>\n<|im_start|>assistant\n", parse_special=True)
open(out + "/pids.txt", "w").write(",".join(map(str, ids)))
print("prompt ids:", len(ids), file=sys.stderr)
PYEOF
P=$(cat "$OUT/pids.txt")

echo "== run A: dump a healthy snapshot =="
echo "GEN 60 $P" | timeout 900 $E $ARGS --nvme-dump $OUT/snap.bin > "$OUT/A.out" 2> "$OUT/A.err"
echo "A: rc=$? tokens=$(grep -c '^T ' "$OUT/A.out")"
[ -f "$OUT/snap.bin" ] || { echo "FAIL: no snapshot"; exit 1; }
L=$(nvme_snapshot_ids "$OUT/snap.bin" "$OUT/snap_ids.txt") || exit 1
REPLY=$(awk '/^T /{print $2}' "$OUT/A.out" | paste -sd, -)
[ -n "$REPLY" ] || { echo "FAIL: run A generated nothing"; exit 1; }
echo "reply tokens: $(awk -F, '{print NF}' <<< "$REPLY")"
echo "snapshot L=$L (header $HDR, version $NVME_VERSION)"
FULL="$P,$REPLY"

cp "$OUT/snap.bin" "$OUT/snap_healthy.bin"   # PART 2 restores a HEALTHY file: the hook must be what fails,
                                             # not an integrity refusal that fires before the transfer pass

cp "$OUT/snap.bin" "$STORE/kv-1-1.bin"   # the STORE copy is the one corrupted; snap_healthy stays pristine

echo "== PART 1 (invalid class): corrupt the stored payload, the PROMOTE must refuse and the engine keep serving =="
# the payload begins after header + ids + image records (no images here): 208 + 4L.  The GDN segment is first.
$NVME_PYTHON - "$OUT" <<PYEOF
import struct, sys
p = "$STORE/kv-1-1.bin"
data = bytearray(open(p, "rb").read())
L = struct.unpack_from("<q", data, 8)[0]
n_imgs = struct.unpack_from("<i", data, 16)[0]
off = 208 + 4 * L + 16 * n_imgs + 64      # inside the GDN segment, past the ids and any image records
for i in range(64):
    data[off + i] ^= 0xA5
open(p, "wb").write(bytes(data))
print("corrupted 64 bytes at", off, file=sys.stderr)
PYEOF
# the request must be STRICTLY longer than the entry: a prompt as long as the entry has nothing to generate,
# and the serve loop never even tries the promote (measured: an exact-length request sat at RESUME 0 silently)
SUFFIX=$($NVME_PYTHON -c "
import json, sys
sys.path.insert(0, 'tools')
import strata_tokenizer as ST
tp = 'packs/iq3_xxs/tokenizer'
vocab = json.load(open(tp + '/vocab.json'))
tokens = [None] * len(vocab)
for t, i in vocab.items(): tokens[i] = t
merges = open(tp + '/merges.txt').read().split('\\n')
types = json.load(open(tp + '/token_type.json'))
tok = ST.Tokenizer(tokens, merges, types)
print(','.join(map(str, tok.encode(' And the lighthouse?<|im_end|>\\n<|im_start|>assistant\\n', parse_special=True))))
")
echo "GEN 60 $FULL,$SUFFIX" | timeout 900 $E $ARGS --kv-nvme "$STORE" --kv-nvme-max 40 > "$OUT/B.out" 2> "$OUT/B.err"
BRC=$?
echo "B: rc=$BRC tokens=$(grep -c '^T ' "$OUT/B.out")"
grep -E "nvme promote|RESUME" "$OUT/B.err" "$OUT/B.out" 2>/dev/null | head -3
STATUS=0
grep -q "nvme promote refused" "$OUT/B.err" || { echo "FAIL: the corrupt snapshot was not refused at promote"; STATUS=1; }
grep -q "^RESUME 0" "$OUT/B.out" || { echo "FAIL: the engine did not fall back to reading the prompt"; STATUS=1; }
[ "$BRC" -eq 0 ] || { echo "FAIL: the engine did not survive the refused promote"; STATUS=1; }
TOK=$(grep -c '^T ' "$OUT/B.out")
[ "$TOK" -ge 8 ] || { echo "FAIL: only $TOK tokens after the refusal - the engine is not serving"; STATUS=1; }
LEFT=$(ls "$STORE" 2>/dev/null | wc -l)
[ ! -f "$STORE/kv-1-1.bin" ] || { echo "FAIL: the refused entry was not dropped"; STATUS=1; }
# THE KV LINE for the refused promote (design §3): the tier served nothing (src=none), the request was refused,
# and it was NOT the fatal class
KV=$(grep -m1 "^KV src=" "$OUT/B.out" 2>/dev/null)
[ -n "$KV" ] || { echo "FAIL: the refused promote printed no KV line"; STATUS=1; }
if grep -qE "^KV src=none .* refused=1 transfer=0 " "$OUT/B.out"; then
  echo "PASS invalid class reports itself on the KV line: $(grep -oE 'src=[a-z]+ refused=[0-9]+ transfer=[0-9]+' "$OUT/B.out" | head -1)"
else
  echo "FAIL: the refusal's KV line is not 'src=none ... refused=1 transfer=0': '$KV'"; STATUS=1
fi
# run B holds the store too, so its own DONE-cascade dump may legitimately be there; the CORRUPT entry must not be
[ "$STATUS" -eq 0 ] && echo "PASS invalid class: refused, dropped, re-prefilled, still serving, store cleaned"

echo "== PART 2 (transfer_failed class): the fatal path, executed =="
for HOOK in gdn spare sync; do
    echo "-- hook: $HOOK --"
    echo "GEN 30 $FULL" | STRATA_TEST_FAIL_CUDA=$HOOK timeout 900 $E $ARGS --nvme-restore "$OUT/snap_healthy.bin" \
        > "$OUT/F_$HOOK.out" 2> "$OUT/F_$HOOK.err"
    RC=$?
    echo "$HOOK: rc=$RC"
    grep -E "ERR|transfer|refused" "$OUT/F_$HOOK.err" "$OUT/F_$HOOK.out" 2>/dev/null | head -3
    [ "$RC" -eq 1 ] || { echo "FAIL ($HOOK): rc=$RC, want 1 - the fatal rule did not fire"; STATUS=1; }
    # the STARTUP path is fatal for both classes but only NAMES the class; the ERR line and the
    # no-clean-reset note belong to the PROMOTE path, exercised in PART 2b below
    grep -q "failed (transfer)" "$OUT/F_$HOOK.err" || { echo "FAIL ($HOOK): the class is not named"; STATUS=1; }
    [ -f "$OUT/snap_healthy.bin" ] || { echo "FAIL ($HOOK): the snapshot was removed - it must be left on disk"; STATUS=1; }
    [ "$STATUS" -eq 0 ] && echo "PASS transfer_failed ($HOOK): exit 1, correct ERR, snapshot left on disk"
done

echo "== PART 2b (transfer_failed, PROMOTE path): the operator-facing fatal branch, with its ERR and note =="
# the serve loop's fatal branch is the one an operator actually hits: a stored snapshot promotes on request,
# the transfer fails, and the contract says STOP - no drop, no re-prefill, no unproven in-process reset.
mkdir -p "$STORE2"; cp "$OUT/snap_healthy.bin" "$STORE2/kv-2-1.bin"
echo "GEN 60 $FULL,$SUFFIX" | STRATA_TEST_FAIL_CUDA=gdn timeout 900 $E $ARGS --kv-nvme "$STORE2" --kv-nvme-max 40 \
    > "$OUT/H.out" 2> "$OUT/H.err"
HRC=$?
echo "H: rc=$HRC tokens=$(grep -c '^T ' "$OUT/H.out")"
grep -E "ERR|clean reset|promote" "$OUT/H.err" "$OUT/H.out" 2>/dev/null | head -3
[ "$HRC" -eq 1 ] || { echo "FAIL (promote path): rc=$HRC, want 1"; STATUS=1; }
grep -q "ERR" "$OUT/H.err" "$OUT/H.out" || { echo "FAIL (promote path): no ERR line"; STATUS=1; }
grep -q "clean reset" "$OUT/H.err" || { echo "FAIL (promote path): the no-clean-reset note is gone"; STATUS=1; }
grep -qiE "failed? \(transfer\)" "$OUT/H.err" "$OUT/H.out" || { echo "FAIL (promote path): the class is not named"; STATUS=1; }
[ -f "$STORE2/kv-2-1.bin" ] || { echo "FAIL (promote path): the snapshot was removed on a TRANSFER failure"; STATUS=1; }
# THE KV LINE must reach the server BEFORE the ERR line and the exit 1: the engine is dying, and this line is
# the only thing that names the class (design §3, ordering note 1)
KVL=$(grep -n "^KV src=" "$OUT/H.out" 2>/dev/null | head -1 | cut -d: -f1)
ERRL=$(grep -n "^ERR" "$OUT/H.out" 2>/dev/null | head -1 | cut -d: -f1)
if [ -z "$KVL" ] || [ -z "$ERRL" ]; then
  echo "FAIL (promote path): no KV line (line '$KVL') or no ERR line (line '$ERRL') on stdout"; STATUS=1
elif [ "$KVL" -ge "$ERRL" ] || ! grep -qE "^KV src=.* refused=0 transfer=1 " "$OUT/H.out"; then
  echo "FAIL (promote path): the KV transfer=1 line did not arrive before the ERR line (KV $KVL, ERR $ERRL)"; STATUS=1
else
  echo "PASS transfer_failed (promote path): KV transfer=1 on stdout line $KVL, before the ERR line $ERRL and the exit"
fi
[ "$STATUS" -eq 0 ] && echo "PASS transfer_failed (promote path): ERR, no clean reset attempted, snapshot kept"

echo "== PART 3: the process AFTER the fatal exit serves normally (the supervisor-restart recovery) =="
echo "GEN 30 $FULL" | timeout 900 $E $ARGS > "$OUT/G.out" 2> "$OUT/G.err"
GRC=$?
echo "G: rc=$GRC tokens=$(grep -c '^T ' "$OUT/G.out")"
[ "$GRC" -eq 0 ] || { echo "FAIL: the fresh process did not come up"; STATUS=1; }
TOK=$(grep -c '^T ' "$OUT/G.out")
[ "$TOK" -ge 8 ] || { echo "FAIL: fresh process generated $TOK tokens"; STATUS=1; }
[ "$STATUS" -eq 0 ] && echo "PASS recovery: a new process with a new CUDA context serves"

[ "$STATUS" -eq 0 ] && echo "ALL PASS" || echo "FAILED"
exit $STATUS
