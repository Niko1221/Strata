#!/bin/bash
# tools/nvme_image_promote_test.sh - the end-to-end oracle for the tier's IMAGE path (docs/nvme-kv-cache-design.md §4).
#
# The image path of the cold tier was unreachable by every recorded test: nvme_p0_test.sh, nvme_steps123_test.sh
# and needle_bench.py are all text, and kv_nvme_host_test asserts kv_nvme_match over synthetic bytes, not the
# engine's request loop.  Two defects lived exactly there - NvmeEntry::imgs was never filled, and a turn-boundary
# dump stored the LIVE image list - so an image conversation could never promote.  This oracle runs the real
# engine over the real path and asserts the PROMOTE.
#
# No vision weights are needed: the engine's --vision path takes a GENI request carrying an embeddings FILE
# (a strata-vision record: int32 'SVE1', n, nx, ny, then n x n_embd floats).  Synthetic embeddings are fine,
# because what the tier keys on is the hash of the grid and the rows, not what the picture depicts - the same
# file on both requests IS the same picture as far as the tier is concerned.
#
#   1: prompt = one user turn holding ONE image -> GENI; the automatic --kv-nvme store dumps at the turn
#      boundary, which is BEFORE the reply, so the key is the user turn: what a chat client re-sends.
#   2: a NEW engine process, same store; prompt = user turn + reply + a new user turn.  The startup scan must
#      PROMOTE the boundary ("nvme promote: resumed"), not re-prefill.
#
# PASS requires: the promote line in run 2, RESUME == the boundary length (strictly less than the full
# conversation, so what resumed is the boundary and not the whole consumed prefix), and n_imgs == 1 in the file.
set -u
ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
cd /local/strata
. "$ROOT/tools/nvme_header_layout.sh"
nvme_header_layout "$ROOT" || exit 1   # sets HDR and NVME_VERSION (and refuses a header it cannot parse)
E=${NVME_ENGINE:-build/strata}
OUT=/tmp/nvme-img
STORE=$OUT/store
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH-}
export STRATA_STATE_HASH=1 STRATA_NVME_HASH=1
NVME_PYTHON=/local/strata/.venv/bin/python

ARGS="--serve --pack packs/iq3_xxs
 --native models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
 --ple-gguf models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
 --expert-profile data/expert-profile.bin --expert-cache auto --prefill 2048
 --spec 4 --spec-min-p 0.5 --mtp mtp/rt --max-context 131072 --kv int8
 --kv-resident 20480 --prompt-cache 12 --adapt-swaps 0 --vision
 --kv-nvme $STORE --kv-nvme-max 40 --kv-delta 0"
ARGS=$(echo "$ARGS" | tr '\n' ' ')

# --kv-delta 0, and it is load-bearing: this oracle asserts the v3 snapshot FILES (the SNAP glob below and
# the NvmeHeader it parses) and the PROMOTE keyed on their image list, and the delta tier has been the DEFAULT
# cascade since Phase 6 (kv_delta = 1, generate.cpp:307).  With `--kv-nvme` alone a turn appends chunks and a
# per-turn state and writes no v3 snapshot at all: the store glob found nothing and the oracle died on
# "the store wrote no snapshot" with a run that had answered the image request perfectly (rc=0, 39 tokens) -
# the same silent wrong-tier bug d2cb8f6 fixed in the three sibling oracles, which left this one behind.
# The image correctness contract is NOT weakened by naming the tier: NvmeEntry::imgs, the two historical image
# defects (a never-filled list, a dump of the LIVE list) and the ImgKey that the negative control below probes
# all live on the v3 path this oracle keeps asserting, at n_imgs == 1, with the promote and the negative control
# unchanged.  The delta cascade's own write path stays gated by tools/nvme_delta_p0_test.sh.
rm -rf "$OUT"; mkdir -p "$STORE"

# the synthetic image and both prompts.  The image is a 2x2 patch grid: four rows the engine strides by
# g.n_embd (qwen4exp.embedding_length).  The ImgKey hash covers the grid and the rows, so the same file on
# both requests is the same picture; a DIFFERENT file must NOT match (negative control).
$NVME_PYTHON - "$OUT" <<'PYEOF'
import json, struct, sys
out = sys.argv[1]
N_EMBD, N, NX, NY = 2560, 4, 2, 2
def image(path, salt):
    with open(path, "wb") as f:
        # record layout per generate.cpp:2768: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats.
        # The engine validates the record's width against g.n_embd (2560) and refuses otherwise.
        f.write(b"SVE1"); f.write(struct.pack("<iiii", N, NX, NY, N_EMBD))
        for i in range(N):
            for c in range(N_EMBD):
                f.write(struct.pack("<f", 0.01 * ((i * 131 + c * 7 + salt) % 251) - 1.25))
image(out + "/emb.bin", 0)
image(out + "/emb_other.bin", 97)          # same grid, different rows -> different ImgKey
tp = "packs/iq3_xxs/tokenizer"
vocab = json.load(open(tp + "/vocab.json"))
tokens = [None] * len(vocab)
for t, i in vocab.items(): tokens[i] = t
merges = open(tp + "/merges.txt").read().split("\n")
types = json.load(open(tp + "/token_type.json"))
import sys as _s; _s.path.insert(0, "tools")
import strata_tokenizer as ST
tok = ST.Tokenizer(tokens, merges, types)
PAD = "<|image_pad|>"
user1 = ("<|im_start|>user\n<|vision_start|>" + PAD * N + "<|vision_end|>\n"
         "Describe this picture in one sentence.<|im_end|>\n<|im_start|>assistant\n")
ids1 = tok.encode(user1, parse_special=True)
open(out + "/ids1.txt", "w").write(",".join(map(str, ids1)))
# The turn boundary is the engine's: the checkpoint holds the user turn up to and including its turn token,
# and the trailing <|im_start|>assistant\n is client template re-sent every time.  Measured: 19 of 22 ids.
# The oracle asserts the INVARIANT, not a hardcoded position: the snapshot key must end at the user turn -
# strictly before the reply - and the promote must resume exactly that key.
BOUNDARY_MAX = len(ids1)
open(out + "/boundary.txt", "w").write(str(BOUNDARY_MAX))
print("ids1:", len(ids1), "pad tokens:", sum(1 for i in ids1 if i == 248056), file=sys.stderr)
PYEOF
IDS1=$(cat "$OUT/ids1.txt")
BOUNDARY_MAX=$(cat "$OUT/boundary.txt")

echo "== run 1: image request, automatic store dumps at the turn boundary =="
echo "GENI 120 $OUT/emb.bin $IDS1" | timeout 900 $E $ARGS > "$OUT/1.out" 2> "$OUT/1.err"
echo "run1: rc=$? tokens=$(grep -c '^T ' "$OUT/1.out")"
grep -E "nvme_dump|kv-nvme dump" "$OUT/1.err" | head -2
SNAP=$(ls "$STORE"/*.bin 2>/dev/null | head -1)
[ -n "$SNAP" ] || { echo "FAIL: the store wrote no snapshot"; exit 1; }
L=$(nvme_snapshot_ids "$SNAP" "$OUT/snap_ids.txt") || exit 1
N_IMGS=$($NVME_PYTHON -c "
import struct, sys
f = open('$SNAP', 'rb'); h = f.read($HDR)
L, n_imgs = struct.unpack_from('<q', h, 8)[0], struct.unpack_from('<i', h, 16)[0]
print(n_imgs)")
echo "snapshot: L=$L n_imgs=$N_IMGS (header $HDR, version $NVME_VERSION)"
# the key must be the BOUNDARY, not the consumed prefix: reply tokens pushed L past the boundary
REPLY=$(awk '/^T /{print $2}' "$OUT/1.out" | paste -sd, -)
[ -n "$REPLY" ] || { echo "FAIL: run 1 generated nothing"; exit 1; }
if [ "$L" -gt 4 ]; then echo "keyed at L=$L of $BOUNDARY_MAX ids"; fi

echo "== run 2: new process, same store, the conversation RE-SENT AS GENI (a chat client re-encodes the image;
#   req_imgs is only built on the GENI path, so a text GEN carrying the pad ids has no image keys at all) =="
IDS2="$IDS1,$REPLY,$(cat "$OUT/ids2_suffix.txt" 2>/dev/null || true)"
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
open(out + "/suffix.txt", "w").write(",".join(map(str, tok.encode(
    "<|im_start|>user\nNow a different question about the weather.<|im_end|>\n<|im_start|>assistant\n",
    parse_special=True))))
PYEOF
IDS2="$IDS1,$REPLY,$(cat "$OUT/suffix.txt")"
echo "GENI 120 $OUT/emb.bin $IDS2" | timeout 900 $E $ARGS > "$OUT/2.out" 2> "$OUT/2.err"
echo "run2: rc=$? tokens=$(grep -c '^T ' "$OUT/2.out")"
PROMOTED=$(grep -oE "nvme promote: resumed [0-9]+ tokens" "$OUT/2.err" | grep -oE "[0-9]+" | head -1)
RESUMED=$(grep -oE "^RESUME [0-9]+" "$OUT/2.out" | awk '{print $2}' | head -1)
echo "promote: ${PROMOTED:-none}   RESUME: ${RESUMED:-none}   conversation length: $(awk -F, '{print NF}' <<< "$IDS2")"

STATUS=0
if [ -z "$PROMOTED" ]; then echo "FAIL: no promote - the image conversation re-prefilled"; STATUS=1; fi
if [ "$PROMOTED" != "$L" ] || [ "$RESUMED" != "$L" ]; then
    echo "FAIL: promoted $PROMOTED / resumed $RESUMED, want the snapshot's own key $L"; STATUS=1
fi
if [ "$L" -gt "$BOUNDARY_MAX" ]; then echo "FAIL: snapshot L=$L is past the user turn ($BOUNDARY_MAX) - the key includes the reply"; STATUS=1; fi
if [ "$L" -le 4 ]; then echo "FAIL: snapshot L=$L is implausibly short"; STATUS=1; fi
if [ "$L" -le "$BOUNDARY_MAX" ] && [ "$L" -gt 4 ]; then echo "PASS boundary key: the snapshot is keyed inside the user turn ($L of $BOUNDARY_MAX ids)"; fi
if [ "$N_IMGS" != "1" ]; then echo "FAIL: snapshot holds $N_IMGS image records, want 1"; STATUS=1; fi
echo "== negative control: a DIFFERENT image in the re-sent prompt must NOT match =="
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
PAD = "<|image_pad|>"
# same geometry, same position, DIFFERENT rows -> a different ImgKey at the same start
user1 = ("<|im_start|>user\n<|vision_start|>" + PAD * 4 + "<|vision_end|>\n"
         "Describe this picture in one sentence.<|im_end|>\n<|im_start|>assistant\n")
ids1 = tok.encode(user1, parse_special=True)
open(out + "/ids1_other.txt", "w").write(",".join(map(str, ids1)))
PYEOF
echo "GENI 120 $OUT/emb_other.bin $(cat "$OUT/ids1_other.txt"),$REPLY,$(cat "$OUT/suffix.txt")" | \
    timeout 900 $E $ARGS > "$OUT/3.out" 2> "$OUT/3.err"
echo "run3: rc=$? tokens=$(grep -c '^T ' "$OUT/3.out")"
if grep -q "nvme promote" "$OUT/3.err"; then
    echo "FAIL: a different picture promoted - the ImgKey is not discriminating"; STATUS=1
else
    echo "PASS negative control: a different image at the same position does not promote"
fi

[ "$STATUS" -eq 0 ] && echo "ALL PASS" || echo "FAILED"
exit $STATUS
