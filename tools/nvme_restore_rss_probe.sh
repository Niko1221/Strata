#!/bin/bash
# C10 evidence: the whole-file staging buffer, measured (docs/nvme-kv-cache-convergence.md C10).
# nvme_restore reads the ENTIRE snapshot into one std::vector before validating anything, so peak RSS during a
# restore should exceed the engine's steady-state RSS by roughly the file size.  This script measures it.
set -u
ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
cd /local/strata
E=${NVME_ENGINE:-/local/strata/.worktrees/converge/build/strata}
OUT=/tmp/nvme-rss
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH-}
NVME_PYTHON=/local/strata/.venv/bin/python
ARGS="--serve --pack packs/iq3_xxs
 --native models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
 --ple-gguf models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
 --expert-profile data/expert-profile.bin --expert-cache auto --prefill 2048
 --spec 4 --spec-min-p 0.5 --mtp mtp/rt --max-context 131072 --kv int8
 --kv-resident 20480 --prompt-cache 12 --adapt-swaps 0"
ARGS=$(echo "$ARGS" | tr '\n' ' ')
rm -rf "$OUT"; mkdir -p "$OUT"

# a ~30k-token prompt: at ~44 KB of snapshot per consumed token (P0 measured 180 MB / 4107 tokens) this is a
# ~1.3 GB snapshot - the scale the convergence doc's C10 section talks about.
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
filler = ("Describe the storm, the ships, the lamp room, the keeper's routine, the logbook, the rocks, "
          "the fog bell, the supply boat, and the winter isolation in vivid detail. ") * 1500
ids = tok.encode("<|im_start|>user\n" + filler + "<|im_end|>\n<|im_start|>assistant\n", parse_special=True)
open(out + "/pids.txt", "w").write(",".join(map(str, ids)))
print("prompt tokens:", len(ids), file=sys.stderr)
PYEOF
P=$(cat "$OUT/pids.txt")

echo "== generate a large snapshot (prefill ~$(( $(awk -F, '{print NF}' "$OUT/pids.txt") / 1000 ))k tokens) =="
echo "GEN 4 $P" | timeout 900 $E $ARGS --nvme-dump $OUT/big.bin > "$OUT/gen.out" 2> "$OUT/gen.err"
echo "gen: rc=$?"
ls -la "$OUT/big.bin"

echo "== restore it, sampling the engine's RSS every 100 ms =="
echo "GEN 4 $P" | $E $ARGS --nvme-restore "$OUT/big.bin" > "$OUT/res.out" 2> "$OUT/res.err" &
EP=$!
PEAK=0; BASE=0; T0=$(date +%s%N)
while kill -0 $EP 2>/dev/null; do
    RSS=$(awk '/VmRSS/{print $2}' "/proc/$EP/status" 2>/dev/null)
    [ -n "$RSS" ] || { sleep 0.05; continue; }
    [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
    [ "$BASE" -eq 0 ] && grep -q "session is up" "$OUT/res.err" 2>/dev/null && BASE=$RSS
    sleep 0.1
done
wait $EP; RC=$?
echo "restore: rc=$RC  peak RSS: $((PEAK / 1024)) MiB  engine steady-state RSS (at 'session is up'): $((BASE / 1024)) MiB"
FILE_KB=$(stat -c %s "$OUT/big.bin"); echo "snapshot size: $((FILE_KB / 1024 / 1024)) MiB"
echo "staging buffer over steady-state: $(( (PEAK - BASE) / 1024 )) MiB"
grep -E "RESUME|nvme_restore" "$OUT/res.out" "$OUT/res.err" 2>/dev/null | head -2
echo "elapsed: $(( ($(date +%s%N) - T0) / 1000000 )) ms"
