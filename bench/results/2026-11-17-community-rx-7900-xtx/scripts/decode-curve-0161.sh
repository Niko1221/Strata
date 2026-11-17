#!/bin/bash
# decode-curve-0161.sh — the UD-IQ3_XXS switch A/B on the CURRENT engine (0.1.61,
# build-0161-hip): two arms x five tiers x 2 reps, one-shot cells, arms interleaved
# within each tier to cancel drift.
#   arm A = GSQ-RCO IQ3_S (the incumbent pack) on build-0161-hip
#   arm B = unsloth UD-IQ3_XXS (the switch candidate) on build-0161-hip
# Protocol = decode-curve-0155.sh verbatim (same pinned env/flags, MemoryMax=66G scopes,
# trap-restore); ONLY the pack/shards differ between arms. The engine delta vs the 11-07
# baseline (0.1.55) is thereby removed from the comparison entirely - same-window,
# same-engine, same-script. Verdict: per-tier paired medians B vs A.
set -u
EXP=~/strata-rocm-experiments/20261003-231142
ROOT=~/projects/strata-amd
E=~/projects/strata-r160/build-0161-hip/strata
UD=/mnt/<disk>/models/flash-next-ud-iq3xxs/UD-IQ3_XXS
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/decode-curve-0161-$STAMP.log
exec > >(tee -a "$LOG") 2>&1
restore_8081() {
    systemctl --user start strata-8081.service 2>/dev/null
    for i in $(seq 1 40); do
        curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && { echo "RESTORE_8081 ok"; return 0; }
        sleep 4
    done
    echo "RESTORE_8081 FAILED"
}
trap restore_8081 EXIT
memav() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
echo "== preflight $(date -u +%H:%M:%SZ); MemAvailable $(memav) MiB =="
"$E" --version
git -C ~/projects/strata-r160 log --oneline -1
systemctl --user stop strata-8081.service
for i in $(seq 1 30); do [ "$(memav)" -ge 68000 ] && break; sleep 4; done
echo "8081: $(systemctl --user is-active strata-8081.service); MemAvailable: $(memav) MiB"

cell() {  # cell <arm> <tier> <rep> <tokens> <maxctx>
    local arm="$1" tier="$2" rep="$3" tok="$4" ctx="$5"
    local PACK NATIVE PLE
    if [ "$arm" = A ]; then
        PACK=$EXP/packs/iq3_s
        NATIVE=~/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
        PLE=~/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
    else
        PACK=$EXP/packs/ud_iq3_xxs
        NATIVE=$UD/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
        PLE=$UD/Qwen3.8-Flash-Next-UD-IQ3_XXS-00004-of-00004.gguf
    fi
    local OUT=$ROOT/artifacts/rocm/logs/decode-curve-0161-$arm-$tier-r$rep-$STAMP.log
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$EXP/sdk/lib \
        STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 STRATA_ROUTE_PRIOR=0.5 STRATA_GDN_CHUNKED=1 \
        STRATA_PF_GEMM=1 STRATA_PF_PAD=1 STRATA_KV_RING_ADAPT=1 STRATA_PREFILL_TIMING=1 \
        systemd-run --user --scope -p MemoryMax=66G \
        "$E" --tokens-file "$tok" --max-new 256 --max-context "$ctx" \
        --pack "$PACK" \
        --native "$NATIVE" \
        --ple-gguf "$PLE" \
        --mtp /mnt/<disk>/models/mtp/rt \
        --expert-profile $EXP/data/expert-profile.bin \
        --expert-cache auto --spec 5 --kv int8 --kv-resident 49152 --prefill 8192 \
        --prompt-cache 6 --spec-min-p 0.70 --vram-reserve-mib 3072 --lookup-chain 2 > "$OUT" 2>&1
    local rc=$? p d
    p=$(grep -oE "prefill +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1 | grep -oE "[0-9.]+")
    d=$(grep -oE "decode +[0-9]+ tokens in [0-9.]+ ms +-> +[0-9.]+ tok/s" "$OUT" | grep -oE "[0-9.]+ tok/s" | head -1 | grep -oE "[0-9.]+")
    echo "CELL $arm $tier r$rep rc=$rc prefill=$p decode=$d"
}
WL=$ROOT/artifacts/rocm/workloads
for r in 1 2; do
    cell A 128k $r $WL/qsa-profile-131072.tokens 139264
    cell B 128k $r $WL/qsa-profile-131072.tokens 139264
    cell A 262k $r $WL/qsa-profile-261888.tokens 262144
    cell B 262k $r $WL/qsa-profile-261888.tokens 262144
    cell A 64k  $r $WL/qsa-profile-65536.tokens  73728
    cell B 64k  $r $WL/qsa-profile-65536.tokens  73728
    cell A 32k  $r $WL/qsa-profile-32768.tokens  40960
    cell B 32k  $r $WL/qsa-profile-32768.tokens  40960
    cell A 1k   $r $WL/qsa-profile-1024.tokens   4096
    cell B 1k   $r $WL/qsa-profile-1024.tokens   4096
done
echo "== MEDIANS + PAIRED VERDICT =="
python3 - "$STAMP" <<'PY'
import glob, re, statistics
stamp = __import__("sys").argv[1]
base = f"~/projects/strata-amd/artifacts/rocm/logs/decode-curve-0161-{{arm}}-{{tier}}-r{{rep}}-{stamp}.log"
def grab(arm, tier, rep):
    t = open(base.format(arm=arm, tier=tier, rep=rep)).read()
    m = re.search(r"prefill +\d+ tokens in [\d.]+ ms +-> +([\d.]+) tok/s", t)
    d = re.search(r"decode +\d+ tokens in [\d.]+ ms +-> +([\d.]+) tok/s", t)
    return (float(m.group(1)) if m else None, float(d.group(1)) if d else None)
print(f"{'tier':5s} {'A decode med':>13s} {'B decode med':>13s} {'B/A %':>7s}   prefill A med | B med")
for tier in ("1k", "32k", "64k", "128k", "262k"):
    decA, decB, preA, preB, pairs = [], [], [], [], []
    for rep in (1, 2):
        pa, da = grab("A", tier, rep)
        pb, db = grab("B", tier, rep)
        if da and db: pairs.append(db / da); decA.append(da); decB.append(db)
        if pa: preA.append(pa)
        if pb: preB.append(pb)
    if pairs:
        med = statistics.median(pairs) * 100
        print(f"{tier:5s} {statistics.median(decA):13.1f} {statistics.median(decB):13.1f} {med:6.2f}%"
              f"   {statistics.median(preA) if preA else 0:7.0f} | {statistics.median(preB) if preB else 0:7.0f}")
    else:
        print(f"{tier:5s} NO CLEAN PAIRS")
PY
echo "== done (trap restores 8081) =="
