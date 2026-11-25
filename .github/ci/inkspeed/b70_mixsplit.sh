#!/bin/bash
# Inkling 2 Mini on the Arc Pro B70 (SYCL), UD-Q4_K_XL only: where the "mix" and "router" groups of a decode step go, per kernel.
#   b70_mixsplit.sh <repo-root> <base-commit> <work-dir>
# Builds HEAD and the base commit, then: 2 alternating pairs of the C1 bench (the finer profiler marks must not change anything when off), the
# profiler with INK_SY_PROF=1 and INK_SY_PROF=2 (finer marks, with the measured cost of an empty wait), and the identical-answers check.
set -uo pipefail
REPO=$1; BASE=$2; W=$3
HF_UNSLOTH=https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main
F=Inkling-2-Mini-UD-Q4_K_XL.gguf
mkdir -p "$W/base-src" "$W/models" "$W/logs"
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
unset OMP_PLACES OMP_PROC_BIND GOMP_CPU_AFFINITY KMP_AFFINITY
echo "== host"; hostname; date; sycl-ls || true; df -h "$W" | tail -1; free -g | head -2
python3 -m venv "$W/venv" && "$W/venv/bin/pip" install -q --disable-pip-version-check cmake ninja
export PATH="$W/venv/bin:$PATH"

build_tree() {  # <src-root> <out-dir>
  local SRC=$1 OUT=$2
  mkdir -p "$OUT"
  cmake -S "$SRC/engine/src/inkling2" -B "$OUT" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DSTRATA_ENABLE_SYCL=ON -DSTRATA_SYCL_TARGETS=spir64,intel_gpu_bmg_g31 > "$OUT/cmake.log" 2>&1 || { tail -40 "$OUT/cmake.log"; return 1; }
  cmake --build "$OUT" --target strata-inkling -j "$(nproc)" > "$OUT/build.log" 2>&1 || { tail -60 "$OUT/build.log"; return 1; }
  ls -la "$OUT/strata-inkling" && "$OUT/strata-inkling" --version
}
echo "== build branch ($(git -C "$REPO" rev-parse --short HEAD))"
build_tree "$REPO" "$W/opt" || exit 11
echo "== build base ($BASE)"
git -C "$REPO" archive "$BASE" | tar -x -C "$W/base-src"
build_tree "$W/base-src" "$W/base" || exit 12
OPT=$W/opt/strata-inkling; BASEX=$W/base/strata-inkling

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0
bench() {  # <tag> <exe> <model> <env "K=V" or -> <bench args...>
  local tag=$1 exe=$2 model=$3 envs=$4; shift 4
  local line
  line=$(env $( [ "$envs" = "-" ] || echo "$envs" ) timeout 1500 "$exe" --bench --native "$model" --draft off --gpus 1 "$@" 2>"$W/logs/err.$tag.log" | grep -E '^\[bench\]' | head -1)
  echo "RESULT $tag | ${line:-FAILED}"
  grep -E '^\[ink_sy_prof\]' "$W/logs/err.$tag.log" | tail -1 | sed "s/^/PROF $tag | /"
  [ -n "$line" ] || tail -5 "$W/logs/err.$tag.log"
}

M="$W/models/$F"
echo "== model: download"
avail=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
[ "$avail" -ge 30 ] || { echo "not enough disk ($avail GB) for $F"; exit 13; }
t0=$(date +%s)
curl -fsSL --retry 3 --connect-timeout 30 --speed-limit 1000 --speed-time 60 -o "$M" "$HF_UNSLOTH/$F" || { echo "download failed $F"; exit 14; }
echo "downloaded $F $(stat -c %s "$M") bytes in $(( $(date +%s) - t0 )) s"
P="--prompt-len 512 --gen 128 --parallel 1"

echo "== 1. base vs branch, default flags, alternating, 2 pairs"
for r in 1 2; do
  bench "base-r$r" "$BASEX" "$M" - $P
  bench "opt-r$r"  "$OPT"   "$M" - $P
done
echo "== 2. profiler: coarse, then fine (each mark waits; the empty wait cost is printed), branch"
bench "opt-prof1" "$OPT" "$M" "INK_SY_PROF=1" $P
bench "opt-prof2" "$OPT" "$M" "INK_SY_PROF=2" $P
bench "opt-prof2b" "$OPT" "$M" "INK_SY_PROF=2" $P
echo "== 3. identical answers: logits of 64 ids (two runs each) and 48 greedy tokens of 96 ids, base against branch"
I64=$(seq -s, 101 164); I96=$(seq -s, 1000 1095); n=0
for tag in base opt base opt; do
  X=$OPT; [ $tag = base ] && X=$BASEX
  n=$((n + 1))
  timeout 1500 "$X" --native "$M" --gpus 1 --max-context 4096 --draft off --prompt-ids "$I64" --gen 16 --dump-ref "$W/logits.bin" > "$W/logs/id64.$n.log" 2>&1
  echo "IDENT logits64 $tag run $n sha=$(sha256sum "$W/logits.bin" 2>/dev/null | cut -c1-16) ids=$(tail -n 1 "$W/logs/id64.$n.log" | cut -c1-70)"
  rm -f "$W/logits.bin"
  timeout 1500 "$X" --native "$M" --gpus 1 --max-context 4096 --draft off --prompt-ids "$I96" --gen 48 > "$W/logs/g96.$n.log" 2>&1
  echo "IDENT greedy96 $tag run $n md5=$(tail -n 1 "$W/logs/g96.$n.log" | md5sum | cut -c1-12) $(tail -n 1 "$W/logs/g96.$n.log" | cut -c1-90)"
done
rm -f "$M"
echo "B70_INKMIX_DONE"
