#!/bin/bash
# Inkling 2 Mini speed on the Arc Pro B70 runner (SYCL): the branch against its base, per model (UD-Q4_K_XL then SQ-20), alternating.
#   b70_speed.sh <repo-root> <base-commit> <work-dir>
# Prints "RESULT <tag> | <bench line>" and "PROF <tag> | <profiler line>"; a failed run prints "RESULT <tag> | FAILED" and its last stderr lines.
set -uo pipefail
REPO=$1; BASE=$2; W=$3
HF_STRATA=https://huggingface.co/strata-engine/Inkling-2-Mini-Strata/resolve/main
HF_UNSLOTH=https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main
mkdir -p "$W/base-src" "$W/models" "$W/logs"
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
unset OMP_PLACES OMP_PROC_BIND GOMP_CPU_AFFINITY KMP_AFFINITY
echo "== host"; hostname; date; sycl-ls || true; df -h "$W" | tail -1; free -g | head -2
echo "dmesg gpu hang/fault count before: $( (dmesg 2>/dev/null | grep -ciE 'GPU HANG|xe.*(reset|timeout)|i915.*hang') || echo n/a)"
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
OPT=$W/opt/strata-inkling; BASEX=$W/base/strata-inkling; ident_n=0

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0
bench() {  # <tag> <exe> <model> <env "K=V K=V" or -> <bench args...>
  local tag=$1 exe=$2 model=$3 envs=$4; shift 4
  local line
  line=$(env $( [ "$envs" = "-" ] || echo "$envs" ) timeout 1500 "$exe" --bench --native "$model" --draft off --gpus 1 "$@" 2>"$W/logs/err.$tag.log" | grep -E '^\[bench\]' | head -1)
  echo "RESULT $tag | ${line:-FAILED}"
  grep -E '^\[ink_sy_prof\]' "$W/logs/err.$tag.log" | tail -1 | sed "s/^/PROF $tag | /"
  grep -E "runner ready|VRAM experts" "$W/logs/err.$tag.log" | head -2 | cut -c1-260 | sed "s/^/INFO $tag | /"
  [ -n "$line" ] || tail -5 "$W/logs/err.$tag.log"
}

run_model() {  # <label> <url> <file>
  local L=$1 URL=$2 F=$3 M="$W/models/$3"
  echo "== model $L: download"
  local avail; avail=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
  [ "$avail" -ge 30 ] || { echo "not enough disk ($avail GB) for $F"; return 1; }
  local t0; t0=$(date +%s)
  curl -fsSL --retry 3 --connect-timeout 30 --speed-limit 1000 --speed-time 60 -o "$M" "$URL" || { echo "download failed $F"; return 1; }
  echo "downloaded $F $(stat -c %s "$M") bytes in $(( $(date +%s) - t0 )) s"
  local P="--prompt-len 512 --gen 128 --parallel 1"

  echo "== $L 1. base vs branch, default flags, alternating, 3 pairs"
  for r in 1 2 3; do
    bench "$L-base-r$r" "$BASEX" "$M" - $P
    bench "$L-opt-r$r"  "$OPT"   "$M" - $P
  done
  echo "== $L 2. where the time goes (branch)"
  bench "$L-opt-timing" "$OPT" "$M" "INK_SY_TIMING=1" $P
  bench "$L-opt-prof"   "$OPT" "$M" "INK_SY_PROF=1"   $P
  echo "== $L 3. identical answers: logits of 64 ids (two runs each) and 48 greedy tokens of 96 ids, base against branch"
  local I64 I96; I64=$(seq -s, 101 164); I96=$(seq -s, 1000 1095)
  for tag in base opt base opt; do
    local X=$OPT; [ $tag = base ] && X=$BASEX
    ident_n=$((ident_n + 1))
    timeout 1500 "$X" --native "$M" --gpus 1 --max-context 4096 --draft off --prompt-ids "$I64" --gen 16 --dump-ref "$W/logits.bin" > "$W/logs/id64.$ident_n.log" 2>&1
    echo "IDENT $L logits64 $tag run $ident_n sha=$(sha256sum "$W/logits.bin" 2>/dev/null | cut -c1-16) ids=$(tail -n 1 "$W/logs/id64.$ident_n.log" | cut -c1-70)"
    rm -f "$W/logits.bin"
    timeout 1500 "$X" --native "$M" --gpus 1 --max-context 4096 --draft off --prompt-ids "$I96" --gen 48 > "$W/logs/g96.$ident_n.log" 2>&1
    echo "IDENT $L greedy96 $tag run $ident_n md5=$(tail -n 1 "$W/logs/g96.$ident_n.log" | md5sum | cut -c1-12) $(tail -n 1 "$W/logs/g96.$ident_n.log" | cut -c1-90)"
  done
  echo "== $L 5. parallel (C8 / C32 total decode), base vs branch"
  for par in 8 32; do
    bench "$L-base-par$par" "$BASEX" "$M" - --prompt-len 512 --gen 64 --parallel $par --max-context 4096
    bench "$L-opt-par$par"  "$OPT"   "$M" - --prompt-len 512 --gen 64 --parallel $par --max-context 4096
  done
  rm -f "$M"
}

run_model q4kxl "$HF_UNSLOTH/Inkling-2-Mini-UD-Q4_K_XL.gguf" Inkling-2-Mini-UD-Q4_K_XL.gguf
run_model sq20  "$HF_STRATA/Inkling-2-Mini-SQ-20.strata"      Inkling-2-Mini-SQ-20.strata
echo "dmesg gpu hang/fault count after: $( (dmesg 2>/dev/null | grep -ciE 'GPU HANG|xe.*(reset|timeout)|i915.*hang') || echo n/a)"
echo "B70_INKSPEED_DONE"
