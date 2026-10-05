#!/bin/sh
# Strata for Linux: the first run installs everything and starts the model; later runs just start it.
# Needs an NVIDIA driver (or, for an AMD Radeon card, the kernel's amdgpu driver: see docs/AMD_HIP.md).
# Pascal (sm_60) and Volta (sm_70) cannot use the CUDA 13 engine. This launcher picks CUDA 12.6 or 12.8
# and compiles for sm_60, sm_70 and sm_86 (RTX 30). Python (with venv) is installed through apt/dnf if missing.
cd "$(dirname "$0")" || exit 1

# Python 3.10+ that can make a venv WITH pip: Debian/Ubuntu ship `venv` without `ensurepip` (that is the separate
# python3-venv package), and a venv made without it has no pip
ok_py() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }

# a .venv from an earlier run that failed half-way has a python but no pip: start it again
if [ -x .venv/bin/python ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && ok_py "$c"; then
      PY=$c; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.10+ with venv is needed; installing it (sudo will ask for your password) ..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y python3 python3-pip
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --noconfirm python python-pip
    fi
    PY=python3
    if ! ok_py "$PY"; then
      echo "Please install Python 3.10 or newer with venv (Ubuntu/Debian: sudo apt install python3-venv), then run"
      echo "./setup.sh again."
      exit 1
    fi
  fi
  # a private environment inside this folder (system Python stays untouched; newer distros refuse global pip)
  "$PY" -m venv .venv || { rm -rf .venv; echo "could not create .venv: sudo apt install python3-venv"; exit 1; }
fi

# --- CUDA 12.6 / 12.8 for sm_60, sm_70, sm_86 ---------------------------------
# setup.py compiles those three into every CUDA 12 engine. sm_100/sm_120 need 12.8; sm_60/sm_70 cannot use CUDA 13.
has_flag() {
  for a in "$@"; do
    case "$a" in
      --cuda|--cuda=*|--build|--backend|--backend=*) return 0 ;;
    esac
  done
  return 1
}

nvcc_minor() {
  # prints 6 or 8 when this nvcc is CUDA 12.6 or 12.8, else nothing
  _nv="$1"
  [ -x "$_nv" ] || return 1
  _rel=$("$_nv" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p' | head -n 1)
  [ "$_rel" = "12 6" ] && { echo 6; return 0; }
  [ "$_rel" = "12 8" ] && { echo 8; return 0; }
  return 1
}

pick_nvcc() {
  # $1 is 6 or 8. echoes the nvcc path.
  _want="$1"
  _cands=""
  if [ -n "$STRATA_NVCC" ]; then
    _cands="$STRATA_NVCC"
  fi
  _cands="$(command -v nvcc 2>/dev/null)"
  if [ -n "$CUDA_HOME" ]; then _cands="$_cands $CUDA_HOME/bin/nvcc"; fi
  if [ -n "$CUDA_PATH" ]; then _cands="$_cands $CUDA_PATH/bin/nvcc"; fi
  _cands="$_cands /usr/local/cuda-12.$_want/bin/nvcc /usr/local/cuda/bin/nvcc /opt/cuda-12.$_want/bin/nvcc /opt/cuda/bin/nvcc"
  for _n in $_cands; do
    [ -n "$_n" ] || continue
    _m=$(nvcc_minor "$_n") || continue
    [ "$_m" = "$_want" ] || continue
    echo "$_n"
    return 0
  done
  return 1
}

need_cuda12=0
have_old=0
have_sm86=0
have_sm100=0
if command -v nvidia-smi >/dev/null 2>&1; then
  _caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')
  for _cap in $_caps; do
    _sm=$(printf '%s' "$_cap" | awk -F. '{printf "%d%d", $1, $2}')
    case "$_sm" in
      60|61|70) have_old=1; need_cuda12=1 ;;
      86) have_sm86=1; need_cuda12=1 ;;
      100|120) have_sm100=1 ;;
    esac
  done
fi

extra=""
if ! has_flag "$@"; then
  if [ "$need_cuda12" -eq 1 ]; then
    # 12.8 if an sm_100/sm_120 card shares the engine, or if only 12.8 is installed; else 12.6.
    _minor=""
    if [ "$have_sm100" -eq 1 ]; then
      _nv=$(pick_nvcc 8) && _minor=8
    else
      _nv=$(pick_nvcc 8) && _minor=8
      if [ -z "$_minor" ]; then
        _nv=$(pick_nvcc 6) && _minor=6
      fi
    fi
    if [ -n "$_minor" ]; then
      export STRATA_NVCC="$_nv"
      export STRATA_CUDA="12.$_minor"
      export STRATA_EXPERIMENTAL_SM60=1
      extra="--build --cuda 12.$_minor"
      echo "GPU needs the CUDA 12 engine (sm_60 / sm_70 / sm_86): compiling with CUDA 12.$_minor ($_nv)"
    elif [ "$have_old" -eq 1 ]; then
      echo "sm_60 or sm_70 detected, but no CUDA 12.6 or 12.8 toolkit was found."
      echo "CUDA 13 cannot compile those cards. Install 12.6 or 12.8 from"
      echo "https://developer.nvidia.com/cuda-toolkit-archive then re-run:"
      echo "  STRATA_NVCC=/usr/local/cuda-12.6/bin/nvcc ./setup.sh --build --cuda 12.6"
      echo "  STRATA_NVCC=/usr/local/cuda-12.8/bin/nvcc ./setup.sh --build --cuda 12.8"
      exit 1
    else
      echo "sm_86 detected and no CUDA 12.6/12.8 toolkit is installed: using the CUDA 13 engine (sm_86 is supported there)."
      echo "For a 12.6 or 12.8 build: STRATA_NVCC=/usr/local/cuda-12.8/bin/nvcc ./setup.sh --build --cuda 12.8"
    fi
  fi
fi

# Stock setup.py only accepts --cuda 12|13|auto. A patched one accepts 12.6 and 12.8.
# Rewrite the flag for the stock parser and keep STRATA_NVCC pointed at that toolkit.
cuda_choice_ok() {
  grep -q '12\.6' setup.py 2>/dev/null && grep -q 'choices=\["12", "12.6", "12.8", "13", "auto"\]' setup.py
}
set -- $extra "$@"
extra=""
if ! cuda_choice_ok; then
  _rewritten=""
  _skip=0
  for _a in "$@"; do
    if [ "$_skip" -eq 1 ]; then
      case "$_a" in
        12.6|12.8)
          _minor=${_a#12.}
          _nv=$(pick_nvcc "$_minor") || _nv="$STRATA_NVCC"
          [ -n "$_nv" ] && export STRATA_NVCC="$_nv"
          export STRATA_EXPERIMENTAL_SM60=1
          unset STRATA_CUDA
          _rewritten="$_rewritten --cuda 12"
          echo "this setup.py does not accept --cuda $_a yet; using --cuda 12 with STRATA_NVCC=$STRATA_NVCC"
          ;;
        *) _rewritten="$_rewritten --cuda $_a" ;;
      esac
      _skip=0
      continue
    fi
    case "$_a" in
      --cuda=12.6|--cuda=12.8)
        _minor=${_a#--cuda=12.}
        _nv=$(pick_nvcc "$_minor") || _nv="$STRATA_NVCC"
        [ -n "$_nv" ] && export STRATA_NVCC="$_nv"
        export STRATA_EXPERIMENTAL_SM60=1
        unset STRATA_CUDA
        _rewritten="$_rewritten --cuda 12"
        echo "this setup.py does not accept $_a yet; using --cuda 12 with STRATA_NVCC=$STRATA_NVCC"
        ;;
      --cuda) _skip=1 ;;
      *) _rewritten="$_rewritten $_a" ;;
    esac
  done
  # shellcheck disable=SC2086
  set -- $_rewritten
fi
# The venv is created empty. An already-installed model skips setup.py's package step, so install
# here too — same packages the first run used to put in .venv, no manual pip.
if [ -f requirements.txt ]; then
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
else
  .venv/bin/python -m pip install --disable-pip-version-check numpy jinja2 regex pyyaml tqdm requests pillow psutil
fi
# shellcheck disable=SC2086
exec .venv/bin/python setup.py "$@"
