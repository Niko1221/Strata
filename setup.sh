#!/bin/sh
# Strata for Linux: the first run installs everything and starts the model; later runs just start it.
# Needs only an NVIDIA driver (or, for an AMD Radeon card, the kernel's amdgpu driver: see docs/AMD_HIP.md).
# Python (with venv) is installed through apt/dnf if it is missing (asks for sudo).
# NixOS (or STRATA_UV=1) with uv installed: uv makes the project venv; every other Linux uses the standard venv.
cd "$(dirname "$0")" || exit 1
# Python 3.10+ that can make a venv WITH pip: Debian/Ubuntu ship `venv` without `ensurepip` (that is the separate
# python3-venv package), and a venv made without it has no pip
ok_py() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
# a .venv from an earlier run that failed half-way has a python but no pip: start it again
if [ -x .venv/bin/python ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  # uv's own Python only on NixOS (its Python cannot load pip wheels' libraries), or when asked: STRATA_UV=1.
  # Everywhere else the standard venv stays the default (#662: a silent toolchain change was not wanted).
  UV=""
  if [ "${STRATA_UV:-}" = "1" ] || grep -qi '^ID=nixos' /etc/os-release 2>/dev/null; then
    UV="$(command -v uv 2>/dev/null || true)"
  fi
  PY=""
  if [ -n "$UV" ]; then
    # Prefer uv's own interpreter: unlike Nix's Python, it works naturally with pip wheels using dlopen.
    PY=3.12
  else
    for c in python3 python; do
      if command -v $c >/dev/null 2>&1 && ok_py $c; then
        PY=$c; break
      fi
    done
  fi
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
    if ! ok_py $PY; then
      echo "Please install Python 3.10 or newer with venv (Ubuntu/Debian: sudo apt install python3-venv), then run"
      echo "./setup.sh again. On NixOS, add uv to environment.systemPackages."
      exit 1
    fi
  fi
  if [ -n "$UV" ]; then
    # Force an uv-managed interpreter: pip wheels may dlopen libraries outside the Nix store.
    "$UV" venv --managed-python --seed --python "$PY" .venv || { echo "Could not create the uv environment in .venv."; exit 1; }
  else
    # a private environment inside this folder (system Python stays untouched; newer distros refuse global pip)
    $PY -m venv .venv || { rm -rf .venv; echo "could not create .venv: sudo apt install python3-venv"; exit 1; }
  fi
fi
# Gentoo: nvcc builds CUDA's host code with the system gcc, and a gcc newer than CUDA supports (Gentoo moved to
# 15) dies in that build. gcc-config's x86_64-pc-linux-gnu-14 entry (or STRATA_GCC_CONFIG=<name>) names the gcc to
# build with; export $(gcc-config -E <index>) points PATH and LIBRARY_PATH at it, so cmake and nvcc see it.
# Without gcc-config (every other Linux) or without a matching entry, nothing changes and the build runs as before.
# STRATA_GCC_CONFIG= (empty) turns the step off even on Gentoo.
if command -v gcc-config >/dev/null 2>&1; then
  gcc_pat="${STRATA_GCC_CONFIG-x86_64-pc-linux-gnu-14}"
  gcc_i="$(gcc-config -l 2>/dev/null | awk -v pat="$gcc_pat" '$0 ~ pat { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+[)]$/) { sub(/[)]$/, "", $i); print $i; exit } }')"
  if [ -n "$gcc_pat" ] && [ -n "$gcc_i" ]; then
    echo "Gentoo: building with gcc-config entry $gcc_i ($gcc_pat)"
    for kv in $(gcc-config -E "$gcc_i" 2>/dev/null | grep '^[A-Za-z_][A-Za-z0-9_]*='); do
      export "$kv"
    done
  fi
fi
exec .venv/bin/python setup.py "$@"
