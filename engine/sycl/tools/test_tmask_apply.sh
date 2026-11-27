#!/usr/bin/env bash
# Regression test for the SYCL device token mask's per-row bitmap (mask alignment lane, branch C).
#
# The T2 decision memo (notes/pr3800-mask-alignment-findings.md, 2026-11-14) pinned the corruption rriggs
# measured on the SYCL build (PR #3800 comment 6873068359: temp 0 6/6 corrupt, fallback only on rows 1-3,
# "took 1" forced lowest-ids; single-row windows clean) to tmask_apply_launch in sycl/src/core/verify.cpp
# reading the MKN bitmap at words[v >> 5] - the row offset t * n_words the call site's own comment promises
# ("row t is words + t * n_words") was missing, so every window row was masked with row 0's bitmap.
#
# The driver extracts the launch function VERBATIM from sycl/src/core/verify.cpp into a generated .inc,
# compiles it into test_tmask_apply.cpp (host shims for the SYCL/dpct surface) and runs it.  It therefore
# always tests the live source text: it fails while the offset is missing and passes once rows index their
# own bitmap row.  No SYCL toolchain or device is needed - plain g++ (or icpx) on the host.
#
#   sycl/tools/test_tmask_apply.sh          # exit 0 pass, 1 fail, 2 environment
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/../src/core/verify.cpp"
CXX=${CXX:-g++}
[ -f "$SRC" ] || { echo "usage: $0  (reads ../src/core/verify.cpp; CXX=g++ by default)"; exit 2; }
command -v "$CXX" >/dev/null 2>&1 || { echo "FAIL: no C++ compiler ($CXX)"; exit 2; }
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

# the function text: from the signature line to the closing brace at column 0.
awk '/^void tmask_apply_launch\(/ {f=1} f {print} f && /^\}/ {exit}' "$SRC" |
    tee "$OUT/tmask_apply_extracted.inc" | head -3
[ -s "$OUT/tmask_apply_extracted.inc" ] || { echo "FAIL: tmask_apply_launch not found in $SRC"; exit 1; }
grep -q 'q->submit' "$OUT/tmask_apply_extracted.inc" || { echo "FAIL: extraction did not capture the kernel"; exit 1; }

if ! "$CXX" -std=c++17 -Wall -I"$OUT" -o "$OUT/test" "$HERE/test_tmask_apply.cpp"; then
    echo "FAIL: the harness did not compile against the extracted kernel"
    exit 1
fi
"$OUT/test"
