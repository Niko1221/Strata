#!/bin/bash
# metal_inner.sh SHA SRC_TGZ SRC_TGZ_SHA256 LABEL : the part of build_metal_feed.sh that runs ON the Mac (m1002), inside both Mac locks (lockrun.py + run_locked.sh).
# Builds the Inkling-2-Mini Metal shape engine (strata-inkling) and the macOS arm64 Terminal archive from the sources in SRC_TGZ (a `git archive` of SHA),
# then makes a TEST-SIGNED local feed in ~/strata-m5/grid0221/feeds/LABEL-metal that the grid's cell_runner.py can install from through install.sh.
# Recipe: phaseA-tests/rel0220/build/mac/rel220_mac_engines.sh and rel220_mac_term.sh (the 0.2.20 release builds), with the commit and version taken from the arguments.
# Optional: WITH_QWEN=1 also builds the Qwen Metal engine package (tools/packaging/mac/build_engine.sh) and puts it in the feed (about 10 more minutes).
# Our folders only (~/strata-m5/grid0221, read-only use of the tool folders in ~/strata-mac); nothing outside them is touched. The test key stays in
# ~/strata-m5/grid0221/feed-test-keys (made once, never printed, never in git).
set -eo pipefail
SHA=${1:?SHA}; TGZ=${2:?SRC_TGZ}; TSHA=${3:?SRC_TGZ_SHA256}; LABEL=${4:?LABEL}
ID12=${SHA:0:12}
G=$HOME/strata-m5/grid0221
W=$G/build-$ID12
OLD=$HOME/strata-mac/rel029
FEED=$G/feeds/$LABEL-metal
KEYS=$G/feed-test-keys
export CMAKE=$HOME/strata-mac/t027/venv/bin/cmake
export NINJA=$HOME/strata-mac/t027/venv/bin/ninja
export METAL_CPP_DIR=$HOME/strata-mac/metal-cpp
export PATH=$OLD/pbs/python/bin:$HOME/strata-mac/t027/venv/bin:$PATH
export STRATA_COMMIT=$SHA

memgate() {
  local lv
  lv=$(sysctl -n kern.memorystatus_level)
  echo "memgate $1: level $lv, swap $(sysctl -n vm.swapusage | cut -c1-60), free disk $(df -h /System/Volumes/Data | tail -1 | awk '{print $4}')"
  [ "$lv" -ge 35 ] || { echo "MEMORY WARNING: level $lv < 35 before stage $1: stopping"; exit 30; }
}

echo "== metal feed build $SHA label $LABEL start $(date)"
s=$(shasum -a 256 "$TGZ" | cut -d' ' -f1)
[ "$s" = "$TSHA" ] || { echo "ABORT: source archive sha256 $s != $TSHA"; exit 2; }
e=$( (gzip -dc "$TGZ" || true) | git get-tar-commit-id 2>/dev/null )
[ "$e" = "$SHA" ] || { echo "ABORT: the archive embeds commit '$e', not $SHA"; exit 2; }
free_gb=$(df -g /System/Volumes/Data | tail -1 | awk '{print $4}')
[ "$free_gb" -ge 100 ] || { echo "ABORT: only $free_gb GB free (floor 100 GB)"; exit 3; }

memgate extract
python3 -c "import shutil; shutil.rmtree('$W', ignore_errors=True)"
mkdir -p "$W/src" "$W/ink-stage" "$W/ink-out" "$W/term-out" "$W/term-cache"
tar -xzf "$TGZ" -C "$W/src"
echo "$ID12" > "$W/src/BUILD_ID"
cd "$W/src"
EV=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' engine/CMakeLists.txt | head -1)
TV=$EV                                    # package_terminal.py names the archive with the engine version of the source (engine/CMakeLists.txt)
echo "engine version $EV, terminal version $TV"
[ -n "$EV" ] && [ -n "$TV" ] || { echo "ABORT: cannot read the versions"; exit 4; }

# ---- 1. strata-inkling (Metal, embedded kernels), the same cmake call as the 0.2.20 release build ----
memgate inkling-engine
echo "== strata-inkling $EV $(date)"
IB=$W/ink-build
mkdir -p "$IB"
"$CMAKE" -S "$W/src/engine/src/inkling2" -B "$IB" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
  -DCMAKE_CXX_FLAGS="-DINKLING_VERSION=\"\\\"$EV\\\"\" -DINK_VERSION=\"\\\"$EV\\\"\"" \
  -DCMAKE_OBJCXX_FLAGS="-DINKLING_VERSION=\"\\\"$EV\\\"\" -DINK_VERSION=\"\\\"$EV\\\"\"" > "$W/ink-cfg.log" 2>&1 || { tail -30 "$W/ink-cfg.log"; exit 5; }
nice -n 19 "$CMAKE" --build "$IB" --target strata-inkling -j 6 > "$W/ink-build.log" 2>&1 || { tail -30 "$W/ink-build.log"; exit 5; }
tail -2 "$W/ink-build.log"
cp "$IB/strata-inkling" "$W/ink-stage/strata-inkling"
chmod 755 "$W/ink-stage/strata-inkling"
python3 -c "import json, pathlib; pathlib.Path('$W/ink-stage/BUILD.json').write_text(json.dumps({'version': '$EV', 'shape': 'inkling2', 'backend': 'metal', 'platform': 'macos-arm64', 'build_id': '${ID12}+metal', 'commit': '$SHA', 'kernels': 'embedded', 'macos_min': '15.0', 'portable': True}, indent=1) + '\n')"
python3 tools/packaging/package_engine.py --src "$W/ink-stage" --shape inkling2 --backend metal --version "$EV" --platform macos-arm64 \
  --entrypoint strata-inkling --arch apple-gpu --build-id "${ID12}+metal" --out "$W/ink-out"
IPKG=$W/ink-out/$EV-macos-arm64-metal
python3 tools/packaging/package_engine.py --verify "$IPKG"
python3 tools/packaging/binary_deps.py "$IPKG"
python3 tools/packaging/binary_deps.py --mac-min-os "$IPKG"
IV=$("$IPKG/strata-inkling" --version)
echo "strata-inkling --version: $IV"
echo "$IV" | grep -q "$EV" || { echo "ABORT: strata-inkling --version does not say $EV"; exit 6; }

# ---- 2. optional Qwen Metal engine ----
QARGS=""
if [ "${WITH_QWEN:-0}" = 1 ]; then
  memgate qwen-engine
  echo "== strata-engine (Qwen Metal) $EV $(date)"
  bash tools/packaging/mac/build_engine.sh --version "$EV" --build-id "$ID12" --jobs 6 --work "$W/qwen-work"
  QARGS="--engine $W/qwen-work/pkg-mac/$EV-macos-arm64-metal"
fi

# ---- 3. the macOS arm64 Terminal archive ----
memgate terminal
echo "== terminal $TV $(date)"
cp -R "$OLD/term-cache/." "$W/term-cache/"
nice -n 19 python3 tools/packaging/package_terminal.py --out "$W/term-out" --target darwin-arm64 --cache "$W/term-cache" --commit "$SHA" > "$W/term-build.log" 2>&1 || { tail -20 "$W/term-build.log"; exit 7; }
TARCH=$W/term-out/strata-terminal-$TV-darwin-arm64.tar.gz
[ -f "$TARCH" ] || { echo "ABORT: no $TARCH"; ls -la "$W/term-out"; exit 7; }
ls -la "$TARCH"

# ---- 4. the test-signed feed: Terminal (+ Qwen) through make_local_feed.py, the Inkling shape engine added with --shapes ----
echo "== feed $FEED $(date)"
mkdir -p "$G/feeds"
# shellcheck disable=SC2086
python3 tools/packaging/make_local_feed.py --out "$FEED" --terminal "$TARCH" $QARGS --keys "$KEYS" --url http://127.0.0.1:8099
rm -rf "$W/ink-signed"
cp -R "$IPKG" "$W/ink-signed"
python3 tools/packaging/feed_tool.py sign-package "$W/ink-signed" --key "$KEYS/feed-key.json"
python3 tools/packaging/feed_tool.py pack-engine "$W/ink-signed" --out "$FEED/shapes" --url-prefix shapes/
EARG=""
[ -z "$QARGS" ] || EARG="--engine $FEED/engine"
# shellcheck disable=SC2086
python3 tools/packaging/feed_tool.py build-feed --out "$FEED" --key "$KEYS/feed-key.json" --sequence 1 $EARG --control "$FEED/control" --shapes "$FEED/shapes" --no-dsv41-carry
python3 tools/packaging/feed_tool.py verify-feed "$FEED" --pub "$KEYS/feed-key.pub.json" || true
ls -la "$FEED" "$FEED/shapes" "$FEED/control"
echo "FEED $FEED"
echo "TRUST $FEED/test-trust.json"
echo "METAL-FEED-DONE $(date)"
