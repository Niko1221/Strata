#!/bin/bash
# build_metal_feed.sh <sha-or-branch> [label] : build the Mac (Metal) side of a release candidate on m1002 and make a test-signed feed for the grid.
#
#   bash tools/release_grid/build/build_metal_feed.sh f-0221-rc0              # label rc-<first 9 of the sha>
#   bash tools/release_grid/build/build_metal_feed.sh a67333492 rc0-a67333492
#   WITH_QWEN=1 bash tools/release_grid/build/build_metal_feed.sh <sha>       # also the Qwen Metal engine
#
# Runs on the owner PC (it only makes the source archive and starts a jobq job; nothing heavy runs here). The job is wrapped as the Mac rules say:
# lockrun.py (FIFO lock with the memory / swap / other-user guards and a watchdog) and run_locked.sh (the GPU lock). It prints the job id; wait with
#   python phaseA-tests/jobq/jobq.py wait <id>      (as a background shell command)
# Result on m1002: ~/strata-m5/grid0221/feeds/<label>-metal  (grid.py run --cand <label>=<that path, absolute> --trust <path>/test-trust.json --boxes m1002 ...).
# Steps and checks: tools/release_grid/build/metal_inner.sh. The same command builds rc1: only the sha changes.
# Environment: GRID_REPO (a git checkout that has the commit; default this repo), GRID_SCRATCH (default C:/w-0221-grid-scratch), GRID_JOBQ, MAC_BOX (default m1002).
set -eu
REF=${1:?usage: build_metal_feed.sh <sha-or-branch> [label]}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${GRID_REPO:-$(cd "$HERE/../../.." && (pwd -W 2>/dev/null || pwd))}
SCRATCH=${GRID_SCRATCH:-C:/w-0221-grid-scratch}
JOBQ=${GRID_JOBQ:-C:/Users/AI-Server/Desktop/Strata/phaseA-tests/jobq/jobq.py}
BOX=${MAC_BOX:-m1002}
export MSYS_NO_PATHCONV=1
SHA=$(git -C "$REPO" rev-parse "$REF^{commit}")
ID12=${SHA:0:12}
LABEL=${2:-rc-${SHA:0:9}}
mkdir -p "$SCRATCH"
TGZ=$SCRATCH/src-$ID12.tgz
git -C "$REPO" archive --format=tar.gz -o "$TGZ" "$SHA"
TSHA=$(sha256sum "$TGZ" | cut -d' ' -f1)
INNER=metal_inner-$(sha256sum "$HERE/metal_inner.sh" | cut -c1-8).sh        # a new name for every version of the script: a running job never sees it change
RB='~/strata-m5/grid0221/build'
ssh -o BatchMode=yes "$BOX" "mkdir -p $RB"
scp -o BatchMode=yes -q "$TGZ" "$BOX:strata-m5/grid0221/build/src-$ID12.tgz"
scp -o BatchMode=yes -q "$HERE/metal_inner.sh" "$BOX:strata-m5/grid0221/build/$INNER"
NAME=grid-metal-$ID12
CMD="python3 ~/strata-m5/locks/lockrun.py $NAME --stream GRID --nice --max-min 170 -- ~/strata-mac/locks/run_locked.sh $NAME env WITH_QWEN=${WITH_QWEN:-0} bash ~/strata-m5/grid0221/build/$INNER $SHA ~/strata-m5/grid0221/build/src-$ID12.tgz $TSHA $LABEL"
echo "source $SHA ($ID12), archive sha256 $TSHA, label $LABEL, script $INNER"
python "$JOBQ" submit "$BOX" --name "$NAME" --timeout 3h --lock none -- "$CMD"
