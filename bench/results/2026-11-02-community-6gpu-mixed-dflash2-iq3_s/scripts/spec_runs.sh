#!/bin/bash
# Wider windows: does DFlash2 win when it may use more of its 7 drafts? Same image and files as the matrix, Swift 1.5 IQ3_S, decode prompts only.
set -uo pipefail
root=/tmp/strata-matrix
sed -n '1,/^docker stop -t 120 strata$/p' "$root/matrix.sh" | sed '$d' > "$root/lib.sh"
. "$root/lib.sh"
variant() { # model new-name spec min-p
  python3 - "$root/cfg/$1.json" "$root/cfg/$2.json" "$3" "$4" "$2" <<'PY'
import json, sys
src, dst, spec, minp, name = sys.argv[1:]
c = json.load(open(src))
a = c['args']
a[a.index('--spec') + 1] = spec
a[a.index('--spec-min-p') + 1] = minp
c['log'] = f'/out/{name}.current.engine.log'
json.dump(c, open(dst, 'w'), indent=1)
PY
}
docker stop -t 120 strata
for v in "s8p5 8 0.5" "s6p5 6 0.5" "s8p3 8 0.3"; do
  set -- $v
  variant swift "swift$1" "$2" "$3"
done
one s4-swift$( : )-mtp-s8p5     swifts8p5 mtp     decode
one s4-swift-dflash2-s8p5       swifts8p5 dflash2 decode
one s4-swift-dflash2-s6p5       swifts6p5 dflash2 decode
one s4-swift-mtp-s6p5           swifts6p5 mtp     decode
one s4-swift-dflash2-s8p3       swifts8p3 dflash2 decode
one s4-swift-mtp-s8p3           swifts8p3 mtp     decode
