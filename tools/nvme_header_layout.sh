# tools/nvme_header_layout.sh - the NVMe snapshot header layout, taken from the header that DEFINES it.
#
# Sourced by tools/nvme_p0_test.sh and tools/nvme_steps123_test.sh.  Both read a snapshot's ids at
# `sizeof(NvmeHeader)`, and until step 4 both were TOLD that number by hand: a hardcoded `104`, then `HDR=208`
# when step 3 widened the header, then `HDR=216` when v4 added the weight-set fingerprint.  Those scripts are
# edited far more often than they are run, and a header whose size moved is exactly the change a hand-written offset
# gets wrong while still printing plausible ids.
#
# It moved twice more, and exactly ONE oracle - tools/nvme_failure_contract_test.sh - still spelled the number out
# by hand.  Worth being precise about what that would have cost at v4, because the obvious guess is wrong and the
# measurement is the reason this file exists: with the real 216-byte header, the old literal computed an offset
# 8 bytes early, which lands 56 bytes into the GDN segment - still inside it.  So that oracle would have kept
# corrupting GDN bytes and kept passing: right by luck, not by construction, and drifting 8 bytes per header
# change on an assertion whose entire job is to be exactly where it says it is.  A passing oracle is not evidence
# that an oracle tested its own name, and "it still passes" is the strongest argument for leaving a hand-written
# constant alone.  It is gone.
#
# So the number has one spelling - `kNvmeHeaderBytes` / `kNvmeFormatVersion` in
# include/strata/platform/kv_nvme.hpp - which the `static_assert`s under `NvmeHeader` pin at build time, and this
# file reads.  If the constants are renamed or reshaped the read fails and the oracles refuse to run rather than
# parsing a snapshot at the wrong offset; and `nvme_snapshot_ids` checks the VERSION FIELD OF THE FILE against
# `kNvmeFormatVersion`, so a snapshot written by a different build is caught too (the engine these scripts run is
# `/local/strata/build/strata`, which is not necessarily the tree these scripts live in).
#
# Sets HDR and NVME_VERSION; `nvme_snapshot_ids FILE IDS_OUT` prints the snapshot's L on stdout.
# `NVME_PYTHON` overrides the interpreter (the oracles use the checkout's own venv).

nvme_header_layout() {   # $1 = repo root whose header the scripts must agree with
  local h="$1/include/strata/platform/kv_nvme.hpp"
  if [ ! -r "$h" ]; then
    echo "FAIL: no NVMe header at $h" >&2
    return 1
  fi
  HDR=$(sed -n 's/^[[:space:]]*inline constexpr uint64_t kNvmeHeaderBytes = \([0-9]\{1,\}\);.*/\1/p' "$h")
  NVME_VERSION=$(sed -n 's/^[[:space:]]*inline constexpr uint32_t kNvmeFormatVersion = \([0-9]\{1,\}\);.*/\1/p' "$h")
  if [ -z "$HDR" ] || [ -z "$NVME_VERSION" ]; then
    echo "FAIL: could not read kNvmeHeaderBytes / kNvmeFormatVersion from $h - the constants the static_asserts" >&2
    echo "      pin changed shape, and this file has to change with them" >&2
    return 1
  fi
}

nvme_snapshot_ids() {   # $1 = snapshot file, $2 = file to write the comma-separated ids to; prints L
  local py="${NVME_PYTHON:-python3}"
  "$py" - "$1" "$2" "$HDR" "$NVME_VERSION" <<'PYEOF'
import struct, sys
path, out, hdr, want = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(path, "rb") as f:
    magic, version, L = struct.unpack("<IIq", f.read(16))
    if magic != 0x5E564D45:
        sys.exit("not a strata NVMe snapshot (%s: magic %08x)" % (path, magic))
    if version != want:
        sys.exit("snapshot %s is format version %d; this build writes %d - the engine that wrote it is not the"
                 " tree these scripts read the header from" % (path, version, want))
    f.seek(hdr)
    ids = struct.unpack("<%di" % L, f.read(4 * L))
open(out, "w").write(",".join(map(str, ids)))
print(L)
PYEOF
}
