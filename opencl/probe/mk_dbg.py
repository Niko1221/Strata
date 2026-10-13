#!/usr/bin/env python3
"""Turn the peer's own generated row kernel into one that also reports what it computed.

The standalone repro shows the kernel text really does produce 447.458 on the peer's bytes, while an
independent re-implementation of the same text produces 322.  So the text reads differently than it looks.
This makes a copy that writes, per lane, the values it used - si, dw, dx, kbx, qa, qb - and the raw
pre-tree accumulator, so the disagreement can be located to a single expression instead of argued about.
"""

import sys

SRC = "ocl_src_IQ4_NL_at_40_calls_row.cl"
DST = "ocl_dbg_IQ4_NL.cl"

src = open(SRC).read()

# 1. the dbg argument, and a new name so both kernels can live in one program
old_sig = "            const uint wrow_bytes, const uint w_extra, const uint act_blk_row, const uint out_stride)"
assert old_sig in src
src = src.replace(
    old_sig,
    "            const uint wrow_bytes, const uint w_extra, const uint act_blk_row, const uint out_stride,\n"
    "            __global float* dbg)")
src = src.replace("void iq_row(", "void iq_row_dbg(")
# group 0 only: without this every group writes the same dbg slots and the table is a race, which is what
# made the first run of this build look like driver nondeterminism.
anchor_g = "    const uint lane = get_local_id(0);\n"
assert anchor_g in src
src = src.replace(anchor_g, anchor_g + "    if (get_group_id(0) != 0u) return;\n")

# 2. what each lane used, for row 0 of the group
anchor = "            acc[r] = fma(dw * dx, (float) si, acc[r]);\n"
assert anchor in src
src = src.replace(anchor, anchor + """            if (r == 0) {
                dbg[lane * 8u + 0u] = (float) si;
                dbg[lane * 8u + 1u] = dw;
                dbg[lane * 8u + 2u] = dx;
                dbg[lane * 8u + 3u] = (float) kbx;
                dbg[lane * 8u + 4u] = (float) k;
                dbg[lane * 8u + 5u] = as_float(qa);
                dbg[lane * 8u + 6u] = as_float(qb);
                dbg[lane * 8u + 7u] = acc[r];
            }
""")

# 3. the raw accumulator of every lane, before the tree
anchor2 = "    for (int r = 0; r < IQ_R; ++r) lred[lane * IQ_R + r] = acc[r];\n    barrier(CLK_LOCAL_MEM_FENCE);\n"
assert anchor2 in src
src = src.replace(anchor2, anchor2 + "    dbg[512u + lane] = acc[0];\n    dbg[576u + lane] = acc[1];\n")

open(DST, "w").write(src)
print(f"wrote {DST}")
sys.exit(0)
