#!/usr/bin/env python3
"""A faithful variant of the peer's IQ4_NL row kernel that reports each lane's own contribution.

The `dbg`-suffixed build was a dead end: its stores come back different on every run while the kernel's
real output is bit-identical every time, so the driver is not keeping them where they were written.  This
one keeps the loop body *byte for byte* from the generated source - only the epilogue changes - and
writes each lane's accumulator straight into the output buffer, which the kernel already proves it writes
deterministically.

    ocl_lane_IQ4_NL.cl   iq_lane: outv[r*IQ_WG + lane] = that lane's contribution to row r
"""

import re
import sys

SRC = "ocl_src_IQ4_NL_at_40_calls_row.cl"
DST = "ocl_lane_IQ4_NL.cl"

src = open(SRC).read()

# the loop body, verbatim: from the k loop down to the accumulator store
body_start = src.index("    for (uint k = lane;")
body_end = src.index("    // The reference tree, spelled out")
body = src[body_start:body_end]
assert "acc[r] = fma(dw * dx" in body, "the accumulator store moved"

# helpers only: everything before the kernel
head_end = src.index("// `wt` is the arena holding")
helpers = src[:head_end]

out = helpers + """__kernel __attribute__((reqd_work_group_size(IQ_WG, 1, 1)))
void iq_lane(__global const uchar* wt, __global const uint* xq,
             __global const uint* t2, __global const uint* t3, __global const uchar* kv,
             __global float* outv, __global const ulong* ent_off, __global const uint* ent_act,
             const uint n_entries, const uint bpe, const uint n_row_e,
             const uint wrow_bytes, const uint w_extra, const uint act_blk_row, const uint out_stride)
{
    const uint lane = get_local_id(0);
    // ONLY group 0: every group would otherwise write the same outv slots and race, which is exactly the
    // trap the first debug build fell into.
    if (get_group_id(0) != 0u) return;
    const uint e    = get_group_id(0) / bpe;
    if (e >= n_entries) return;
    const uint row0 = (get_group_id(0) - e * bpe) * (uint) IQ_R;

    __global const uchar* W  = wt + ent_off[e] + w_extra;
    __global const uint*  XW = xq + (size_t) ent_act[e] * act_blk_row * 9u;

    float acc[IQ_R];
    #pragma unroll
    for (int r = 0; r < IQ_R; ++r) acc[r] = 0.0f;

""" + body + """
    // the epilogue the real kernel replaces with the tree: every lane's own number, in the clear
    #pragma unroll
    for (int r = 0; r < IQ_R; ++r)
        outv[(size_t) r * IQ_WG + lane] = acc[r];
}
"""

open(DST, "w").write(out)
print(f"wrote {DST}: body {len(body)} chars")
sys.exit(0)
