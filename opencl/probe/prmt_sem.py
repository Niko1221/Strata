#!/usr/bin/env python3
"""What does CUDA's get_int_from_table_16 actually return?

It is the one helper behind every IQ4 decode (IQ4_NL and IQ4_XS) and it is built out of
__byte_perm, whose selector semantics are easy to misread: four 4-bit selectors, and bit 3 of
each one means "replicate the sign of the selected byte" rather than "byte index 8-15".
Emulating prmt exactly settles the nibble -> result-byte mapping, which is what the OpenCL
transcription has to reproduce.
"""


def prmt(a, b, s):
    out = 0
    for j in range(4):
        n = (s >> (4 * j)) & 0xF
        idx = n & 0x7
        byte = (a >> (8 * idx)) & 0xFF if idx < 4 else (b >> (8 * (idx - 4))) & 0xFF
        if n & 0x8:                                   # sign replicate
            byte = 0xFF if (byte & 0x80) else 0x00
        out |= byte << (8 * j)
    return out


def u32(v):
    return v & 0xFFFFFFFF


def get_int_from_table_16(q4, table):
    """table: 16 int8 values.  q4: the 32-bit word holding 8 nibbles."""
    t32 = []
    for i in range(4):
        w = 0
        for k in range(4):
            w |= (table[4 * i + k] & 0xFF) << (8 * k)
        t32.append(w)
    lhi = u32(0x32103210 | ((q4 & 0x88888888) >> 1))
    tmp = []
    for i in range(2):
        shift = 16 * i
        low = prmt(t32[0], t32[1], u32(q4 >> shift))
        high = prmt(t32[2], t32[3], u32(q4 >> shift))
        tmp.append(prmt(low, high, u32(lhi >> shift)))
    x = prmt(tmp[0], tmp[1], 0x6420)
    y = prmt(tmp[0], tmp[1], 0x7531)
    return x, y


def as_int8s(w):
    return [(w >> (8 * i)) & 0xFF for i in range(4)]


def main():
    table = list(range(16))                      # values[k] == k, so the result prints the indices
    q4 = 0x76543210                             # nibbles n0..n7 = 0,1,2,3,4,5,6,7
    x, y = get_int_from_table_16(q4, table)
    print("table = identity, q4 = 0x76543210 (nibbles 0..7 in byte order)")
    print("  v.x bytes =", as_int8s(x), " -> weight positions", as_int8s(x))
    print("  v.y bytes =", as_int8s(y), " -> weight positions", as_int8s(y))
    print()
    print("read as: weight position p (0..15) pairs with the result byte that holds p")
    print()

    # the same thing with the real codebook, so a sign bug would show up too
    kvalues_iq4nl = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
    x, y = get_int_from_table_16(q4, kvalues_iq4nl)
    print("with kvalues_iq4nl: v.x =", [kvalues_iq4nl[i] for i in as_int8s(x)],
          " v.y =", [kvalues_iq4nl[i] for i in as_int8s(y)])

    # and the shuffled case: nibble 3 in the high half, to see the "select the other half" path
    q4b = 0x7654321A                          # n0 = 0xA (>= 8) -> must come from values[8..15]
    x, y = get_int_from_table_16(q4b, table)
    print("q4 = 0x7654321A (n0 = 10):  v.x =", as_int8s(x), " v.y =", as_int8s(y))


if __name__ == "__main__":
    main()
