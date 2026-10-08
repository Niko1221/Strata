#!/usr/bin/env python3
"""Exhaustive packed Q2_0 signed-byte witness; no model or GPU required."""
def original(q):
    return sum((((q >> (2*i)) & 3) - 1 & 255) << (8*i) for i in range(4))

def spread(q):
    c = (q | (q << 12)) & 0x000f000f
    c = (c | (c << 6)) & 0x03030303
    return ((c + 0x7f7f7f7f) ^ 0x80808080) & 0xffffffff

for word in range(65536):
    for q in (word, word >> 8):
        assert spread(q) == original(q), (word, q)
print("PASS: 131072 packed outputs match for all 65536 words")
