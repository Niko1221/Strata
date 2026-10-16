"""tools/exl3 - reference implementation of the ExLlamaV3 EXL3 quantized weight format.

This is the Python specification of the format the engine's C++/HIP backend must match, in the
same spirit as tools/strata_pack.py: written first, in the language where it is cheap to test,
then ported.  See docs/EXL3.md for the format description and the source references.
"""
from .codebook import (CB_MCG, CB_MUL1, CB_3INST, BitsK, bits_from_K, k2_from_K,
                       decode, unpack_tile, windows_tile, windows_tile_fast,
                       codebook_lut, decode_lut)                                # noqa: F401
from .reconstruct import (HAD_BLOCK, tile_perm, tile_perm_inv, had128,
                          decode_weight_hat, weight_from_what, folded_from_what,
                          reconstruct_weight, folded_forward)                   # noqa: F401
