"""Explicit identity for the paper's research codec; no backend compatibility.

Groups default to eight values, scales use the input floating dtype, signed
codes have 2**bits - 1 levels, and storage uses little-endian packed bytes.
Prism, TheTom and TurboQuant format identifiers must never refer to this codec.
"""

from .core import dequantize as dequantize_reference_absmax
from .core import quantize as quantize_reference_absmax

CODEC_ID = "paper-reference-absmax-group8-v1"

__all__ = [
    "CODEC_ID",
    "quantize_reference_absmax",
    "dequantize_reference_absmax",
]
