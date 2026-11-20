"""strata_control/gguf_dialects.py - Strict GGUF quant dialect typing, byte computation, and overflow bounds."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DialectType(str, Enum):
    STANDARD = "standard"
    PRISM = "prism"
    TURBOQUANT = "turboquant"


@dataclass(frozen=True)
class QuantDialect:
    type_id: int
    name: str
    dialect: DialectType
    block_size: int
    type_size: int


# Standard GGML types
STANDARD_DIALECTS = {
    0: QuantDialect(0, "F32", DialectType.STANDARD, 1, 4),
    1: QuantDialect(1, "F16", DialectType.STANDARD, 1, 2),
    2: QuantDialect(2, "Q4_0", DialectType.STANDARD, 32, 18),
    3: QuantDialect(3, "Q4_1", DialectType.STANDARD, 32, 20),
    8: QuantDialect(8, "Q8_0", DialectType.STANDARD, 32, 34),
}

# Prism / Bonsai / TurboQuant custom type mappings
PRISM_DIALECTS = {
    142: QuantDialect(142, "PQ2_0", DialectType.PRISM, 128, 34),
    143: QuantDialect(143, "PTQ1_0", DialectType.PRISM, 128, 28),
    1001: QuantDialect(1001, "PQ2_0", DialectType.PRISM, 64, 18),
    1002: QuantDialect(1002, "PTQ1_0", DialectType.PRISM, 64, 10),
    2001: QuantDialect(2001, "TQ4_1S", DialectType.TURBOQUANT, 32, 20),
    2002: QuantDialect(2002, "TQ3_0S", DialectType.TURBOQUANT, 32, 15),
}

TURBOQUANT_DIALECTS = {k: v for k, v in PRISM_DIALECTS.items() if v.dialect == DialectType.TURBOQUANT}


def resolve_dialect_type(type_id: int, explicit_dialect: str | None = None) -> QuantDialect:
    """Resolve a numeric quant type id into its QuantDialect, requiring explicit dialect on collisions."""
    if explicit_dialect == "prism" and type_id in PRISM_DIALECTS:
        return PRISM_DIALECTS[type_id]
    if explicit_dialect == "turboquant" and type_id in PRISM_DIALECTS:
        return PRISM_DIALECTS[type_id]
    if explicit_dialect is None:
        if type_id in STANDARD_DIALECTS:
            return STANDARD_DIALECTS[type_id]
        if type_id in PRISM_DIALECTS:
            return PRISM_DIALECTS[type_id]
    elif type_id in PRISM_DIALECTS and PRISM_DIALECTS[type_id].dialect.value == explicit_dialect:
        return PRISM_DIALECTS[type_id]
    elif type_id in STANDARD_DIALECTS and explicit_dialect == "standard":
        return STANDARD_DIALECTS[type_id]

    raise ValueError(f"Unknown quantization type id: {type_id} (explicit_dialect={explicit_dialect})")


# Alias for compatibility
get_dialect_descriptor = resolve_dialect_type


def compute_tensor_bytes(dims: list[int], block_size: int, type_size: int) -> int:
    """Checked computation of tensor byte footprint with overflow and block alignment guards."""
    if not dims:
        raise ValueError("Tensor dimensions must not be empty")
    if block_size <= 0 or type_size <= 0:
        raise ValueError("block_size and type_size must be positive")

    ne0 = dims[0]
    if ne0 % block_size != 0:
        raise ValueError(f"Leading dimension {ne0} is not a multiple of block size {block_size}")

    row_blocks = ne0 // block_size
    row_bytes = row_blocks * type_size

    # Check subsequent dimensions
    total_rows = 1
    max_elements = (1 << 63) - 1

    for d in dims[1:]:
        if d <= 0 or d > 1_000_000_000:
            raise ValueError(f"Invalid dimension {d}")
        if total_rows > max_elements // d:
            raise ValueError("Tensor dimension multiplication overflow")
        total_rows *= d

    if row_bytes > max_elements // total_rows:
        raise ValueError("Tensor total bytes overflow")

    return row_bytes * total_rows
