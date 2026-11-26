"""Which Ornith-1.5-35B-A3B (GGUF architecture qwen35moe) files the engine can load: the same engine path as Qwen3.6-35B-A3B,
so the loader checks are q36's (strata_control/q36_types.py, `problems`) and are reused as they are.  This module is the data
half: the family's sizes and its file name pattern, and the size a file under another name is taken as.

The files are the plain quantizations (Q4_K_M, Q5_K_M, Q6_K, Q8_0), not Unsloth's Dynamic ones, so the size a file is read as
follows its routed experts' gate type directly: Q8_0 -> Q8_0, Q6_K -> Q6_K, Q5_K -> Q5_K_M, anything else -> Q4_K_M.
"""

from __future__ import annotations

#: setup.py OWN_PATH_FAMILIES["ornith"]["models"] and its file name pattern (the same rule as q36_types.py's)
FAMILY = "ornith"
SIZES = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
FILE = "Ornith-1.5-35B-{q}.gguf"


def size_for(file_name: str, types: dict[str, int]) -> tuple[str, bool]:
    """(the catalog size setup installs this file as, whether that is the published file's own name). A file under another name is
    taken as the size whose routed experts it is closest to: Q8_0 gate/up -> Q8_0, Q6_K -> Q6_K, Q5_K -> Q5_K_M, anything else ->
    Q4_K_M (setup uses the size only for its RAM and context advice; the file is read as it is)."""
    for q in SIZES:
        if file_name == FILE.format(q=q):
            return q, True
    gate = {t for n, t in types.items() if n.endswith("ffn_gate_exps.weight")}
    if gate == {8}:
        return "Q8_0", False
    if gate == {14}:
        return "Q6_K", False
    if gate == {13}:
        return "Q5_K_M", False
    return "Q4_K_M", False
