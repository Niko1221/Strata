"""The one fit label of a model on this computer (owner, 2026-11-08): the model's weight files, WITHOUT the n-gram / PLE lookup table (it stays on the
SSD and is not counted), against the graphics memory plus the RAM.  They fit: "fits".  They do not: "may run slower".

    size  = the catalog's memory figure (`memory_gb`, the "Needs (RAM + VRAM)" column): the experts and the dense weights, the lookup table left out
            (DeepSeek: the plan's `weights_gb`, the set without its Engram table)
    room  = RAM + the card's memory (the largest card that counts; a chip that shares the RAM only its own carve-out; DeepSeek: every card)

No reserve is taken off either side: the same figures the screens already show.  Nothing here refuses anything; a model that "may run slower" can still be started.
"""

from __future__ import annotations

FITS = "fits"
SLOWER = "may run slower"


def card_room_gb(cards: list[dict] | None) -> float:
    """The graphics memory that counts for one model: the largest card, an integrated chip only for its own carve-out (its shared memory is the RAM)."""
    best = 0.0
    for c in cards or []:
        gb = float(c.get("dedicated_gb") or 0.0) if c.get("uma") else float(c.get("vram_gb") or 0.0)
        best = max(best, gb)
    return best


def label(size_gb: float, ram_gb: float | None, vram_gb: float | None) -> str:
    """"fits" when the weights are no bigger than the RAM and the card's memory together, else "may run slower"."""
    return FITS if float(size_gb) <= float(ram_gb or 0.0) + float(vram_gb or 0.0) else SLOWER
