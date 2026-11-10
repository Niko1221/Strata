"""The picture reader and a graphics card that has no room for it: what its output says, and the words.

The GPU picture reader (strata-vision, llama.cpp's mtmd) asks the card for its weights and work buffers when it starts.  A card that another program
has filled answers "cudaMalloc failed: out of memory" (the same text on AMD cards, whose runtime ggml names the same way) and ggml aborts: the reader cannot
print an answer of its own.  Its lines end up in the engine log (its stderr is the log):

    ggml_backend_cuda_buffer_type_alloc_buffer: allocating 865.48 MiB on device 0: cudaMalloc failed: out of memory
    alloc_tensor_range: failed to allocate CUDA0 buffer of size 907524736
    ggml-backend.cpp:188: GGML_ASSERT(buffer) failed

`find` reads them (card number inside the reader's own numbering, and how much it asked for); the two sentences below are what the person reads, in the
Terminal's launch summary (the reader moved to the processor) and in the error (the launch ended because of it).  One wording for the server and the
Terminal.  Pure functions, no I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_ALLOC = re.compile(r"allocating (\d+(?:\.\d+)?) MiB on device (\d+): (?:cuda|hip)Malloc failed: out of memory", re.I)
_BUFFER = re.compile(r"failed to allocate (?:CUDA|ROCm|HIP|MUSA)(\d+) buffer of size (\d+)", re.I)
_GENERIC = re.compile(r"(?:cuda|hip)Malloc failed: out of memory", re.I)


@dataclass(frozen=True)
class NoRoom:
    """`device`: the card's number as the reader counts (the cards its environment shows it), None when the lines do not say; `need_mib`: what it asked for."""
    device: int | None = None
    need_mib: float | None = None


def find(text_or_lines) -> NoRoom | None:
    """The reader's out-of-memory lines in `text_or_lines` (a string, or a list of lines), or None.  The last allocation that failed is the one reported."""
    lines = text_or_lines.splitlines() if isinstance(text_or_lines, str) else [str(x) for x in text_or_lines or []]
    alloc = buffer = None
    generic = False
    for line in lines:
        m = _ALLOC.search(line)
        if m:
            alloc = NoRoom(int(m.group(2)), float(m.group(1)))
            continue
        m = _BUFFER.search(line)
        if m:
            buffer = NoRoom(int(m.group(1)), int(m.group(2)) / 1048576.0)
            continue
        if _GENERIC.search(line):
            generic = True
    return alloc or buffer or (NoRoom() if generic else None)


def physical_card(device: int | None, visible) -> int | None:
    """The card's number as nvidia-smi counts, from the reader's number and the list of cards its environment shows it (CUDA_VISIBLE_DEVICES, the engine's
    card order); the reader's own number when it sees them all.  None when it cannot be said."""
    if device is None:
        return None
    if isinstance(visible, str):
        visible = [x for x in re.split(r"[,\s]+", visible.strip()) if x]
    if visible:
        try:
            return int(str(visible[device]).strip())
        except (IndexError, ValueError):
            return None
    return device


def _gb(mib: float) -> str:
    gb = mib / 1024.0
    return f"{gb:.1f} GB" if gb >= 0.05 else "less than 0.1 GB"


def _facts(need_mib: float | None, free_mib: float | None) -> str:
    """The figures in brackets: what the reader needs and what the card has; only what is known, nothing when neither is."""
    parts = []
    if need_mib:
        parts.append(f"it needs about {_gb(need_mib)}")
    if free_mib is not None:
        parts.append(f"{_gb(free_mib)} are free")
    return " (" + "; ".join(parts) + ")" if parts else ""


def _card(card: int | None) -> str:
    return f"graphics card {card}" if card is not None else "the graphics card"


def on_processor(card: int | None = None, need_mib: float | None = None, free_mib: float | None = None) -> str:
    """The launch goes on and the pictures are read on the processor (the Terminal's summary line)."""
    return (f"Pictures are read on the processor, which is slower: {_card(card)} has no room for the picture reader{_facts(need_mib, free_mib)}. "
            "To read them on the card, close the other programs that use it and start again.")


def could_not_start(card: int | None = None, need_mib: float | None = None, free_mib: float | None = None) -> tuple[str, str]:
    """(what happened, the next step) when the launch ends because the reader could not start."""
    return (f"The picture reader could not start: {_card(card)} has no room for it{_facts(need_mib, free_mib)}.",
            "Close the other programs that use the card and start again, or add --vision off to start without pictures.")
