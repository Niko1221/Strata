"""AMD GPUs on Windows by PCI device id and by marketing name, as one table.

setup.py's GPU listing and engine_component's engine matching read this file, so both answer the same for
the same card.  The PCI device id answers first; a card's name answers only when its id is not in
DEVICE_ARCH, and only for desktop and workstation names: a laptop SKU that shares a desktop part's number
(an RX 6800M is not an RX 6800) says nothing about its chip.
"""

from __future__ import annotations

import re

#: Dedicated AMD GPUs by PCI device id (VEN_1002): the architecture Strata runs the card as.
DEVICE_ARCH = {0x744C: "gfx1100", 0x7448: "gfx1100", 0x745E: "gfx1100",       # RX 7900 XTX/XT/GRE, W7900, W7800
               0x747E: "gfx1101",                                              # RX 7800 XT / 7700 XT
               0x7480: "gfx1102",                                              # RX 7600 / 7600 XT
               0x7590: "gfx1200",                                              # RX 9060 XT
               0x7550: "gfx1201", 0x7551: "gfx1201",                           # RX 9070 / 9070 XT, AI PRO R9700
               0x73BF: "gfx1030", 0x73AF: "gfx1030", 0x73A5: "gfx1030",        # RX 6800 / 6800 XT / 6900 XT / 6950 XT
               0x73A3: "gfx1030",                                              # Radeon PRO W6800 (pci.ids)
               0x73DF: "gfx1031"}                                              # RX 6700 XT / 6750 XT / 6800M (#881)

#: The integrated Radeon Strata runs on: Strix Halo, unified memory (docs/STRIX_HALO.md).  A card like any other.
UMA_ARCH = {0x1586: "gfx1151"}                                                 # pci.ids: one id for 8040S / 8050S / 8060S

#: Integrated Radeons with no Strata build for their gfx, with the gfx they are: never "the GPU" of a PC.
IGPU_ARCH = {0x150E: "gfx1150",                                                # Strix Point: Radeon 880M / 890M
             0x1114: "gfx1152", 0x1902: "gfx1152",                             # Krackan Point (Krackan2): Radeon 840M / 860M
             0x15BF: "gfx1103", 0x15C8: "gfx1103", 0x164F: "gfx1103",          # Phoenix: Radeon 740M / 760M / 780M
             0x1900: "gfx1103", 0x1901: "gfx1103"}                             # Hawk Point

#: More integrated Radeons seen on Windows.  No gfx is claimed for them; like IGPU_ARCH they can only say
#: "not this one" - they never decide which engine a PC needs.
IGPU_OLD_IDS = frozenset({0x1681, 0x1638, 0x1636, 0x15D8, 0x15E7,
                          0x13C0})                                           # Granite Ridge: the integrated Radeon Graphics of a Ryzen 9000 desktop CPU

#: Every integrated Radeon id (with or without a known gfx): these never decide which engine a PC needs.
IGPU_IDS = frozenset(IGPU_ARCH) | IGPU_OLD_IDS


def arch_from_id(device_id: int | None) -> str:
    """The architecture a Windows AMD PCI device id (VEN_1002) says the adapter is ("" when the id is unknown)."""
    return DEVICE_ARCH.get(device_id) or UMA_ARCH.get(device_id) or IGPU_ARCH.get(device_id) or ""


#: Desktop and workstation names -> the architecture they imply, checked in order when the PCI device id is
#: not in DEVICE_ARCH.  Laptop suffixes stay unlisted on purpose: the number alone would name the wrong
#: chip (RX 6800M, RX 7700S).
NAME_ARCH = (
    (re.compile(r"\b80[456]0S\b"), "gfx1151"),                                 # Radeon 8040S / 8050S / 8060S (Strix Halo)
    (re.compile(r"RYZEN AI MAX"), "gfx1151"),
    (re.compile(r"\b8[89]0M\b"), "gfx1150"),                                   # Radeon 880M / 890M (Strix Point)
    (re.compile(r"\b8[46]0M\b"), "gfx1152"),                                   # Radeon 840M / 860M (Krackan Point)
    (re.compile(r"\b7[468]0M\b"), "gfx1103"),                                  # Radeon 740M / 760M / 780M (Phoenix, Hawk Point)
    (re.compile(r"\bRX 9070\b|\b9070\b|\bR9700\b|\bR97\d\d\b|\bAI PRO R97"), "gfx1201"),   # RX 9070 / 9070 XT, AI PRO R9700
    (re.compile(r"\bRX 906\d\b|\b9060\b|\bR96\d\d\b|\bAI PRO R96"), "gfx1200"),            # RX 9060 XT
    (re.compile(r"RX\s*7900|W7900|W7800"), "gfx1100"),                         # RX 7900 XTX/XT/GRE, Radeon PRO W7900 / W7800
    (re.compile(r"RX\s*7800|RX\s*7700(?!S)|W7700"), "gfx1101"),                # RX 7800 XT / 7700 XT, Radeon PRO W7700
    (re.compile(r"RX\s*7600|RX\s*7500|W7600|W7500"), "gfx1102"),               # RX 7600 / 7600 XT, Radeon PRO W7600 / W7500
    (re.compile(r"RX\s*6800(?!\s*[MS])|RX\s*6900|RX\s*6950|W6800"), "gfx1030"),            # RX 6800 / 6900 / 6950 series, Radeon PRO W6800;
    #   the 6800M / 6800S laptops are the one exception: that number is not Navi 21 there (#881), so their name says nothing
    (re.compile(r"RX\s*6700|RX\s*6750"), "gfx1031"),                 # RX 6700 XT / 6750 XT / 6750 GRE, laptop SKUs too (same chip)
    (re.compile(r"RX\s*66\d\d"), "gfx1032"),                         # RX 6600 / 6650: listed, no Strata build for gfx1032
    (re.compile(r"RX\s*6500|RX\s*6400"), "gfx1034"),                 # RX 6500 / 6400: listed, no Strata build for gfx1034
)


def arch_from_name(name: str) -> str:
    """The AMD architecture a card's marketing name implies ("" when the name says nothing: an integrated
    Radeon, a laptop SKU, an unknown card).  The last hint when the PCI device id is not in DEVICE_ARCH."""
    text = re.sub(r"\s+", " ", str(name or "")).upper()
    if not any(word in text for word in ("AMD", "RADEON", "RYZEN")):
        return ""
    return next((gfx for rx, gfx in NAME_ARCH if rx.search(text)), "")
