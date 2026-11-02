"""Which drafter a model uses to guess ahead: DFlash2, MTP or none. One place, read by the Terminal and the app.

One drafter per run. The engine is started with the DFlash2 drafter or with the MTP layer, never with both (owner, 2026-11-02): the one that is not
chosen is not loaded, so its graphics memory goes to the expert cache.

The choice is the person's, and it is saved in the model's run config as ``"drafting"``: ``auto`` (the default), ``mtp``, ``dflash2`` or ``off``.
``strata launch --drafting ...`` writes it, the app's Models screen writes it, and ``serve/server.py`` applies it at every start (like ``"mtp": false``,
strata_control/draft_layer.py), so a setup run that rewrites the engine arguments cannot undo it. An older config that names both drafters (``--mtp`` and
``--draft dflash2``) starts with the one chosen.

  dflash2  ``--draft dflash2 --draft-model <pack> --draft-fallback none --mtp-draft-vocab <file> --spec 6 --draft-conf 0.35`` and no ``--mtp``: the DFlash2
           block drafter (incoai/Qwen3.8-Flash-Next-DFlash2, Q4_0 pack) for Qwen3.8-Flash-Next and Gyro, with the window and confidence that measured best
           (RESULTS-DRAFT.md section 4); a ``--spec`` or ``--draft-conf`` the person set by hand (``user_args``) stays
  mtp      the model's own draft layer (``--mtp <folder>``) and none of the DFlash2 flags
  off      no draft layer and no DFlash2 (the same as ``"mtp": false``: the engine still looks ahead in the text with ``--spec``)
  auto     DFlash2 wherever the engine has it (DFLASH2_BACKENDS) and the model is Qwen3.8-Flash-Next or Gyro, the MTP everywhere else

Recommend, never force: a choice that cannot be applied (DFlash2 in a layer split or with several requests at once, no drafter file yet) does not stop the
start. The other drafter runs if it is on this computer, else none, and ``Decision.note`` says why in one short line. Nothing is loaded twice and the
engine never switches drafter mid-run.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODES = ("auto", "mtp", "dflash2", "off")
DEFAULT_MODE = "auto"
LABELS = {"auto": "Auto (recommended)", "mtp": "MTP", "dflash2": "DFlash2", "off": "Off"}
DRAFTERS = ("mtp", "dflash2", "off")

#: The engine families that have the DFlash2 code (CMakeLists.txt STRATA_HAVE_DFLASH2). Auto picks DFlash2 on these; a computer with any other engine keeps the
#: MTP. Intel SYCL joined with the DF2-SYCL engine (Arc A750 measured: a tie with the MTP, section 8-9 of phaseA-tests/blackwell/RESULTS-DF2SYCL.md). Metal: the DF2-METAL engine.
DFLASH2_BACKENDS = ("cuda", "hip", "metal", "sycl")

#: Graphics memory each drafter takes on Qwen3.8-Flash-Next IQ3_S (GB; phaseA-tests/blackwell/RESULTS-DF2MEM.md section 1, l620, engine free memory at
#: the first graph capture): DFlash2 alone 890 MiB, the MTP layer 1082 MiB. The fit, the plan and the slot estimates use the one that is chosen.
DRAFTER_MIB = {"dflash2": 890, "mtp": 1082, "off": 0}
DRAFTER_GB = {name: round(mib / 1000, 2) for name, mib in DRAFTER_MIB.items()}      # 0.89 and 1.08, the figures the apps show

#: GPU architecture (what nvidia-smi's compute capability and the kernel driver report) -> card class
ARCH_CLASS = (
    (re.compile(r"^gfx1151$"), "strix_halo"),
    (re.compile(r"^gfx120[01]$"), "rdna4"),
    (re.compile(r"^sm_12\d$"), "blackwell"),
    (re.compile(r"^sm_86$"), "ampere"),
)


def card_class(archs: Any) -> str:
    """The class of the first card whose architecture the table knows (``sm_120``, ``gfx1151``...); ``other`` for an unknown or empty list."""
    for arch in ([archs] if isinstance(archs, str) else list(archs or [])):
        for pattern, name in ARCH_CLASS:
            if pattern.match(str(arch).strip().lower()):
                return name
    return "other"


# --------------------------------------------------------------------------------------------------- where the engine can run it
#
# The DFlash2 code (src/core/dflash2.cpp, src/kernels/cuda/dflash2.cu) is compiled into the engines of DFLASH2_BACKENDS (CMakeLists.txt, STRATA_HAVE_DFLASH2);
# the others have none, and a PC with no supported graphics card has no engine to run it. Where the code exists, three words say what was
# measured (DFlash2 against MTP, same prompts, same flags; the numbers are in the comment above SUGGESTED and in phaseA-tests/blackwell/RESULTS-DRAFT.md):
#   tested    measured on this kind of card: faster on some kinds of text, slower on others
#   slower    measured on this kind of card: slower on every prompt
#   untested  the engine has the code for this card but it was never measured on it

#: What the Terminal's Drafting step and the app's paragraph under the dropdown say (the same words; apps/host/src/locales/df2-en.json, tests keep them equal).
#: The numbers are measured: l620, RTX PRO 5000, Qwen3.8-Flash-Next IQ3_S, DFlash2 `--spec 6 --draft-conf 0.35` against MTP (phaseA-tests/blackwell/RESULTS-DRAFT.md section 4).
EXPLAIN_NUMBERS = ("MTP is the default. DFlash2 is another way to guess the next words ahead, for Qwen3.8-Flash-Next only. On an RTX PRO 5000 it was 7.8% faster than MTP "
                   "on 19 short prompts and 1.1% faster on long thinking, and faster on 21 of 23 prompts. On the short prompts the answers were identical.")
EXPLAIN_LIMITS = "It downloads about 1.2 GB once, works with one graphics card and one request at a time, and uses MTP otherwise."

LINE_TESTED = "On graphics cards like yours, DFlash2 was faster than MTP on some kinds of text and slower on others in our tests."
LINE_SLOWER = "On graphics cards like yours, DFlash2 was slower than MTP in our tests."
LINE_UNTESTED = "DFlash2 has not been tested on your graphics card yet."


@dataclass(frozen=True)
class Support:
    can_run: bool        # this engine build has the DFlash2 code for the card
    level: str           # none | tested | slower | untested
    line: str            # plain words for the person ("" when there is nothing to say)


_TESTED = {("cuda", "sm_120"), ("cuda", "sm_121"), ("hip", "gfx1151"), ("hip", "gfx1201"), ("sycl", "arc_a")}
_SLOWER = {("cuda", "sm_86")}


_HARDWARE: list = []

# --- Intel: the SYCL path gives no architecture, so the card is told by its name (Windows display adapters) or its PCI id (Linux sysfs).
#   arc_a        Arc A-series, Alchemist: A3xx / A5xx / A7xx (the A750 was measured: RESULTS-DF2SYCL.md), PCI ids 0x56A0..0x56AF and the mobile 0x5690..0x5692
#   arc_a_pro    Arc Pro A-series (A40, A50, A60): not measured
#   arc_b        Arc B-series, Battlemage (B580, Pro B60, B65, B70...): not measured
#   intel_igpu   integrated graphics (UHD, Iris Xe, "Arc Graphics" of Core Ultra): not measured
_ARC_NAME = re.compile(r"\bArc\b(?:\(TM\))?\s*(Pro\s+)?([AB])\s?(\d{2,3})(?!\d)", re.I)


def sycl_arch_of_name(name: str) -> str:
    """The class of an Intel adapter from its Windows name: ``Intel(R) Arc(TM) A750 Graphics`` is ``arc_a``; ``intel_igpu`` for any other Intel name."""
    m = _ARC_NAME.search(str(name or ""))
    if not m:
        return "intel_igpu"
    pro, letter, number = bool(m.group(1)), m.group(2).upper(), int(m.group(3))
    if letter == "B":
        return "arc_b"
    return "arc_a" if not pro and number >= 300 else "arc_a_pro"   # A310 / A380 / A580 / A750 / A770 and the mobile A3xx-A7xx; A40 / A50 / A60 and every Pro are not measured


def sycl_arch_of_pci(driver: str, devid: str) -> str:
    """The class of an Intel card from its Linux driver and PCI device id (strata_control/intel_spec.py's cards): i915 ids 0x56A0.. are Alchemist, 0x56B0.. the Pro
    A-series, xe ids 0xE2xx Battlemage."""
    try:
        n = int(str(devid).lower().replace("0x", ""), 16)
    except ValueError:
        return "intel_igpu"
    if driver == "xe" and 0xE200 <= n <= 0xE2FF:
        return "arc_b"
    if driver == "i915" and 0x5690 <= n <= 0x56AF:
        return "arc_a"
    if driver == "i915" and 0x56B0 <= n <= 0x56BF:
        return "arc_a_pro"
    return "intel_igpu"


def sycl_archs_present() -> list[str]:
    """One class (``arc_a`` ...) per Intel card: from the display adapters on Windows, from sysfs on Linux. Empty when neither can say."""
    from strata_control import engine_component, intel_spec
    found: list[str] = []
    if os.name == "nt":
        for adapter in engine_component.windows_display_adapters():
            if str(adapter.get("vendor", "")).lower() == "8086":
                found.append(sycl_arch_of_name(adapter.get("name", "")))
    else:
        for card in intel_spec.sysfs_cards():
            found.append(sycl_arch_of_pci(card.get("driver", ""), card.get("devid", "")))
    return found


def detect_hardware() -> tuple[str | None, list[str]]:
    """(engine family, card architectures) of this computer, looked up once: ``("cuda", ["sm_120"])``, ``("hip", ["gfx1151"])``, ``("metal", [])``;
    ``(None, [])`` when nothing answers. Read-only probes of tools that are already installed; never raises, never downloads."""
    if not _HARDWARE:
        backend: str | None = None
        archs: list[str] = []
        try:
            from strata_control import engine_component
            backend = engine_component.detect_backend()[0]
            archs = list(engine_component.cuda_archs_present()) + list(engine_component.hip_archs_present())
            if backend == "sycl" and not archs:
                archs = sycl_archs_present()
        except Exception:                                    # noqa: BLE001 - an unknown computer is "no DFlash2", never an error
            pass
        _HARDWARE.append((backend, archs))
    return _HARDWARE[0]


def hardware_support(backend: str | None, archs: Any = None) -> Support:
    """Can the engine for this backend and card run DFlash2, and what was measured on it. ``backend`` is the engine family (cuda | hip | sycl | metal);
    ``archs`` the cards' architectures (``sm_120``, ``gfx1151``...), the first one counts like card_class does."""
    make = str(backend or "").strip().lower()
    if make not in DFLASH2_BACKENDS:
        return Support(False, "none", "")
    first = ([archs] if isinstance(archs, str) else list(archs or []))
    arch = str(first[0]).strip().lower() if first else ""
    if (make, arch) in _TESTED:
        return Support(True, "tested", LINE_TESTED)
    if (make, arch) in _SLOWER:
        return Support(True, "slower", LINE_SLOWER)
    return Support(True, "untested", LINE_UNTESTED)


# --------------------------------------------------------------------------------------------------- the drafter file
#
# incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0), pinned to one revision. The safetensors file is 1.2 GB; tools/dflash2_pack.py --q4_0 turns it, on the
# person's own computer, into the engine's runtime file (a Q4_0 pack, 522,458,112 bytes; the repository publishes only the bf16 file and we host nothing).
# The sizes and hashes are the repository's own for that revision: a download is refused if either differs. The pack itself is checked too: the same
# bytes come out on x86 Windows, x86 Linux and aarch64 Linux (numpy 2.3.3 and 2.5.3), so its size and SHA-256 are pinned and a pack that differs is deleted.
# A Q8 pack (760,353,792 bytes) made by an earlier Strata keeps working and is never fetched again; Q4_0 and Q8 ran the same speed (-0.3%, RESULTS-DRAFT.md).

REPO = "incoai/Qwen3.8-Flash-Next-DFlash2"
REVISION = "a22a1744e85c17364dcc54a6a420af7135d4c7f6"
LICENSE = "Apache-2.0"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
FILES = (
    ("config.json", 1236, "2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9"),
    ("model.safetensors", 1206140008, "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f"),
)
DOWNLOAD_GB = 1.2            # what is fetched (the safetensors file)
PACK_GB = 0.52               # what stays on disk (the source file is removed after packing)
PACK_NAME = "qwen3.8-flash-next-dflash2.q4_0.pack"
PACK_BYTES = 522458112       # the Q4_0 pack tools/dflash2_pack.py --q4_0 makes from the pinned safetensors file, and its SHA-256
PACK_SHA256 = "cda40d62d0eb8f37ed6e272142641a1d2f0de4a32f9e14d0416e7a71534849f1"
LEGACY_PACK_NAME = "qwen3.8-flash-next-dflash2.q8.pack"      # what Strata 0.2.9 test builds packed: still used when it is the one on disk
VRAM_GB = DRAFTER_GB["dflash2"]   # graphics memory with the MTP layer not loaded: 890 MiB (RESULTS-DF2MEM.md); the engine reports "drafter 799 MiB" for the pack and buffers alone
FOLDER = "dflash2"           # under the model data folder, next to ``mtp``
SPEC_WINDOW = 6              # --spec with DFlash2: the widest window of drafted words per round (measured best on Blackwell; the engine caps it per card)
MTP_WINDOW = 4               # the MTP's usual --spec: the window it runs when DFlash2 falls back to it, unless the config names another
DRAFT_CONF = 0.35            # --draft-conf with DFlash2: the window ends where the chance that every draft so far holds falls below this (the card holds most experts)

# --draft-conf by how many expert slots the card holds once the drafter's 0.8 GB is taken (RESULTS-DF2Q4.md, fork F's mimic on l620, IQ3_S, 7 classes
# at 9,300 / 5,000 slots and 5 classes at 1,000 / 300, 3 launches per cell; the DFlash2 engine holds the drafter, so it had 420 slots fewer than MTP).
# Speed against MTP spec 4 by confidence 0.35 / 0.6 / 0.7 / 0.8:  9,300 slots +4.3 / +5.2 / +2.1 / +0.7%;  5,000: -4.7 / -0.3 / +1.6 / -0.7%;
# 1,000: -10.0 / -2.8 / -2.5 / +0.2%;  300: -14.0 / -10.4 / -7.6 / -7.5%.  R9700 at 12,570 slots measured +7.2% at 0.35 (RESULTS-DF2HIP.md).
# Steps, rounded: the more experts come over PCIe, the dearer a row, the shorter the window should be.
CONF_STEPS = ((9000, 0.35), (3000, 0.7), (0, 0.8))      # (slots at least, --draft-conf)
# A backend whose card class was measured on its own has ONE value that replaces the slot steps (they were calibrated on CUDA cards with 300 to 9,300 slots):
# Intel Arc (SYCL, A750 with 298 expert slots, IQ3_XXS, serve sweeps with suffix drafting on): 0.6 was the best of 0.35 / 0.6 / 0.8 and the other two lost 4 to 10%
# (RESULTS-DF2SYCL.md sections 6-8).
BACKEND_CONF = {"sycl": 0.6}
SLOT_MIB = 1.95              # one expert slot of Qwen3.8-Flash-Next IQ3_S: the engine's "46.84 GiB for 24576 slots" (l620) and 23.87 GiB for 12570 (R9700)
SLOT_BASE_MIB = 2718         # dense weights and the engine's fixed buffers beyond vram_fit's own terms: 3800 from the R9700 start (128K context, 12570 slots on 32 GiB, which loaded the
                             # MTP layer beside DFlash2) less the MTP layer's 1082 MiB, now that DFlash2 runs alone (RESULTS-DF2MEM.md section 1)
SLOT_RESERVE_MIB = 700       # the engine's default --vram-reserve-mib
DEFAULT_CONTEXT = 131072     # --max-context when the arguments name none (the 128K default)

#: model families (strata launch --family) the drafter was trained for and measured on: Qwen3.8-Flash-Next and its Gyro build.
SUPPORTED_FAMILIES = ("qwen", "gyro")
_SUPPORTED_NAME = re.compile(r"^qwen3\.8-flash-next(-gyro-[a-z0-9._-]+|-(q|iq|tq)\d[a-z0-9_.]*)?$", re.IGNORECASE)


def supported_model(model_name: str | None = None, family: str | None = None) -> bool:
    """DFlash2 is for Qwen3.8-Flash-Next only: by family when it is known (the Terminal), else by the run config's model name."""
    if family is not None:
        return family.lower() in SUPPORTED_FAMILIES
    return bool(model_name and _SUPPORTED_NAME.match(str(model_name)))


def _here(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > (1 << 20)
    except OSError:
        return False


def target_path(data_dir: str | Path) -> Path:
    """Where a fetch writes the Q4_0 pack."""
    return Path(data_dir) / FOLDER / PACK_NAME


def pack_path(data_dir: str | Path) -> Path:
    """The pack the engine is started with: the Q4_0 pack, else a Q8 pack an earlier Strata made (no download for it), else where the Q4_0 pack will be."""
    for name in (PACK_NAME, LEGACY_PACK_NAME):
        if _here(Path(data_dir) / FOLDER / name):
            return Path(data_dir) / FOLDER / name
    return target_path(data_dir)


def pack_ready(data_dir: str | Path | None) -> bool:
    try:
        return bool(data_dir) and _here(pack_path(data_dir))
    except OSError:
        return False


# --------------------------------------------------------------------------------------------------- a decision


@dataclass
class Decision:
    asked: str                   # what the person chose (auto | mtp | dflash2 | off)
    drafter: str                 # what the engine is started with: mtp | dflash2 | off
    note: str = ""               # plain words when the choice could not be applied; "" otherwise
    flags: tuple[str, ...] = ()  # engine flags added


def normalize(value: Any) -> str:
    text = str(value if value is not None else DEFAULT_MODE).strip().lower()
    return text if text in MODES else DEFAULT_MODE


#: Does the engine in this release run DFlash2 when it serves (the app and `strata launch`)? On this branch it does: `--draft dflash2` works with --serve for one
#: card and one request at a time (src/program/generate.cpp); --batch, --pipeline-windows and a layer split cannot (DF2-MGPU is to lift that). A parked or
#: edited conversation starts the drafter with an empty context (a log line; the answers are the same). Flip back to False to hide the choice again.
ENGINE_SERVES_DFLASH2 = True

#: Why another drafter (or none) runs when the one chosen cannot: a closed set of short reasons, each one is a sentence in the interface catalog in all
#: nine languages (the line is "Drafting: MTP (<reason>)", apps/host/src/locales/df2-<lang>.json and en.json).
DIDNT_FIT = "DFlash2 didn't fit"
NOT_HERE = "DFlash2 isn't available in this mode"
NOT_SUPPORTED = "DFlash2 is for Qwen3.8-Flash-Next only"
NOT_DOWNLOADED = "the DFlash2 drafter isn't downloaded yet"
NOT_THIS_CARD = "DFlash2 isn't available on this graphics card"
NO_MTP = "the MTP drafter isn't downloaded yet"
NONE_HERE = "no drafter is downloaded yet"
NOT_SERVED = NOT_HERE
REASONS = (DIDNT_FIT, NOT_HERE, NOT_SUPPORTED, NOT_DOWNLOADED, NOT_THIS_CARD, NO_MTP, NONE_HERE)


def line(drafter: str, reason: str = "") -> str:
    """``Drafting: DFlash2``, ``Drafting: MTP (DFlash2 isn't available in this mode)``, ``Drafting: off (no drafter is downloaded yet)``."""
    word = {"mtp": "MTP", "dflash2": "DFlash2", "off": "off"}.get(drafter, drafter)
    return f"Drafting: {word} ({reason})" if reason else f"Drafting: {word}"


def mtp_line(reason: str) -> str:
    return line("mtp", reason)


def engine_flags(pack: str | Path, vocab: str | Path | None = None) -> tuple[str, ...]:
    """The engine arguments of a DFlash2 start: the drafter, its file, no MTP layer, the draft vocabulary, the window and the confidence."""
    return ("--draft", "dflash2", "--draft-model", str(pack), "--draft-fallback", "none", *(("--mtp-draft-vocab", str(vocab)) if vocab else ()),
            "--spec", str(SPEC_WINDOW), "--draft-conf", f"{DRAFT_CONF:g}")


def conf_for(backend: str | None, slots: int | float | None) -> float:
    """The ``--draft-conf`` for this engine family and a card that holds ``slots`` experts: the backend's own value (BACKEND_CONF) when it has one, else the slot steps."""
    own = BACKEND_CONF.get(str(backend or "").strip().lower())
    return own if own is not None else conf_for_slots(slots)


def conf_for_slots(slots: int | float | None) -> float:
    """The ``--draft-conf`` for a card that holds ``slots`` experts (CONF_STEPS); DRAFT_CONF when the number is unknown."""
    if slots is None:
        return DRAFT_CONF
    for at_least, conf in CONF_STEPS:
        if slots >= at_least:
            return conf
    return CONF_STEPS[-1][1]


def expected_slots(vram_mib: float, args: list[Any]) -> int:
    """The expert slots a card of ``vram_mib`` MiB is expected to hold with the drafter on, the same arithmetic as strata_control/vram_fit.py (the card, the
    runtime's gap, the dense weights, the KV session of ``--max-context`` / ``--kv``, the prompt buffers, the reserve) minus the DFlash2 drafter's VRAM_GB
    (the MTP layer is not loaded beside it), over one slot. An estimate good to about 10% (two engine starts); the steps of CONF_STEPS are wider than that."""
    from strata_control import vram_fit
    ctx = _flag_value(args, "--max-context")
    ctx = int(ctx) if ctx and ctx.isdigit() else DEFAULT_CONTEXT
    kv = _flag_value(args, "--kv") or "int8"
    reserve = _flag_value(args, "--vram-reserve-mib")
    reserve = int(reserve) if reserve and reserve.isdigit() else SLOT_RESERVE_MIB
    prefill = _flag_value(args, "--prefill")
    chunk = vram_fit.chunk_of(prefill.split(":")[-1] if prefill else "auto")
    free = (float(vram_mib) - vram_fit.GAP_OTHER_MIB - SLOT_BASE_MIB - vram_fit.session_mib(ctx, kv) - vram_fit.HEAD_MIB - reserve
            - (vram_fit.PROMPT_BASE_MIB + vram_fit.PROMPT_PER_TOKEN_MIB * chunk) - DRAFTER_MIB["dflash2"])
    return max(0, int(free / SLOT_MIB))


def slots_of(args: list[Any], vram_mib: float | None = None) -> int | None:
    """How many expert slots this start holds: the person's own ``--expert-cache N``, else the estimate for the card (``vram_mib``), else None."""
    given = _flag_value(args, "--expert-cache")
    if given is not None and given.isdigit():
        return int(given)
    if vram_mib:
        return expected_slots(vram_mib, args)
    return None


def blocked_by(args: list[Any], *, split: bool = False, slots: int = 0) -> bool:
    """Does this start have something the engine cannot run DFlash2 beside today (DF2-MGPU is to lift it): a layer split over several cards, several requests at
    once (``--batch``, ``--parallel``), ``--pipeline-windows``. The caller says what it adds later (``split``, ``slots``); the arguments are looked at too."""
    items = [str(a) for a in args or []]
    return bool(split or slots > 1 or any(a in items for a in ("--layer-split", "--batch", "--slots", "--pipeline-windows", "--parallel")))


def mtp_ready(args: list[Any]) -> bool | None:
    """Is the MTP layer this start would load on this computer? ``--mtp <folder>`` with its ``experts.bin`` (or, on a Mac, ``mtp-q2_0.gguf`` beside the folder).
    None (not known, taken as there) when the arguments name no layer at all (an Intel route, or a layer switched off) or a draft depth instead of a folder (Qwen3.6's
    ``--mtp 4``: that layer is its own). A DFlash2-only install keeps ``--mtp <folder>`` in the saved arguments for a later switch to the MTP, so the folder is looked at."""
    items = [str(a) for a in args or []]
    if "--mtp" not in items:
        return None
    value = _flag_value(args, "--mtp")
    if value is None or value.isdigit():
        return None
    folder = Path(value)
    try:
        return (folder / "experts.bin").is_file() or (folder.parent / "mtp-q2_0.gguf").is_file()
    except OSError:
        return False


VOCAB_NAME = "draft_vocab.bin"


def shipped_vocab() -> Path:
    """The default draft vocabulary Strata ships with (``data/draft_vocab.bin``, also in the packaged Control)."""
    return Path(__file__).resolve().parents[1] / "data" / VOCAB_NAME


def vocab_target(data_dir: str | Path) -> Path:
    """Where the DFlash2 install keeps its copy of the draft vocabulary: next to the pack, so an install that never downloaded the MTP layer has one."""
    return Path(data_dir) / FOLDER / VOCAB_NAME


def provide_vocab(data_dir: str | Path | None) -> Path | None:
    """Make sure the DFlash2 folder holds the draft vocabulary (a local copy of the file Strata ships; no network). Called by the drafter's fetch and by
    setup's DFlash2 step, so a fresh install that selected only DFlash2 has everything the engine needs. None when it cannot be had."""
    if not data_dir:
        return None
    target = vocab_target(data_dir)
    try:
        if target.is_file() and target.stat().st_size > 0:
            return target
        source = shipped_vocab()
        if not source.is_file():
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(source.read_bytes())
        tmp.replace(target)
        return target
    except OSError:
        return None


def vocab_path(args: list[Any], data_dir: str | Path | None = None) -> Path | None:
    """The draft vocabulary DFlash2 reads its draft head from: the person's own ``--mtp-draft-vocab``, else the one in the model's MTP folder (the subset the
    person chose in setup), else the copy the DFlash2 install keeps, else the default subset Strata ships. None when none of them is on this computer."""
    given = _flag_value(args, "--mtp-draft-vocab")
    if given and Path(given).is_file():
        return Path(given)
    folder = _flag_value(args, "--mtp")
    if folder and not folder.isdigit() and (Path(folder) / VOCAB_NAME).is_file():
        return Path(folder) / VOCAB_NAME
    if data_dir and vocab_target(data_dir).is_file():
        return vocab_target(data_dir)
    shipped = shipped_vocab()
    return shipped if shipped.is_file() else None


def can_run_dflash2(model_name: str | None, family: str | None, backend: str | None, archs: Any = None) -> bool:
    """Is this model one the drafter was made for, and does this computer's engine have the DFlash2 code? (The disk and the mode of the start are not looked at.)"""
    return bool(supported_model(model_name, family) and ENGINE_SERVES_DFLASH2 and backend is not None and hardware_support(backend, archs).can_run)


def choice_of(mode: Any, *, model_name: str | None = None, family: str | None = None, backend: str | None = None, archs: Any = None) -> str:
    """The drafter the choice means for this model on this computer, before anything is looked up on disk: ``dflash2``, ``mtp`` or ``off``. Auto is DFlash2 wherever
    the engine has it (DFLASH2_BACKENDS) and the model is one it was made for, the MTP everywhere else; DFlash2 asked for by name where it cannot run is MTP."""
    asked = normalize(mode)
    if asked == "off":
        return "off"
    if asked == "mtp":
        return "mtp"
    return "dflash2" if can_run_dflash2(model_name, family, backend, archs) else "mtp"


def decide(mode: Any, *, klass: str = "other", model_name: str | None = None, family: str | None = None, data_dir: str | Path | None = None,
           backend: str | None = None, archs: Any = None, mtp_here: bool | None = None, blocked: bool = False, vocab_here: bool = True) -> Decision:
    """The drafter a start uses. Never raises. One drafter, never two. A choice that cannot be applied becomes the other drafter when that one is on this computer,
    else none, with a note in plain words. ``mtp_here`` says whether the MTP layer is on disk (None: not known, taken as there), ``blocked`` that this start is a layer
    split or has several requests at once (see blocked_by), ``vocab_here`` that the draft vocabulary DFlash2 needs is there. ``klass`` is not used any more (Auto
    does not depend on the kind of card) and stays only so the callers keep working."""
    asked = normalize(mode)
    if asked == "off":
        return Decision(asked, "off")
    want = choice_of(asked, model_name=model_name, family=family, backend=backend, archs=archs)
    have_mtp = mtp_here is not False
    runnable = can_run_dflash2(model_name, family, backend, archs)

    if want == "dflash2":
        reason = NOT_HERE if (blocked or not vocab_here) else NOT_DOWNLOADED if not pack_ready(data_dir) else ""
        if not reason:
            return Decision(asked, "dflash2", flags=engine_flags(pack_path(data_dir)))
        return Decision(asked, "mtp", reason) if have_mtp else Decision(asked, "off", NONE_HERE)
    # the MTP: asked for, or all there is where DFlash2 cannot run
    if have_mtp:
        why = ""
        if asked == "dflash2":                      # named, and it cannot run on this model or this computer
            why = NOT_SUPPORTED if not supported_model(model_name, family) else NOT_SERVED if not ENGINE_SERVES_DFLASH2 else NOT_THIS_CARD
        return Decision(asked, "mtp", why)
    # the layer is not on this computer: DFlash2 instead when it can run and its pack is here (said), else none (said)
    if runnable and not blocked and vocab_here and pack_ready(data_dir):
        return Decision(asked, "dflash2", NO_MTP, engine_flags(pack_path(data_dir)))
    return Decision(asked, "off", NONE_HERE)


ONE_REQUEST_NOTE = "DFlash2 answers one request at a time, so Strata set this model to one request at a time."


def wants_one_request(mode: Any, *, klass: str = "other", model_name: str | None = None, family: str | None = None, backend: str | None = None,
                      archs: Any = None, **_ignored: Any) -> bool:
    """The choice means DFlash2 on this model and computer (named, or Auto where the engine has it): the model then starts with one request at a time, because the
    engine's request slots (``--batch``, which Strata fills in by itself on a card with room) cannot run beside it. A number of requests the person chose stays as
    they chose it (the start then loads the MTP and says so); the callers check that."""
    return choice_of(mode, model_name=model_name, family=family, backend=backend, archs=archs) == "dflash2"


def needs_pack(mode: Any, *, klass: str = "other", model_name: str | None = None, family: str | None = None, data_dir: str | Path | None = None,
               backend: str | None = None, archs: Any = None, **_ignored: Any) -> bool:
    """Would this choice use the DFlash2 drafter if its file were here, and is it not here yet? (Then it is fetched before the start, once.)"""
    return choice_of(mode, model_name=model_name, family=family, backend=backend, archs=archs) == "dflash2" and not pack_ready(data_dir)


def needs_mtp(mode: Any, *, klass: str = "other", model_name: str | None = None, family: str | None = None, backend: str | None = None,
              archs: Any = None, mtp_here: bool | None = None, **_ignored: Any) -> bool:
    """Would this choice use the MTP layer, and is it not on this computer yet? (A new install fetches only the drafter it will use; this is what a later switch
    to the MTP fetches, once.)"""
    return choice_of(mode, model_name=model_name, family=family, backend=backend, archs=archs) == "mtp" and mtp_here is False


#: The engine options of the DFlash2 drafter; none of them goes with the MTP.
DRAFT_FLAGS = {"--draft": True, "--draft-model": True, "--draft-conf": True, "--draft-mtp-spec": True, "--draft-fallback": True, "--draft-cap": True,
               "--draft-rule": True, "--draft-mtp-min-p": True}
#: The engine options of the MTP layer; none of them goes with DFlash2 (``--mtp-max-t`` is left: DFlash2 reads it too).
MTP_ONLY_FLAGS = {"--mtp": True, "--mtp-q4": True, "--mtp-hnorm": True, "--mtp-window": True, "--mtp-draft-vocab": True, "--spec-min-p": True, "--batch-mtp": False}


def _strip(args: list[Any], table: dict[str, bool]) -> list[str]:
    out: list[str] = []
    items = [str(a) for a in args]
    i = 0
    while i < len(items):
        if items[i] in table:
            i += 2 if table[items[i]] and i + 1 < len(items) and not items[i + 1].startswith("--") else 1
            continue
        out.append(items[i])
        i += 1
    return out


def strip_draft_flags(args: list[Any]) -> list[str]:
    """The arguments without the DFlash2 drafter's options (``--draft``, ``--draft-model``, ``--draft-fallback``, ``--draft-conf``... and their values)."""
    return _strip(args, DRAFT_FLAGS)


def strip_mtp_flags(args: list[Any]) -> list[str]:
    """The arguments without the MTP layer's options (``--mtp``, ``--mtp-q4``, ``--spec-min-p``, ``--batch-mtp``... and their values)."""
    return _strip(args, MTP_ONLY_FLAGS)


def data_dir_of(args: list[Any]) -> Path | None:
    """The model data folder an argument list belongs to: `--pack <data>/packs/<tag>`, else `--mtp <data>/mtp/rt`."""
    items = [str(a) for a in args]
    for flag, up in (("--pack", 2), ("--mtp", 2)):
        if flag in items and items.index(flag) + 1 < len(items):
            path = Path(items[items.index(flag) + 1])
            if flag == "--mtp" and path.name.isdigit():
                continue
            return path.parents[up - 1] if len(path.parents) >= up else None
    return None


def _flag_value(args: list[Any], flag: str) -> str | None:
    items = [str(a) for a in args or []]
    return items[items.index(flag) + 1] if flag in items and items.index(flag) + 1 < len(items) else None


def _without_flag(args: list[str], flag: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            i += 2 if i + 1 < len(args) and not args[i + 1].startswith("--") else 1
            continue
        out.append(args[i])
        i += 1
    return out


def apply(cfg: dict[str, Any], *, klass: str = "other", data_dir: str | Path | None = None, family: str | None = None,
          backend: str | None = None, archs: Any = None, vram_mib: float | None = None, split: bool = False, slots: int = 0) -> Decision:
    """Put the config's ``"drafting"`` choice on its engine arguments (in place): ONE drafter's options and none of the other's. DFlash2: no ``--mtp``, no MTP option,
    ``--draft dflash2 --draft-model <pack> --draft-fallback none --mtp-draft-vocab <file> --spec 6 --draft-conf C``; the person's own ``--spec`` or ``--draft-conf``
    in ``user_args`` stays. C is 0.35 where the card holds most experts and higher where fewer slots fit (conf_for_slots): from the person's own ``--expert-cache N``,
    else from ``vram_mib``, the memory of the one card (None: several cards, a shared-memory chip, unknown: 0.35). MTP: the arguments without any DFlash2 option.
    ``off``: neither (the draft layer's own switch, ``"mtp": false``). ``split`` and ``slots`` say that this start is a layer split or serves several requests at
    once, which DFlash2 cannot do yet: the MTP then, said in one line. A config from before this rule that names both drafters starts with the one chosen."""
    args = [str(a) for a in cfg.get("args") or []]
    if not supported_model(cfg.get("model_name"), family):    # another model: its drafting is not this choice's business (DeepSeek, Swift, Coder, Qwen3.6, Unsloth...)
        asked = normalize(cfg.get("drafting"))
        if asked == "dflash2":
            cfg["args"] = strip_draft_flags(args)
            return Decision(asked, "mtp", NOT_SUPPORTED)
        return Decision(asked, "mtp")
    mtp_here = mtp_ready(args)
    vocab = vocab_path(args, data_dir)
    mode = cfg.get("drafting")
    if cfg.get("mtp") is False and normalize(mode) == "auto":
        mode = "off"                                          # "Draft layer (MTP)" turned off by the person, and no drafter chosen since: no drafter
    decision = decide(mode, model_name=cfg.get("model_name"), family=family, data_dir=data_dir, backend=backend, archs=archs,
                      mtp_here=mtp_here, blocked=blocked_by(strip_draft_flags(args), split=split, slots=slots), vocab_here=vocab is not None)
    mine = cfg.get("user_args") or []
    if decision.drafter == "off":
        cfg["args"] = strip_mtp_flags(strip_draft_flags(args))
        cfg["mtp"] = False
    elif decision.drafter == "mtp":
        cfg["args"] = strip_draft_flags(args)
        if cfg.get("drafting") is not None and normalize(cfg.get("drafting")) != "off" and cfg.get("mtp") is False:
            cfg["mtp"] = True                                 # an explicit drafter choice overrides an older "mtp": false
    else:
        base = strip_mtp_flags(strip_draft_flags(args))
        flags = list(engine_flags(pack_path(data_dir), vocab))
        kept_conf = _flag_value(mine, "--draft-conf")
        flags[flags.index("--draft-conf") + 1] = kept_conf if kept_conf is not None else f"{conf_for(backend, slots_of(base, vram_mib)):g}"
        if _flag_value(mine, "--spec") is None:
            base = _without_flag(base, "--spec")              # the DFlash2 window; the person's own --spec stays
        else:
            i = flags.index("--spec")
            del flags[i:i + 2]
        cfg["args"] = base + flags
    return decision


# --------------------------------------------------------------------------------------------------- what the engine said

_ON = re.compile(r"DFlash2 drafter on\b")


def fallback_note(log_text: str) -> str:
    """From one engine start's log: ``Drafting: MTP (DFlash2 didn't fit)`` when the engine fell back from DFlash2, else ``""``. The engine
    names the reason in its own words; two are told apart: no room in graphics memory, and a mode this build does not support."""
    text = str(log_text or "")
    if _ON.search(text):
        return ""
    if "--draft dflash2" not in text:
        return ""
    low = text.lower()
    if "it is for one card and the cli run" in low or "needs a native pack" in low or "it needs --mtp" in low:
        reason = NOT_HERE
    else:
        reason = DIDNT_FIT
    return mtp_line(reason)


def summary_line(decision: Decision, engine_note: str = "") -> str:
    """The line the Terminal and the app show: what drafts."""
    if engine_note:
        return engine_note
    if decision.note:
        return line(decision.drafter, decision.note)
    return f"Drafting: {LABELS.get(decision.drafter, decision.drafter) if decision.drafter != 'mtp' else 'MTP'}"


# --------------------------------------------------------------------------------------------------- what a start did

_pending_note = ""


def remember_note(note: str) -> None:
    """The server calls this when it builds the engine arguments: why the choice was not applied, for the engine that starts next."""
    global _pending_note
    _pending_note = note or ""


def take_note() -> str:
    global _pending_note
    note, _pending_note = _pending_note, ""
    return note


def engine_info(args: list[Any], log_text: str = "", note: str = "") -> dict[str, str]:
    """The facts of one engine start for /metrics and the apps: ``drafting`` (mtp | dflash2 | off) and ``drafting_note`` (plain words, or ""): the one drafter this
    start loads (``--draft dflash2`` without ``--mtp``, ``--mtp`` without ``--draft``), and why it is not the one asked for. The engine's own log wins over the choice."""
    items = [str(a) for a in args]
    asked_dflash2 = any(items[i] == "--draft" and i + 1 < len(items) and items[i + 1] == "dflash2" for i in range(len(items)))
    fell_back = fallback_note(log_text) if asked_dflash2 else ""
    drafter = "mtp" if "--mtp" in items else "off"
    if asked_dflash2 and not fell_back:
        drafter = "dflash2"
    text = fell_back or (line(drafter, note) if note else "")
    return {"drafting": drafter, "drafting_note": text}
