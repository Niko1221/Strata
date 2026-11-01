"""Which drafter a model uses to guess ahead: MTP, DFlash2 or none. One place, read by the Terminal and the app.

The choice is the person's, and it is saved in the model's run config as ``"drafting"``: ``auto`` (the default), ``mtp``, ``dflash2`` or
``off``. ``strata launch --drafting ...`` writes it, the app's Models screen writes it, and ``serve/server.py`` applies it at every start
(like ``"mtp": false``, strata_control/draft_layer.py), so a setup run that rewrites the engine arguments cannot undo it.

  mtp      the engine's default drafter, the model's own draft layer (no extra flags)
  dflash2  ``--draft dflash2 --draft-model <pack> --spec 6 --draft-conf 0.35``: the DFlash2 block drafter (incoai/Qwen3.8-Flash-Next-DFlash2) for
           Qwen3.8-Flash-Next, with the window and confidence that measured best (RESULTS-DRAFT.md section 4); a ``--spec`` or ``--draft-conf`` the
           person set by hand (``user_args``) stays
  off      no draft layer (the same as ``"mtp": false``: the engine still looks ahead in the text with ``--spec``)
  auto     the table below says which of mtp / dflash2 this card class gets

Recommend, never force: a choice that cannot be applied (DFlash2 for another model, no drafter file yet) does not stop the start. The
model starts with the MTP and ``Decision.note`` says why in plain words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODES = ("auto", "mtp", "dflash2", "off")
DEFAULT_MODE = "auto"
LABELS = {"auto": "Auto (recommended)", "mtp": "MTP", "dflash2": "DFlash2", "off": "Off"}
DRAFTERS = ("mtp", "dflash2", "off")

# --------------------------------------------------------------------------------------------------- the suggestion table
#
# "Suggest DFlash2 where the win is 100%": Auto picks DFlash2 only on a card class where it measured faster than the MTP on EVERY tested
# prompt (``faster_everywhere``). A class goes here when it was measured; a row with ``faster_everywhere`` False is MTP whatever the average
# says. Fill this in as measurements come in (and change nothing else: the Terminal and the app both read it, tests/terminal/test_drafting.py
# checks that a DFlash2 row has the evidence).
#
# Evidence so far (phaseA-tests/notes-dflash2-p1.md, -p02.md, -ports.md; DFlash2 against MTP, tokens per second, same prompts and flags):
#   strix_halo  Radeon 8060S (gfx1151): faster on chat, code and json, SLOWER on tool_call (-6%).
#   blackwell   RTX 5070 (sm_120, 12 GB): faster on code, SLOWER on chat (-1%).
#   rdna4       Radeon AI PRO R9700 (gfx1201): slower on every prompt.
#   ampere      RTX 3060 (sm_86, 12 GB): slower on every prompt.
#   other       not measured.


@dataclass(frozen=True)
class Suggestion:
    drafter: str                 # "mtp" | "dflash2"
    faster_everywhere: bool      # DFlash2 measured faster than the MTP on every tested prompt of this class
    evidence: str                # one line: what was measured


SUGGESTED: dict[str, Suggestion] = {
    "strix_halo": Suggestion("mtp", False, "gfx1151: faster on chat, code and json; slower on tool_call (-6%)"),
    "blackwell": Suggestion("mtp", False, "sm_120 (RTX 5070, 12 GB): faster on code; slower on chat (-1%)"),
    "rdna4": Suggestion("mtp", False, "gfx1201 (R9700): slower on every prompt"),
    "ampere": Suggestion("mtp", False, "sm_86 (RTX 3060, 12 GB): slower on every prompt"),
    "other": Suggestion("mtp", False, "not measured on this kind of card"),
}

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


def suggest(klass: str) -> str:
    """What Auto picks on this card class: DFlash2 only where it was faster on every tested prompt, MTP otherwise."""
    row = SUGGESTED.get(klass) or SUGGESTED["other"]
    return "dflash2" if row.drafter == "dflash2" and row.faster_everywhere else "mtp"


# --------------------------------------------------------------------------------------------------- where the engine can run it
#
# The DFlash2 code (src/core/dflash2.cpp, src/kernels/cuda/dflash2.cu) is compiled into the CUDA and HIP engines only (CMakeLists.txt, STRATA_HAVE_DFLASH2);
# the Metal and SYCL engines have none, and a PC with no supported graphics card has no engine to run it. Where the code exists, three words say what was
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


_TESTED = {("cuda", "sm_120"), ("cuda", "sm_121"), ("hip", "gfx1151")}
_SLOWER = {("cuda", "sm_86"), ("hip", "gfx1201")}


_HARDWARE: list = []


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
        except Exception:                                    # noqa: BLE001 - an unknown computer is "no DFlash2", never an error
            pass
        _HARDWARE.append((backend, archs))
    return _HARDWARE[0]


def hardware_support(backend: str | None, archs: Any = None) -> Support:
    """Can the engine for this backend and card run DFlash2, and what was measured on it. ``backend`` is the engine family (cuda | hip | sycl | metal);
    ``archs`` the cards' architectures (``sm_120``, ``gfx1151``...), the first one counts like card_class does."""
    make = str(backend or "").strip().lower()
    if make not in ("cuda", "hip"):
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
# incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0), pinned to one revision. The safetensors file is 1.2 GB; tools/dflash2_pack.py turns it into the
# engine's runtime file (Q8 pack, about 0.76 GB). The sizes and hashes are the repository's own for that revision: a download is refused
# if either differs.

REPO = "incoai/Qwen3.8-Flash-Next-DFlash2"
REVISION = "a22a1744e85c17364dcc54a6a420af7135d4c7f6"
LICENSE = "Apache-2.0"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
FILES = (
    ("config.json", 1236, "2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9"),
    ("model.safetensors", 1206140008, "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f"),
)
DOWNLOAD_GB = 1.2            # what is fetched (the safetensors file)
PACK_GB = 0.76               # what stays on disk (the source file is removed after packing)
PACK_NAME = "qwen3.8-flash-next-dflash2.q8.pack"
VRAM_GB = 1.2                # what the drafter needs in graphics memory (the engine's own estimate)
FOLDER = "dflash2"           # under the model data folder, next to ``mtp``
SPEC_WINDOW = 6              # --spec with DFlash2: the widest window of drafted words per round (measured best on Blackwell; the engine caps it per card)
DRAFT_CONF = 0.35            # --draft-conf with DFlash2: the window ends where the chance that every draft so far holds falls below this

#: model families (strata launch --family) the drafter was trained for and measured on: Qwen3.8-Flash-Next and its Gyro build.
SUPPORTED_FAMILIES = ("qwen", "gyro")
_SUPPORTED_NAME = re.compile(r"^qwen3\.8-flash-next(-gyro-[a-z0-9._-]+|-(q|iq|tq)\d[a-z0-9_.]*)?$", re.IGNORECASE)


def supported_model(model_name: str | None = None, family: str | None = None) -> bool:
    """DFlash2 is for Qwen3.8-Flash-Next only: by family when it is known (the Terminal), else by the run config's model name."""
    if family is not None:
        return family.lower() in SUPPORTED_FAMILIES
    return bool(model_name and _SUPPORTED_NAME.match(str(model_name)))


def pack_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / FOLDER / PACK_NAME


def pack_ready(data_dir: str | Path | None) -> bool:
    try:
        return bool(data_dir) and pack_path(data_dir).is_file() and pack_path(data_dir).stat().st_size > (1 << 20)
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
#: card and one request at a time (src/program/generate.cpp); --batch, --pipeline-windows and a layer split keep the MTP and the engine says so. A parked or edited
#: conversation starts the drafter with an empty context (a log line; the answers are the same). Flip back to False to hide the choice again.
ENGINE_SERVES_DFLASH2 = True

#: Why the MTP runs when something else was asked for: a closed set of short reasons, each one is a sentence in the interface catalog in all eight
#: languages (the line is "Drafting: MTP (<reason>)").
DIDNT_FIT = "DFlash2 didn't fit"
NOT_HERE = "DFlash2 isn't available in this mode"
NOT_SUPPORTED = "DFlash2 is for Qwen3.8-Flash-Next only"
NOT_DOWNLOADED = "the DFlash2 drafter isn't downloaded yet"
NOT_THIS_CARD = "DFlash2 isn't available on this graphics card"
NOT_SERVED = NOT_HERE
REASONS = (DIDNT_FIT, NOT_HERE, NOT_SUPPORTED, NOT_DOWNLOADED, NOT_THIS_CARD)


def mtp_line(reason: str) -> str:
    return f"Drafting: MTP ({reason})"


def engine_flags(pack: str | Path) -> tuple[str, ...]:
    """The engine arguments of a DFlash2 start: the drafter, its file, the window and the confidence."""
    return ("--draft", "dflash2", "--draft-model", str(pack), "--spec", str(SPEC_WINDOW), "--draft-conf", f"{DRAFT_CONF:g}")


def decide(mode: Any, *, klass: str, model_name: str | None = None, family: str | None = None, data_dir: str | Path | None = None,
           backend: str | None = None, archs: Any = None) -> Decision:
    """The drafter a start uses. Never raises: a choice that cannot be applied becomes MTP with a note. ``backend`` (cuda | hip | sycl | metal)
    is optional: given, a build without the DFlash2 code (Metal, SYCL) keeps the MTP."""
    asked = normalize(mode)
    if asked == "off":
        return Decision(asked, "off")
    want = suggest(klass) if asked == "auto" else asked
    if want != "dflash2":
        return Decision(asked, "mtp")
    if not supported_model(model_name, family):
        return Decision(asked, "mtp", NOT_SUPPORTED if asked == "dflash2" else "")
    if not ENGINE_SERVES_DFLASH2:
        return Decision(asked, "mtp", NOT_SERVED if asked == "dflash2" else "")
    if backend is not None and not hardware_support(backend, archs).can_run:
        return Decision(asked, "mtp", NOT_THIS_CARD if asked == "dflash2" else "")
    if not pack_ready(data_dir):
        return Decision(asked, "mtp", NOT_DOWNLOADED)
    return Decision(asked, "dflash2", flags=engine_flags(pack_path(data_dir)))


def needs_pack(mode: Any, *, klass: str, model_name: str | None = None, family: str | None = None, data_dir: str | Path | None = None,
               backend: str | None = None, archs: Any = None) -> bool:
    """Would this choice use the DFlash2 drafter if its file were here, and is it not here yet? (Then it is fetched before the start.)"""
    asked = normalize(mode)
    want = suggest(klass) if asked == "auto" else asked
    if backend is not None and not hardware_support(backend, archs).can_run:
        return False
    return want == "dflash2" and ENGINE_SERVES_DFLASH2 and supported_model(model_name, family) and not pack_ready(data_dir)


DRAFT_FLAGS = {"--draft": True, "--draft-model": True, "--draft-conf": True}


def strip_draft_flags(args: list[Any]) -> list[str]:
    """The arguments without ``--draft`` / ``--draft-model`` / ``--draft-conf`` (and their values)."""
    out: list[str] = []
    items = [str(a) for a in args]
    i = 0
    while i < len(items):
        if items[i] in DRAFT_FLAGS:
            i += 2 if i + 1 < len(items) and not items[i + 1].startswith("--") else 1
            continue
        out.append(items[i])
        i += 1
    return out


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


def apply(cfg: dict[str, Any], *, klass: str, data_dir: str | Path | None = None, family: str | None = None,
          backend: str | None = None, archs: Any = None) -> Decision:
    """Put the config's ``"drafting"`` choice on its engine arguments (in place). ``off`` is the draft layer's own switch (``"mtp": false``).
    DFlash2 sets the window (``--spec 6``) and the confidence (``--draft-conf 0.35``) unless the person set either one by hand in ``user_args``."""
    from strata_control import draft_layer
    decision = decide(cfg.get("drafting"), klass=klass, model_name=cfg.get("model_name"), family=family, data_dir=data_dir, backend=backend, archs=archs)
    if cfg.get("drafting") is None and decision.drafter == "mtp":
        return decision                                       # nobody chose and Auto is the MTP: the arguments stay exactly as they are
    mine = cfg.get("user_args") or []
    kept_conf = _flag_value(mine, "--draft-conf")
    args = strip_draft_flags(cfg.get("args") or [])
    if decision.drafter == "off":
        cfg["mtp"] = False
        cfg["args"] = args
        draft_layer.apply(cfg)
    else:
        if cfg.get("drafting") is not None and normalize(cfg.get("drafting")) != "off" and cfg.get("mtp") is False:
            cfg["mtp"] = True                                 # an explicit drafter choice overrides an older "mtp": false
        flags = list(decision.flags)
        if decision.drafter == "dflash2":
            if kept_conf is not None:
                flags[flags.index("--draft-conf") + 1] = kept_conf
            if _flag_value(mine, "--spec") is None:
                args = _without_flag(args, "--spec")          # the person's own --spec stays; otherwise the window is the DFlash2 one
            else:
                i = flags.index("--spec")
                del flags[i:i + 2]
        cfg["args"] = args + flags
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
        return mtp_line(decision.note)
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
    """The facts of one engine start for /metrics and the apps: ``drafting`` (mtp | dflash2 | off) and ``drafting_note`` (plain words, or "").
    The engine's own log wins over the choice: if it says it fell back, that is what ran."""
    items = [str(a) for a in args]
    asked_dflash2 = any(items[i] == "--draft" and i + 1 < len(items) and items[i + 1] == "dflash2" for i in range(len(items)))
    fell_back = fallback_note(log_text) if asked_dflash2 else ""
    if asked_dflash2 and not fell_back:
        return {"drafting": "dflash2", "drafting_note": ""}
    drafter = "mtp" if "--mtp" in items else "off"
    text = fell_back or (mtp_line(note) if note and drafter == "mtp" else "")
    return {"drafting": drafter, "drafting_note": text}
