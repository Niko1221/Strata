"""The Inkling-2-Mini family: exact verified sizes & SHA-256 hashes, shape engine component, defaults, and sources.

Pure data and small helpers; the hardware auto-selection and placement rules live in inkling_fit.py.
Exact byte sizes and SHA-256 digests are pinned from the verified R2 `.strata` v1.1 checkpoints (`SHA256SUMS.txt`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from strata_control import inkling_fit

FAMILY = "inkling-2-mini"
TAG = "inkling-2-mini-"               # run config file: strata-inkling-2-mini-sq-20.json
ARCH = "inkling2_loop"                # run config's "arch"; serve & Control route by it
SHAPE = "inkling2"                    # shape engine component: component:strata-inkling (strata_control/shape_engine.py)
COMPONENT = "strata-inkling"
EXE = "strata-inkling.exe" if os.name == "nt" else "strata-inkling"
TITLE = "Inkling-2-Mini"
BY = "Strata's native .strata v1.1 build of Thinking Machines' Inkling-2-Mini"
#: The picture reader (Vision shard). 0.2.23 ships the 16 KiB-aligned file under a new name, used with every pack (SQ-10, SQ-15, SQ-20); its weights are the same as the 0.2.22 file
#: (422 of 422 tensors identical, phaseA-tests/rel0223/reader16k). The old file stays on Hugging Face for older apps, stays accepted wherever it is on disk (VISION_FILES: the new name
#: first), is never deleted, and an install that runs it is told that a newer file is available.
VISION_FILE = "Inkling-2-Mini-Vision-v2.strata"
VISION_FILE_V1 = "Inkling-2-Mini-Vision.strata"            # 0.2.22's file: 4 KiB layout, 636,526,592 bytes
VISION_FILES = (VISION_FILE, VISION_FILE_V1)
#: Not offered and not fetched any more (0.2.23: DFlash 2 is the only drafter); the names stay so that a file an install already has is recognised and kept.
DSPARK_GGUF_FILE = "dspark-Inkling-2-Mini-Q8_0.gguf"
MMPROJ_BF16_FILE = "mmproj-BF16.gguf"
MMPROJ_F16_FILE = "mmproj-F16.gguf"
TOKENIZER_FILE = "tokenizer.json"
CHAT_TEMPLATE_FILE = "chat_template.jinja"
DEFAULT_CONTEXT = inkling_fit.DEFAULT_CONTEXT
CONTEXTS = inkling_fit.CONTEXTS
#: The one drafter of every Inkling-2-Mini build on every device (owner, 2026-11-26): DFlash 2, read from dflash2.safetensors beside the model. DSpark is not downloaded and
#: not offered. Without the sidecar file the model runs with no drafter (dflash2_sidecar() is None -> `--draft off`).
DEFAULT_DRAFTER = "dflash2"
DRAFTERS = ("off", "dflash2")
#: Words an older run config may hold for "drafting on" ("dspark" was the word until 0.2.22): read as DFlash 2.
LEGACY_DRAFTING_WORDS = ("dspark", "on")
#: Window shape the engine is told to use when drafting is on (`--draft-beam`): 1 = one linear chain, the block drafter; 4 = the beam tree.
#: Measured 2026-11-23 (0221-DSPARK): the chain is as fast as the tree on CUDA (within 3 percent either way) and simpler.
DRAFT_BEAM = 1
#: `--draft-adapt`: per request, drafting goes off for 40 steps when the measured gain (tokens per step over the step cost) falls below 0.93.
DRAFT_ADAPT = True


def default_drafting(backend: str | None = None) -> str:
    """What a run config gets when the person chose nothing: "auto" = a measured probe on this card and placement decides
    (strata_control/inkling_probe.py: drafting only when it gains MARGIN, 1.10x, on a code + chat probe; no valid probe yet = off).
    Owner decision 2026-11-23: a slightly different answer is acceptable, a loss in speed is not, so the gate is the measured gain on every
    backend, not bit-identical answers. `--drafting dflash2` always turns drafting on (when the DFlash 2 file is there), `--drafting off` always turns it off."""
    return "auto"

DEFAULT_VISION = True
DEFAULT_PARALLEL = inkling_fit.DEFAULT_PARALLEL
PARALLEL_CHOICES = inkling_fit.PARALLEL_CHOICES
MAX_PARALLEL = inkling_fit.MAX_PARALLEL

#: ---- The 0.2.23 lineup (owner decision 2026-11-26): the three Strata packs, then GSQ-RCO IQ3_S, then Unsloth's UD-Q4_K_XL. Nothing else is offered. ----------------
#: Every download is pinned to a Hugging Face commit, so a later upload to a repository never changes what an install fetches.
STRATA_REPO = "strata-engine/Inkling-2-Mini-Strata"
#: The commit of strata-engine/Inkling-2-Mini-Strata (2026-11-26, 19:24 UTC, owner-approved uploads) that holds SQv2-20 (first uploaded in 16b217027a71, unchanged since) and Vision-v2,
#: and still holds the 0.2.22 files (SQ-20, SQ-15, SQ-10, Vision, the tokenizer files). Every .strata download is pinned to it, so a later upload to the repository never changes what an install fetches.
STRATA_PACK_COMMIT = "5623ceb47d7a01db827460ce1c23e7818a75661b"
SQ20_FILE = "Inkling-2-Mini-SQv2-20.strata"                # the re-adjusted 16 KiB-layout SQ-20; the shown name and the model id stay SQ-20 / inkling-2-mini-sq-20
SQ20_FILE_V1 = "Inkling-2-Mini-SQ-20.strata"               # 0.2.22's file: still accepted where an install already has it
#: build -> the file names an older Strata wrote for it, still accepted on disk (never deleted, never offered for download)
LEGACY_BUILD_FILES = {"SQ-20": (SQ20_FILE_V1,)}

UNSLOTH_REPO = "unsloth/Inkling-2-Mini-GGUF"
#: head of unsloth/Inkling-2-Mini-GGUF on 2026-11-26 (last commit 2026-11-23 16:29 UTC): its UD-Q4_K_XL is the refreshed file (17,629,760,736 bytes, mainline llama.cpp layout).
UNSLOTH_COMMIT = "8078d149286eec7193795413b9e0643da6d11fa1"
UNSLOTH_OLD_COMMIT = "800c946d7b9c830c86290df49afbbe91db8455f5"        # the commit 0.2.22 pinned: the files of UNSLOTH_GGUF_FILES below
UNSLOTH_OFFERED_FILE = "Inkling-2-Mini-UD-Q4_K_XL.gguf"
UNSLOTH_OFFERED_PIN = (17_629_760_736, "e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3")

GSQ_REPO = "ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF"                  # Apache-2.0, not gated; GSQ + RCO by IST Austria DASLab (arXiv 2604.18556, 2605.00649)
GSQ_COMMIT = "357051619a8805c032e58a9976cea9bddb0832cd"                # head on 2026-11-26 (16:20 UTC)
GSQ_IQ3S_FILE = "Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf"
GSQ_IQ3S_PIN = (11_425_259_648, "8a3515374c3cd9b65a1677b17e805bb1e32df8e98756e5f0d2fe1837d62aa4cb")
#: The card's vision projector (llama.cpp's clip file). Strata's engine does not read it (pictures use the Vision shard); the pin is here so a copy that is already on disk is recognised.
GSQ_MMPROJ_FILE = "mmproj-Inkling-2-Mini-BF16.gguf"
GSQ_MMPROJ_PIN = (641_806_176, "4f8608d079a581c585b502b456630d935066e2382c53aa4b2aef0a231505b4ba")
#: What the screens show next to a build's line (Apache-2.0 repositories; the licence text is the repository's own).
BUILD_CREDIT = {
    "GSQ-RCO-IQ3_S": "GSQ + RCO by IST Austria DASLab, Apache-2.0",
    "UD-Q4_K_XL": "Unsloth Dynamic quantization, Apache-2.0",
}

#: Unsloth Dynamic GGUF files as 0.2.22 pinned them (commit UNSLOTH_OLD_COMMIT; exact byte sizes & LFS SHA-256 digests). Since 0.2.23 only UD-Q4_K_XL is offered, from the refreshed
#: commit (FILES); this table is what an install that already downloaded another quant, or the old Q4_K_XL, is checked against (pins_for). Nothing here is downloaded for a new install.
UNSLOTH_GGUF_FILES = {
    "Inkling-2-Mini-UD-Q8_K_XL.gguf": (22_867_451_616, "dfb04bf37bee457f98800989dedf93ec442ede8636dbea9433992f1e527e9a41"),
    "Inkling-2-Mini-UD-Q4_K_XL.gguf": (17_629_759_456, "e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15"),
    "Inkling-2-Mini-UD-Q4_K_M.gguf": (16_610_543_328, "f90c5a4df44d335ac9f9eea2a15c31d6c5a377b1b3cd9ff7fc2cf123f3cdf7e1"),
    "Inkling-2-Mini-UD-IQ4_XS.gguf": (15_733_755_232, "a861d21e7c6d7bc69d8cdcdfb9eb195fd37ef1cd2ab31499db61cd7d30d9b3a1"),
    "Inkling-2-Mini-UD-Q3_K_XL.gguf": (13_631_778_016, "f72f396d1b479b14d76346df80baa449c8b1f399653bff38fa4764b359a40048"),
    "Inkling-2-Mini-UD-IQ3_XXS.gguf": (11_663_309_504, "bd7ef0865adf119b5183e45e741c629bafa649769b33b5ebf689bb4bcc4690d6"),
    "Inkling-2-Mini-UD-Q2_K_XL.gguf": (9_138_715_328, "20292e8142b69535ca88a5d04439d0fdeb4dc698d44c002dca706dcf92901441"),
    "Inkling-2-Mini-UD-IQ2_XXS.gguf": (8_486_236_352, "4fe958f73a667d8082f911a4347e52bc34696a7816cf3d5b9f7c8a81d5c5ceaa"),
    "Inkling-2-Mini-UD-IQ1_M.gguf": (8_024_441_024, "544e0b2412ba2392272b0510b0c8f53a1ab872b73c2ba4f7488de0d5d61467d9"),
    "Inkling-2-Mini-UD-IQ1_S.gguf": (7_764_291_776, "c09c028b050c108624682baeb28ac6e0cdb936af021ac637c8c3609f923ea4b8"),
    DSPARK_GGUF_FILE: (402_278_560, "2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773"),
    MMPROJ_BF16_FILE: (641_805_760, "2c165c78790e78a3f45a2d1e9e931083e52ee69d45ff2a0dc9c281fdf1a666e7"),
    MMPROJ_F16_FILE: (639_348_160, "3bff75ffeb2d6e0371ba7bd42e30d26a7d98aea681fa44b03b5a20891cfa2c5f"),
}

#: The files Strata downloads, file name -> (bytes, sha256 or None). SQv2-20 and Vision-v2: size and SHA-256 of the owner's files, checked against the Hugging Face LFS oids at the pinned commit (2026-11-26); SQ-15 and SQ-10 are unchanged.
FILES = {
    SQ20_FILE: (19_374_768_128, "429671f8eb5b50e593856e6eb82ebb7da9d1f07e99b553438b196f72cc86a6d6"),
    "Inkling-2-Mini-SQ-15.strata": (14_348_316_672, "2ea32eed533afba257e2cbee7b073dc2c3da80fe55d1d2d794779b5498e30737"),
    "Inkling-2-Mini-SQ-10.strata": (10_471_526_400, "ac4d452ad03e4930abd9d46f9c5eec22b2f75a1da45a1e49bd24ccd58b065934"),
    VISION_FILE: (636_551_168, "34a176505276729b58744fc9ee40a5005bbff9a590d8e1f7dbe82baf33ef07d5"),
    GSQ_IQ3S_FILE: GSQ_IQ3S_PIN,
    UNSLOTH_OFFERED_FILE: UNSLOTH_OFFERED_PIN,
}
#: Files that are no longer downloaded but are still recognised (an install that has them keeps working): the 0.2.22 SQ-20 and Vision, every other Unsloth quant,
#: the 0.2.22 pin of UD-Q4_K_XL, Unsloth's DSpark and picture-projector files, the GSQ-RCO projector.
LEGACY_FILES = {
    SQ20_FILE_V1: (19_346_685_952, "0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97"),
    VISION_FILE_V1: (636_526_592, "6589eff47f9c20bc3a032faa3d449f10bf4f04a84cc6b2ee8a44d51d2b6fc677"),
    **UNSLOTH_GGUF_FILES,
    GSQ_MMPROJ_FILE: GSQ_MMPROJ_PIN,
}
ALL_FILES = {**LEGACY_FILES, **FILES}
#: DSpark's old drafter file (0.2.22 fetched it beside a GGUF): the ORIGINAL checkpoint file of Thinking Machines (Apache-2.0), pinned to the commit of the upload.
#: BF16, 635,801,416 bytes. NOT fetched since 0.2.23 (DFlash 2 is the only drafter); the pin stays so that a copy an install already has is recognised and kept, and
#: its sources stay so that `drafter_sources` keeps answering. An old run config that passes it with `--dspark` still starts (see serve/inkling_kind.py).
MTP_FILE = "mtp.safetensors"
MTP_COMMIT = "1fb4134b9a74f9681054f4ef82e4bf1f458a2337"
MTP_BASE = f"https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/{MTP_COMMIT}"
#: An engine reads mtp.safetensors (BF16 norms / biases widened to F32) only when its `--info` line lists this key (0221-DSPARK, measured on Metal: an
#: engine without the widen gets 1.0 tokens per step from the same file, every draft rejected, while Unsloth's Q8_0 file gives 3 to 7). The flag, not
#: the version, decides: builds of any age and custom builds are told apart by what they say.
ENGINE_MTP_FLAG = "dspark_f32_widen"


def engine_reads_mtp(info: dict | None) -> bool:
    return str((info or {}).get(ENGINE_MTP_FLAG)) == "1"


#: The one drafter of every Inkling-2-Mini build (the engine's default, and `--draft` on in Control only when the file is there): incoai/Inkling-2-Mini-DFlash2 (Apache-2.0, not
#: gated, anonymous download), pinned to a commit. The engine reads it from a file beside the model named dflash2.safetensors (the repository calls it model.safetensors).
#: BF16, 989,975,616 bytes (the engine quantizes the block weights at load: STRATA_INK_DFLASH2_WQ). Setup and update fetch it beside EVERY Inkling model that Strata
#: downloaded (the .strata packs and the GGUFs). The DFlash 2 port is by tjm8874. See notices/.
DFLASH2_FILE = "dflash2.safetensors"
DFLASH2_COMMIT = "4f9088ea66c21f2b99ed657d1941c306b03e84d2"
DFLASH2_URL = f"https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/resolve/{DFLASH2_COMMIT}/model.safetensors"
DFLASH2_SOURCE_ENV = "STRATA_INKLING_DFLASH2_SOURCE"
#: Strata's pre-quantized copy of the same weights (engine format "strata-dflash2-q1", variant "mix": attention Q8_ROW, feed-forward SF4, the rest unchanged), written by the
#: engine (STRATA_INK_DFLASH2_DUMP) and reproduced byte for byte by hf-package/make_mix_from_original.py. 551,600,700 bytes against 989,975,616: 44 percent less to download, measured to
#: keep the drafter's acceptance within about 2 percent on an RTX 5090 and an RTX 5060 Ti (phaseA-tests/rel0223/dflash2/RESULTS.md). Both pins are accepted, whichever the file in the folder is.
DFLASH2_Q_PIN = (551_600_700, "cf7dbb08560dfd99544af0cdbab2915406fc571f516b19aff0fe113b9458d387")
#: Where our own copies are planned (the package is phaseA-tests/rel0223/dflash2/hf-package/): repository strata-engine/Inkling-2-Mini-DFlash2, a commit pinned like the others.
#: The commit and the choice are EMPTY until the upload is done and the owner has chosen; empty means that source does not exist and nothing is tried.
#:   option A ("mix"): DFLASH2_Q_FILE (the pre-quantized mix, DFLASH2_Q_PIN) is fetched first;
#:   option B ("original"): DFLASH2_MIRROR_FILE (a byte-for-byte copy of the original, the first pin) is fetched before the third party's repository.
DFLASH2_REPO = "strata-engine/Inkling-2-Mini-DFlash2"
DFLASH2_Q_FILE = "Inkling-2-Mini-DFlash2-mix.safetensors"
DFLASH2_MIRROR_FILE = "model.safetensors"
DFLASH2_OWN_COMMIT = "04ecd36faef74e696f0a7b6b3c119264442a6234"   # the HF commit that holds the files above (uploaded 2026-11-27: Inkling-2-Mini-DFlash2-mix.safetensors, 551,600,700 bytes)
DFLASH2_OWN_COPY = "mix"       # the owner picked option A (2026-11-27): the pre-quantized mix; "original" = option B, empty = no copy of ours is used


def _own_url(file: str, kind: str) -> str:
    return f"https://huggingface.co/{DFLASH2_REPO}/resolve/{DFLASH2_OWN_COMMIT}/{file}" if (DFLASH2_OWN_COMMIT and DFLASH2_OWN_COPY == kind) else ""


DFLASH2_Q_URL = _own_url(DFLASH2_Q_FILE, "mix")
DFLASH2_MIRROR_URL = _own_url(DFLASH2_MIRROR_FILE, "original")
DRAFTER_FILES = {
    MTP_FILE: (635_801_416, "7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2"),
    DFLASH2_FILE: (989_975_616, "217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee"),
}
#: Auxiliary tokenizer files hosted beside the `.strata` weights (Section 0 embeds chat_template + config; tokenizer.json holds the 200K o200k BPE vocab/merges).
TOKENIZER_FILES = {
    TOKENIZER_FILE: (26_178_678, None),
    CHAT_TEMPLATE_FILE: (6_868, None),
}
BUILD_ABOUT = {
    "SQ-20": "bit-exact lossless native SF4+MX8 build: 100% GPU on 24 GiB cards and 32+ GiB unified memory; DFlash 2 drafter",
    "SQ-15": "flagship quality (beats UD-Q4_K_XL): 100% GPU on 20+ GiB cards / 24 GiB Macs, or fast 3x loop-pinned hybrid on 12–16 GiB cards; DFlash 2 drafter",
    "SQ-10": "compact high-speed build: 100% GPU on 12 GiB cards, 16 GiB Macs, 4–8 GiB hybrid cards, and Radeon 780M / Strix Point 890M iGPUs; DFlash 2 drafter",
    "GSQ-RCO-IQ3_S": "3.5-bit GGUF (11.4 GB), keeps 99.2% of the base model's task average (model card): for 12–16 GiB GPUs or hybrid RAM; DFlash 2 drafter",
    "UD-Q4_K_XL": "Unsloth Dynamic ~4.5-bit GGUF: high-quality GGUF for 16–24 GiB GPUs or hybrid RAM (basic support); DFlash 2 drafter",
}
#: The quants Strata 0.2.22 offered that 0.2.23 does not. They are never listed for download; an install that already has one keeps running it and the lists show it
#: as installed, "no longer offered" (their text below is what such a row says).
LEGACY_BUILD_ABOUT = {
    "UD-Q8_K_XL": "Unsloth Dynamic ~8-bit GGUF (no longer offered; an install that has it keeps working)",
    "UD-Q4_K_M": "Unsloth Dynamic 4-bit medium GGUF (no longer offered; an install that has it keeps working)",
    "UD-IQ4_XS": "Unsloth Dynamic 4-bit i-quant GGUF (no longer offered; an install that has it keeps working)",
    "UD-Q3_K_XL": "Unsloth Dynamic ~3.5-bit GGUF (no longer offered; an install that has it keeps working)",
    "UD-IQ3_XXS": "Unsloth Dynamic 3-bit i-quant GGUF (no longer offered; an install that has it keeps working)",
    "UD-Q2_K_XL": "Unsloth Dynamic ~2.5-bit GGUF (no longer offered; an install that has it keeps working)",
    "UD-IQ2_XXS": "Unsloth Dynamic 2-bit i-quant GGUF (no longer offered; an install that has it keeps working)",
    "UD-IQ1_M": "Unsloth Dynamic ~1.7-bit GGUF (no longer offered; an install that has it keeps working)",
    "UD-IQ1_S": "Unsloth Dynamic ~1.5-bit GGUF (no longer offered; an install that has it keeps working)",
}


def is_gguf_build(build: str | None) -> bool:
    return inkling_fit.is_gguf_build(build)


def is_offered(build: str | None) -> bool:
    """True for the five builds 0.2.23 offers (inkling_fit.ORDER); False for a legacy quant that may only be installed already."""
    return str(build or "") in inkling_fit.ORDER


def about_of(build: str) -> str:
    return BUILD_ABOUT.get(build) or LEGACY_BUILD_ABOUT.get(build) or ""


def build_file(build: str) -> str:
    return inkling_fit.BUILDS[build]["file"]


def files_of(build: str, vision: bool = DEFAULT_VISION, dspark: bool = False, drafter: bool = False) -> list[str]:
    """The files for one build: the model file + the Vision shard when enabled + (drafter=True) the DFlash 2 sidecar. `dspark` is kept for old callers and ignored: no DSpark file
    is fetched any more. The sidecar is optional (a missing one means no drafter), which is why setup fetches it separately and asks for it here only to count the download."""
    out = [build_file(build)]
    if vision:
        out.append(VISION_FILE)
    if drafter:
        out.append(DFLASH2_FILE)
    return out


#: The same main GGUFs as Unsloth re-uploaded them on 2026-11-23 14:54 UTC (commit 19bf97a66719d0165f6f8082a9d59581912a3863 "Support mainline llama.cpp"):
#: +1,280 bytes each, a renamed layout (arch inkling2_loop, 24 physical blocks, loop_*_ctrl_* tensors). The engine loads and answers from them, so a file a
#: person already has in this form is accepted next to the 0.2.22 pin. Since 0.2.23 the UD-Q4_K_XL entry is also the offered download's own pin (UNSLOTH_OFFERED_PIN, commit
#: UNSLOTH_COMMIT); the other quants are no longer offered and are only recognised. The re-uploaded drafter (503,889,952 bytes, dflash layout) is not listed.
ALT_GGUF_FILES = {
    "Inkling-2-Mini-UD-Q8_K_XL.gguf": (22_867_452_896, "2da51788d43a17e1f033bec420fa83ba9cb87ac1f0130dc9135a4d2ed2678b68"),
    "Inkling-2-Mini-UD-Q4_K_XL.gguf": (17_629_760_736, "e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3"),
    "Inkling-2-Mini-UD-Q4_K_M.gguf": (16_610_544_608, "78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae"),
    "Inkling-2-Mini-UD-IQ4_XS.gguf": (15_733_756_512, "374970bf6f151fc50da49419751c3e77b82e373a3b9445ded617bd386cd12163"),
    "Inkling-2-Mini-UD-Q3_K_XL.gguf": (13_631_779_296, "e2d03e8c92ac21cff0e3309b95f40dfd27d1825184f0d2bc09db249797273639"),
    "Inkling-2-Mini-UD-IQ3_XXS.gguf": (11_663_310_784, "fd9ccedae17c1bb89744612fbb46730b5275f2a342133d0419776bd1ae311657"),
    "Inkling-2-Mini-UD-Q2_K_XL.gguf": (9_138_716_608, "b42841608ea380ddde5a3f1125081357128a3c71b80d006246de4e31292fc6c1"),
    "Inkling-2-Mini-UD-IQ2_XXS.gguf": (8_486_237_632, "2f953a44281cf861b19ccfdb2ce65f7648e1e92bda4af3729893604fd907c544"),
    "Inkling-2-Mini-UD-IQ1_M.gguf": (8_024_442_304, "af7b45e1c0097fb2602d4c69abf72a009cad78e54201da2265853f5ef11a514a"),
    "Inkling-2-Mini-UD-IQ1_S.gguf": (7_764_293_056, "17a7ff34acb0e1fe878309962458e0633172c7f93ffe3b1c4f5f6b317dc317db"),
}


def _pin_of(name: str) -> tuple[int, str | None] | None:
    return FILES.get(name) or TOKENIZER_FILES.get(name) or DRAFTER_FILES.get(name) or LEGACY_FILES.get(name)


def pins_for(name: str) -> list[tuple[int, str | None]]:
    """Every (bytes, sha256) a file of this name may have: the current pin first, then the pins an older Strata wrote or Unsloth re-uploaded (a file in
    either layout is accepted, so an install that has the old one keeps it)."""
    out = []
    for got in (FILES.get(name) or TOKENIZER_FILES.get(name) or DRAFTER_FILES.get(name), LEGACY_FILES.get(name), ALT_GGUF_FILES.get(name)):
        if got and got not in out:
            out.append(got)
    if name == DFLASH2_FILE and DFLASH2_Q_PIN not in out:
        out.append(DFLASH2_Q_PIN)
    return out


def size_of(name: str) -> int | None:
    if name == DFLASH2_FILE and DFLASH2_Q_URL:
        return DFLASH2_Q_PIN[0]                                 # the first source's file: what the download step shows
    got = _pin_of(name)
    return got[0] if got else None


def sha256_of(name: str) -> str | None:
    got = _pin_of(name)
    return got[1] if got else None


def model_key(build: str) -> str:
    """"SQ-20" -> "inkling-2-mini-sq-20", "GSQ-RCO-IQ3_S" -> "inkling-2-mini-gsq-rco-iq3_s", "UD-Q4_K_XL" -> "inkling-2-mini-ud-q4_k_xl"."""
    return TAG + build.lower()


def model_file_in(folder: str | os.PathLike, build: str) -> Path | None:
    """The model file of `build` that is in `folder`: the current name, else the name an older Strata used (SQ-20 of 0.2.22). None when there is neither."""
    base = Path(folder)
    for name in (build_file(build), *LEGACY_BUILD_FILES.get(build, ())):
        if (base / name).is_file():
            return base / name
    return None


def has_newer_file(build: str, model_path: str | os.PathLike | None, vision_path: str | os.PathLike | None = None) -> bool:
    """True when `model_path` is a model file an older Strata wrote for `build` (SQ-20 of 0.2.22) or `vision_path` is 0.2.22's Vision file, and this Strata has a newer one
    (SQv2-20, Vision-v2): the lists then say "update available". The old files are never deleted and keep running until the person installs the new ones."""
    old_model = bool(model_path) and Path(str(model_path)).name in LEGACY_BUILD_FILES.get(build, ())
    old_vision = bool(vision_path) and Path(str(vision_path)).name == VISION_FILE_V1
    return old_model or old_vision


def vision_in(*folders: str | os.PathLike) -> Path | None:
    """The Vision shard in the first of `folders` that has one: the 0.2.23 file, else 0.2.22's."""
    for folder in folders:
        for name in VISION_FILES:
            if (Path(folder) / name).is_file():
                return Path(folder) / name
    return None


def dflash2_sidecar(model: str | os.PathLike | None, env: dict | None = None) -> Path | None:
    """The DFlash 2 weights this run can use, or None (the run then has no drafter): a file named by STRATA_INK_DFLASH2 (the engine's own override), else
    dflash2.safetensors beside the model. A file that does not have a pinned size is not used (a partial download, another checkpoint)."""
    env = os.environ if env is None else env
    given = str(env.get("STRATA_INK_DFLASH2") or "").strip()
    try:
        if given and Path(given).is_file():
            return Path(given)
        if model:
            p = Path(str(model)).parent / DFLASH2_FILE
            if p.is_file() and p.stat().st_size in {pin[0] for pin in pins_for(DFLASH2_FILE)}:
                return p
    except OSError:
        pass
    return None


SOURCES_FILE = Path(__file__).resolve().parents[1] / "data" / "inkling-sources.json"
SOURCE_ENV = "STRATA_INKLING_SOURCE"
GGUF_SOURCE_ENV = "STRATA_INKLING_GGUF_SOURCE"
GSQ_SOURCE_ENV = "STRATA_INKLING_GSQ_SOURCE"


def _is_commit(text: str) -> bool:
    return len(text) == 40 and all(c in "0123456789abcdef" for c in text)


def hf_base(repo: str, revision: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}"


def pack_base() -> str:
    """The shipped download address of the .strata packs, the Vision shard and the tokenizer files: the pinned commit (STRATA_PACK_COMMIT)."""
    return hf_base(STRATA_REPO, STRATA_PACK_COMMIT)


def pack_commit_is_set() -> bool:
    return _is_commit(STRATA_PACK_COMMIT)


def _shipped(data_key: str, built_in: dict, shipped: Path | None) -> list[dict]:
    """The built-in pinned source, then any extra mirrors listed under `data_key` in the shipped inkling-sources.json (empty by default)."""
    out = _normalise(built_in)
    try:
        data = json.loads((shipped or SOURCES_FILE).read_text(encoding="utf-8"))
        extra = data.get(data_key) if isinstance(data, dict) else (data if data_key == "sources" else None)
        out += _normalise(extra)
    except (OSError, ValueError):
        pass
    return out


def sources(env: dict | None = None, cfg: dict | None = None, shipped: Path | None = None) -> list[dict]:
    """Download sources for native `.strata` / tokenizer files in priority order: config `inkling_source`, env `STRATA_INKLING_SOURCE`, then the pinned Strata repository
    (pack_base()), then any mirror listed in the shipped `inkling-sources.json`."""
    env = os.environ if env is None else env
    out: list[dict] = []
    for item in ((cfg or {}).get("inkling_source"),):
        out += _normalise(item)
    if env.get(SOURCE_ENV):
        out += _normalise(env[SOURCE_ENV])
    return out + _shipped("sources", {"kind": "http", "base": pack_base()}, shipped)


def gguf_sources(env: dict | None = None, cfg: dict | None = None, shipped: Path | None = None) -> list[dict]:
    """Download sources for Unsloth GGUF files in priority order: config/env overrides first, then the pinned Unsloth commit (UNSLOTH_COMMIT)."""
    env = os.environ if env is None else env
    out: list[dict] = []
    for item in ((cfg or {}).get("inkling_gguf_source"), (cfg or {}).get("inkling_source")):
        out += _normalise(item)
    if env.get(GGUF_SOURCE_ENV):
        out += _normalise(env[GGUF_SOURCE_ENV])
    if env.get(SOURCE_ENV):
        out += _normalise(env[SOURCE_ENV])
    return out + _shipped("gguf_sources", {"kind": "http", "base": hf_base(UNSLOTH_REPO, UNSLOTH_COMMIT)}, shipped)


def gsq_sources(env: dict | None = None, cfg: dict | None = None, shipped: Path | None = None) -> list[dict]:
    """Download sources for the GSQ-RCO GGUF: config `inkling_gsq_source`, env STRATA_INKLING_GSQ_SOURCE, STRATA_INKLING_SOURCE (a local mirror of everything), then the pinned
    ISTA-DASLab commit (GSQ_COMMIT)."""
    env = os.environ if env is None else env
    out = _normalise((cfg or {}).get("inkling_gsq_source"))
    if env.get(GSQ_SOURCE_ENV):
        out += _normalise(env[GSQ_SOURCE_ENV])
    if env.get(SOURCE_ENV):
        out += _normalise(env[SOURCE_ENV])
    return out + _shipped("gsq_sources", {"kind": "http", "base": hf_base(GSQ_REPO, GSQ_COMMIT)}, shipped)


DRAFTER_SOURCE_ENV = "STRATA_INKLING_DRAFTER_SOURCE"


def drafter_sources(env: dict | None = None, cfg: dict | None = None) -> list[dict]:
    """Download sources for mtp.safetensors: config inkling_drafter_source, env STRATA_INKLING_DRAFTER_SOURCE, STRATA_INKLING_SOURCE, then the pinned original repository."""
    env = os.environ if env is None else env
    out = _normalise((cfg or {}).get("inkling_drafter_source"))
    if env.get(DRAFTER_SOURCE_ENV):
        out += _normalise(env[DRAFTER_SOURCE_ENV])
    if env.get(SOURCE_ENV):                                    # a local mirror of everything (tests, offline installs)
        out += _normalise(env[SOURCE_ENV])
    return out + [{"kind": "http", "base": MTP_BASE, "files": {}}]


def sources_for(name: str, env: dict | None = None, cfg: dict | None = None, shipped: Path | None = None) -> list[dict]:
    """Return the prioritized download sources for `name` (Unsloth GGUF repo for GGUF files, Strata repo for .strata/tokenizer)."""
    if name == DFLASH2_FILE:
        env = os.environ if env is None else env
        mirror = _normalise(env.get(DFLASH2_SOURCE_ENV)) if env.get(DFLASH2_SOURCE_ENV) else []
        own = [{"kind": "http", "base": u.rsplit("/", 1)[0], "files": {DFLASH2_FILE: u}} for u in (DFLASH2_Q_URL, DFLASH2_MIRROR_URL) if u]
        return mirror + own + [{"kind": "http", "base": DFLASH2_URL.rsplit("/", 1)[0], "files": {DFLASH2_FILE: DFLASH2_URL}}]
    if name in DRAFTER_FILES:
        return drafter_sources(env=env, cfg=cfg)
    if name in (GSQ_IQ3S_FILE, GSQ_MMPROJ_FILE):
        return gsq_sources(env=env, cfg=cfg, shipped=shipped)
    if name in UNSLOTH_GGUF_FILES and name != UNSLOTH_OFFERED_FILE:      # a quant that is no longer offered: its 0.2.22 commit (a file that is already there is checked against it)
        return [s if s.get("base") != hf_base(UNSLOTH_REPO, UNSLOTH_COMMIT) else {**s, "base": hf_base(UNSLOTH_REPO, UNSLOTH_OLD_COMMIT)}
                for s in gguf_sources(env=env, cfg=cfg, shipped=shipped)]
    if name in UNSLOTH_GGUF_FILES or str(name).lower().endswith(".gguf"):
        return gguf_sources(env=env, cfg=cfg, shipped=shipped)
    return sources(env=env, cfg=cfg, shipped=shipped)


def _normalise(item) -> list[dict]:
    if isinstance(item, str):
        return [{"kind": "http", "base": item.rstrip("/")}] if item.startswith(("http://", "https://")) else []
    if isinstance(item, dict):
        base = str(item.get("base") or "").rstrip("/")
        kind = item.get("kind") if item.get("kind") in ("http", "http-parts") else "http"
        files = item.get("files") if isinstance(item.get("files"), dict) else {}
        if base.startswith(("http://", "https://")) or files:
            return [{"kind": kind, "base": base, "files": files}]
        return []
    if isinstance(item, list):
        return [x for sub in item for x in _normalise(sub)]
    return []


def urls_for(name: str, source: dict) -> list[str]:
    listed = (source.get("files") or {}).get(name)
    if listed:
        return [listed] if isinstance(listed, str) else list(listed)
    base = source.get("base") or ""
    if not base:
        return []
    return [f"{base}/{name}"]


def local_files(folder: str | os.PathLike, build: str | None = None) -> dict[str, Path]:
    """Inkling-2-Mini .strata / .gguf files already present in `folder` with a pinned size (the current files and the ones an older Strata wrote)."""
    found: dict[str, Path] = {}
    base = Path(folder)
    wanted = None
    if build is not None and build in inkling_fit.BUILDS:
        wanted = {build_file(build), *LEGACY_BUILD_FILES.get(build, ())}
    for name in ALL_FILES:
        if wanted is not None and name.startswith("Inkling-2-Mini-") and name not in VISION_FILES and name not in wanted:
            continue
        p = base / name
        try:
            if p.is_file() and p.stat().st_size in {pin[0] for pin in pins_for(name)}:
                found[name] = p
        except OSError:
            continue
    return found
