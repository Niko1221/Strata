"""Local hy4-flash launch configuration. The HTTP server and runtime supervisor stay shared."""
from __future__ import annotations

import math
import mmap
import os
import re
import subprocess
import sys
from pathlib import Path


PACK_FILES = ("index.txt", "dense.bin", "native_experts.txt", "engram_token_map.bin",
              "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/token_type.json",
              "tokenizer/tokenizer.json", "tokenizer/chat_template.jinja")

RUNTIME_KEYS = (
    "STRATA_HY4_FLASH_GPU_SAMPLE", "STRATA_HY4_FLASH_WINDOW", "STRATA_HY4_FLASH_WINDOW_GRAPH",
    "STRATA_HY4_FLASH_WINDOW_COMPACT_ROWS", "STRATA_HY4_FLASH_MTP_GRAPH",
    "STRATA_HY4_FLASH_MTP_CACHE_ONLY", "STRATA_HY4_FLASH_WINDOW_SKIP_EMPTY",
)


def runtime_capabilities(exe: str) -> set[str]:
    """Read compiled feature markers without starting CUDA or executing an old binary."""
    try:
        with open(exe, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            return {name for name, marker in (
                ("sample", b"HY4_FLASH_GPU_SAMPLE enabled"),
                ("window", b"HY4_FLASH_WINDOW_SUPPLY initialized"),
                ("window_cpu", b"HY4_FLASH_WINDOW_CPU enabled"),
                ("window_borrow", b"prefill borrow: verify window drain failed"),
                ("device_residual", b"STRATA_HY4_FLASH_PREFILL_LM_R"),
                ("draft", b"HY4_FLASH_MTP_GRAPH captured"),
                ("conditional", b"window conditional body capture failed"),
                ("vision", b"hy4-flash requires a hy_v4_flash mmproj GGUF"),
                ("kv_q8", b"HY4_FLASH_KV_Q8 enabled"),
                ("metal", b"STRATA_METAL_ARENA_GB"),
                ("dfly5", b"dfly: --draft-max exceeds the GGUF block size minus the anchor"),
                ("dfly", b"--draft dfly requires F16 or Q8_0 KV"),
            ) if data.find(marker) >= 0}
    except (OSError, ValueError):
        return set()


def mtp_model_layout(path: str) -> tuple[bool, int]:
    """Return draft compatibility and the expert count recorded in this GGUF."""
    from tools.gguf_reader import GGUFFile
    model = GGUFFile(model_file(path))
    meta = model.metadata
    prefix = f"blk.{int(meta.get('hy_v4_flash.block_count', 0))-1}."
    types = {t.name: t.type_id for t in model.tensors}
    supported = (meta.get("hy_v4_flash.nextn_predict_layers") == 1
            and meta.get("hy_v4_flash.attention.indexer.kpool") == 4
            and meta.get("hy_v4_flash.expert_used_count") == 16
            and types.get("token_embd.weight") == 8
            and types.get(prefix + "ffn_gate_exps.weight") == 8)
    return supported, int(meta.get("hy_v4_flash.expert_count", 0))


def is_48g_4090(card: dict) -> bool:
    return (card.get("vendor") == "nvidia" and re.search(r"\bRTX 4090\b", card.get("name", ""), re.I) is not None
            and 45 <= float(card.get("vram_gb") or 0) <= 50)


def prefill_loan_profile(s, card, available: float):
    """Return (chunk, minimum pool, reserve credit) for measured ordinary 4090 plans."""
    if (not card or not is_48g_4090(card) or s.resident_experts
            or str(s.extra.get("STRATA_HY4_FLASH_PREFILL_LAYER_MAJOR", "1")) != "1"
            or str(s.extra.get("STRATA_HY4_FLASH_PREFILL_BORROW", "1")) != "1"):
        return None
    # Smaller cards need room for the largest layer plus the chosen chunk inside
    # the loan. At 16 GiB (no draft) 832 MiB of the legacy reserve is reusable.
    if 15 <= available <= 16.0625 and s.mtp == "off":
        profile = (1024, 4.9, 0.8125)
    elif 23 <= available <= 24.0625:
        profile = (2048, 7.4, 0.0)
    elif 31 <= available <= 32.0625:
        profile = (2048, 7.4, 0.0)
    elif available >= 47:
        profile = (4096, 13.0, 0.0)
    else:
        return None
    requested = s.extra.get("STRATA_HY4_FLASH_PREFILL_CHUNK", profile[0])
    return profile if str(requested) == str(profile[0]) else None


def apply_runtime_defaults(cfg: dict, s, computer) -> None:
    """Measured single-GPU defaults after launch overrides; explicit settings always win.

    Keep provenance so a second resolution (e.g. enabling CPU experts or changing
    cards) can recompute our defaults without erasing the user's overrides.
    """
    env = cfg.setdefault("env", {})
    previous = cfg.pop("hy4_flash_runtime_defaults", {})
    for key, value in previous.items():
        if key not in s.extra and env.get(key) == value:
            base = serving_environment(s.mtp == "on" or bool(s.dfly_model))
            if key in base:
                env[key] = base[key]
            else:
                env.pop(key, None)
    selected = s.gpus or [0]
    card = next((c for c in computer.usable_gpus if len(selected) == 1
                 and int(c.get("index", -1)) == selected[0] and c.get("vendor") == "nvidia"), None)
    h20 = card is not None and re.search(r"\bH20\b", card.get("name", ""), re.I) is not None
    ada_4090 = card is not None and is_48g_4090(card)
    # Feature defaults follow what the engine build supports, on any single CUDA card.
    caps = runtime_capabilities(cfg.get("exe", "")) if len(selected) == 1 and cfg.get("backend") == "cuda" else set()
    auto = {}
    def default(key: str, enabled: bool):
        if key in s.extra:
            # derive_config already applies explicit --set, including removals.
            return
        if key not in env:
            env[key] = auto[key] = "1" if enabled else "0"
    def on(key: str, fallback="0") -> bool:
        return str(env.get(key, fallback)) not in ("0", "", "None")
    default("STRATA_HY4_FLASH_GPU_SAMPLE", "sample" in caps)
    mtp = s.mtp == "on"
    mtp_shape_ok = mtp and "draft" in caps and mtp_model_layout(s.model_path)[0]
    chunk_key = "STRATA_HY4_FLASH_PREFILL_CHUNK"
    borrow_key = "STRATA_HY4_FLASH_PREFILL_BORROW"
    available = available_vram_gib(card) if ada_4090 and "window_borrow" in caps else 0
    profile = prefill_loan_profile(s, card, available)
    if profile and s.expert_vram is not None and s.expert_vram >= profile[1]:
        chunk, _, credit = profile
        if chunk_key not in s.extra:
            env[chunk_key] = auto[chunk_key] = str(chunk)
        if borrow_key not in s.extra:
            env[borrow_key] = auto[borrow_key] = "1"
        residual = "STRATA_HY4_FLASH_PREFILL_LM_R"
        superchunk = str(env.get("STRATA_HY4_FLASH_PREFILL_SUPERCHUNK", "8192"))
        reserve = trunk_reserve_gib(s.context or 4096, s.kv or "fp16") + dfly_reserve_gib(s)
        if mtp:
            reserve += draft_workspace_gib(s.context or 4096, s.kv or "fp16") + cfg.get("mtp_resident_bytes", 0) / 2**30
        if (residual not in s.extra and residual not in env and on(borrow_key)
                and "device_residual" in caps and superchunk.isdecimal() and 0 < int(superchunk) <= 8192
                and s.expert_vram <= available - reserve + credit + 0.005):
            env[residual] = auto[residual] = "vram"
    if ada_4090:
        default("STRATA_HY4_FLASH_ATTN_MMA_PREC", True)
    if chunk_key not in s.extra and chunk_key not in auto and env.get(chunk_key) == "1024":
        if s.expert_vram == 0:
            env[chunk_key] = auto[chunk_key] = "256"
        elif card is not None and s.expert_vram is not None:
            context = s.context or 4096
            available = available_vram_gib(card)
            spare = available - s.expert_vram - trunk_reserve_gib(context, s.kv or "fp16") - dfly_reserve_gib(s)
            if mtp:
                spare -= draft_workspace_gib(context, s.kv or "fp16") + cfg.get("mtp_resident_bytes", 0) / 2**30
            # Launch settings express the expert budget to the nearest 0.01 GiB.
            preferred = preferred_prefill_chunk(available)
            if preferred == 2048 and spare >= PREFILL_2048_EXTRA_GIB - 0.005:
                env[chunk_key] = auto[chunk_key] = "2048"
            elif preferred == 512 or spare < PREFILL_1024_EXTRA_GIB - 0.005:
                small_host_draft = mtp and mtp_host_experts(s) and s.expert_vram < MIN_EXPERT_VRAM_GIB
                env[chunk_key] = auto[chunk_key] = "256" if small_host_draft else "512"
    cpu_key = "STRATA_HY4_FLASH_CPU_MISS_MAX"
    if mtp and "window_cpu" not in caps and cpu_key not in s.extra and env.get(cpu_key, "0") != "0":
        # An engine without CPU verify windows keeps MTP on the GPU rather than losing the window.
        env[cpu_key] = auto[cpu_key] = "0"
    compatible = ((mtp_shape_ok or (bool(s.dfly_model) and "dfly" in caps)) and (str(env.get(cpu_key, "0")) == "0" or "window_cpu" in caps)
                  and on("STRATA_HY4_FLASH_ASYNC_MOE") and (s.dfly_model or s.resident_experts or on("STRATA_HY4_FLASH_PIN_EXPERT_MMAP"))
                  and not on("STRATA_DF_PDL") and
                  (not on("STRATA_HY4_FLASH_PREFILL_BORROW") or "window_borrow" in caps))
    default("STRATA_HY4_FLASH_WINDOW", compatible and "window" in caps)
    default("STRATA_HY4_FLASH_WINDOW_GRAPH", compatible and on("STRATA_HY4_FLASH_WINDOW") and on("STRATA_HY4_FLASH_GRAPHS"))
    default("STRATA_HY4_FLASH_WINDOW_COMPACT_ROWS", compatible and on("STRATA_HY4_FLASH_WINDOW"))
    default("STRATA_HY4_FLASH_MTP_GRAPH", mtp_shape_ok and not on("STRATA_DF_PDL"))
    default("STRATA_HY4_FLASH_MTP_CACHE_ONLY", mtp_shape_ok)
    default("STRATA_HY4_FLASH_WINDOW_SKIP_EMPTY", h20 and "conditional" in caps and on("STRATA_HY4_FLASH_WINDOW_GRAPH"))
    cfg["hy4_flash_runtime_defaults"] = auto


def available_vram_gib(card: dict) -> float:
    # The catalog rounds the physical capacity to 0.1 GiB. NVML free memory
    # already excludes the driver reservation included in the 4090 fit.
    capacity = float(card.get("vram_gb") or 0)
    available = max(0.0, capacity - 0.62)
    if card.get("uma") or not card.get("driver"):
        return available
    try:
        measured = subprocess.check_output(
            ["nvidia-smi", "-i", str(card["index"]), "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=3)
        return min(available, int(measured.strip()) / 1024)
    except (OSError, ValueError, subprocess.SubprocessError):
        return available


def vision_shard(s) -> Path:
    configured = s.extra.get("STRATA_HY4_FLASH_VISION_SHARD") or os.environ.get("STRATA_HY4_FLASH_VISION_SHARD")
    if configured:
        path = Path(configured).expanduser().absolute()
    elif s.model_path:
        folder = model_file(s.model_path).parent
        ggufs = sorted(p for p in folder.glob("mmproj*.gguf") if p.is_file())
        if not ggufs:
            raise ValueError(f"Missing hy4-flash mmproj GGUF in {folder}. Set STRATA_HY4_FLASH_VISION_SHARD to its path.")
        if len(ggufs) > 1:
            raise ValueError("Multiple mmproj GGUFs found; select one with STRATA_HY4_FLASH_VISION_SHARD.")
        path = ggufs[0]
    else:
        raise ValueError("hy4-flash pictures require the model and its vision shard.")
    if path.suffix.lower() != ".gguf":
        raise ValueError(f"hy4-flash pictures require a mmproj GGUF file: {path}")
    if not path.is_file():
        raise ValueError(f"Missing hy4-flash mmproj GGUF: {path}")
    return path


def defaults(s, computer, src: Path | None = None):
    """Resolve supported choices before a plan or a start; never silently discard a request."""
    if s.parallel not in (None, 1):
        raise ValueError("hy4-flash currently serves one inference request at a time; use --parallel 1 (HTTP requests can queue).")
    pictures = s.vision in ("on", "gpu")
    if s.vision not in (None, "off", "on", "gpu"):
        raise ValueError("hy4-flash pictures require the CUDA encoder; use --vision on or off.")
    if pictures:
        exe = find_engine(s, src or Path(__file__).resolve().parents[1])
        caps = runtime_capabilities(str(exe)) if exe else set()
        if "vision" not in caps:
            raise ValueError("hy4-flash pictures require an engine built with CUDA mmproj GGUF support.")
        vision_shard(s)
    if s.dfly_model:
        if s.mtp == "on" or s.drafting not in (None, "auto", "dfly"):
            raise ValueError("--dfly-model selects the external drafter; remove the conflicting drafting choice")
        exe = find_engine(s, src or Path(__file__).resolve().parents[1])
        draft_caps = runtime_capabilities(str(exe)) if exe else set()
        if "dfly" not in draft_caps:
            raise ValueError("--dfly-model requires an engine built with hy4-flash Dfly and Q8_0 KV support")
        from tools.gguf_reader import GGUFFile
        path = Path(s.dfly_model).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Missing hy4-flash Dfly GGUF: {path}")
        draft = GGUFFile(path)
        if (draft.metadata.get("general.architecture") != "hy_v4_next_dfly"
                or not draft.tensors or any(t.type_id not in (0, 8) for t in draft.tensors)):
            raise ValueError("--dfly-model needs a hy_v4_next_dfly GGUF with Q8_0 matrices; convert BF16 with tools/hy4_flash/quantize_mtp.py --dfly")
        s.dfly_model = str(path)
        if s.draft_tokens is None:
            s.draft_tokens = 3
        if not 1 <= s.draft_tokens <= 5:
            raise ValueError("hy4-flash Dfly --draft-tokens must be 1..5")
        if s.draft_tokens > 3 and "dfly5" not in draft_caps:
            raise ValueError("Rebuild the hy4-flash Dfly engine to use more than 3 draft tokens")
    elif s.drafting == "dfly":
        raise ValueError("--drafting dfly requires --dfly-model GGUF")
    elif s.drafting not in (None, "off", "mtp", "auto"):
        raise ValueError("hy4-flash supports its MTP drafter; DFlash2 is not supported.")
    if s.mtp == "off" and s.drafting == "mtp" or s.mtp == "on" and s.drafting == "off":
        raise ValueError("Conflicting hy4-flash MTP and drafting choices")
    mtp = s.mtp == "on" or s.drafting == "mtp"
    if s.draft_tokens is not None and not 1 <= s.draft_tokens <= 7:
        raise ValueError("hy4-flash --draft-tokens must be 1..7")
    if s.kv not in (None, *KV_FORMATS):
        raise ValueError("hy4-flash keeps its latent KV cache as int8 (Q8_0) or fp16; use --kv int8 or --kv fp16.")
    if s.tune != "skip" or s.split_engine not in (None, "off") or s.gpu_roles:
        raise ValueError("hy4-flash uses one CUDA GPU, without the Qwen tuner or GPU roles; use --gpu N --tune skip --split-engine off.")
    if s.gpus and len(s.gpus) != 1:
        raise ValueError("hy4-flash currently uses one GPU; specify one index with --gpu N.")
    if "--expert-host" in s.extra:
        raise ValueError("The saved hy4-flash expert-host budget has been retired. Choose --resident-experts "
                         "for split GPU/RAM storage or --no-resident-experts for a full host copy.")
    if "STRATA_HY4_FLASH_EXPERT_HOST_GIB" in s.extra or "STRATA_HY4_FLASH_EXPERT_HOST_GIB" in os.environ:
        raise ValueError("Remove STRATA_HY4_FLASH_EXPERT_HOST_GIB and select the hy4-flash storage mode with --resident-experts.")
    advanced = {"--temp", "--top-k", "--top-p", "--repeat-penalty", "--repeat-last", "--seed", "--greedy",
                "--conversation-cache-mib", "--conversation-cache-slots"}
    for flag in s.extra:
        if flag.startswith("--") and flag not in advanced:
            raise ValueError(f"hy4-flash launch does not support --set {flag}; use its named launch options or hy4-flash environment settings.")
    cards = [g for g in computer.usable_gpus if g.get("vendor") == "nvidia"]
    if not cards and sys.platform == "darwin":
        # Apple silicon: the engine is the native Metal one, whose expert budget is the unified-memory arena
        # it plans for itself, not a card's VRAM. Everything below this point reads a CUDA card, so the Mac
        # path resolves its own small set of choices and returns; off Darwin nothing here changes.
        _metal_defaults(s, src)
        return
    if not cards:
        raise ValueError("hy4-flash requires a CUDA GPU and a local hy4-flash-enabled engine build.")
    if not s.gpus:
        s.gpus = [int(cards[0].get("index", 0))]
    card = next((g for g in cards if int(g.get("index", 0)) == s.gpus[0]), None)
    if card is None:
        raise ValueError(f"hy4-flash GPU {s.gpus[0]} is not an available CUDA GPU.")
    s.context = s.context or 4096
    s.kv = kv_format(s, src)
    vram = available_vram_gib(card) if s.expert_vram is None or s.mtp is None else None
    draft_reserve = draft_workspace_gib(s.context, s.kv)
    reserve = trunk_reserve_gib(s.context, s.kv) + (2.0 if pictures else 0.0)
    wide_reserve = 0.0
    if vram is not None:
        chunk = str(s.extra.get("STRATA_HY4_FLASH_PREFILL_CHUNK", preferred_prefill_chunk(vram)))
        if chunk.isdecimal() and int(chunk) >= 2048:
            wide_reserve = PREFILL_2048_EXTRA_GIB
        elif (chunk.isdecimal() and int(chunk) >= 1024) or chunk == "auto":
            wide_reserve = PREFILL_1024_EXTRA_GIB
        reserve += wide_reserve
    auto_mtp = not s.dfly_model and s.mtp is None and s.drafting in (None, "auto")
    resolve_layout = auto_mtp or (mtp and s.draft_tokens is None)
    supported, expert_count = mtp_model_layout(s.model_path) if s.model_path and resolve_layout else (False, 0)
    reap25 = supported and expert_count == 384
    if auto_mtp and supported:
        # Under tight budgets host drafts slowed five of six 16 GiB decode cases.
        # Enable speculative decode by default when the GPU can hold both the draft
        # and the trunk pool. An explicit MTP request can still use host experts.
        trunk_pool = MIN_EXPERT_VRAM_GIB if s.expert_vram is None else s.expert_vram
        mtp = (not mtp_host_experts(s) and
               vram - reserve - draft_reserve
               - mtp_weight_bytes(model_file(s.model_path)) / 2**30 >= trunk_pool)
    s.parallel, s.vision, s.mtp, s.drafting, s.split_engine = 1, "on" if pictures else "off", "on" if mtp else "off", "mtp" if mtp else "off", "off"
    if s.dfly_model:
        s.drafting = "dfly"
    ada_draft = mtp and is_48g_4090(card) and supported
    if mtp and s.draft_tokens is None:
        cpu_window = (int(s.extra.get("STRATA_HY4_FLASH_CPU_MISS_MAX", environment()["STRATA_HY4_FLASH_CPU_MISS_MAX"]) or 0) > 0
                      and str(s.extra.get("STRATA_HY4_FLASH_WINDOW", "1")) != "0")
        s.draft_tokens = 1 if reap25 or (ada_draft and not cpu_window) else 3
    if s.expert_vram is None:
        # Assign the memory left after the estimated working peak to the expert pool.
        if mtp:
            reserve += draft_reserve
            host_key = "STRATA_HY4_FLASH_MTP_HOST_EXPERTS"
            if (host_key not in s.extra and host_key not in os.environ and
                    vram - reserve - mtp_weight_bytes(model_file(s.model_path)) / 2**30 < MIN_EXPERT_VRAM_GIB):
                # A 12 GiB card cannot also hold the draft experts: map them from host memory.
                s.extra[host_key] = "1"
            reserve += mtp_weight_bytes(model_file(s.model_path), host_experts=mtp_host_experts(s)) / 2**30
        pool = vram - reserve - dfly_reserve_gib(s)
        profile = prefill_loan_profile(s, card, vram)
        if profile and pool + wide_reserve + profile[2] >= profile[1]:
            exe = find_engine(s, src or Path(__file__).resolve().parents[1])
            if exe is not None and "window_borrow" in runtime_capabilities(str(exe)):
                pool += wide_reserve + profile[2]
        s.expert_vram = round(max(0, pool), 2)
    if not math.isfinite(s.expert_vram) or s.expert_vram < 0:
        raise ValueError("--expert-vram must be a finite, non-negative GiB budget.")


def _metal_defaults(s, src: Path | None) -> None:
    """The Apple-silicon choices: what the native Metal engine actually takes, and nothing invented.

    The expert budget is deliberately NOT set here. STRATA_METAL_ARENA_GB short-circuits the engine's own
    arena plan and with it the live shrink/grow guards, so a number chosen here would be a fixed budget with
    the self-correction switched off - and the right number is the machine's, not this file's. A person who
    wants a fixed arena passes --set STRATA_METAL_ARENA_GB=N, which reaches the engine through `extra`.
    """
    pictures = s.vision in ("on", "gpu")
    if pictures:
        # The Metal engine has its own tower now (engine/metal/src/hy4flash_gguf/vis_tower.cpp).  The capability marker is
        # the gate, not the platform: a binary without it refuses --vision for the same reason the CUDA
        # path does, and an older Metal build is one of those binaries.
        exe = find_engine(s, src or Path(__file__).resolve().parents[1])
        if not exe or "vision" not in runtime_capabilities(str(exe)):
            raise ValueError("This Metal engine was built without the hy4-flash vision tower; use --vision off.")
        vision_shard(s)
    s.context = s.context or 4096
    s.kv = kv_format(s, src)           # the binary is probed for Q8_0 KV, so a Metal build resolves to fp16
    if s.kv != "fp16":
        raise ValueError("The Metal hy4-flash engine keeps its latent KV cache as fp16; use --kv fp16.")
    mtp = s.mtp == "on" or s.drafting == "mtp"
    s.parallel, s.vision = 1, "on" if pictures else "off"
    s.mtp, s.drafting = ("on", "mtp") if mtp else ("off", "off")
    s.split_engine = "off"
    s.gpus = s.gpus or [0]
    # Accepted and ignored by this engine (its experts live in the arena, not in a card's VRAM); kept finite so
    # the caller's own validation of the field still means something.
    s.expert_vram = 0.0


def find_engine(s, src: Path) -> Path | None:
    explicit = s.engine_path or os.environ.get("STRATA_BIN")
    candidates = [Path(explicit).expanduser()] if explicit else [
        src.parent / "engine/build/strata", src.parent / "build/strata", src / "build/strata",
        src / "build-4090/strata", src / "build-local-h20/strata",
    ]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    return None


def engine_path(s, src: Path) -> Path:
    path = find_engine(s, src)
    if path is not None:
        s.engine_path = str(path)
        return path
    raise ValueError("hy4-flash needs a local engine build: pass --engine-path /path/to/build/strata. See the README build command.")


def model_file(path: str) -> Path:
    # Preserve shard filenames in symlinked snapshots for sibling discovery.
    p = Path(path).expanduser().absolute()
    if p.is_dir():
        candidates = sorted(p.glob("*.gguf"))
        candidates = [f for f in candidates if not f.name.startswith("mmproj")]
        from strata_control.models import SHARD
        candidates = [f for f in candidates if not SHARD.fullmatch(f.name) or
                      int(SHARD.fullmatch(f.name).group(2)) == 1]
        if len(candidates) != 1:
            raise ValueError("For hy4-flash, pass --model-path to one GGUF file when the folder contains several models.")
        p = candidates[0]
    if not p.is_file():
        raise ValueError(f"hy4-flash model is missing: {p}")
    from strata_control.models import SHARD
    match = SHARD.fullmatch(p.name)
    if match:
        total = int(match.group(3))
        if not 1 <= int(match.group(2)) <= total:
            raise ValueError("Invalid hy4-flash GGUF shard numbering")
        for i in range(1, total + 1):
            sibling = p.with_name(f"{match.group(1)}-{i:05d}-of-{total:05d}.gguf")
            if not sibling.is_file():
                raise ValueError(f"hy4-flash model shard is missing: {sibling}")
        return p.with_name(f"{match.group(1)}-00001-of-{total:05d}.gguf")
    return p


def pack_path(s) -> Path:
    if s.pack_path:
        s.pack_path = str(Path(s.pack_path).expanduser().resolve())
        return Path(s.pack_path)
    beside = model_file(s.model_path).parent / "pack"
    if beside.is_dir() or not s.data_dir:
        return beside
    return Path(s.data_dir).expanduser().resolve() / "packs/hy4-flash-q4_k_m"


def missing_pack(pack: Path) -> list[str]:
    return [name for name in PACK_FILES if not (pack / name).is_file()]


def mtp_host_experts(s) -> bool:
    key = "STRATA_HY4_FLASH_MTP_HOST_EXPERTS"
    value = s.extra.get(key, os.environ.get(key, "0")).lower()
    if value in ("off", "unset", "none", "0"):
        return False
    if value != "1":
        raise ValueError(f"{key} must be 0 or 1")
    return True


def mtp_weight_bytes(path: Path, *, host_experts: bool = False) -> int:
    """GPU-resident bytes of the draft layer, measured from both shard directories."""
    from tools.gguf_reader import GGUFFile
    from strata_control.models import SHARD
    first = GGUFFile(path)
    blocks = first.metadata.get("hy_v4_flash.block_count", 0)
    count = first.metadata.get("hy_v4_flash.nextn_predict_layers", 0)
    if count != 1:
        raise ValueError("hy4-flash MTP requires one nextn layer in the GGUF")
    match = SHARD.fullmatch(path.name)
    files = [path] if not match else [path.with_name(f"{match.group(1)}-{i:05d}-of-{int(match.group(3)):05d}.gguf")
                                    for i in range(1, int(match.group(3))+1)]
    prefix = f"blk.{blocks-count}."
    mapped = {prefix + f"ffn_{role}_exps.weight" for role in ("gate", "up", "down")} if host_experts else set()
    return sum(t.expected_bytes() for p in files for t in (first if p == path else GGUFFile(p)).tensors
               if t.name.startswith(prefix) and t.name not in mapped)


# Headroom above the measured peak for allocator fragmentation and other processes.
VRAM_MARGIN_GIB = 0.25
# Below this trunk pool, draft experts move to host memory (measured: GPU drafts win at 24 GiB).
MIN_EXPERT_VRAM_GIB = 2.0
# `--kv` -> the engine's STRATA_HY4_FLASH_KV. int8 keeps the 512-wide latent as Q8_0 (32-wide F16 scales) and the
# 64-wide rope key as F16: 672 B a token a block against 1152 at fp16. 42 trunk blocks plus the draft's one.
KV_FORMATS = {"int8": "q8_0", "fp16": "f16"}
KV_SAVED_PER_TOKEN = {"int8": 42 * (1152 - 672), "fp16": 0}
KV_SAVED_PER_DRAFT_TOKEN = {"int8": 1152 - 672, "fp16": 0}


def kv_format(s, src: Path | None) -> str:
    """Resolve the effective latent format before budgeting memory for the selected engine."""
    forced = str(s.extra.get("STRATA_HY4_FLASH_KV", "")).lower()   # --set STRATA_HY4_FLASH_KV=..., which wins in the env
    requested = {"q8_0": "int8", "q8": "int8", "f16": "fp16", "fp16": "fp16"}.get(forced, s.kv)
    if requested == "fp16":
        return "fp16"
    exe = find_engine(s, src or Path(__file__).resolve().parents[1])
    if exe is not None and "kv_q8" in runtime_capabilities(str(exe)):
        return "int8"
    if requested == "int8" and exe is not None:
        raise ValueError(f"This hy4-flash engine does not support Q8_0 KV: {exe}. Select --kv fp16 or a Q8_0 build.")
    return requested or "fp16"


# Additional space above the fitted reserves to allow a wider prefill.
PREFILL_1024_EXTRA_GIB = 1.0
PREFILL_2048_EXTRA_GIB = 3.5


def preferred_prefill_chunk(available_gib: float) -> int:
    # Match the 32 and 48 GiB tiers, allowing for driver reservations.
    if 31 <= available_gib <= 32.0625 or 47 <= available_gib <= 48.0625:
        return 2048
    return 512 if available_gib <= 16 else 1024


def trunk_reserve_gib(context: int, kv: str) -> float:
    return VRAM_MARGIN_GIB + 9.65 + context * (54e3 - KV_SAVED_PER_TOKEN[kv]) / 2**30


def dfly_reserve_gib(s) -> float:
    # The drafter shares the target KV. Allow for its uploaded weights, fixed scratch
    # and one int32 key-position table spanning the requested context.
    if not s.dfly_model:
        return 0.0
    return (Path(s.dfly_model).stat().st_size + 4 * (s.context or 4096)) / 2**30 + 0.125


def draft_workspace_gib(context: int, kv: str) -> float:
    return 0.12 + context * (20e3 - KV_SAVED_PER_DRAFT_TOKEN[kv]) / 2**30


def environment() -> dict[str, str]:
    """Default engine environment for serving. Use launch --set to override a value."""
    on = """STRATA_MMQ_KQ STRATA_HY4_FLASH_INDEXER STRATA_HY4_FLASH_DENSE_ARENA STRATA_HY4_FLASH_DEVICE_EPOCH
        STRATA_HY4_FLASH_COMPACT_PAIR_GU STRATA_HY4_FLASH_RESIDENT_PAIR_GU STRATA_HY4_FLASH_PIN_EXPERT_MMAP
        STRATA_HY4_FLASH_GRAPHS STRATA_HY4_FLASH_TOKEN_GRAPH STRATA_HY4_FLASH_ASYNC_MOE STRATA_HY4_FLASH_HOT_CACHE
        STRATA_HY4_FLASH_ASYNC_COMPACT_MISS STRATA_HY4_FLASH_ASYNC_DEVICE_PLAN STRATA_HY4_FLASH_ASYNC_ROUTER_PUBLISH
        STRATA_HY4_FLASH_ASYNC_SHARED_STREAM STRATA_HY4_FLASH_CPU_CACHE_FIRST STRATA_HY4_FLASH_CPU_DIRECT_PARTS
        STRATA_CPU_Q81_VECTOR_QUANT STRATA_CPU_Q81_PAIR_GU STRATA_HY4_FLASH_PREFILL_PIPELINE
        STRATA_HY4_FLASH_PREFILL_RESIDENT_MMQ STRATA_HY4_FLASH_PREFILL_ROW_PLAN STRATA_HY4_FLASH_PREFILL_ROW_SCATTER
        STRATA_HY4_FLASH_PREFILL_QUANT_REUSE STRATA_HY4_FLASH_PREFILL_HOST_ASYNC STRATA_HY4_FLASH_PREFILL_DIRECT_DMA
        STRATA_HY4_FLASH_CPU_NATIVE_Q8K STRATA_HY4_FLASH_PREFILL_LAYER_MAJOR"""
    off = """STRATA_HY4_FLASH_ASYNC_MAP_SYNC_ONCE STRATA_HY4_FLASH_ASYNC_SHARED_EARLY STRATA_HY4_FLASH_ASYNC_SINGLE_PHASE
        HY4_FLASH_ASYNC_PARALLEL_PIPELINE HY4_FLASH_ASYNC_STAGE_PIPELINE
        STRATA_HY4_FLASH_HOT_CPU_COLD_FIRST STRATA_CPU_Q81_PAIR_VNNI STRATA_HY4_FLASH_PREFILL_BORROW"""
    env = {k: "1" for k in on.split()}
    env.update({k: "0" for k in off.split()})
    # Every routed slot may go to the CPU; the auto split decides how many actually do.
    env.update(STRATA_HY4_FLASH_CPU_SPLIT="auto", STRATA_HY4_FLASH_CPU_MISS_MAX="16", STRATA_HY4_FLASH_CPU_WORKERS="32",
               STRATA_HY4_FLASH_PREFILL_OFFLOAD_MMQ="64", STRATA_HY4_FLASH_PREFILL_GROUP_ROWS="1024",
               STRATA_HY4_FLASH_PREFILL_CHUNK="1024", STRATA_HY4_FLASH_PREFILL_SUPERCHUNK="8192", HY4_FLASH_ASYNC_STAGE_MAX_MISSES="2")
    # PDL is an H20 tuning choice, not required for serving; the engine default remains
    # appropriate on other CUDA architectures. Users may enable it with --set.
    return env


def serving_environment(mtp: bool) -> dict[str, str]:
    env = environment()
    if mtp:
        env.update(STRATA_HY4_FLASH_VERIFY_DIRECT_DMA="1", STRATA_HY4_FLASH_VERIFY_HEADS="1",
                   STRATA_HY4_FLASH_VERIFY_DECODE_EQUIVALENT="1", STRATA_HY4_FLASH_VERIFY_BATCH_DENSE="1")
    return env


#: Prefill positions a chunk covers on Apple silicon. A multiple of the indexer's kpool, which the engine
#: re-checks: a chunk ending inside a group leaves that group to be pooled by the next one.
METAL_PREFILL_CHUNK = 4096


def _metal_config(s, src: Path, model, pack, mtp: bool) -> dict:
    """The Metal engine's command line. None of the CUDA serving environment applies here - those switches
    belong to src/hy4flash_gguf and this engine rejects the ones that would change its arithmetic - so only what a
    person passed with --set is forwarded.

    --expert-vram and --resident-experts are left out: this engine accepts and ignores them, and an argument
    that reads as a budget while having no effect is worse than its absence. --draft-tokens is left out too,
    so the engine's own default stands unless a person names one.
    """
    chunk = int(s.extra.get("HY4_FLASH_PREFILL_CHUNK", METAL_PREFILL_CHUNK))
    args = ["--native", str(model), "--pack", str(pack), "--context", str(s.context or 4096),
            "--prefill-chunk", str(chunk), "--serve"]
    # terminal/serve/server.py selects Hy4FlashPictures on `--vision-shard` being in these args, so this is also what
    # turns pictures on for the server side.
    if s.vision == "on":
        args += ["--vision-shard", str(vision_shard(s))]
    if mtp:
        args.append("--mtp")
        if s.draft_tokens is not None:
            args += ["--draft-tokens", str(s.draft_tokens)]
    env = {k: str(v) for k, v in s.extra.items() if not k.startswith("--")}
    return {"exe": str(engine_path(s, src)), "cwd": str(src), "arch": "hy4-flash", "backend": "metal",
            "model_name": "hy4-flash-metal", "tokenizer": str(pack / "tokenizer"),
            "args": args, "env": env, "kv": s.kv or "fp16",
            "gpu": s.gpus or [0], "parallel": 1, "mtp": mtp, "mtp_resident_bytes": 0,
            "drafting": "mtp" if mtp else "off", "split_engine": "off"}


def config(s, src: Path) -> dict:
    model, pack = model_file(s.model_path), pack_path(s)
    missing = missing_pack(pack)
    if missing:
        raise ValueError(f"hy4-flash pack {pack} is incomplete: {', '.join(missing)}")
    mtp = s.mtp == "on"
    # The ENGINE decides this, not the platform.  `defaults()` takes the Metal path only when there is
    # no CUDA card as well, so keying this on sys.platform alone made the two disagree: planning a CUDA
    # target on a Mac got CUDA defaults and a Metal config.  A Metal build is the one that reads the
    # unified-memory arena, so its own marker is the discriminator - the same rule the vision gate uses.
    if "metal" in runtime_capabilities(str(engine_path(s, src))):
        return _metal_config(s, src, model, pack, mtp)
    draft_bytes = mtp_weight_bytes(model, host_experts=mtp_host_experts(s)) if mtp else 0
    env = serving_environment(mtp or bool(s.dfly_model))
    exe = str(engine_path(s, src))
    vision_args = ["--vision-shard", str(vision_shard(s))] if s.vision == "on" else []
    return {"exe": exe, "cwd": str(src), "arch": "hy4-flash", "backend": "cuda",
            "model_name": "hy4-flash", "tokenizer": str(pack / "tokenizer"),
            "args": ["--native", str(model), "--pack", str(pack), "--context", str(s.context or 4096),
                     "--expert-vram", str(s.expert_vram)] +
                    vision_args +
                    (["--resident-experts"] if s.resident_experts else []) +
                    (["--mtp", "--draft-tokens", str(s.draft_tokens or 3)] if mtp else []) +
                    (["--draft", "dfly", "--draft-model", s.dfly_model, "--draft-max", str(s.draft_tokens)] if s.dfly_model else []),
            "env": {**env, "HY4_FLASH_DIRECT_CONTEXT": str(s.context or 4096),
                    "STRATA_HY4_FLASH_KV": KV_FORMATS[s.kv or "fp16"]},
            "kv": s.kv or "fp16",
            "gpu": s.gpus or [0], "parallel": 1, "mtp": mtp, "mtp_resident_bytes": draft_bytes,
            "drafting": "dfly" if s.dfly_model else "mtp" if mtp else "off", "split_engine": "off"}
