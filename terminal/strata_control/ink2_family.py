"""Inkling-2-Mini (Thinking Machines' checkpoint as published on Hugging Face): its files (exact sizes and SHA-256 at a pinned revision), its engine
program (strata-ink2, built from engine/src/inkling2) and the rule that turns this PC's card, RAM and CPU into the engine's arguments.  Pure data and
small functions; setup.py's install_ink2 does the downloads, the build and the questions.  In setup it is one source ("model") of the
Inkling family (setup.py INKLING_FAMILIES, whose SQ builds run upstream's strata-inkling).

The pins: the four LFS files' SHA-256 are Hugging Face's own (the repository tree's `lfs.oid`); the small files' SHA-256 were computed from copies whose
git blob id matched the tree's `oid` at REVISION, so they are that revision's bytes.  README.md, .gitattributes and tiktoken/tokenizer.model are not
downloaded: neither the engine nor the tokenizer pack reads them.
"""

from __future__ import annotations

FAMILY = "inkling2"                    # setup.sh --family inkling2: --family inkling-2-mini --model official
ARCH = "inkling2"                      # the run config's "arch" (serve: the family's sampling defaults); never strata-inkling's inkling2_loop
MODEL = "official"                     # its "model" in the Inkling family; run config file: strata-inkling-2-mini-official.json
# the run config's model_name (the API's model id): not upstream's "inkling-2-mini" prefix, which strata launch's agent harness, tune and pictures
# take for strata-inkling (thinking off, --parallel, the Vision shard)
NAME = "inkling2-mini"
EXE = "strata-ink2"                    # Linux only: the engine includes sys/mman.h and unistd.h
ABOUT = ("a 128-expert mixture of experts with NVFP4 / MXFP8 experts, read as published (19.5 GB); "
         "NVIDIA (CUDA) and AMD (ROCm/HIP) cards on Linux, the experts that do not fit the card stay in RAM")
REPO = "thinkingmachines/Inkling-2-Mini"
REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"   # 2026-11-20 ("Declare the dtype at the top level of config.json")
#: Pictures: strata-ink2 reads them itself (its ViT, ink2_vision.cuh) and says INFO images=1 at its start, which is what the server enables image
#: input from (Service.engine_reads_images): nothing to write in the run config.  The engine takes no file for it beyond the checkpoint (its
#: preprocessing constants are processor_config.json's, built in); the file stays in the download set as part of the published checkpoint
VISION = True
#: the ViT weights, pinned in host memory at the first picture (634 MB BF16 + 6 MB fp32)
VISION_RAM_GIB = 0.65

#: file name -> (bytes, sha256), in download order (the big ones first: a disk that fills up says so early)
FILES = {
    "model-00001-of-00001.safetensors": (19_526_019_168, "65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1"),
    "mtp.safetensors": (635_801_416, "7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2"),
    "tokenizer.json": (26_178_678, "7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6"),
    "model.safetensors.index.json": (109_300, "c1631065b7b00787993522dddc66a68b3abee869192977fb3dcb796e63bc5838"),
    "hf_quant_config.json": (67_924, "7c656207d3ce37879c7be209a036b0dde43bf01f78159bff06880b868ed57143"),
    "tokenizer_config.json": (13_235, "bf09342d285139fd37ce10c70878af1d52cb390e9901e10248654293a80ee636"),
    "chat_template.jinja": (6_868, "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc"),
    "config.json": (2_722, "a7a349601f0984c4abf999c480ef775d8567d21f5f84b4e8a3e035b44f288dc2"),
    "special_tokens_map.json": (529, "c16f44b3594563d82743ddb0417a2be1bfe3e2000dfad2da4b8f32554593e1a0"),
    "processor_config.json": (461, "d7350adce18c8be4b607f261c8f0f7eabbf720bb7aaf648a43d4f8372bbacc2d"),
    "generation_config.json": (98, "a08737c07a65a8fdf69f7cf086576210e5aad4b241437090b26d7b0c4f7905f7"),
}
DOWNLOAD_GB = sum(s for s, _ in FILES.values()) / 1e9
#: the tokenizer pack (tools/strata_tokenizer.py extract_hf): the o200k split rule
PRE_TOKENIZER = "gpt-4o"

# ---- the engine's memory (engine/src/inkling2/ink2_engine.cu)
#: K/V per context token, fp16 (--kv f16): a full-attention layer (L % 6 == 5: 5, 11, 17, 23) keeps the whole context, the others a 1024-token
#: window whose size does not grow with it.  Layers 6-17 run three times, each pass with its own K/V, so the full-attention visits are 5, 11, 17, 23
#: + 11, 17 twice more = 8; each holds K and V of 8 heads x 128 at 2 bytes.  --spec 1 adds the drafter's one global layer (mtp.safetensors: 3 layers,
#: 0 and 2 local).  Measured on an RTX 3090 (nvidia-smi, --spec 1, --chunk 1024): 8K -> 32K context +864 MiB = 36864 bytes a token.
KV_HEADS, HEAD_DIM, KV_FULL_VISITS, DRAFT_FULL_LAYERS = 8, 128, 8, 1
#: bytes of one full-attention visit's K and V per token in each --kv format (the engine's kv_cell_bytes): f16 2 x 1024 x 2; i8r, i8 and k8v8 all
#: int8 K and V with an fp16 scale per 64 values, 2 x (1024 + 32).  Of those three, i8r is the closest to f32 K/V (KL_VS_F32 below)
KV_CELL_BYTES = {"f16": 4096, "i8r": 2112}
#: KL(f32 K/V || format) per position, the routing forced to one f32 recording, served flags (--tc 1 --chunk 1024): (mean, p99, top-1 %).
#: p1600: 1600 tokens of prose; c23k: 23,265 tokens of mixed prose, French, C++ and Python.  f16 matched f32 bit for bit on both
KL_VS_F32 = {"p1600": {"i8": (1.00e-4, 1.59e-3, 99.50), "i8r": (2.04e-5, 2.41e-4, 99.88), "k8v8": (3.39e-5, 4.66e-4, 99.94)},
             "c23k": {"i8": (1.32e-4, 1.69e-3, 99.61), "i8r": (3.52e-5, 4.09e-4, 99.80), "k8v8": (5.34e-5, 6.41e-4, 99.73)}}


def kv_bytes_per_token(spec: bool, kv: str = "f16") -> int:
    return (KV_FULL_VISITS + (DRAFT_FULL_LAYERS if spec else 0)) * KV_CELL_BYTES[kv] if kv != "f16" else \
        (KV_FULL_VISITS + (DRAFT_FULL_LAYERS if spec else 0)) * 2 * KV_HEADS * HEAD_DIM * 2


#: VRAM with every expert on the card (nvidia-smi memory.used, CUDA context included) less the K/V: --spec 1, --chunk 1024, --kv f16, after a
#: 1.5K-token request (RTX 3090: 19720 MiB at 8K, 20584 at 32K, 21740 at 64K context).
RESIDENT_FIXED_MIB = 19436
#: kept free beside it so a longer prompt or the driver does not end the load in an out-of-memory error.  CBA: one card measured (RTX 3090); a
#: card with another CUDA context size moves this by ~100 MiB, the margin covers it
RESIDENT_MARGIN_MIB = 1024
#: --chunk: prompt tokens per step (the measured default of the hand-tuned config)
CHUNK = 1024
#: the contexts offered (the model was trained to 1M; past 256K is untested here)
CONTEXTS = [8192, 16384, 32768, 49152, 65536, 131072, 262144]
DEFAULT_CONTEXT = 32768                # where nothing larger fits on the card with every expert
#: the RAM of the offload mode: every expert is kept in RAM (pinned, placed on the NUMA nodes) while the card holds the most used.  Measured: the
#: engine's peak resident memory (VmHWM) 16.8 GiB on an RTX 3090 set up as an 8 GB card; 32 GB leaves the OS, the server and the browser room
OFFLOAD_RAM_GIB = 16.8
RESIDENT_RAM_GIB = 3.2                 # the same peak with every expert on the card (24 GB, 64K context)
#: the OS, the server and a browser beside the engine.  CBA: one figure for every PC, not measured per system
OS_RAM_GIB = 6.0
RAM_GB = 32

#: The conversation cache (--conversation-cache-mib, the engine's default 0 = off): prompt-end checkpoints and parked conversations in pinned
#: host memory.  Without it a client that drops the reasoning from its history (every agent client does) re-reads its whole prompt every turn: the
#: rewind point it needs lives in that cache.  The largest of CACHE_MIB that leaves CACHE_SPARE_MIN_GIB of RAM beside the engine, the pictures'
#: weights and the OS; off below that.
CACHE_MIB = (4096, 2048)
CACHE_SPARE_MIN_GIB = 1.5


def conversation_cache_mib(ram_gib: float, offload: bool) -> tuple:
    """(MiB for --conversation-cache-mib or 0, the RAM left beside the engine, the pictures and the OS in GiB)."""
    spare = ram_gib - (OFFLOAD_RAM_GIB if offload else RESIDENT_RAM_GIB) - VISION_RAM_GIB - OS_RAM_GIB
    return next((m for m in CACHE_MIB if spare - m / 1024 >= CACHE_SPARE_MIN_GIB), 0), spare

#: The expert offload (--vram-reserve-mib) runs with --kv-grow 1: the full-attention K/V is mapped in 8192-cell steps as a conversation grows and
#: the prompt buffers only while a prompt is read, both taken from the expert pools and given back after.  The engine keeps each pool at its
#: 64-slot minimum: a prompt reads in smaller chunks (down to 256 tokens) when the buffers would not fit, and a prompt that does not fit even then
#: ends with "the context is full" (a 400 to the client, the engine keeps running).  So the context setup writes is the one whose deepest point -
#: the whole K/V and a 256-token chunk's buffers - still leaves the pools their minimum.  Measured on an RTX 3090 set up as an 8 GB card with a
#: 500 MiB desktop (the engine's "offload:" and "kv-grow: N cells, expert slots a + b" lines): at rest it holds GROW_BASE_MIB beside the pools
#: and the K/V (--chunk 1024); a prompt maps up to 1000 MiB more for 1024-token chunks, and PROMPT_BUF_MIN_MIB where pf_fit has shrunk the
#: chunk (the parts laid out for a full chunk stay mapped whole: the largest drop beside the K/V seen at the floor was 728 MiB).  Checked there
#: with i8r: a 30K-token prompt read in 83 s (chunks shrinking near the end; --chunk 256 throughout took 240 s), a 55K one refused at 40,960
#: tokens with the engine still up.  CBA: one card measured; another card's CUDA context moves it by ~100 MiB
GROW_BASE_MIB = 3451
PROMPT_BUF_MIN_MIB = 730
#: the offload's K/V formats, in order: the first that reaches TARGET_CONTEXT is used (f16 first: the reference)
OFFLOAD_KV = ["f16", "i8r"]
TARGET_CONTEXT = 32768                 # agentic coding needs 30-60K
OFFLOAD_MARGIN_MIB = 128               # the engine's INK2_OFF_MARGIN_MIB, kept beside the reserve
#: the least room the pools may have: the engine refuses to start with fewer than 64 slots in either pool (FETCH_MAX), the MXFP8 pool gets 20% of
#: the room and an MXFP8 expert is 8211456 bytes: 64 x 8211456 / 0.2 bytes = 2506 MiB, rounded up for the 2-decimal GiB of the measurement
MIN_POOLS_MIB = 2560
#: kept beside the pools' minimum when setup sizes the offload: the measurement is one card's (another card's CUDA context differs by ~100 MiB) and
#: a desktop's VRAM grows after setup looked at it; with less the engine can stop at start ("need >= 64" slots)
OFFLOAD_SLACK_MIB = 256


#: --spec 1 beside the expert offload (with CPU experts: a verify block's missing experts run on the CPU pools).  The drafter takes VRAM from the
#: expert pools: its weights (SPEC_DRAFTER_MIB), the prompt buffers, which stay mapped with it (SPEC_PROMPT_BUF_MIB), and its K/V layer for the whole
#: context, 4096 bytes a token in every --kv format, not grown.  Measured on the RTX 3090 (the engine's "offload:" line with and without --spec 1):
#: 1884 MiB at 256K without --kv-grow (the buffers mapped either way), 2120 MiB at 64K and 2376 MiB at 128K with it.
#: A verify block computes 8 tokens' experts and keeps 3 to 5 of them, so every missing expert costs more CPU time a token than in plain
#: decoding, and the smaller pools miss more.  SPEC_OFFLOAD_SPEEDS: per card shape, the share of the experts the pools hold with the drafter (%),
#: then decode tok/s plain and with --spec 1 on the 257-token chat prompt, plain and with --spec 1 on a 24-token prompt (the 3090 limited to the
#: card size, 2 x 22 threads, 512 greedy tokens at the start of the context, the mean of two runs; 16 GB: one run).  With most experts on the
#: card it does what it does with all of them (88.6 against 90.9 on the chat prompt, whose drafts are rarely kept, and well ahead where they
#: are); at 45% and less the chat prompt loses 10% and more.  So the drafter is on where the pools still hold SPEC_OFFLOAD_POOLS_MIB (70% of
#: the experts) beside it and the whole context's K/V.  CBA: one box measured (fewer or slower CPU threads move the break-even up); nothing
#: between 45% and 70%
SPEC_DRAFTER_MIB = 860
SPEC_PROMPT_BUF_MIB = 1004
SPEC_OFFLOAD_POOLS_MIB = 10500
SPEC_OFFLOAD_SPEEDS = {"8 GB, 32K": (19, 66.9, 51.8, 64.6, 59.9), "12 GB, 64K": (45, 80.6, 72.3, 81.5, 87.9),
                       "16 GB, 128K": (70, 84.1, 84.4, 85.7, 100.2), "24 GB, 128K": (98, 85.9, 83.9, 86.7, 103.1),
                       "24 GB, 256K, i8r, no --kv-grow": (92, 82.9, 84.7, 83.3, 100.0)}


def offload_spec(vram_mib: float, ctx: int, reserve_mib: int = 0, used_mib: float = 0, kv: str = "f16") -> bool:
    """The offload mode runs the drafter (--spec 1) too: the expert pools keep SPEC_OFFLOAD_POOLS_MIB beside it at the deepest point (the whole
    K/V of `ctx`)."""
    spec_mib = SPEC_DRAFTER_MIB + SPEC_PROMPT_BUF_MIB + -(-ctx * DRAFT_FULL_LAYERS * KV_CELL_BYTES["f16"] // 2**20)
    pools = int(vram_mib - used_mib) - reserve_mib - GROW_BASE_MIB - -(-ctx * kv_bytes_per_token(False, kv) // 2**20) - spec_mib
    return pools >= SPEC_OFFLOAD_POOLS_MIB


def resident_need_mib(ctx: int) -> int:
    """VRAM (MiB) the resident mode (every expert on the card, the drafter on) takes at this context."""
    return RESIDENT_FIXED_MIB + -(-ctx * kv_bytes_per_token(True) // 2**20)


def resident_fits(vram_mib: float, ctx: int, reserve_mib: int = 0) -> bool:
    """Every expert, the drafter and the K/V of `ctx` fit on a card of `vram_mib` with `reserve_mib` kept free for other programs."""
    return resident_need_mib(ctx) + RESIDENT_MARGIN_MIB + reserve_mib <= vram_mib


def offload_room_mib(vram_mib: float, ctx: int, used_mib: float = 0, kv: str = "f16") -> int:
    """The VRAM a reserve can take in the offload mode at its deepest point (the whole K/V of `ctx` in format `kv` and a 256-token chunk's
    buffers), with the expert pools at their minimum and the slack kept; below 0 a prompt that long ends with "the context is full".
    `used_mib`: what other programs (a desktop) hold on the card now: the engine sizes its pools from the FREE VRAM and keeps the reserve on top
    of that."""
    return (int(vram_mib - used_mib) - GROW_BASE_MIB - -(-ctx * kv_bytes_per_token(False, kv) // 2**20) - PROMPT_BUF_MIN_MIB
            - OFFLOAD_MARGIN_MIB - MIN_POOLS_MIB - OFFLOAD_SLACK_MIB)


def offload_kv(vram_mib: float, reserve_mib: int = 0, used_mib: float = 0, ctx: int = TARGET_CONTEXT) -> str:
    """The offload's K/V format at this context: the first that leaves the reserve its room; else the first that fits at all (setup then cuts
    the reserve to its room); else the last (the smallest)."""
    return (next((k for k in OFFLOAD_KV if offload_room_mib(vram_mib, ctx, used_mib, k) >= reserve_mib), None)
            or next((k for k in OFFLOAD_KV if offload_room_mib(vram_mib, ctx, used_mib, k) >= 0), OFFLOAD_KV[-1]))


def recommended_context(vram_mib: float, reserve_mib: int = 0, used_mib: float = 0) -> int:
    """The largest context up to 128K that runs with every expert on the card (the reserve covers what a desktop holds there).  When none does,
    the experts share the card: the largest context up to 128K whose deepest point leaves the pools their room beside the reserve (offload_kv's
    format).  A reserve that would hold the context under TARGET_CONTEXT gives way: TARGET_CONTEXT (or the largest that fits) with whatever room
    is left for the reserve (setup cuts it and says so); 8K at the least."""
    fits = [c for c in CONTEXTS if DEFAULT_CONTEXT <= c <= 131072 and resident_fits(vram_mib, c, reserve_mib)]
    if fits:
        return max(fits)
    def best(reserve):
        kv = offload_kv(vram_mib, reserve, used_mib)
        return max([c for c in CONTEXTS if c <= 131072 and offload_room_mib(vram_mib, c, used_mib, kv) >= reserve], default=8192)
    ctx = best(reserve_mib)
    return ctx if ctx >= TARGET_CONTEXT or not reserve_mib else max(ctx, min(best(0), TARGET_CONTEXT))


def engine_args(model_dir: str, ctx: int, vram_mib: float, reserve_mib: int, cpu_threads: int | None, used_mib: float = 0,
                cache_mib: int = 0) -> list[str]:
    """The engine's arguments.  Every expert fits on the card: resident, with the drafter (--spec 1).  Otherwise the expert offload with --kv-grow 1:
    the card keeps `reserve_mib` free and fills the rest with the most used experts, the others stay in RAM; `cpu_threads` (per NUMA node; None: no
    CPU experts) computes a token's missing experts on the CPU instead of copying them over PCIe; the K/V format is offload_kv's.  With CPU
    experts the drafter stays on where the pools keep most of the experts beside it (offload_spec).  CBA: offload without CPU experts (no AVX2)
    goes without the drafter: its verify blocks would copy their missing experts over PCIe (measured at 8 GB: 16 tok/s)"""
    cache = ["--conversation-cache-mib", str(int(cache_mib))] if cache_mib else []   # (conversation_cache_mib)
    if resident_fits(vram_mib, ctx, reserve_mib):
        return ["--model", model_dir, "--max-context", str(ctx), "--chunk", str(CHUNK), "--kv", "f16", "--tc", "1", "--spec", "1", *cache]
    args = ["--model", model_dir, "--max-context", str(ctx), "--chunk", str(CHUNK), "--kv", offload_kv(vram_mib, reserve_mib, used_mib, ctx),
            "--tc", "1", "--kv-grow", "1", "--vram-reserve-mib", str(int(reserve_mib))]
    if cpu_threads:
        args += ["--cpu-threads", str(int(cpu_threads))]
        if offload_spec(vram_mib, ctx, reserve_mib, used_mib, args[args.index("--kv") + 1]):
            args += ["--spec", "1"]
    return args + cache
