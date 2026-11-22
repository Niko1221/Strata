"""Inkling-2-Mini (Thinking Machines' checkpoint as published on Hugging Face): its files (exact sizes and SHA-256 at a pinned revision), its engine
program (strata-ink2, built from engine/src/inkling2) and the rule that turns this PC's card, RAM and CPU into the engine's arguments.  Pure data and
small functions; setup.py's install_inkling2 does the downloads, the build and the questions.

The pins: the four LFS files' SHA-256 are Hugging Face's own (the repository tree's `lfs.oid`); the small files' SHA-256 were computed from copies whose
git blob id matched the tree's `oid` at REVISION, so they are that revision's bytes.  README.md, .gitattributes and tiktoken/tokenizer.model are not
downloaded: neither the engine nor the tokenizer pack reads them.
"""

from __future__ import annotations

FAMILY = "inkling2"
TAG = "inkling2-"                      # run config file: strata-inkling2-mini.json
ARCH = "inkling2"                      # the run config's "arch" (serve: the family's sampling defaults)
MODEL = "mini"                         # the one size
NAME = "inkling-2-mini"                # the run config's model_name
EXE = "strata-ink2"                    # Linux only: the engine includes sys/mman.h and unistd.h
TITLE = "Inkling-2-Mini"
BY = "Thinking Machines Lab (the official checkpoint)"
ABOUT = ("a 128-expert mixture of experts with NVFP4 / MXFP8 experts, read as published (19.5 GB); "
         "NVIDIA cards on Linux, the experts that do not fit the card stay in RAM")
LICENSE = "Apache 2.0: https://huggingface.co/thinkingmachines/Inkling-2-Mini"
REPO = "thinkingmachines/Inkling-2-Mini"
REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"   # 2026-11-20 ("Declare the dtype at the top level of config.json")
#: Pictures: off until the engine reads them (another change adds image input).  The hook: set this True and add "processor_config.json"'s reader to
#: the run config the way install_own_path writes cfg["vision"]; the file is already in the download set.
VISION = False

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
#: bytes of one full-attention visit's K and V per token in each --kv format (the engine's kv_cell_bytes): f16 2 x 1024 x 2; k8v8 int8 K and V with
#: an fp16 scale per 64 values (2 x (1024 + 32)).  k8v8: mean KL 1.24e-3, top-1 99.48% against an f32 K/V run (the engine commit that added it)
KV_CELL_BYTES = {"f16": 4096, "k8v8": 2112}


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
RAM_GB = 32

#: The expert offload (--vram-reserve-mib) runs with --kv-grow 1: the full-attention K/V is mapped in 8192-cell steps as a conversation grows and
#: the prompt buffers only while a prompt is read, both taken from the expert pools and given back after.  When a step would leave a pool under its
#: minimum the engine stops (503 to the client, restarted by the next request), so the context setup writes is the one whose deepest point - the
#: whole K/V and the prompt buffers - still leaves the pools their minimum.  Measured on an RTX 3090 set up as an 8 GB card (the engine's
#: "offload:" and "kv-grow: N cells, expert slots a + b" lines): at rest it holds GROW_BASE_MIB beside the pools and the K/V; reading a prompt maps
#: PROMPT_BUF_MIB more.  Checked: f16 / --chunk 1024 stops at the first step (16K), f16 / 256 at 32K, k8v8 / 256 reads a 49K-token prompt at
#: exactly 64 MXFP8 slots.  --chunk 256 reads prompts about 3x slower than 1024 (a 30K-token prompt: 180 s against 57 s), so it is used only where
#: 1024 would not reach 32K.  CBA: one card measured; another card's CUDA context moves it by ~100 MiB
GROW_BASE_MIB = {1024: 3451, 256: 3523}
PROMPT_BUF_MIB = {1024: 1000, 256: 524}
#: the offload's choices, in order: the first that reaches TARGET_CONTEXT is used (f16 K/V and big prompt chunks first)
OFFLOAD_PLANS = [("f16", 1024), ("k8v8", 1024), ("k8v8", 256)]
TARGET_CONTEXT = 32768                 # agentic coding needs 30-60K
OFFLOAD_MARGIN_MIB = 128               # the engine's INK2_OFF_MARGIN_MIB, kept beside the reserve
#: the least room the pools may have: the engine refuses to start with fewer than 64 slots in either pool (FETCH_MAX), the MXFP8 pool gets 20% of
#: the room and an MXFP8 expert is 8211456 bytes: 64 x 8211456 / 0.2 bytes = 2506 MiB, rounded up for the 2-decimal GiB of the measurement
MIN_POOLS_MIB = 2560
#: kept beside the pools' minimum when setup sizes the offload: the measurement is one card's (another card's CUDA context differs by ~100 MiB) and
#: a desktop's VRAM grows after setup looked at it; with less the engine can stop at start ("need >= 64" slots)
OFFLOAD_SLACK_MIB = 256


def resident_need_mib(ctx: int) -> int:
    """VRAM (MiB) the resident mode (every expert on the card, the drafter on) takes at this context."""
    return RESIDENT_FIXED_MIB + -(-ctx * kv_bytes_per_token(True) // 2**20)


def resident_fits(vram_mib: float, ctx: int, reserve_mib: int = 0) -> bool:
    """Every expert, the drafter and the K/V of `ctx` fit on a card of `vram_mib` with `reserve_mib` kept free for other programs."""
    return resident_need_mib(ctx) + RESIDENT_MARGIN_MIB + reserve_mib <= vram_mib


def offload_room_mib(vram_mib: float, ctx: int, used_mib: float = 0, plan: tuple = OFFLOAD_PLANS[0]) -> int:
    """The VRAM a reserve can take in the offload mode at its deepest point (the whole K/V of `ctx` and a prompt's buffers mapped), with the
    expert pools at their minimum and the slack kept; below 0 the engine may stop at this context.  `used_mib`: what other programs (a desktop)
    hold on the card now: the engine sizes its pools from the FREE VRAM and keeps the reserve on top of that.  `plan`: (K/V format, --chunk)."""
    kv, chunk = plan
    return (int(vram_mib - used_mib) - GROW_BASE_MIB[chunk] - -(-ctx * kv_bytes_per_token(False, kv) // 2**20) - PROMPT_BUF_MIB[chunk]
            - OFFLOAD_MARGIN_MIB - MIN_POOLS_MIB - OFFLOAD_SLACK_MIB)


def offload_plan(vram_mib: float, reserve_mib: int = 0, used_mib: float = 0, ctx: int = TARGET_CONTEXT) -> tuple:
    """(K/V format, --chunk) of the offload at this context: the first plan that leaves the reserve its room; else the first that fits at all
    (setup then cuts the reserve to its room); else the last (the smallest)."""
    return (next((p for p in OFFLOAD_PLANS if offload_room_mib(vram_mib, ctx, used_mib, p) >= reserve_mib), None)
            or next((p for p in OFFLOAD_PLANS if offload_room_mib(vram_mib, ctx, used_mib, p) >= 0), OFFLOAD_PLANS[-1]))


def recommended_context(vram_mib: float, reserve_mib: int = 0, used_mib: float = 0) -> int:
    """The largest context up to 128K that runs with every expert on the card (the reserve covers what a desktop holds there).  When none does,
    the experts share the card: the largest context up to 128K whose deepest point leaves the pools their room beside the reserve (offload_plan's
    K/V format and chunk).  A reserve that would hold the context under TARGET_CONTEXT gives way: TARGET_CONTEXT (or the largest that fits)
    with whatever room is left for the reserve (setup cuts it and says so); 8K at the least."""
    fits = [c for c in CONTEXTS if DEFAULT_CONTEXT <= c <= 131072 and resident_fits(vram_mib, c, reserve_mib)]
    if fits:
        return max(fits)
    def best(reserve):
        plan = offload_plan(vram_mib, reserve, used_mib)
        return max([c for c in CONTEXTS if c <= 131072 and offload_room_mib(vram_mib, c, used_mib, plan) >= reserve], default=8192)
    ctx = best(reserve_mib)
    return ctx if ctx >= TARGET_CONTEXT or not reserve_mib else max(ctx, min(best(0), TARGET_CONTEXT))


def engine_args(model_dir: str, ctx: int, vram_mib: float, reserve_mib: int, cpu_threads: int | None, used_mib: float = 0) -> list[str]:
    """The engine's arguments.  Every expert fits on the card: resident, with the drafter (--spec 1).  Otherwise the expert offload with --kv-grow 1:
    the card keeps `reserve_mib` free and fills the rest with the most used experts, the others stay in RAM; `cpu_threads` (per NUMA node; None: no
    CPU experts) computes a token's missing experts on the CPU instead of copying them over PCIe; the K/V format and the prompt chunk are
    offload_plan's.  The engine refuses --cpu-threads with --spec 1, so the offload mode has no drafter.  CBA: offload without CPU experts (no AVX2)
    also goes without the drafter, unmeasured either way"""
    if resident_fits(vram_mib, ctx, reserve_mib):
        return ["--model", model_dir, "--max-context", str(ctx), "--chunk", str(CHUNK), "--kv", "f16", "--tc", "1", "--spec", "1"]
    kv, chunk = offload_plan(vram_mib, reserve_mib, used_mib, ctx)
    args = ["--model", model_dir, "--max-context", str(ctx), "--chunk", str(chunk), "--kv", kv, "--tc", "1", "--kv-grow", "1",
            "--vram-reserve-mib", str(int(reserve_mib))]
    if cpu_threads:
        args += ["--cpu-threads", str(int(cpu_threads))]
    return args
