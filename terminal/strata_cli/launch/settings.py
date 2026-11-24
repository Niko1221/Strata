"""What a launch is made of, as plain data: the choices of the setup screens, the rules that turn them into warnings, and the
flags that say the same thing without the TUI.

No Textual, no network, standard library only: the screens, the flags and the saved presets all use this one description,
so `strata launch --preset X` and clicking through the screens cannot drift apart.

Rules of this file: recommend, never force. A choice that does not fit this computer is a warning with the reason; only
a value that cannot be run at all (a port that is not a number, an unknown access kind) is an error.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any

SCHEMA = "strata.launch/1"

#: setup.py's CONTEXTS (kept here only as the fallback when the installer files are not next to this terminal).
FALLBACK_CONTEXTS = (8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288)

ACCESS_KINDS = ("local", "lan", "address", "cloudflare", "tailscale")
#: Where the picture reader runs, in the words the plan, the summary and the running-server line all use (#0221 APPTUI: the plan said "the reader runs on the processor", the running line "pictures gpu").
PICTURE_WORDS = {"off": "off", "on": "on", "cpu": "on (the picture reader runs on the processor)", "gpu": "on (the picture reader runs on the graphics card)"}
PICTURE_SHORT = {"off": "off", "on": "on", "cpu": "on, read by the processor", "gpu": "on, read by the graphics card"}   # the same, for inside a bracket
ACCESS_WORDS = {"local": "This computer only", "lan": "My local network", "address": "One address only", "cloudflare": "Cloudflare tunnel",
                "tailscale": "Tailscale"}
TUNE_MODES = ("quick", "thorough", "skip")
VISION_MODES = ("off", "on", "gpu", "cpu")
MTP_MODES = ("on", "off")
#: strata_control/split_engine.py MODES: the all-cards split engine on NVIDIA and AMD gfx1201 cards (auto = by itself where it is supported, the default)
SPLIT_MODES = ("auto", "on", "off")
#: strata_control/drafting.py MODES (tests/terminal/test_drafting.py keeps the two the same)
DRAFTING_MODES = ("auto", "mtp", "dflash2", "off")
#: setup.py DRAFT_VOCABS (tests/terminal/test_launch_draft_vocab.py keeps the two the same)
DRAFT_VOCAB_MODES = ("cjk", "en", "cyrillic", "fr", "latin", "arabic")
KV_MODES = ("int8", "q4_0", "k8v4", "fp16", "fp8", "k8v8")
INKLING_KV_MODES_CUDA = ("fp16", "fp8", "k8v8", "k8v4")
INKLING_KV_MODES_SAFE = ("fp16", "fp8")        # #6681: HIP/Metal/SYCL support fp16 and fp8 only (k8v4/k8v8 OOM on HIP)


def inkling_kv_modes(backend: str | None = None) -> tuple[str, ...]:
    """KV cache formats supported by strata-inkling for `backend` (fp16/fp8 on HIP/Metal/SYCL; fp16/fp8/k8v8/k8v4 on CUDA)."""
    return INKLING_KV_MODES_SAFE if str(backend or "").strip().lower() in ("hip", "metal", "sycl") else INKLING_KV_MODES_CUDA


def safe_inkling_kv(kv: str | None, backend: str | None = None) -> tuple[str | None, bool]:
    """Return (effective_kv, clamped) for Inkling-2-Mini: clamps k8v4/k8v8 -> fp8 on HIP/Metal/SYCL (#6681)."""
    if not kv:
        return None, False
    val = str(kv).strip().lower()
    allowed = inkling_kv_modes(backend)
    if val in allowed:
        return val, False
    if val in ("k8v4", "k8v8"):
        return "fp8", True
    return None, False


def inkling_kv_note(kv: str | None, backend: str | None = None) -> str | None:
    """The sentence for a `--kv` that Inkling-2-Mini does not take as it was typed: a Qwen format (int8, q4_0) or a format this card's backend lacks (k8v4 / k8v8
    on HIP, Metal and SYCL, which use fp8 instead). None when the value goes through as it is, or none was asked. Said at every launch, never dropped quietly
    (0.2.20 accepted `--kv k8v4` for an Inkling GGUF and wrote none: the 8-bit comparison it was meant for ran on the default)."""
    if not kv:
        return None
    val = str(kv).strip().lower()
    eff, clamped = safe_inkling_kv(val, backend)
    allowed = ", ".join(inkling_kv_modes(backend))
    if eff is None:
        return f"--kv {val} is not a memory format of Inkling-2-Mini (it takes {allowed}), so the engine's own default is used."
    if clamped:
        return f"Inkling-2-Mini on {backend or 'this card'} does not support --kv {val} (it takes {allowed}); using --kv {eff} instead."
    return None
DEFAULT_PORT = 8080

#: How many requests the card can answer at once (the spec's "fit to card" default): by graphics memory.
PARALLEL_BY_VRAM = ((32.0, 4), (24.0, 3), (20.0, 2), (0.0, 1))


def fit_parallel(vram_gb: float | None) -> int:
    """12 GB -> 1, 20 -> 2, 24 -> 3, 32 and more -> 4 (each step starts at the listed size)."""
    gb = float(vram_gb or 0.0)
    for floor, n in PARALLEL_BY_VRAM:
        if gb >= floor - 0.5:
            return n
    return 1


def parse_context(text: Any) -> int:
    """`128k`, `128K`, `131072` or `64 K` -> tokens. 1K is 1024 tokens (the engine's and setup's unit)."""
    raw = str(text).strip().lower().replace(" ", "").replace("_", "").replace(",", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(k|m)?", raw)
    if not m:
        raise ValueError(f"'{text}' is not a context size. Use a number of tokens or a size such as 32k, 64k, 128k, 256k.")
    value = float(m.group(1)) * {"k": 1024, "m": 1024 * 1024, None: 1}[m.group(2)]
    n = int(round(value))
    if n < 1024:
        raise ValueError(f"A context of {n} tokens is too small to be useful; the smallest size offered is 8k.")
    return n


def context_label(tokens: int | None) -> str:
    if not tokens:
        return "default"
    if tokens >= 1048576 and tokens % 1048576 == 0:        # 1,048,576 positions: "1M"
        return f"{tokens // 1048576}M"
    if tokens % 1024 == 0:
        return f"{tokens // 1024}K"
    if tokens >= 1000:
        return f"{tokens / 1000:g}K"
    return str(tokens)


def words_for(tokens: int) -> str:
    """About how many English words fit (0.75 words per token, rounded to two figures)."""
    words = tokens * 0.75
    mag = 10 ** max(0, len(str(int(words))) - 2)
    return f"about {int(round(words / mag) * mag):,} words"


@dataclass
class LaunchSettings:
    """Every choice of `strata launch`. `None` means "the recommendation for this computer"."""

    name: str | None = None                    # the saved preset's name
    family: str | None = None                  # catalog family id (qwen, swift, coder, unsloth, gyro, nvfp4...)
    model: str | None = None                   # catalog size (IQ3_XXS ...)
    model_path: str | None = None              # "Use model files I already have": a .gguf file or a folder
    data_dir: str | None = None                # where model files are stored (None: the Strata default)
    gpus: list[int] | None = None              # None = the engine's choice (all usable cards)
    gpu_roles: dict[str, str] | None = None    # {"1": "main", "0": "helper", "2": "off"}: a role for every card; None = the recommendation
    context: int | None = None
    kv: str | None = None                      # None = int8 above 8K, as setup
    vision: str | None = None                  # None = the default (on when the model can read pictures and the engine has the reader), off | on | gpu | cpu
    parallel: int | None = None                # None = fit to the card
    drafting: str | None = None                # None = Auto (recommended): the table in strata_control/drafting.py; mtp | dflash2 | off choose one. Saved in the launch config
    draft_vocab: str | None = None              # setup's token subset for the MTP/DFlash2 draft head
    split_engine: str | None = None            # None = auto (the all-cards split engine runs by itself where it is supported); on | off. Saved in the preset and the launch config
    mtp: str | None = None                     # None = as installed (draft layer on); "off" starts without it (less graphics memory and RAM, slower); "on" overrides an installed "off"
    tune: str = "skip"                         # quick | thorough | skip
    tune_allow_small: bool = False             # the "answer-changing" tick, off by default, as in the app
    access: str = "local"
    listen: str | None = None                  # access "address" (#5917): the one address of this computer the server listens on
    port: int = DEFAULT_PORT
    api_key: str | None = None                 # never shown whole; stored only in the private preset file
    no_key: bool = False                       # served beyond this computer without a key, on purpose (--no-api-key); a warning, never a refusal
    extra: dict[str, str] = field(default_factory=dict)    # Advanced: --set key=value (flags and STRATA_* variables)
    # filled in by the run, kept in the preset
    tuned_preset: str | None = None            # id of the tuning preset the tuner saved
    tuned_at: str | None = None
    last_used: str | None = None
    created: str | None = None

    @property
    def is_tuned(self) -> bool:
        """Tuning was done for this preset: it saved faster settings (tuned_preset), or it measured that the standard settings are already the
        fastest (only tuned_at is set). Either way the next start must not tune again."""
        return bool(self.tuned_preset or self.tuned_at)

    def to_dict(self, *, with_key: bool = True) -> dict[str, Any]:
        d = {"schema": SCHEMA, **asdict(self)}
        if not with_key:
            d["api_key"] = "set" if self.api_key else None
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LaunchSettings":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in (data or {}).items() if k in known}
        if isinstance(clean.get("extra"), dict):
            clean["extra"] = {str(k): str(v) for k, v in clean["extra"].items()}
        else:
            clean.pop("extra", None)
        return cls(**clean)

    def copy(self, **changes: Any) -> "LaunchSettings":
        d = asdict(self)
        d.update(changes)
        return LaunchSettings.from_dict(d)


# ----------------------------------------------------------------------------------------------------- checks

@dataclass
class Notice:
    level: str        # "warn" | "error"
    text: str         # plain words: what it means
    next: str = ""    # plain words: what to do
    after_write: bool = False   # true only once the thing it reports has really been written: the launch says it then (Runner.announce_key_written), not in the plan or before a refusal

    def line(self) -> str:
        return self.text + (f" {self.next}" if self.next else "")


def closest(word: str, options: list[str]) -> str | None:
    hit = difflib.get_close_matches(word.lower(), [o.lower() for o in options], n=1, cutoff=0.5)
    if not hit:
        return None
    return next(o for o in options if o.lower() == hit[0])


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Whether something answers on host:port (the check the launch makes before it starts: Runner.check_port_free)."""
    import socket
    try:
        with socket.create_connection((host, int(port)), timeout=0.5):
            return True
    except OSError:
        return False


def next_free_port(port: int, host: str = "127.0.0.1", tries: int = 50) -> int | None:
    """The first port after `port` that nothing answers on (and that a server could take), or None within `tries` ports (T11)."""
    import socket
    for p in range(int(port) + 1, min(65535, int(port) + tries) + 1):
        if port_in_use(p, host):
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((host if host not in ("", "0.0.0.0") else "127.0.0.1", p))
        except OSError:
            continue
        return p
    return None


def check_port(port: Any) -> Notice | None:
    try:
        p = int(port)
    except (TypeError, ValueError):
        return Notice("error", f"'{port}' is not a port number.", "Use a number from 1 to 65535, for example 8080.")
    if not 1 <= p <= 65535:
        return Notice("error", f"Port {p} does not exist.", "Use a number from 1 to 65535, for example 8080.")
    if p < 1024:
        return Notice("warn", f"Port {p} is below 1024; on many computers only an administrator can use it.",
                      "8080 works without special rights.")
    return None


NO_KEY_NETWORK = "No API key: anyone who can reach this port can use the model and its tools (they can read and change files on this computer)."
NO_KEY_PUBLIC = ("No API key on a public address: anyone on the internet who finds it can use the model and its tools "
                 "(they can read and change files on this computer).")


def check_access(s: LaunchSettings) -> list[Notice]:
    out: list[Notice] = []
    if s.access not in ACCESS_KINDS:
        near = closest(s.access, list(ACCESS_KINDS))
        out.append(Notice("error", f"'{s.access}' is not a way to share Strata.",
                          f"Choose one of: {', '.join(ACCESS_KINDS)}." + (f" Did you mean '{near}'?" if near else "")))
        return out
    port = check_port(s.port)
    if port:
        out.append(port)
    if s.access != "local" and not s.api_key:
        if s.access == "cloudflare":
            out.append(Notice("warn", NO_KEY_PUBLIC, "Leave --no-api-key out to get a key."))
        elif s.no_key:
            out.append(Notice("warn", NO_KEY_NETWORK, "Leave --no-api-key out to get a key."))
        else:
            out.append(Notice("warn", f"{ACCESS_WORDS[s.access]} without an API key lets anyone who can reach it use your computer's model.",
                              "Add an API key (Strata makes one for you in the Access step)."))
    return out


#: The most requests one engine answers at once, by model family (0221 APPTUI, from the engines: strata-inkling kMaxParallelSlots = 64; strata-hy4 has no slots;
#: the Strata engine's batch window holds 16 rows in the CUDA and HIP builds and 8 in the others (engine/include/strata/kernels/window_limits.hpp), and more
#: only as pipelined groups on a layer split, which a lone card does not have).
INKLING_MAX_PARALLEL = 64
QWEN_WINDOW = {"nvidia": 16, "amd": 16}
QWEN_WINDOW_OTHER = 8


def max_parallel(family: str | None, vendor: str | None = None) -> int:
    """The most slots the engine of this family really runs (one card)."""
    if family == "inkling-2-mini":
        return INKLING_MAX_PARALLEL
    if family == "hy4-flash":
        return 1
    return QWEN_WINDOW.get(str(vendor or "").lower(), QWEN_WINDOW_OTHER)


def inkling_slots_notices(s: LaunchSettings, *, vram_gb: float | None, ram_gb: float | None, vendor: str | None = None) -> list[Notice]:
    """Inkling-2-Mini: can the card hold N conversations (--parallel N) at the chosen context? The planner's own arithmetic (strata_control/inkling_fit.py) with N slots against
    one: an error when the slots' memory alone does not fit the card (the engine would stop at the start), a warning when the experts in graphics memory shrink."""
    out: list[Notice] = []
    n = int(s.parallel or 1)
    if n <= 1 or not vram_gb or not s.model:
        return out
    try:
        from strata_control import inkling_fit as F
        if s.model not in F.BUILDS:
            return out
        ctx = int(s.context or F.DEFAULT_CONTEXT)
        backend = {"amd": "hip", "intel": "sycl"}.get(str(vendor or "").lower())
        ram = float(ram_gb or 0.0)
        one = F.place(s.model, vram_gb, ram, False, ctx, False, parallel=1, backend=backend)
        many = F.place(s.model, vram_gb, ram, False, ctx, False, parallel=n, backend=backend)
        fp8 = F.auto_fp8_kv(vram_gb, ctx, n)
        dense = F.BUILDS[s.model]["dense_gb"]
        host_kv = (backend not in ("hip", "sycl")) and F.global_kv_gb(ctx, n, fp8_kv=fp8) > 1.5
        need = dense + F.non_expert_gb(ctx, n, dspark=True, fp8_kv=fp8, hybrid=True, host_offload_global_kv=host_kv)
        have = float(vram_gb) * F.GIB_TO_GB
    except Exception:                                         # noqa: BLE001 - a plan that cannot work this out stays silent rather than wrong
        return out
    ctx_k = f"{ctx // 1024}K"
    if need > have:
        out.append(Notice("error", f"{n} requests at once at {ctx_k} need about {need:.1f} GB of graphics memory for the conversations alone (before any expert); "
                                   f"this card has {have:.1f} GB, so the engine cannot start.",
                          f"Use fewer requests (--parallel), a shorter context (--context), or a card with more memory. {n} at {min(ctx, 8192) // 1024}K is a smaller ask."))
    elif many.get("vram_gb", 0.0) < one.get("vram_gb", 0.0) - 0.05 or many.get("ssd_gb", 0.0) > one.get("ssd_gb", 0.0) + 0.05:
        out.append(Notice("warn", f"{n} requests at once at {ctx_k} leave room for {many.get('vram_gb', 0.0):.1f} GB of the model's experts in graphics memory "
                                  f"(one request: {one.get('vram_gb', 0.0):.1f} GB); the rest runs from RAM" + (f" and the SSD ({many['ssd_gb']:.1f} GB)" if many.get("ssd_gb", 0.0) > 0.05 else "") + ".",
                          "Each answer is slower than with fewer requests. A shorter context or fewer requests keeps more experts on the card."))
    return out


def check_parallel(s: LaunchSettings, *, vram_gb: float | None, fitted: int | None = None, ram_gb: float | None = None, vendor: str | None = None) -> list[Notice]:
    out: list[Notice] = []
    if s.parallel is None:
        return out
    max_p = 64                                                # what any engine takes at all; the family's own limit is below
    if not 1 <= int(s.parallel) <= max_p:
        out.append(Notice("error", f"{s.parallel} requests at once is outside what the engine can do (1 to {max_p}).",
                          f"Pick a number from 1 to {max_p}."))
        return out
    if s.family == "inkling-2-mini":
        return inkling_slots_notices(s, vram_gb=vram_gb, ram_gb=ram_gb, vendor=vendor)
    if s.family == "hy4-flash":
        if int(s.parallel) > 1:
            out.append(Notice("warn", f"Hy4-Flash answers one request at a time, so --parallel {s.parallel} is not used.",
                              "Other requests wait their turn. Inkling-2-Mini runs up to 64 at once."))
        return out
    cap = max_parallel(s.family, vendor)
    if int(s.parallel) > cap:
        out.append(Notice("warn", f"{s.parallel} requests at once is more than this engine runs on one card: it keeps {cap} (its batch window holds {cap} requests).",
                          "The others wait their turn. Inkling-2-Mini runs up to 64 at once; two or more cards in a layer split can run more."))
        return out
    fitted = fit_parallel(vram_gb) if fitted is None else fitted
    if vram_gb and s.parallel > fitted:
        out.append(Notice("warn", f"{s.parallel} requests at once is more than a {vram_gb:.0f} GB card usually holds "
                                  f"(it fits {fitted}).",
                          "It can still start; the engine keeps less of the model in graphics memory, so each answer is slower."))
    return out


def check_context(s: LaunchSettings, *, max_fit_tokens: int | None = None) -> list[Notice]:
    out: list[Notice] = []
    if s.context is not None and s.context > 262144:
        out.append(Notice("warn", f"{context_label(s.context)} is more than the model was trained on (256K).",
                          "Setup adds rope scaling for it (experimental); answers may get less reliable."))
    if max_fit_tokens and s.context and s.context > max_fit_tokens:
        out.append(Notice("warn", f"{context_label(s.context)} is bigger than this computer's memory comfortably holds.",
                          f"{context_label(max_fit_tokens)} is the recommended size."))
    return out


# ----------------------------------------------------------------------------------------------------- advanced settings

_ENV_NAME = re.compile(r"^(?=[A-Za-z0-9_]*[A-Z])[A-Za-z_][A-Za-z0-9_]*$")     # any name with a capital letter: STRATA_X, IGC_EnableDPEmulation, ZES_ENABLE_SYSMAN
_FLAG_NAME = re.compile(r"^[a-z][a-z0-9-]*$")


def split_setting(text: str) -> tuple[str, str]:
    """`prefill=65536`, `--prefill=65536`, `STRATA_PIN_BY_NEED=1` -> (key, value); the key is a `--flag` or an ENV name."""
    key, sep, value = text.partition("=")
    key = key.strip()
    if not key:
        raise ValueError("an empty setting name")
    if _ENV_NAME.match(key):
        return key, value.strip()
    flag = key[2:] if key.startswith("--") else key
    flag = flag.replace("_", "-")
    if not _FLAG_NAME.match(flag):
        raise ValueError(f"'{text}' is not a setting name. Write it like prefill=65536 or STRATA_PIN_BY_NEED=1.")
    return "--" + flag, value.strip()


def settings_from_pairs(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs or []:
        k, v = split_setting(p)
        out[k] = v
    return out


#: Options whose value is a WORD the engine reads (on, off, auto, none): the word goes through as typed (none since DeepSeek's --dspark left, #3959).
WORD_VALUE_FLAGS: frozenset[str] = frozenset()


def as_argv_changes(extra: dict[str, str], bool_flags: frozenset[str],
                    word_flags: frozenset[str] = WORD_VALUE_FLAGS) -> tuple[dict[str, Any], dict[str, str | None]]:
    """Turn the Advanced table's values into `tune_knobs.apply_to_argv` changes and environment changes.

    A flag without a value (or on/true/yes) is "present"; off/false/no/none removes it; anything else is a value.
    A word-valued option (WORD_VALUE_FLAGS) keeps its word."""
    args: dict[str, Any] = {}
    env: dict[str, str | None] = {}
    for key, value in extra.items():
        if key.startswith("--"):
            low = value.lower()
            if key in word_flags and key not in bool_flags:
                if low:
                    args[key] = low
                continue                          # an empty value is no change, as for every option that takes one (#3504)
            if low == "" and key not in bool_flags:
                continue                          # an empty value for an option that takes one is no change, never a bare flag (#3504)
            if key in bool_flags or low in ("on", "true", "yes", ""):
                args[key] = None if low in ("off", "false", "no", "none") else True
            elif low in ("off", "false", "no", "none"):
                args[key] = None
            else:
                args[key] = value
        else:
            env[key] = None if value.lower() in ("off", "unset", "none") else value
    return args, env


def flags_text(s: LaunchSettings, *, preset: str | None = None) -> str:
    """The `strata launch ...` line that says the same as these choices (copyable)."""
    parts = ["strata launch"]
    if preset:
        parts.append(f'--preset "{preset}"')
    else:
        if s.model_path:
            parts.append(f'--model-path "{s.model_path}"')
        if s.model:                                   # own files the table recognised say which model they are, so the line stays exact
            parts.append(f"--model {s.model.lower()}" + (f" --family {s.family}" if s.family and s.family != "qwen" else ""))
        if s.data_dir:
            parts.append(f'--data-dir "{s.data_dir}"')
        if s.context:
            parts.append(f"--context {context_label(s.context).lower()}")
        if s.kv:
            parts.append(f"--kv {s.kv}")
        if s.vision:
            parts.append(f"--vision {s.vision}")
        if s.parallel:
            parts.append(f"--parallel {s.parallel}")
        if s.mtp:
            parts.append(f"--mtp {s.mtp}")
        if s.drafting:
            parts.append(f"--drafting {s.drafting}")
        if s.draft_vocab:
            parts.append(f"--draft-vocab {s.draft_vocab}")
        if s.split_engine:
            parts.append(f"--split-engine {s.split_engine}")
        parts.append(f"--tune {s.tune}")
        if s.tune_allow_small:
            parts.append("--tune-allow-small")
        parts.append(f"--listen {s.listen}" if s.access == "address" and s.listen else f"--access {s.access}")
        parts.append(f"--port {s.port}")
        if s.gpu_roles:
            parts.append("--gpu-roles " + ",".join(f"{k}={v}" for k, v in sorted(s.gpu_roles.items(), key=lambda kv: int(kv[0]))))
        elif s.gpus:
            parts.append("--gpu " + ",".join(str(g) for g in s.gpus))
    for k, v in sorted(s.extra.items()):
        parts.append(f"--set {k[2:] if k.startswith('--') else k}={v}" if v else f"--set {k[2:] if k.startswith('--') else k}")
    return " ".join(parts)


def slug(name: str) -> str:
    """A name safe for profile ids and file names: lowercase letters, digits, - and _, starting with a letter."""
    text = re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9_-]+", "-", name.lower())).strip("-_") or "launch"
    if not text[0].isalpha():
        text = "l" + text
    return text[:48]
