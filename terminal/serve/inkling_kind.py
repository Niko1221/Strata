"""The Inkling-2-Mini engine as serve sees it: its own shape engine program (`strata-inkling`, component `strata-inkling`),
with native DSpark speculative drafting, on-demand in-process Vision, `--gpus 2` dual-GPU EP, and `1..64` continuous-batching parallelism (`--parallel`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ARCH = "inkling2_loop"
EXE_PREFIX = "strata-inkling"
REFERENCE = "component:strata-inkling"
END_TEXTS = ("<|content_model_end_sampling|>", "<|endoftext|>")
IMAGE_PAD_TOKEN = "<|unused_200054|>"
REFUSED = {
    "logprobs": "logprobs: not supported by the Inkling-2-Mini engine",
    "logit_bias": "logit_bias: not supported by the Inkling-2-Mini engine",
}


def _base(exe) -> str:
    return str(exe or "").replace("\\", "/").rsplit("/", 1)[-1].lower()


def is_inkling(cfg: dict | None) -> bool:
    if not isinstance(cfg, dict):
        return False
    if cfg.get("arch") == ARCH:
        return True
    exe = str(cfg.get("exe") or "")
    return exe == REFERENCE or _base(exe).startswith(EXE_PREFIX)


def is_inkling_exe(exe) -> bool:
    return _base(exe).startswith(EXE_PREFIX)


NO_DRAFTER_NOTE = "Inkling-2-Mini: this engine has no DSpark drafter yet, so it starts without it"
NO_IMAGES_NOTE = "Inkling-2-Mini: this engine does not read images yet, so it starts without the image reader"
NO_IMAGES_REPLY = "this Inkling-2-Mini engine does not read images yet; update Strata or attach the Vision shard"
_PROBES: dict = {}


def wants_drafter(cfg: dict) -> bool:
    from strata_control import inkling_family
    raw = cfg.get("drafting")
    if raw is False:
        return False
    choice = str(raw if raw not in (None, "") else inkling_family.default_drafting(cfg.get("backend"))).strip().lower()
    if choice == "auto":
        # measured at the first start of this placement (strata_control/inkling_probe.py); no valid probe yet = off, and needs_drafting_probe() says so
        from strata_control import inkling_probe
        return inkling_probe.decision_of(cfg) == "on"
    return choice not in ("off", "none", "false", "0")


def needs_drafting_probe(cfg: dict) -> bool:
    """True when this run config wants the measured drafting choice ("drafting": "auto", or nothing chosen) and
    has no probe valid for its current placement: the launcher should run strata_control.inkling_probe.probe_run_config before the real start."""
    from strata_control import inkling_family, inkling_probe
    raw = cfg.get("drafting")
    choice = str(raw if raw not in (None, "") else inkling_family.default_drafting(cfg.get("backend"))).strip().lower()
    return (raw is not False and choice == "auto"
            and inkling_probe.is_stale(cfg))


_NOTE = {"text": ""}   # the probe's sentence for the next engine start's /metrics (drafting_note), set by engine_args


def parse_info(text: str) -> dict:
    for line in str(text or "").splitlines():
        if line.startswith("arch="):
            return dict(p.split("=", 1) for p in line.split() if "=" in p)
    return {}


def probe(exe: str | None, args: list, cfg: dict | None = None, timeout: float = 30.0) -> dict | None:
    import subprocess
    model = next((args[i + 1] for i, a in enumerate(args) if a in ("--native", "--model") and i + 1 < len(args)), None)
    if not exe or not model or not Path(str(exe)).is_file():
        return None
    vision = next((args[i + 1] for i, a in enumerate(args) if a == "--vision" and i + 1 < len(args)), None)
    dspark = next((args[i + 1] for i, a in enumerate(args) if a == "--dspark" and i + 1 < len(args)), None)
    try:
        st = Path(str(exe)).stat()
        key = (str(exe), st.st_mtime_ns, st.st_size, str(model), str(vision or ""), str(dspark or ""))
    except OSError:
        return None
    if key in _PROBES:
        return _PROBES[key]
    from strata_control.childenv import quiet_crash_env
    env = quiet_crash_env(dict(os.environ))
    dirs = [d for d in ((cfg or {}).get("lib_dirs") or []) if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    cmd = [str(exe), "--info", str(model)]
    if vision:
        cmd += ["--vision", str(vision)]
    if dspark:
        cmd += ["--dspark", str(dspark)]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                              timeout=timeout, env=env, cwd=str(Path(str(exe)).parent))
        info = parse_info(done.stdout) if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        info = None
    _PROBES[key] = info or None
    return _PROBES[key]


def drafter_args(cfg: dict, info: dict | None = None) -> list[str]:
    from strata_control import inkling_family
    if not wants_drafter(cfg):
        return ["--draft", "off"]
    if (info or {}).get("drafter") != "dspark":
        return []
    out = ["--draft", "dspark"]
    # an engine that lists spec_opts=1 takes the window shape: one linear chain (the block drafter) unless the run config says otherwise
    if (info or {}).get("spec_opts") == "1":
        beam = cfg.get("draft_beam", inkling_family.DRAFT_BEAM)
        if isinstance(beam, int) and not isinstance(beam, bool) and 1 <= beam <= 8:
            out += ["--draft-beam", str(beam)]
        if cfg.get("draft_adapt", inkling_family.DRAFT_ADAPT) is True:
            out += ["--draft-adapt"]
    return out


def drafting_info(args: list, info: dict | None = None) -> dict[str, str]:
    """Metrics/status view of Inkling-2-Mini's built-in DSpark drafter (`drafting`: `"dspark"` | `"off"`)."""
    items = [str(a) for a in (args or ())]
    mode = None
    for i in range(len(items) - 1):
        if items[i] == "--draft":
            mode = items[i + 1].strip().lower()
    text, _NOTE["text"] = _NOTE["text"], ""
    if mode == "off":
        return {"drafting": "off", "drafting_note": text}
    # #6741: the engine says what it started with (INFO ... drafting=dspark|off). It leaves DSpark off when the CUDA layers are split across
    # cards (and when it has no drafter or no room for it); the status used to repeat what was asked for and say "dspark" over an engine
    # that ran without it. An older engine that does not say falls back to the asked mode, as before.
    said = str((info or {}).get("drafting") or "").strip().lower()
    if said == "dspark":
        return {"drafting": "dspark", "drafting_note": text}
    if said == "off":
        asked = mode == "dspark" or (info or {}).get("drafter") == "dspark"
        return {"drafting": "off", "drafting_note": _drafting_off_note(items, info) if asked else text}
    if mode == "dspark" or (info or {}).get("drafter") == "dspark":
        return {"drafting": "dspark", "drafting_note": text}
    return {"drafting": "off", "drafting_note": text}


DRAFTING_SPLIT_NOTE = ("Drafting is off: DSpark does not run yet when the model's layers are split across several cards (CUDA). "
                       "On one card, with the rest of the experts in RAM, it does (--gpu N).")
DRAFTING_NO_DRAFTER_NOTE = "Drafting is off: no DSpark drafter was found next to the model."
DRAFTING_LEFT_OFF_NOTE = "Drafting is off: the engine started without the DSpark drafter (see its log for why, often too little free graphics memory)."


def _drafting_off_note(items: list[str], info: dict | None) -> str:
    """Plain words for 'DSpark was asked for and the engine left it off' (#6741)."""
    if (info or {}).get("drafter") == "none":
        return DRAFTING_NO_DRAFTER_NOTE
    if "--layer-split" in items or str((info or {}).get("backend") or "").endswith("-ls"):
        return DRAFTING_SPLIT_NOTE
    return DRAFTING_LEFT_OFF_NOTE


def _local_gpu_ordinals(n_gpus: int, max_g: int) -> str:
    """Map Helper..Main visibility order (`CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`/`ZE_AFFINITY_MASK`)
    to Main-first local ordinals for `strata-inkling --gpus` (`gpu_ids[0]` is primary/Main)."""
    use = min(max_g, n_gpus)
    main = n_gpus - 1
    helpers = list(range(n_gpus - use, main))
    return ",".join(str(i) for i in [main, *helpers])


def _strip_flag(args: list[str], flag: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            i += 2 if i + 1 < len(args) and not str(args[i + 1]).startswith("--") else 1
            continue
        out.append(args[i])
        i += 1
    return out


def fit_dspark(args: list, info: dict | None, announce: bool = True) -> tuple[list, dict | None]:
    """The drafter file this engine can read. An engine that does not list dspark_f32_widen=1 in its --info line cannot use the original
    mtp.safetensors (it drafts nothing the target accepts): the Unsloth Q8_0 file beside it is passed instead, and with none the run starts without a
    drafter. The file stays on disk, and a newer engine (the GUI engine picker) takes it again at the next start."""
    from strata_control import inkling_family
    if not info or "--dspark" not in args:
        return args, info
    i = args.index("--dspark")
    if i + 1 >= len(args):
        return args, info
    path = Path(str(args[i + 1]))
    if path.name.lower() != inkling_family.MTP_FILE or inkling_family.engine_reads_mtp(info):
        return args, info
    out = list(args)
    alt = path.with_name(inkling_family.DSPARK_GGUF_FILE)
    if alt.is_file():
        out[i + 1] = str(alt)
        if announce:
            print(f"[strata] this engine cannot read {inkling_family.MTP_FILE}; the drafter {alt.name} is used", flush=True)
        return out, info
    del out[i:i + 2]
    if announce:
        print(f"[strata] this engine cannot read {inkling_family.MTP_FILE} and there is no {alt.name} beside it: starting without the drafter", flush=True)
    return out, {**info, "drafter": "none"}


def engine_args(cfg: dict, exe: str | None = None, *, announce: bool = True) -> list[str]:
    """Build start args for `strata-inkling`, preserving `--parallel 1..64`, `--max-context`, `--vision`,
    Main-first `--gpus` + `--layer-split`, and `--draft dspark|off`."""
    from strata_control import inkling_family, user_args
    args = list(cfg["args"])
    mine = user_args.capture(cfg)
    if mine:
        args = user_args.merge(args, mine)
    # #6497: honor a configured CPU thread limit (`--pool-workers N`) on Inkling by translating it to `--threads N`.
    if "--pool-workers" in args:
        pw_val = None
        out_args: list[str] = []
        i = 0
        while i < len(args):
            if args[i] == "--pool-workers":
                if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                    pw_val = str(args[i + 1])
                    i += 2
                else:
                    i += 1
            else:
                out_args.append(args[i])
                i += 1
        args = out_args
        if pw_val and pw_val.isdigit() and int(pw_val) > 0 and "--threads" not in args:
            args += ["--threads", pw_val]
    # Propagate top-level cfg["parallel"] (1..64) into `--parallel <N>` when not explicitly in args
    par = cfg.get("parallel")
    if isinstance(par, int) and 1 <= par <= inkling_family.MAX_PARALLEL and "--parallel" not in args:
        if par > 1:
            args += ["--parallel", str(par)]
    # Propagate 2..4 GPU selection (`cfg["gpu"]` or `gpu_roles`) into Main-first `--gpus` and `--layer-split` (#6592, #6631).
    # child_env() exposes GPUs in Helper..Main order, so local ordinal N-1 is Main and 0..N-2 are Helpers;
    # strata-inkling puts gpu_ids[0] as primary GPU, so pass Main first (`1,0`, `2,0,1`, `3,0,1,2`).
    from serve.server_parts.engines import gpu_list
    n_gpus = len(gpu_list(cfg))
    is_cuda = cfg.get("backend") not in ("hip", "sycl", "metal", "cpu")
    mine_flags = {str(a) for a in (mine or ()) if str(a).startswith("--")}
    if n_gpus >= 2:
        max_g = 4 if is_cuda else 2
        local_spec = _local_gpu_ordinals(n_gpus, max_g)
        if "--gpus" not in args:
            args += ["--gpus", local_spec]
        elif "--gpus" not in mine_flags:
            at = args.index("--gpus")
            if at + 1 < len(args) and "," not in str(args[at + 1]):
                args[at + 1] = local_spec
        if is_cuda and "--layer-split" not in args and "--ep-split" not in args:
            args += ["--layer-split", str(cfg["layer_split"]) if cfg.get("layer_split") else "auto"]
        # Drop single-GPU expert caps when running multi-GPU unless the user explicitly pinned them in user_args.
        if "--vram-experts-gb" not in mine_flags:
            args = _strip_flag(args, "--vram-experts-gb")
        if "--ram-experts-gb" not in mine_flags:
            args = _strip_flag(args, "--ram-experts-gb")
    # strata-inkling reads --ram-experts-gb 0 as "no RAM tier, GPU + SSD only", but setup used to write 0.00 for a plan with every
    # expert on the card: drop that 0 (unless the person set it, or the plan streams from the SSD) so experts that do not fit go to RAM
    if "--ram-experts-gb" in args and "--ram-experts-gb" not in mine_flags and "--ssd-experts" not in args:
        at = args.index("--ram-experts-gb")
        try:
            zero = at + 1 < len(args) and float(args[at + 1]) <= 0.0
        except ValueError:
            zero = False
        if zero:
            args = _strip_flag(args, "--ram-experts-gb")
    wants_images = "--vision" in args
    want_draft = wants_drafter(cfg)
    need_draft = "--draft" not in args and want_draft
    info = probe(exe, args, cfg) if (exe and (wants_images or need_draft)) else None
    args, info = fit_dspark(args, info, announce)
    if "--draft" not in args and not want_draft:
        args += ["--draft", "off"]
    elif need_draft:
        add = drafter_args(cfg, info)
        args += add
        if not add and announce:
            print(f"[strata] {NO_DRAFTER_NOTE}", flush=True)
    if announce and isinstance(cfg.get("drafting_probe"), dict):          # why drafting is on or off here, for /metrics and `strata launch --status`
        from strata_control import inkling_family, inkling_probe
        text = inkling_probe.note(cfg)
        _NOTE["text"] = text
    if wants_images and (info or {}).get("images") != "1":
        at = args.index("--vision")
        del args[at:at + 2]
        if announce:
            print(f"[strata] {NO_IMAGES_NOTE}", flush=True)
    return args


def encode_path(path) -> str:
    return str(path).replace("%", "%25").replace(" ", "%20")


def decode_path(text: str) -> str:
    return str(text).replace("%20", " ").replace("%25", "%")


# ---- #7043: the picture reader on the CPU --------------------------------------------------------------------------------------------
# strata-inkling reads a picture with its GPU encoder where it has one (HIP, SYCL, Metal) and otherwise, CUDA included, with the CPU one, which has
# no size limit below 4096 x 4096 pixels: a 4535 x 3004 photo is 27,900 patches with full attention in 23 layers on one CPU thread of control, no
# progress line and no cancel, minutes in which the engine answers nothing (and the 300 s watchdog ends it).  The engine has no option for the size,
# so serve scales a picture down before it writes the file the engine reads: where the reader is the CPU (backend cuda or cpu), to at most
# DEFAULT_MAX_PIXELS, 0.8 MP: where the measured CPU reader takes about a minute (a 1024 x 768 picture is not touched, a 1920 x 1080 screenshot is scaled to about 1190 x 670).  `inkling_image_max_mp` in the run config (or STRATA_INKLING_IMAGE_MAX_MP) sets
# another size in megapixels, 0 turns the cap off; a value above 0 applies on every backend.
DEFAULT_MAX_PIXELS = 800_000
GPU_PICTURE_BACKENDS = ("hip", "sycl", "metal")


def image_cap_pixels(cfg: dict | None, environ=None) -> int:
    """The most pixels a picture may have when it is handed to the engine: 0 = no cap.  ValueError for a setting that is not a number >= 0."""
    environ = os.environ if environ is None else environ
    cfg = cfg or {}
    raw = cfg.get("inkling_image_max_mp")
    if raw is None or raw == "":
        raw = (environ.get("STRATA_INKLING_IMAGE_MAX_MP") or "").strip() or None
    if raw is None:
        return DEFAULT_MAX_PIXELS if cfg.get("backend") not in GPU_PICTURE_BACKENDS else 0
    try:
        mp = float(raw)
        if isinstance(raw, bool) or mp != mp or mp < 0 or mp > 1e6:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f'"inkling_image_max_mp" must be a number of megapixels, 0 for no cap, not {raw!r}') from None
    return int(mp * 1_000_000)


def cap_picture(data: bytes, max_pixels: int):
    """(the picture, or a smaller one with the same proportions when it has more than `max_pixels` pixels; a note saying so or None).
    Needs Pillow (the server already uses it for other formats); without it, or for a picture it cannot read, the picture is passed on as it is."""
    if not max_pixels or max_pixels < 1:
        return data, None
    try:
        import io
        from PIL import Image, ImageOps
    except ImportError:
        return data, None
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            if w * h <= max_pixels:
                return data, None
            jpeg = data[:3] == b"\xff\xd8\xff"
            im.load()
            im = ImageOps.exif_transpose(im)
            scale = (max_pixels / float(w * h)) ** 0.5
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            while nw * nh > max_pixels and nw > 1:               # (a sliver: one pixel high is the least it can be)
                nw = max(1, max_pixels // nh)
            small = im.resize((nw, nh), Image.LANCZOS)
            out = io.BytesIO()
            if jpeg:
                small.convert("RGB").save(out, format="JPEG", quality=95)
            else:
                small.save(out, format="PNG")
    except Exception:  # noqa: BLE001 - a picture Pillow cannot read goes on as it is: the engine says what it thinks of it
        return data, None
    note = (f"picture {w}x{h} ({w * h / 1e6:.1f} MP) scaled to {nw}x{nh} ({nw * nh / 1e6:.1f} MP) before the engine reads it "
            f"(its picture reader runs on the CPU; inkling_image_max_mp in the run config sets the size, 0 turns this off)")
    return out.getvalue(), note


# What the CPU reader costs: one picture, prompt to first token, nothing else using the CPU, owner PC (6 cores / 12 threads, Inkling SQ-10 + Vision, engine 0.1.67; measured
# 11-24, rel02212/RESULTS-02212.md section 6): (megapixels, seconds).  Between two points the time is read on a log-log line, beyond the last one it grows as the square
# (the measured exponent from 2 to 3 MP is 2.1).  A request with pictures on that path gets an allowance of ALLOW_MARGIN times the sum of its pictures' times, plus
# ALLOW_BASE_S, at least the server's own engine_silence_s (the person's value stays when it is larger) and at most ALLOW_MAX_S, so a big picture is slow and answered,
# never ended by the silence watchdog.  DEFAULT_MAX_PIXELS (0.8 MP) is where the table says about a minute.
READ_CURVE = ((0.04, 3.1), (0.25, 14.9), (0.499, 32.0), (0.749, 55.0), (0.999, 74.0), (1.499, 159.8), (2.066, 333.9), (2.996, 743.2))
ALLOW_MARGIN = 3.0
ALLOW_BASE_S = 60.0
ALLOW_MAX_S = 3 * 3600.0


def cpu_picture_reader(cfg: dict | None) -> bool:
    """True where the engine reads pictures on the CPU (CUDA, cpu); HIP, SYCL and Metal have a GPU encoder."""
    return (cfg or {}).get("backend") not in GPU_PICTURE_BACKENDS


def picture_pixels(data: bytes) -> int:
    """The pixels of a picture (0 when Pillow cannot say)."""
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            return int(im.size[0]) * int(im.size[1])
    except Exception:  # noqa: BLE001
        return 0


def picture_read_seconds(pixels: int) -> float:
    """The measured time of the CPU reader for one picture of this many pixels (READ_CURVE)."""
    import math
    mp = max(pixels, 0) / 1e6
    pts = READ_CURVE
    if mp <= 0:
        return 0.0
    if mp <= pts[0][0]:
        return pts[0][1] * mp / pts[0][0]
    for (x0, t0), (x1, t1) in zip(pts, pts[1:]):
        if mp <= x1:
            k = math.log(t1 / t0) / math.log(x1 / x0)
            return t0 * (mp / x0) ** k
    x, t = pts[-1]
    return t * (mp / x) ** 2.1


def picture_allowance_s(pixel_counts, cap: int = 0) -> float:
    """How long the engine may stay silent while it reads these pictures (the sizes the engine will get).  A picture whose size could not be read
    counts as the cap in pixels, or 4096 x 4096 (the engine's own limit) when there is no cap."""
    total = 0.0
    for px in pixel_counts:
        total += picture_read_seconds(px if px > 0 else (cap or 4096 * 4096))
    return min(ALLOW_MAX_S, ALLOW_MARGIN * total + ALLOW_BASE_S) if total > 0 else 0.0


def write_images(sources, load, normalize, cancel=None, folder=None) -> list[str]:
    """Write normalized request images under unique per-request names (`strata-inkling-images`)."""
    import hashlib
    import secrets
    import tempfile
    base = Path(folder) if folder else Path(tempfile.gettempdir()) / "strata-inkling-images"
    base.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(6)
    out: list[str] = []
    try:
        for idx, src in enumerate(sources):
            if cancel is not None and cancel.is_set():
                raise ValueError("the request was cancelled")
            data = normalize(src if isinstance(src, bytes) else load(src))
            ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg" if data[:3] == b"\xff\xd8\xff" else ".bmp" if data[:2] == b"BM" else ".gif" if data[:3] == b"GIF" else ".img"
            p = base / f"{hashlib.sha256(data).hexdigest()[:24]}-{nonce}-{idx}{ext}"
            p.write_bytes(data)
            out.append(str(p))
    except Exception:
        for p in out:
            Path(p).unlink(missing_ok=True)
        raise
    return out


def refuse(engine, req: dict) -> None:
    from serve.server_parts.core import top_logprobs
    if getattr(engine, "inkling_engine", False) is True and top_logprobs(req) is not None:
        raise ValueError(REFUSED["logprobs"])


def literal_tags(tokenizer) -> list[str]:
    types = getattr(tokenizer, "token_types", None) or []
    toks = getattr(tokenizer, "tokens", None) or []
    return [toks[i] for i, ty in enumerate(types) if ty == 4 and i < len(toks)]


def end_ids(tokenizer) -> set[int]:
    ids = set()
    for text in END_TEXTS:
        got = tokenizer.encode(text, parse_special=True)
        if len(got) == 1:
            ids.add(int(got[0]))
    return ids


def model_path(cfg: dict) -> Path | None:
    args = cfg.get("args") or []
    for key in ("--native", "--model"):
        if key in args and args.index(key) + 1 < len(args):
            return Path(args[args.index(key) + 1])
    return None
