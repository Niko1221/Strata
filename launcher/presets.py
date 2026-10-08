"""A Strata preset: one model, one set of setup choices, a name.

A preset is what the launcher shows instead of a wall of command-line flags: it names a model (family + size), the
context, the KV precision, images, and the rest of the choices `setup.py` takes, and it can be saved, edited,
duplicated and started again later.  It is not a new kind of configuration: applying a preset runs setup's own
install path with these choices, and starting one starts the model's own `strata-<model>.json` the way
`run-<model>.bat` does.  Nothing here invents a setting - every field maps to a `setup.py` flag, and the flags are
the ones `setup.py` documents (its `--help` stays the reference).

Presets live in `.strata-launcher/presets.json` next to the MCP server's own state: plain JSON, so a preset can be
copied to another PC or reviewed in a text editor.  The built-in ones are not stored: they are derived from setup's
tables and its recommendation for this PC, so they follow the repository when its model list changes.
"""
from __future__ import annotations

import json
import os
import re
import time

# The choices setup.py takes, with the words the launcher shows for them.  The values are setup's; only the labels
# are ours.  A field the user leaves at its default is not passed at all, so setup's own default (which may depend
# on the PC) decides.
KV = {"int8": "8-bit (the default: precise, and what setup picks)",
      "q4_0": "4-bit (half the memory, about 4% faster at 128K, less precise on long documents)",
      "k8v4": "hybrid K8/V4 (23% less than 8-bit; it streams its KV cache too since 0.1.40)"}
VISION = {"no": "text only", "gpu": "images - the encoder on the GPU", "cpu": "images - the encoder on the CPU"}
LOW_RAM = {"auto": "auto (setup's rule: only when the RAM needs it)", "on": "on", "off": "off",
           "resident": "resident (keep a fixed budget of experts in RAM)",
           "mmap": "mmap (read the experts through the OS file cache)"}
KV_STREAMING = {"auto": "auto (from a 64K context, when the RAM has room)", "on": "on", "off": "off"}
# setup's --backend takes cuda, hip or sycl; it has no "auto" value, so "leave it to setup" is the empty field and
# no flag is written - which is also what makes the preset portable to a PC with another card.
BACKEND = {"": "auto (setup picks: NVIDIA, or AMD when the PC has no NVIDIA it can use)",
           "cuda": "NVIDIA (cuda)", "hip": "AMD (hip) - RX 7900 / 7800 / 7700 XT, RX 9060 XT / 9070",
           "sycl": "Intel Arc (sycl) - EXPERIMENTAL, Linux, built from source"}
CALIBRATE = {"ask": "ask before the first start of a new model", "always": "always measure this PC first",
             "never": "never (keep the shipped defaults)"}

# The context lengths the page offers: setup's own menu plus the size between its 128K and 256K.  setup takes any
# --context integer (its CONTEXTS list is only what its questions offer interactively), and 196,608 = 192K is still
# under the model's trained 262,144 positions, so no rope scaling comes into it.  A value of your own is taken
# between these bounds; below or above them setup's engine has no use for it.
EXTRA_CONTEXTS = [196608]
CTX_MIN, CTX_MAX = 1024, 1048576


def offered_contexts(catalog: dict) -> list:
    """The context lengths the page offers, in order."""
    return sorted(set(catalog["contexts"]) | set(EXTRA_CONTEXTS))
ESP = {"off": "off (the default)", "on": "on - EXPERIMENTAL: the speed-projection vector in data/"}

NAME_MAX = 60
ID_MAX = 60
TRAINED_CONTEXT = 262144          # setup.py's own trained length; the page marks a longer context as scaled
GPUS_PATTERN = re.compile(r"^(all|\d{1,2}(,\d{1,2}){1,7})$")
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,59}$")

# A size that names no families in setup's table belongs to the original model *and* to its fine-tunes: that is
# setup.py's own default (its `MODELS[m].get("families", ("qwen", "swift"))`, and tools/strata_mcp.py's sizes_of).
# Swift 1.5's IQ2_XS and IQ3_XXS are stored that way, so a launcher reading the same table with ("qwen",) alone
# offers a size on the page and then refuses to save it ("Swift 1.5 has no IQ3_XXS chosen; its sizes:").
DEFAULT_FAMILIES = ("qwen", "swift")


def sizes_of(catalog: dict, family: str) -> list:
    """The sizes a family publishes: the controller's answer when the catalog carries it (the launcher's catalog
    does, from tools/strata_mcp.py), setup's tables read the same way otherwise.  One rule, so the list the page
    shows and the list a save is checked against cannot drift apart."""
    listed = (catalog.get("family_sizes") or {}).get(family)
    if listed is not None:
        return list(listed)
    return [m for m in catalog["models"] if family in catalog["models"][m].get("families", DEFAULT_FAMILIES)]

# The fields, in the order the launcher shows them.  `flag` is the setup.py flag it writes; a field without one is
# launcher-side only (calibrate is a choice about the first start, not a setup flag).  `default` "" / None means
# "do not pass it": setup's own default, which often depends on the PC, decides.
FIELDS = [
    {"key": "family", "label": "Model", "type": "family", "required": True,
     "help": "which release of the model"},
    {"key": "model", "label": "Size", "type": "size", "required": True,
     "help": "the quantization: smaller is faster, bigger is better"},
    {"key": "context", "label": "Context", "type": "context", "required": True, "flag": "--context",
     "help": "how much of the conversation it keeps; longer needs more VRAM and RAM"},
    {"key": "kv", "label": "KV cache", "type": "select", "choices": KV, "flag": "--kv",
     "help": "how the context's attention values are stored (above an 8K context)"},
    {"key": "vision", "label": "Images", "type": "select", "choices": VISION, "flag": "--vision",
     "default": "no"},
    {"key": "kv_streaming", "label": "KV streaming", "type": "select", "choices": KV_STREAMING,
     "flag": "--kv-streaming", "help": "from 64K: keep the KV cache in RAM so more experts fit on the GPU"},
    {"key": "low_ram", "label": "Low-RAM mode", "type": "select", "choices": LOW_RAM, "flag": "--low-ram",
     "help": "when the RAM is short, the GPU holds part of the experts (slower)"},
    {"key": "resident_budget_gib", "label": "Experts kept in RAM (GiB)", "type": "number",
     "flag": "--resident-budget-gib", "help": "Unsloth's 4-bit file only: the rest is read from the SSD"},
    {"key": "vram_reserve_mib", "label": "VRAM left free (MiB)", "type": "number", "flag": "--vram-reserve-mib",
     "help": "for other programs; the engine's own default is 700"},
    {"key": "speed_projection", "label": "Speed projection", "type": "select", "choices": ESP,
     "flag": "--experimental-speed-projection", "help": "EXPERIMENTAL, off by default"},
    {"key": "gpu", "label": "GPU", "type": "number", "flag": "--gpu",
     "help": "one card, numbered as nvidia-smi numbers them; blank: the one with the most VRAM"},
    {"key": "gpus", "label": "GPUs (layer split)", "type": "text", "flag": "--gpus",
     "help": "several cards sharing one model: \"0,1\" or \"all\" (experimental)"},
    {"key": "layer_split", "label": "Layer split", "type": "text", "flag": "--layer-split",
     "help": "with several cards: where each later card's layers start (\"18\", \"16,32\")"},
    {"key": "backend", "label": "Backend", "type": "select", "choices": BACKEND, "flag": "--backend"},
    {"key": "port", "label": "Port", "type": "number", "flag": "--port", "default": 8080,
     "help": "the address is always 127.0.0.1 unless you set up a network address with a key"},
    {"key": "host", "label": "Listen address", "type": "text", "flag": "--host",
     "help": "0.0.0.0 also answers other devices - only with an API key"},
    {"key": "api_key", "label": "API key", "type": "text", "flag": "--api-key", "secret": True,
     "help": "required by clients when the address is not this PC only"},
    {"key": "calibrate", "label": "Measure this PC", "type": "select", "choices": CALIBRATE, "default": "ask",
     "help": "setup's --calibrate: 5-10 minutes on the PCIe share, the draft depth and the CPU threads"},
]


class PresetError(ValueError):
    """A preset that setup.py would refuse, or that names nothing the launcher can install."""


def slug(text: str) -> str:
    """A stable id from a name: lowercase, one dash between words.  Two names that slug the same get a number."""
    s = re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-")
    return (s or "preset")[:ID_MAX]


def defaults() -> dict:
    return {f["key"]: f.get("default", "") for f in FIELDS}


def blank(catalog: dict, recommendation: dict | None = None) -> dict:
    """A new preset, pre-filled the way setup pre-fills its own questions: the size and context it recommends for
    this PC, images off, everything else left to setup."""
    p = defaults()
    p["name"] = ""
    p["note"] = ""
    rec = recommendation or {}
    p["family"] = rec.get("family") or ""
    p["model"] = rec.get("model") or ""
    p["context"] = rec.get("context") or 32768
    if rec.get("backend") in BACKEND:
        p["backend"] = rec["backend"]
    return p


def builtin(catalog: dict, recommendation: dict | None) -> list[dict]:
    """The presets the launcher offers before anyone makes one: setup's recommendation for this PC, and the fastest
    and the best-fitting size of the original model.  Built from setup's own tables, so a repository that adds a
    model adds it here too; a size that does not fit this PC is left out."""
    models, families = catalog["models"], catalog["families"]
    contexts = catalog["contexts"]
    rec = recommendation or {}
    ram = catalog.get("ram_gb") or 0

    def fits(m):
        return ram >= models[m]["ram_gb"] - 4

    def pick(family, want):
        sizes = sizes_of(catalog, family)
        for m in want:
            if m in sizes and (ram == 0 or fits(m)):
                return family, m
        return None, None

    wanted = []
    if rec.get("model") and rec.get("model") in models and rec.get("family") in families:
        wanted.append(("Recommended for this PC", rec["family"], rec["model"], rec["context"],
                       rec.get("why", ""), "setup's own pick for this PC"))
    for title, family, sizes, note in (
            ("Fastest answers", "qwen", ["Q2_0", "IQ2_XS"], "the shortest answers, the least quality"),
            ("Best quality that fits", "qwen", ["IQ3_S", "IQ3_XXS", "IQ2_XS"],
             "the biggest size this PC has room for"),
            ("Coding", "coder", ["IQ1_M"], "the Coder: half the experts, ~32 GB of RAM, made for code")):
        family, model = pick(family, sizes)
        if model:
            wanted.append((title, family, model, None, note, ""))
    out = []
    for title, family, model, context, why, note in wanted:
        # the context a built-in preset starts from is the one setup would pick for this card; a longer one is the
        # user's choice, made in the form
        ctx = context or rec.get("context") or next((c for c in contexts if c == 32768), contexts[0])
        p = dict(defaults(), name=title, note=(why + (" - " + note if note else ""))[:200],
                 family=family, model=model, context=ctx, vision="no", id=f"builtin-{slug(title)}", builtin=True)
        if rec.get("backend") in BACKEND:
            p["backend"] = rec["backend"]
        out.append(p)
    return out


def _as_number(value):
    if value in ("", None):
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        raise PresetError(f"{value!r} is not a number")
    return int(n) if n.is_integer() else n


def clean(raw: dict, catalog: dict, keep_id: str | None = None) -> dict:
    """`raw` as a stored preset: unknown keys dropped, numbers checked, and every choice checked against setup's
    tables.  Raises PresetError with the whole list of problems, so the form can show them at once."""
    bad = []
    p = defaults()
    name = str(raw.get("name") or "").strip()
    if not name:
        bad.append("a preset needs a name")
    if len(name) > NAME_MAX:
        bad.append(f"the name is longer than {NAME_MAX} characters")
    p["name"] = name
    p["note"] = str(raw.get("note") or "").strip()[:500]
    p["id"] = str(raw.get("id") or keep_id or "") or None
    if p["id"] and not ID_PATTERN.match(p["id"]):
        bad.append(f"the preset id {p['id']!r} may use lowercase letters, digits, '-' and '_'")

    models, families = catalog["models"], catalog["families"]
    family, model, context = str(raw.get("family") or ""), str(raw.get("model") or ""), raw.get("context")
    if family not in families:
        bad.append(f"{family or 'nothing'} is not a model family setup offers"
                   + (f" (it offers {', '.join(families)})" if families else ""))
        p["family"], p["model"] = family, model
    else:
        p["family"] = family
        sizes = sizes_of(catalog, family)
        if model not in sizes:
            bad.append(f"{families[family]['title']} has no {model or 'size'} chosen; its sizes: {', '.join(sizes)}")
            p["model"] = model
        else:
            p["model"] = model
            if models[model].get("families") and family not in models[model]["families"]:
                bad.append(f"{model} is not published for {families[family]['title']}")
    try:
        context = int(context)
    except (TypeError, ValueError):
        bad.append("the context must be a whole number of tokens: "
                   f"{', '.join(str(c) for c in offered_contexts(catalog))}, or a value of your own")
        context = None
    if context is not None and not CTX_MIN <= context <= CTX_MAX:
        bad.append(f"a {context:,} token context is outside what Strata takes ({CTX_MIN:,} to {CTX_MAX:,}); "
                   "setup's --context is a token count")
        context = None
    p["context"] = context or ""

    for f in FIELDS:
        k = f["key"]
        if k in ("family", "model", "context"):
            continue
        value = raw.get(k, p.get(k))
        if f["type"] == "number":
            try:
                value = _as_number(value)
            except PresetError as e:
                bad.append(f"{f['label']}: {e}")
                value = None
            if isinstance(value, int) and k in ("port", "vram_reserve_mib", "gpu", "resident_budget_gib"):
                if k == "port" and not 1024 <= value <= 65535:
                    bad.append("the port must be between 1024 and 65535")
                if k in ("vram_reserve_mib", "resident_budget_gib", "gpu") and value < 0:
                    bad.append(f"{f['label']} must be 0 or more")
            p[k] = value if value not in (None,) else ""
        elif f["type"] == "select":
            value = "" if value is None else str(value)
            if k == "backend" and value == "auto":
                value = ""      # a preset saved when the page still called this "auto": it meant "leave it to setup"
            if value and value not in f["choices"]:
                bad.append(f"{f['label']}: {value!r} is not one of {', '.join(x for x in f['choices'] if x)}")
            p[k] = value
        else:
            p[k] = str(value or "").strip()

    # choices that only make sense together - the same rules setup.py applies to its flags
    if p["gpus"] and not GPUS_PATTERN.match(p["gpus"]):
        bad.append(f"the GPU list {p['gpus']!r} looks like \"0,1\" or \"all\"")
    if p["gpus"] and p["gpu"] not in ("", None):
        bad.append("choose one GPU or a list of them, not both")
    if p["vision"] in ("yes", "gpu", "cpu") and families.get(family, {}).get("vision") is False:
        bad.append(f"images are not available with {families[family]['title']} yet")
    if p["vision"] in ("yes", "gpu", "cpu") and p["backend"] == "hip":
        bad.append("the AMD backend has no image encoder yet: turn images off")
    if p["resident_budget_gib"] not in ("", None) and not models.get(model, {}).get("budget"):
        bad.append(f"{model or 'this size'} keeps all of its experts in RAM: the RAM budget is only for "
                   "Unsloth's UD-Q4_K_XL")
    if p["host"] and p["host"] not in ("127.0.0.1", "localhost") and not p["api_key"]:
        bad.append("an address other than 127.0.0.1 needs an API key")
    if p["calibrate"] not in CALIBRATE:
        p["calibrate"] = "ask"
    if bad:
        raise PresetError("\n".join(bad))
    p["id"] = p["id"] or slug(p["name"])
    p["updated"] = time.strftime("%Y-%m-%d %H:%M")
    return p


def from_config(desc: dict, cfg: dict, catalog: dict, name: str | None = None) -> dict:
    """A preset that describes an installed model, so it can be re-created on another PC or changed and installed
    again.  Reads what the model's own strata-<model>.json says (setup's choices_from_config already did that
    reading; here it is only mapped onto fields)."""
    p = defaults()
    tag = str(desc.get("model") or "")
    family = next((f for f, d in catalog["families"].items() if d.get("tag") and tag.startswith(d["tag"])), "qwen")
    size = tag[len(catalog["families"][family]["tag"]):] if family != "qwen" else tag
    p["name"] = name or f"{tag.upper()} (from the installed model)"
    p["note"] = "made from " + str(desc.get("config") or "an installed model")
    p["family"], p["model"] = family, (size or tag).upper()
    p["context"] = desc.get("context") or ""
    p["kv"] = desc.get("kv") or ""
    p["vision"] = "" if desc.get("images") in (None, "off") else desc["images"]
    p["port"] = desc.get("port") or ""
    p["host"] = desc.get("host") or ""
    p["api_key"] = ""
    args = cfg.get("args") if isinstance(cfg.get("args"), list) else []

    def val(flag):
        return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None

    for flag, key, cast in (("--vram-reserve-mib", "vram_reserve_mib", int), ("--kv-streaming", "kv_streaming", str),
                            ("--low-ram", "low_ram", str)):
        v = val(flag)
        if v is not None:
            try:
                p[key] = cast(v)
            except (TypeError, ValueError):
                p[key] = v
    if val("--resident-experts") is not None or val("--mmap-experts") is not None:
        p["low_ram"] = "resident" if val("--resident-experts") is not None else "mmap"
    if val("--control-vector-scaled"):
        p["speed_projection"] = "on"
    if isinstance(desc.get("gpu"), list) and len(desc["gpu"]) > 1:
        p["gpus"] = ",".join(str(g) for g in desc["gpu"])
        if desc.get("layer_split"):
            p["layer_split"] = ",".join(str(x) for x in desc["layer_split"])
    elif isinstance(desc.get("gpu"), int):
        p["gpu"] = desc["gpu"]
    p["calibrate"] = "ask"
    p["id"] = slug(p["name"])
    return p


def setup_args(p: dict) -> list[str]:
    """The preset as setup.py's flags.  A field left empty is not passed, so setup's own default - which depends on
    the PC - decides; that is what makes a preset portable."""
    args = ["--yes", "--no-start", "--family", str(p["family"]), "--model", str(p["model"]),
            "--context", str(p["context"])]
    for f in FIELDS:
        flag, key = f.get("flag"), f["key"]
        if not flag or key in ("family", "model", "context"):
            continue
        value = p.get(key)
        if value in ("", None):
            continue
        if key == "vision" and value == "yes":
            value = "gpu"
        args += [flag, str(value)]
    return args


def merge_args(base: list[str], extra: list[str]) -> list[str]:
    """`base` with `extra`'s flags applied: a flag that appears in both is set to the extra value, so the preset
    wins over what another builder picked.  Only single-value flags appear in either list."""
    out, i = [], 0
    taken = {}
    while i < len(extra):
        if str(extra[i]).startswith("--") and i + 1 < len(extra) and not str(extra[i + 1]).startswith("--"):
            taken[extra[i]] = extra[i + 1]
            i += 2
        else:
            i += 1
    drop = {f for f, v in taken.items() if v is None}
    i = 0
    while i < len(base):
        flag = base[i]
        if str(flag).startswith("--") and i + 1 < len(base) and not str(base[i + 1]).startswith("--"):
            if flag in drop:
                i += 2
                continue
            out += [flag, taken.get(flag, base[i + 1])]
            taken.pop(flag, None)
            i += 2
            continue
        out.append(flag)
        i += 1
    for flag, value in taken.items():
        out += [flag, str(value)]
    return out


def summary(p: dict, catalog: dict | None = None) -> str:
    """One line: what this preset is, in the words the model page uses."""
    fam = ((catalog or {}).get("families", {}).get(p.get("family") or "", {}) or {}).get("title") or p.get("family")
    bits = [f"{fam} {p.get('model')}", f"{(p.get('context') or 0) // 1024}K"]
    if p.get("kv"):
        bits.append(f"kv {p['kv']}")
    if p.get("vision") in ("yes", "gpu", "cpu"):
        bits.append("images")
    if p.get("gpus"):
        bits.append(f"GPUs {p['gpus']}")
    if p.get("low_ram") not in ("", "auto"):
        bits.append(f"low-RAM {p['low_ram']}")
    if p.get("vram_reserve_mib") not in ("", None):
        bits.append(f"{p['vram_reserve_mib']} MiB free")
    if p.get("speed_projection") == "on":
        bits.append("speed projection")
    return " · ".join(str(x) for x in bits)


def model_id(p: dict, catalog: dict) -> str:
    """The id of the model this preset installs - the tag of its strata-<model>.json and of run-<model>.bat."""
    tag = catalog["families"].get(p["family"], {}).get("tag", "")
    return f"{tag}{p['model']}".lower()


def model_name(p: dict, catalog: dict) -> str:
    """The name the model answers to in the API and in its own config (setup writes "<family name>-<size>"): what a
    running server reports as its model, and what a calibration record is keyed by."""
    fam = catalog["families"].get(p["family"], {})
    return f"{fam.get('name', p['family'])}-{str(p['model']).lower()}"


class Store:
    """`.strata-launcher/presets.json`: the user's presets. Written whole or not at all, like setup's configs
    (#459), so a launcher closed half-way never leaves an empty file behind."""

    def __init__(self, path):
        self.path = path

    def load(self) -> list[dict]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        items = raw.get("presets") if isinstance(raw, dict) else raw
        return [p for p in (items or []) if isinstance(p, dict) and p.get("name")]

    def save(self, presets: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"presets": presets}, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def put(self, preset: dict) -> dict:
        """Add or replace by id; a new name that collides with another preset's id gets a numbered id, so two
        presets never overwrite each other."""
        items = self.load()
        by_id = {p.get("id"): p for p in items}
        pid = preset["id"]
        if pid in by_id and by_id[pid].get("name") != preset["name"]:
            n = 2
            while f"{pid}-{n}" in by_id:
                n += 1
            preset = dict(preset, id=f"{pid}-{n}")
        by_id[preset["id"]] = preset
        self.save(list(by_id.values()))
        return preset

    def delete(self, pid: str) -> bool:
        items = self.load()
        keep = [p for p in items if p.get("id") != pid]
        if len(keep) == len(items):
            return False
        self.save(keep)
        return True

    def get(self, pid: str) -> dict | None:
        return next((p for p in self.load() if p.get("id") == pid), None)
