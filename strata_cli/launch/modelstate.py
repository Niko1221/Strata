"""What a model still needs on this computer, in plain words: the one place the Folder step, the download question and the final summary ask.

A model that is already installed is "already downloaded". What can still be missing are helper files, each with its own size: the
draft layer (MTP, makes answers faster) and the picture reader. Only when the weights themselves are gone does it say the model
is downloaded again. Nothing here reads more than a run config and file existence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from strata_control import discovery
from strata_control.discovery import MMPROJ_GB, MTP_FETCH_GB


@dataclass
class Part:
    kind: str                        # weights | mtp | mmproj
    label: str                       # "the draft layer"
    gb: float


@dataclass
class ModelState:
    installed: bool = False          # a run config for this model exists and names files that are there
    own_files: bool = False          # the person's own model files
    total_gb: float = 0.0            # the whole model (the catalog's figure)
    parts: list[Part] = field(default_factory=list)       # what is missing and would be downloaded
    folder: Path | None = None       # where the model lives now (its data folder), None when not installed
    on_disk_gb: float = 0.0          # weights already in the model's folder (a partial download)
    notes: list[str] = field(default_factory=list)

    @property
    def remaining_gb(self) -> float:
        return sum(p.gb for p in self.parts)

    @property
    def only_helpers(self) -> bool:
        return bool(self.parts) and all(p.kind != "weights" for p in self.parts)

    def _names(self) -> str:
        names = [f"{p.label} ({p.gb:.1f} GB)" for p in self.parts]
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

    def headline(self) -> str:
        """One plain sentence for the size line of the Folder step."""
        if self.own_files:
            if not self.parts:
                return "Your model files are used where they are. Nothing needs to be downloaded."
            return f"Your model files are used where they are. Strata still needs to fetch {self._names()}."
        if self.installed and not self.parts:
            return f"Already downloaded ({self.total_gb:.0f} GB on this computer). Nothing more to download."
        if self.installed and self.only_helpers:
            first = self.parts[0]
            if len(self.parts) == 1:
                return f"The model is already downloaded ({self.total_gb:.0f} GB). Only {first.label} is missing ({first.gb:.1f} GB)."
            return f"The model is already downloaded ({self.total_gb:.0f} GB). Only {self._names()} are missing."
        if self.installed:
            return (f"The model's files are no longer all on this computer. About {self.remaining_gb:.0f} GB would be downloaded "
                    f"({self._names()}).")
        if self.on_disk_gb >= 1:
            return (f"{self.on_disk_gb:.0f} GB of the model is already in the folder. About {self.remaining_gb:.0f} GB left to download "
                    f"(the whole model is {self.total_gb:.0f} GB).")
        return f"Not downloaded yet: about {self.remaining_gb:.0f} GB to download."

    def question(self) -> tuple[str, str]:
        """(title, text) for the "Download?" pop-up before a start that needs a download."""
        gb = self.remaining_gb
        if self.only_helpers:
            return (f"Download {self._names()}?",
                    "The model itself is already on this computer and is not downloaded again. Strata needs this extra file to start "
                    "with the choices you made; an interrupted download continues where it stopped.")
        return (f"Download about {gb:.0f} GB?",
                f"This model is not on this computer yet. Strata will download about {gb:.0f} GB now; "
                "an interrupted download continues where it stopped.")


#: The families whose setup also fetches the draft layer and the picture reader after the model (setup.py steps 5 and 6). NVFP4 packs bring
#: theirs inside the download, Qwen3.6 has its own engine path.
HELPER_FAMILIES = ("qwen", "swift", "coder", "unsloth", "gyro")


def resolved_drafting(drafting: str | None, family: str | None) -> str:
    """The drafter a choice means for this family on this computer: ``dflash2`` | ``mtp`` | ``off`` (strata_control/drafting.py: Auto is DFlash2 where the engine
    has it). Never raises; an unknown computer is the MTP."""
    try:
        from strata_control import drafting as _d
        backend, archs = _d.detect_hardware()
        return _d.choice_of(drafting, family=family, backend=backend, archs=archs)
    except Exception:                                        # noqa: BLE001
        return "mtp"


def drafting_for_family(choice: str | None, family: str | None) -> str | None:
    """The Drafting choice to keep when the model changes to `family`: DFlash2/MTP/Off belong to Qwen3.8-Flash-Next (qwen, gyro, unsloth); any other model (DeepSeek, which has its
    own DSpark and no MTP or DFlash2; Coder; Swift; NVFP4) goes back to Auto (None), so a choice made for one model never rides along to another. An unknown family keeps it."""
    from strata_control import drafting as _d
    return choice if (not family or family in _d.SUPPORTED_FAMILIES) else None


def drafting_to_save(key: str, saved: object) -> str | None:
    """What the Drafting step writes into the launch settings for the row the person chose. Auto stays "nobody chose" (None: a saved preset writes nothing), except when the
    model's own settings file already holds another choice (the app's Models screen writes its Drafting choice there and a launch starts from that file): then Auto is
    written as "auto", so it overrides the older choice instead of silently running it."""
    if key != "auto":
        return key
    from strata_control import drafting as _d
    return "auto" if saved is not None and _d.normalize(saved) != "auto" else None


def mtp_choice(mtp: str | None, drafting: str | None = None) -> str | None:
    """The draft-layer choice for what to fetch: off when the person said --mtp off, or the drafter is off or DFlash2 (``drafting`` is the resolved drafter,
    resolved_drafting(); nothing then uses the layer, so a new install does not download it), else as chosen."""
    if str(drafting or "").strip().lower() in ("off", "dflash2"):
        return "off"
    return mtp


def mmproj_name(catalog, family: str | None) -> str:
    try:
        _m, families, _c, _s = catalog.tables()
        name = (families.get(family or "") or {}).get("mmproj")
        if name:
            return str(name)
        mod = catalog.strata.setup_module()
        if mod is not None and getattr(mod, "MMPROJ", None):
            return str(mod.MMPROJ)
    except Exception:                                        # noqa: BLE001
        pass
    return "mmproj-Qwen3.8-Flash-Next-BF16.gguf"


def assess(catalog, family: str | None, model: str | None, *, vision: str = "off", model_path: str | None = None,
           data_dir: str | Path | None = None, config: Path | None = None, desc: dict | None = None, rows: list | None = None,
           mtp: str | None = None, drafting: str | None = None) -> ModelState:
    """The state of `family model` for these choices. `config` / `desc` are the installed run config and its description (looked up when
    not given)."""
    st = ModelState()
    mtp = mtp_choice(mtp, drafting)
    row = next((r for r in (rows if rows is not None else catalog.rows()) if r.family == family and r.model == model), None) if (family and model) else None
    st.total_gb = float(row.download_gb) if row else 0.0
    # DeepSeek reads pictures with its own engine's tower: nothing separate to fetch or to start (the tower's file is part of its setup)
    multi = getattr(getattr(catalog, "strata", None), "multi_family", None)
    own_tower = bool(family) and multi is not None and bool(multi(family))
    wants_pictures = vision != "off" and (row is None or row.images) and not own_tower
    data = Path(data_dir) if data_dir else None
    if model_path:
        st.own_files = True
        first = _first_gguf(Path(model_path))
        beside = discovery.prepared_beside(first) if first else {}
        st.folder = beside.get("data_dir")
        if beside.get("pack") is not None and beside.get("mtp") is None and mtp != "off":
            st.parts.append(Part("mtp", "the draft layer", MTP_FETCH_GB))
        if wants_pictures and beside.get("pack") is not None and not beside.get("mmproj"):
            st.parts.append(Part("mmproj", "the picture reader", MMPROJ_GB))
        return st
    if config is None and family and model:
        config = catalog.installed_config(catalog.tag_of(family, model))
    cfg = None
    if config is not None:
        try:
            import json
            cfg = json.loads(Path(config).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            cfg = None
    if cfg is not None:
        fp = discovery.footprint(cfg, Path(config))
        st.installed = True
        st.folder = fp.data_dir
        gone = [m for m in fp.missing() if m.kind == "weights"]
        if gone:
            st.installed = False
            st.parts.append(Part("weights", "the model", max(st.total_gb, 0.0)))
            return st
        for m in fp.missing():
            if m.kind == "mtp" and mtp != "off":
                st.parts.append(Part("mtp", "the draft layer", MTP_FETCH_GB))
            elif m.kind == "mmproj" and wants_pictures:
                st.parts.append(Part("mmproj", "the picture reader", MMPROJ_GB))
        images = (desc or {}).get("images") if desc is not None else ("cpu" if fp.mmproj else "off")
        if wants_pictures and (images in (None, "off")) and not any(p.kind == "mmproj" for p in st.parts):
            base = st.folder / "models" if st.folder else data
            have = (base / mmproj_name(catalog, family)) if base else None
            if have is None or not have.exists():
                st.parts.append(Part("mmproj", "the picture reader", MMPROJ_GB))
        return st
    # not installed: whatever of the weights is in the model's folder counts
    if family and model:
        base = data or Path(catalog.data_dir())
        folder = catalog.model_folder(base, family, model)
        got = 0
        try:
            got = sum(f.stat().st_size for f in folder.glob("*.gguf") if not f.name.lower().startswith("mmproj"))
        except OSError:
            pass
        st.on_disk_gb = got / 1e9
        st.parts.append(Part("weights", "the model", max(st.total_gb - st.on_disk_gb, 0.0)))
        if family in HELPER_FAMILIES:
            # setup fetches the draft layer and the picture reader after the model, so the question and the disk check count them too
            if mtp != "off" and not (base / "mtp" / "rt" / "experts.bin").is_file():
                st.parts.append(Part("mtp", "the draft layer", MTP_FETCH_GB))
            if wants_pictures and not _has_reader(base):
                st.parts.append(Part("mmproj", "the picture reader", MMPROJ_GB))
    return st


def _has_reader(base: Path) -> bool:
    for folder in (base / "models", base / "Models"):
        try:
            if any(folder.glob("mmproj*.gguf")):
                return True
        except OSError:
            pass
    return False


def _first_gguf(path: Path) -> Path | None:
    try:
        if path.is_file():
            return path
        found = sorted(p for p in path.glob("*.gguf") if not p.name.lower().startswith("mmproj"))
        return found[0] if found else None
    except OSError:
        return None
