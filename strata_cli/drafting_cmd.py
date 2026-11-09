"""``strata drafting plan|fetch``: the DFlash2 drafter file (strata_control/drafting_fetch.py). The app calls it with ``--output jsonl``.

With ``--config`` (a model's run config) and ``--mode`` (the choice, default: the config's own ``drafting``) the command decides what this
model needs: ``needed`` is true only when the choice would use the drafter and its file is not here yet. ``fetch --if-needed`` then downloads
only in that case, so the app can call it before every load without asking twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from strata_control import acquisition, drafting
from strata_control.discovery import MTP_FETCH_GB


def add_parser(sp) -> None:
    drafting_p = sp.add_parser("drafting", help="the DFlash2 drafter file: what it would download, and fetch it")
    d_sp = drafting_p.add_subparsers(dest="drafting_command", required=True)
    for name, text in (("plan", "say what fetching would download; nothing is changed"),
                       ("fetch", "download the drafter (pinned, checked) and pack it for the engine")):
        one = d_sp.add_parser(name, help=text)
        one.add_argument("--data-dir", metavar="DIR", help="the model data folder (default: the one Strata uses)")
        one.add_argument("--config", metavar="FILE", help="a model's run config: its data folder, model and drafting choice decide what is needed")
        one.add_argument("--family", metavar="NAME", help="the model family when there is no run config yet (qwen, gyro, coder, ...)")
        one.add_argument("--mode", choices=drafting.MODES, help="the choice to decide for (default: the run config's own)")
        one.add_argument("--no-split", action="store_true",
                         help="decide without the all-cards split engine, as if it will not take the start (a person's choice; the app asks this only after the draft layer for the split could not be downloaded, so the plan is DFlash2's again)")
        one.add_argument("--engines-unloading", action="store_true",
                         help=argparse.SUPPRESS)
        if name == "fetch":
            one.add_argument("--if-needed", action="store_true", help="download only when the choice would use the drafter and it is not here")
        one.add_argument("--output", choices=("human", "json", "jsonl"), default="human")


def _hardware() -> tuple[str | None, list[str]]:
    return drafting.detect_hardware()


def _split_expected(cfg: dict, args: list, unloading: bool = False) -> bool:
    """Would the all-cards split engine take a start of this run config (strata_control/split_engine.expected, asked before the draft layer is on disk)?"""
    try:
        from strata_control import split_engine as SE
        flags = [str(x) for x in args]
        flag = lambda name: next((flags[i + 1] for i, x in enumerate(flags[:-1]) if x == name), None)   # noqa: E731
        gpus = cfg.get("gpu")
        cards = [int(g) for g in gpus] if isinstance(gpus, list) else ([int(gpus)] if isinstance(gpus, int) and not isinstance(gpus, bool) else None)
        probe = None
        if unloading:                                        # memory the app's own engines hold is given back before the new start: the cards count by their size (as Runner.card_probe does)
            import dataclasses
            probe = lambda idx: [dataclasses.replace(c, free_mib=None) for c in SE.probe_cards(idx)]   # noqa: E731
        return SE.expected({"split_engine": cfg.get("split_engine"), "drafting": cfg.get("drafting"), "user_args": cfg.get("user_args"),
                            "replicas": cfg.get("replicas")}, indexes=cards, native=flag("--native"),
                           context=int(flag("--max-context") or SE.TESTED_CONTEXT), kv=flag("--kv"), probe=probe)
    except Exception:                                        # noqa: BLE001 - an early question is never a reason to stop
        return False


def _inputs(a) -> dict:
    cfg: dict = {}
    if a.config:
        try:
            cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            cfg = {}
    if a.data_dir:
        data = Path(a.data_dir)
    else:
        data = drafting.data_dir_of(cfg.get("args") or [])
        if data is None:
            from strata_cli.launch.catalog import Catalog
            data = Catalog().data_dir()
    backend, archs = _hardware()
    args = cfg.get("args") or []
    mode = a.mode if a.mode else cfg.get("drafting")
    if cfg.get("mtp") is False and drafting.normalize(mode) == "auto":
        mode = "off"                                         # the Draft layer switch is off and no drafter was chosen since
    elif drafting.normalize(mode) in ("auto", "mtp") and not getattr(a, "no_split", False) and _split_expected({**cfg, "drafting": mode}, args, bool(getattr(a, "engines_unloading", False))):
        a._split = True                                      # the plan says so (the app's Models screen and its load read it); asked with the choice in hand, as the Terminal does
        mode = "mtp"                                         # the all-cards split engine runs the MTP draft layer: the app fetches that, not DFlash2's drafter
    return {"mode": mode, "klass": drafting.card_class(archs), "model_name": cfg.get("model_name"), "family": a.family, "data_dir": data,
            "backend": backend, "archs": archs, "mtp_here": drafting.mtp_ready(args) if cfg else None,
            "blocked": drafting.blocked_by(args, slots=cfg.get("parallel") if isinstance(cfg.get("parallel"), int) else 0, named=drafting.normalize(mode) == "dflash2", backend=backend)}


def _signals(a, want: dict) -> dict:
    """The arguments and the one card's memory drafting.tuned_applies reads (as the server reads them for the start); {} when there is no run config to read them from."""
    try:
        cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig")) if a.config else {}
        args = [str(x) for x in cfg.get("args") or []]
        if cfg.get("user_args"):
            from strata_control import user_args
            args = user_args.merge(args, cfg["user_args"])
        return {"args": args, "vram_mib": drafting.card_vram_mib(cfg, want["backend"], want["archs"])}
    except Exception:                                        # noqa: BLE001 - no signal: the tuned drafter is not offered
        return {"args": [], "vram_mib": None}


def run(a) -> int:
    from strata_control import drafting_fetch      # here, not at the top: it pulls in the network code, which a started engine's command line must not load
    machine = a.output in ("json", "jsonl")
    want = _inputs(a)
    folder = want["data_dir"]
    sig = _signals(a, want)
    needed = drafting.needs_pack(**want, **sig) or drafting.needs_tuned(**want, **sig)       # (the tuned drafter is a download of its own, where the set and the card use it)
    mtp_needed = drafting.needs_mtp(**want, **sig)
    if a.drafting_command == "plan":
        decision = drafting.decide(**want, **sig)
        support = drafting.hardware_support(want["backend"], want["archs"], want["model_name"])
        # serves: this computer's engine can run DFlash2 (the build has the code and the card is one it is made for); the app offers the choice only then
        result = {**drafting_fetch.plan(folder, drafting.installed_engine_exe(), model_name=want["model_name"], backend=want["backend"], archs=want["archs"], **sig), "needed": needed, "mtp_here": want["mtp_here"], "mtp_needed": mtp_needed, "mtp_gb": MTP_FETCH_GB if mtp_needed else 0.0,
                  "serves": drafting.ENGINE_SERVES_DFLASH2 and support.can_run, "drafter": decision.drafter,
                  "choice": drafting.choice_of(want["mode"], model_name=want["model_name"], family=want["family"], backend=want["backend"], archs=want["archs"], **sig),
                  "split": bool(getattr(a, "_split", False)), "note": decision.note, "line": drafting.summary_line(decision), "support": support.level, "support_line": support.line}
        if result["split"]:                                # the Terminal's words for the same plan (runner.drafting_plan_line)
            from strata_control import split_engine as SE
            result["line"] = SE.split_plan_line(result["mtp_gb"] or None)
        if (a.config or a.mode) and not needed:           # asked about a model: nothing to download unless the choice uses the drafter
            result["gb"] = 0.0
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, **result}, ensure_ascii=False))
        else:
            print(result["text"] if needed or not decision.note else result["line"])
        return 0
    if getattr(a, "if_needed", False) and not needed:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, "event": "result", "pack": None, "skipped": True}, ensure_ascii=False))
        else:
            print("Nothing to download.")
        return 0

    def show(step: str, done: int, total: int) -> None:
        if a.output == "jsonl":
            print(json.dumps({"event": "progress", "step": step, "done": done, "total": total}, ensure_ascii=False), flush=True)
        elif not machine and total:
            print(f"\r{step} {done / 1e9:.2f} of {total / 1e9:.2f} GB", end="", flush=True)

    try:
        out = drafting_fetch.ensure_for(folder, model_name=want["model_name"], backend=want["backend"], progress=show, archs=want["archs"], **sig)
        if out is None:                                    # (a set with a tuned drafter, and nothing answered: an explicit fetch says so)
            raise drafting_fetch.OfflineError("There is no internet connection, so the DFlash2 drafter can't be downloaded now.")
    except (drafting_fetch.DraftingError, acquisition.AcquisitionRefused) as exc:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": "drafting_failed", "message": str(exc)}}, ensure_ascii=False))
        else:
            print(f"\nstrata: {exc}", file=sys.stderr)
        return 5
    if machine:
        print(json.dumps({"schema_version": 1, "ok": True, "event": "result", "pack": str(out)}, ensure_ascii=False))
    else:
        print(f"\nThe DFlash2 drafter is ready: {out}")
    return 0
