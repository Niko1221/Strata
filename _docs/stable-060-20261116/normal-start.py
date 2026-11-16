"""Read-only normal-profile verification after an independently authorised start.

This controller does not launch or stop the persistent user app. A live app must
not become a child of a short-lived QA Job. No configuration PUT is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import sys
import urllib.request

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("normal060_delivery", HERE / "delivery.py")
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)
release, native = delivery.release, delivery.native
PROFILE_PATH_ENV = (
    "USERPROFILE",
    "HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "STRATA_HOME",
    "STRATA_SETTINGS_FILE",
    "STRATA_TERMINAL_SETTINGS_FILE",
    "STRATA_DESKTOP_DATA_DIR",
    "STRATA_PROFILE_DIR",
    "STRATA_RUNTIME_DIR",
    "STRATA_COMPONENTS_DIR",
    "STRATA_INSTALL_DIR",
    "STRATA_STATE_DIR",
    "STRATA_DATA_DIR",
    "STRATA_PRESET_DIR",
    "STRATA_WORKSPACE",
    "STRATA_MODELS_DIR",
    "STRATA_DOCUMENTS_DIR",
    "STRATA_TUNE_DIR",
    "STRATA_TIMESKIP_DIR",
    "STRATA_APP_SESSION_FILE",
    "STRATA_HARNESS_DIR",
    "STRATA_CONTROL_DIR",
    "STRATA_ENGINE_BUNDLE_DIR",
    "STRATA_NATIVE_REPO",
    "STRATA_APPDIR",
    "STRATA_DIST",
    "DSH_HOME",
    "DSH_AGENTS_HOME",
    "HERMES_HOME",
    "WEBVIEW2_USER_DATA_FOLDER",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_DOCUMENTS_DIR",
)


def path_identity(value: str, cwd: str, *, user: str | None = None) -> dict:
    """Keep lexical and physical identity; case is not identity on this Windows host."""
    release.require(
        isinstance(value, str) and bool(value) and not any(c in value for c in "\r\n\0"),
        "A nonempty profile path is required",
    )
    if user and (value == "~" or value.startswith(("~/", "~\\"))):
        value = str(Path(user) / value[2:]) if len(value) > 1 else user
    path = Path(value)
    if not path.is_absolute():
        release.require(not path.anchor, "Ambiguous rooted or drive-relative profile path")
        path = Path(cwd) / path
    release.require(path.is_absolute(), "Absolute profile path resolution required")
    return {
        "lexical": os.path.normcase(os.path.normpath(str(path))),
        "physical": os.path.normcase(str(path.resolve(strict=False))),
    }


def filtered_paths(environment: dict, cwd: str, *, agents_cwd: str | None = None) -> dict:
    """Only allowlisted nonsecret path values survive the process boundary."""
    selected = {key: environment.get(key) for key in PROFILE_PATH_ENV}
    # HOMEDRIVE/HOMEPATH are fragments, not independently resolvable paths.
    for key in ("HOMEDRIVE", "HOMEPATH"):
        value = selected[key]
        selected[key] = os.path.normcase(value) if value is not None else None
    for key in set(PROFILE_PATH_ENV) - {"HOMEDRIVE", "HOMEPATH"}:
        value = selected[key]
        if key == "DSH_AGENTS_HOME" and isinstance(value, str) and not value.strip():
            continue  # The pinned SDK treats empty/whitespace-only overrides as absent.
        if value:
            if key == "HERMES_HOME":
                release.require(
                    not value.startswith("~") or value == "~" or value.startswith(("~/", "~\\")),
                    "Only current-user tilde is valid for HERMES_HOME",
                )
            selected[key] = path_identity(
                value,
                agents_cwd if key == "DSH_AGENTS_HOME" and agents_cwd else cwd,
                user=(
                    environment.get("USERPROFILE")
                    if key
                    in {
                        "HERMES_HOME",
                        "DSH_AGENTS_HOME",
                        "STRATA_PROFILE_DIR",
                        "STRATA_RUNTIME_DIR",
                        "STRATA_PRESET_DIR",
                    }
                    else None
                ),
            )
    return selected


def effective_profile(
    shell_env: dict,
    host_env: dict,
    shell_cwd: str,
    host_cwd: str,
    app_paths: dict,
    workspace: str,
    *,
    legacy_agents: bool = False,
) -> dict:
    """Observe actual host fallback outcomes; derive the remaining Windows path rules."""
    user = host_env.get("USERPROFILE")
    release.require(bool(user), "Explicit observed Windows USERPROFILE required")
    local = host_env.get("LOCALAPPDATA") or str(Path(user) / "AppData/Local")
    roaming = host_env.get("APPDATA") or str(Path(user) / "AppData/Roaming")
    components = host_env.get("STRATA_COMPONENTS_DIR") or str(Path(local) / "Strata/components")
    profiles = host_env.get("STRATA_PROFILE_DIR") or str(Path(roaming) / "Strata/profiles")
    profiles_path = path_identity(profiles, host_cwd, user=user)["lexical"]
    components_path = path_identity(components, host_cwd)["lexical"]
    host_paths = {
        "dataDir": app_paths.get("dataDir"),
        "modelsDir": app_paths.get("modelsDir"),
        "workspace": workspace,
        "components": components,
        "controlSettings": host_env.get("STRATA_SETTINGS_FILE")
        or str(Path(components_path).parent / "app-settings.json"),
        "terminalSettings": host_env.get("STRATA_TERMINAL_SETTINGS_FILE")
        or str(Path(roaming) / "Strata/settings.json"),
        "profiles": profiles,
        "runtime": host_env.get("STRATA_RUNTIME_DIR") or str(Path(local) / "Strata/runtime"),
        "presets": host_env.get("STRATA_PRESET_DIR") or str(Path(profiles_path).parent / "presets"),
        "tuning": host_env.get("STRATA_TUNE_DIR") or str(Path(local) / "Strata/tuning"),
        "timeSkip": host_env.get("STRATA_TIMESKIP_DIR") or str(Path(local) / "Strata/time-skip"),
    }
    # DATA_DIR is returned by /api/setup after store.resolveHome has selected its
    # writable fallback. Re-running store.mjs would create files and is forbidden.
    data = host_paths["dataDir"]
    release.require(isinstance(data, str) and bool(data), "Host effective dataDir unavailable")
    host_paths.update(
        nativePreferences=str(Path(data) / "native-host.json"),
        harnessHome=str(Path(data) / "agent/home"),
        skills=str(Path(data) / "skills"),
    )
    # HarnessProcess starts in data/agent/run. The upstream SDK expands only
    # current-user tilde prefixes, keeps nonblank whitespace, and falls back to
    # USERPROFILE/.agents for an absent, empty or whitespace-only override.
    agents_cwd = str(Path(data) / "agent/run")
    agents_override = host_env.get("DSH_AGENTS_HOME")
    host_paths["agentsHome"] = (
        agents_override
        if isinstance(agents_override, str) and agents_override.strip()
        else str(Path(user) / ".agents")
    )
    if legacy_agents:
        # Installed 0.5.9 strips this environment override. Refuse a baseline
        # whose unchanged environment would select another root in 0.6.0.
        legacy_root = str(Path(user) / ".agents")
        release.require(
            path_identity(host_paths["agentsHome"], agents_cwd, user=user)
            == path_identity(legacy_root, agents_cwd),
            "Legacy agents-home override would change the effective instruction root",
        )
        host_paths["agentsHome"] = legacy_root
    shell_user = shell_env.get("USERPROFILE")
    release.require(bool(shell_user), "Observed shell USERPROFILE required")
    desktop = shell_env.get("STRATA_DESKTOP_DATA_DIR") or str(
        Path(shell_env.get("LOCALAPPDATA") or str(Path(shell_user) / "AppData/Local")) / "Strata"
    )
    return {
        "shellCwd": path_identity(shell_cwd, shell_cwd),
        "hostCwd": path_identity(host_cwd, host_cwd),
        "shellPaths": filtered_paths(shell_env, shell_cwd, agents_cwd=agents_cwd),
        "hostPaths": filtered_paths(host_env, host_cwd, agents_cwd=agents_cwd),
        "effectiveHost": {
            key: path_identity(
                value,
                agents_cwd if key == "agentsHome" else host_cwd,
                user=user if key in {"profiles", "runtime", "presets", "agentsHome"} else None,
            )
            for key, value in host_paths.items()
        },
        "effectiveShellDesktop": path_identity(desktop, shell_cwd),
        "packagedHostEntry": path_identity(str(delivery.APP / "host/server/index.mjs"), host_cwd),
    }


def read_api(base: str, path: str) -> dict | list:
    request = urllib.request.Request(base + path, method="GET", headers={"Origin": base})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def choices(base: str) -> dict:
    """Stable settings and private skill content; only a digest leaves memory."""
    learning = read_api(base, "/api/skills/learning")
    skills = read_api(base, "/api/skills")
    release.require(isinstance(skills, list), "Skill response must be a list")
    return {
        "tools": read_api(base, "/api/agent/tools"),
        "learning": {name: learning[name] for name in ("enabled", "autoApply")},
        "skills": sorted(
            ({name: row.get(name) for name in ("id", "enabled", "body", "learning")} for row in skills),
            key=lambda row: str(row["id"]),
        ),
    }


def choices_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def owned_processes(args: argparse.Namespace) -> tuple:
    """Bind PID, creation time, executable, lineage and listening socket."""
    import psutil

    shell, host = psutil.Process(args.shell_pid), psutil.Process(args.host_pid)
    release.require(
        shell.create_time() == args.shell_created and host.create_time() == args.host_created,
        "PID was reused or owner changed",
    )
    release.require(
        Path(shell.exe()) == delivery.APP / "strata-desktop.exe"
        and Path(host.exe()) == delivery.APP / "node/node.exe",
        "Unexpected app owner executable",
    )
    release.require(shell.pid in [p.pid for p in host.parents()], "Host does not belong to this shell")
    release.require(len(shell.cmdline()) == 1, "Normal shell must have no debug arguments")
    host_args = host.cmdline()
    release.require(
        len(host_args) == 2
        and Path(host_args[0]) == delivery.APP / "node/node.exe"
        and Path(host_args[1]) == delivery.APP / "host/server/index.mjs",
        "Fixed packaged host CLI required; raw arguments suppressed",
    )
    environment = shell.environ()
    release.require(
        not environment.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS")
        and not environment.get("STRATA_AGENT")
        and not environment.get("TAURI_CONFIG"),
        "Normal shell carries debug or injected configuration",
    )
    listeners = [
        c
        for c in psutil.net_connections(kind="tcp")
        if c.status == psutil.CONN_LISTEN and c.laddr.port == args.port
    ]
    release.require(
        listeners and all(c.pid == host.pid and c.laddr.ip == "127.0.0.1" for c in listeners),
        "Loopback port ownership differs",
    )
    return shell, host


def profile_identity(args: argparse.Namespace, base: str, *, legacy_agents: bool = False) -> dict:
    shell, host = owned_processes(args)
    # Neither full process environment nor API response is retained or logged.
    shell_env = {key: value for key, value in shell.environ().items() if key in PROFILE_PATH_ENV}
    host_env = {key: value for key, value in host.environ().items() if key in PROFILE_PATH_ENV}
    app = read_api(base, "/api/setup")["app"]
    app_paths = {key: app[key] for key in ("dataDir", "modelsDir")}
    workspace = read_api(base, "/api/workspace")["path"]
    result = effective_profile(
        shell_env, host_env, shell.cwd(), host.cwd(), app_paths, workspace, legacy_agents=legacy_agents
    )
    owned_processes(args)
    return result


def original_baseline(args: argparse.Namespace) -> dict:
    record = delivery.prior(args, "original", "original-profile")
    path = Path(record.get("privateProof", ""))
    release.require(
        path.name == "original-profile.json"
        and path.parent.parent == delivery.PRIVATE
        and re.fullmatch(r"original-profile-r[1-9][0-9]*", path.parent.name),
        "Private original proof path required",
    )
    proof = delivery.pinned(path, record.get("privateProofSHA256"))
    release.require(
        proof.get("observedVersion") == "0.5.9"
        and proof.get("choicesSHA256") == args.choices_sha256
        and proof.get("profileIdentitySHA256") == choices_digest(proof["profileIdentity"]),
        "Original profile or choices proof differs",
    )
    return proof


def capture(args: argparse.Namespace, work: Path) -> dict:
    base = f"http://127.0.0.1:{args.port}"
    owned_processes(args)
    release.require(
        read_api(base, "/api/app/build") == {"host": "0.5.9", "shell": "0.5.9"},
        "Capture requires the original 0.5.9 app",
    )
    profile = profile_identity(args, base, legacy_agents=True)
    selected = choices(base)
    release.require(
        profile_identity(args, base, legacy_agents=True) == profile and choices(base) == selected,
        "Original profile changed during capture",
    )
    private = delivery.PRIVATE / ("original-profile-" + args.attempt)
    delivery.private_directory(private)
    proof = {
        "observedVersion": "0.5.9",
        "profileIdentity": profile,
        "profileIdentitySHA256": choices_digest(profile),
        "choicesSHA256": choices_digest(selected),
    }
    path = private / "original-profile.json"
    release.save(path, proof)
    return {
        "privateProof": str(path),
        "privateProofSHA256": release.sha(path),
        "choicesSHA256": proof["choicesSHA256"],
        "profileIdentitySHA256": proof["profileIdentitySHA256"],
        "settingsWritten": False,
        "observationOnly": True,
        "observedVersion": "0.5.9",
    }


def verify(args: argparse.Namespace, work: Path) -> dict:
    original = original_baseline(args)
    installed = delivery.prior(args, "install")
    release.require(
        installed.get("installed") is True and installed.get("exactPayloadMatch") is True,
        "Successful protected install required",
    )
    release.require(delivery.payload(delivery.APP) == installed["payload"], "Installed bytes changed")
    owned_processes(args)
    base = f"http://127.0.0.1:{args.port}"
    release.require(
        profile_identity(args, base) == original["profileIdentity"],
        "Original effective profile identity differs",
    )
    release.require(
        read_api(base, "/api/app/build") == {"host": "0.6.0", "shell": "0.6.0"}, "Normal app version differs"
    )
    before = choices(base)
    release.require(choices_digest(before) == args.choices_sha256, "Original choices or skills differ")
    for route in ("/api/native", "/api/runtime"):
        release.require(
            read_api(base, route).get("instances") == [], "An active runtime requires separate owner handling"
        )
    release.require(
        read_api(base, "/api/chat/active").get("turns") == [], "A new active turn requires separate handling"
    )
    engine = read_api(base, "/api/health")["engine"]
    release.require(
        engine.get("status") == "stopped" and engine.get("running") is False, "Engine not unloaded"
    )
    release.require(choices(base) == before, "Choices changed during normal observation")
    release.require(
        profile_identity(args, base) == original["profileIdentity"],
        "Profile changed during normal observation",
    )
    owned_processes(args)
    return {
        "installerSHA256": installed["installerSHA256"],
        "normalStartup": True,
        "settingsWritten": False,
        "originalChoicesAndSkillsUnchanged": True,
        "originalEffectiveProfileUnchanged": True,
        "profileIdentitySHA256": original["profileIdentitySHA256"],
        "choicesSHA256": args.choices_sha256,
        "modelUnloaded": True,
        "activeTurns": 0,
        "shellPID": args.shell_pid,
        "hostPID": args.host_pid,
        "persistentAppStartedByThisController": False,
        "observationOnly": True,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = native.parser(__doc__)
    cli.add_argument("--install-receipt", type=Path)
    cli.add_argument("--install-receipt-sha256")
    cli.add_argument("--choices-sha256")
    cli.add_argument("--capture-original", action="store_true")
    cli.add_argument("--original-receipt", type=Path)
    cli.add_argument("--original-receipt-sha256")
    cli.add_argument("--shell-pid", type=int)
    cli.add_argument("--shell-created", type=float)
    cli.add_argument("--host-pid", type=int)
    cli.add_argument("--host-created", type=float)
    cli.add_argument("--port", type=int)
    args = cli.parse_args(argv)
    args.phase = "original-profile" if args.capture_original else "normal-start"
    if args.execute:
        release.require("--attempt" in argv, "Explicit fresh attempt required")
        release.require(
            args.capture_original or re.fullmatch(r"[a-f0-9]{64}", args.choices_sha256 or ""),
            "Original choice digest required",
        )
        release.require(
            all(
                value and value > 0
                for value in (args.shell_pid, args.host_pid, args.shell_created, args.host_created)
            )
            and args.port is not None
            and 1 <= args.port <= 65535,
            "Explicit fresh owners and port required",
        )
    return native.run_controller(
        args, HERE / "normal-start.py", capture if args.capture_original else verify, argv
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.error(
            "Normal observation failed (%s); app left unchanged; raw error suppressed", type(error).__name__
        )
        raise SystemExit(1) from None
