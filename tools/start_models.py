"""Start Strata's page without loading native weights; optionally start installed Ollama."""
from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from strata_mcp import Strata, Tools, spawn_detached, proc_identity  # noqa: E402


def reachable(url, api_key=None):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(url)
        if api_key:
            request.add_header('Authorization', 'Bearer ' + api_key)
        with opener.open(request, timeout=3) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--model', help='installed config tag or model name; default: most recently used config')
    parser.add_argument('--start-ollama', action='store_true', help='also start Ollama if installed')
    return parser.parse_args(argv)


def model_config(manager, requested):
    if requested is not None:
        config = manager.find_config(requested)
    else:
        config = next((path for path in manager.configs()
                       if not path.name.lower().endswith('.shared-settings.json')), None)
        if config is None:
            raise RuntimeError('No native model configuration is installed. Run Strata setup first.')
    if config.name.lower().endswith('.shared-settings.json'):
        raise RuntimeError('Select an installed model configuration, not its shared sampling settings.')
    return config


def main(argv=None):
    args = parse_args(argv)
    log_dir = ROOT / ".local"
    log_dir.mkdir(exist_ok=True)
    manager = Strata(ROOT)
    def api_key():
        key = os.environ.get('STRATA_API_KEY')
        if key:
            return key
        state = manager.state('server')
        name = state.get('config') if isinstance(state, dict) else None
        if (isinstance(name, str) and Path(name).name == name and name.startswith('strata-')
                and name.endswith('.json') and not name.lower().endswith('.shared-settings.json')
                and (ROOT / name).is_file()):
            return manager.read_config(ROOT / name).get('api_key') or None
        return manager.api_key_for(8080)
    def frontend_health():
        return reachable('http://127.0.0.1:8080/health', api_key())
    if args.stop:
        if frontend_health():
            request = urllib.request.Request("http://127.0.0.1:8080/api/providers/select",
                data=b'{"id":null}', headers={"Content-Type": "application/json"})
            key = api_key()
            if key:
                request.add_header("Authorization", "Bearer " + key)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=60) as response:
                json.load(response)
        original_api_key_for = manager.api_key_for
        active_key = api_key()
        manager.api_key_for = lambda port: active_key if port == 8080 else original_api_key_for(port)
        try:
            print(Tools(manager).strata_stop()["summary"])
        finally:
            manager.api_key_for = original_api_key_for
        return 0
    if args.start_ollama and not reachable("http://127.0.0.1:11434/api/version"):
        binary = shutil.which('ollama')
        if not binary and os.name == 'nt':
            installed = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
            binary = str(installed) if installed.is_file() else None
        if not binary:
            raise RuntimeError("Ollama is missing. Install it from https://ollama.com before adding Ollama models.")
        env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434")
        spawn_detached([binary, "serve"], ROOT, log_dir / "ollama-provider.log", env)
    health = frontend_health()
    if health and health.get("service") != "strata":
        raise RuntimeError("Another application is using port 8080.")
    if not health:
        tracked = manager.tracked_server()
        if tracked and not tracked.get("ended"):
            raise RuntimeError("Strata is already starting. Wait for it to become ready.")
        config = model_config(manager, args.model)
        cmd = [manager.run_python(), str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
               str(config), "--host", "127.0.0.1", "--port", "8080", "--lazy"]
        process = spawn_detached(cmd, ROOT, manager.state_dir / "server.log",
                                 dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"))
        manager.save_state("server", {"pid": process.pid, "ident": proc_identity(process.pid), "port": 8080,
                           "model": config.stem.removeprefix('strata-'), "config": config.name,
                           "log": str(manager.state_dir / "server.log"), "started": time.strftime("%Y-%m-%d %H:%M:%S")})
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if frontend_health():
                break
            if process.poll() is not None:
                raise RuntimeError("Strata could not start. Check .strata-mcp/server.log.")
            time.sleep(0.5)
        else:
            raise RuntimeError("Strata is still starting. Check STATUS-Strata.bat.")
    if not args.no_browser:
        webbrowser.open("http://127.0.0.1:8080")
    print("Ready: http://127.0.0.1:8080 - select a connection or add a model in About.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
