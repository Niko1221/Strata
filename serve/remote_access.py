"""Keep this installation's opt-in remote gateway available while Strata runs."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import http.client
import ipaddress
import json
import os
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from serve.keepawake import KeepAwake

TAILSCALE = Path(r"C:\Program Files\Tailscale\tailscale.exe")


def funnel_ownership(configuration, port):
    """Funnel publishes the whole HTTPS port, so another path must never be exposed as a side effect."""
    if configuration is None:
        return "empty"
    if not isinstance(configuration, dict):
        return "other"
    web = configuration.get("Web") if configuration.get("Web") is not None else {}
    tcp_ports = configuration.get("TCP") if configuration.get("TCP") is not None else {}
    if not isinstance(web, dict) or not isinstance(tcp_ports, dict):
        return "other"
    foreground = configuration.get("Foreground")
    if foreground is not None and (
        not isinstance(foreground, dict) or
        any(funnel_ownership(child, port) != "empty" for child in foreground.values())
    ):
        return "other"
    tcp = tcp_ports.get("443")
    if tcp is not None and (not isinstance(tcp, dict) or not tcp.get("HTTPS") or tcp.get("TCPForward")):
        return "other"
    bindings = [value for host, value in web.items() if host.endswith(":443")]
    if not bindings:
        return "other" if tcp is not None else "empty"
    target = "http://127.0.0.1:" + str(port)
    for binding in bindings:
        handlers = binding.get("Handlers") if isinstance(binding, dict) else None
        if not isinstance(handlers, dict) or set(handlers) != {"/"} or \
                not isinstance(handlers["/"], dict) or handlers["/"].get("Proxy") != target:
            return "other"
    return "owned"


class RemoteMonitor:
    def __init__(self, runtime, health, public, repair, *, clock=time.monotonic, say=print):
        self.runtime = runtime
        self.health = health
        self.public = public
        self.repair = repair
        self.clock = clock
        self.say = say
        self.next_public_check = 0
        self.public_failures = 0
        self.last_message = None

    def report(self, message):
        if message != self.last_message:
            self.say(message)
            self.last_message = message

    def tick(self):
        try:
            if (self.runtime / "disabled").exists():
                self.public_failures = 0
                self.next_public_check = 0
                self.report("Remote access paused by Stop-Remote.ps1.")
                return
            healthy, reason = self.health()
            if not healthy:
                self.public_failures = 0
                self.report(reason + "; repairing remote access.")
                self.repair(False)
                self.next_public_check = 0
                return
            if self.clock() < self.next_public_check:
                return
            self.next_public_check = self.clock() + 60
            available, reason = self.public()
            if available is True:
                self.public_failures = 0
                self.report("Public API healthy; automatic recovery remains active.")
            elif available is None:
                self.public_failures = 0
                self.report(reason + "; leaving the configured tunnel in place.")
            else:
                self.public_failures += 1
                self.report(reason + "; checking again before republishing.")
                if self.public_failures >= 2:
                    self.report("Public ingress failed twice; republishing the owned Funnel endpoint.")
                    self.repair(True)
                    self.public_failures = 0
        except Exception as error:  # noqa: BLE001 - a failed check must not stop recovery
            self.report(f"Remote health check failed ({type(error).__name__}); will retry.")


class PublicConnection(http.client.HTTPSConnection):
    def __init__(self, host, address):
        self.address = address
        self.tls_context = ssl.create_default_context()
        super().__init__(host, timeout=8, context=self.tls_context)

    def connect(self):
        # Dial public ingress, not this PC's private Tailscale DNS address.
        sock = socket.create_connection((self.address, 443), self.timeout)
        try:
            self.sock = self.tls_context.wrap_socket(sock, server_hostname=self.host)
        except Exception:
            sock.close()
            raise


class RuntimeChecks:
    def __init__(self, root, runtime):
        self.root = root
        self.runtime = runtime
        self.host = None
        self.config = None

    def tailscale_json(self, *arguments):
        result = subprocess.run(
            [str(TAILSCALE), *arguments], check=True, capture_output=True,
            encoding="utf-8", timeout=10, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return json.loads(result.stdout)

    def health(self):
        try:
            self.config = json.loads((self.runtime / "gateway.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return False, "Gateway configuration unavailable"
        connection = http.client.HTTPConnection("127.0.0.1", self.config.get("port", 8787), timeout=3)
        try:
            connection.request("OPTIONS", "/v1/models", headers={"Authorization": "Bearer " + self.config["apiKey"]})
            response = connection.getresponse()
            if response.status != 405:
                return False, "Gateway credential check failed"
            if response.getheader('x-strata-gateway-version') != '2':
                return False, 'Gateway revision needs an update'
        except (OSError, http.client.HTTPException):
            return False, "Gateway unavailable"
        finally:
            connection.close()
        try:
            status = self.tailscale_json("status", "--json")
            if status.get("BackendState") != "Running":
                return False, "Tailscale is not connected"
            self.host = status["Self"]["DNSName"].rstrip(".")
            funnel = self.tailscale_json("funnel", "status", "--json")
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            return False, "Tailscale status unavailable"
        endpoint = self.host + ":443"
        if funnel_ownership(funnel, self.config.get("port", 8787)) != "owned":
            return False, "Funnel configuration is missing or includes unrelated HTTPS handlers"
        handler = funnel.get("Web", {}).get(endpoint, {}).get("Handlers", {}).get("/", {})
        target = "http://127.0.0.1:" + str(self.config.get("port", 8787))
        if not funnel.get("AllowFunnel", {}).get(endpoint) or handler.get("Proxy") != target:
            return False, "Public Funnel configuration missing"
        return True, "ready"

    def public(self):
        query = urllib.parse.urlencode({"name": self.host, "type": "A"})
        try:
            with urllib.request.urlopen("https://dns.google/resolve?" + query, timeout=8) as response:
                dns = json.loads(response.read(65536))
            if dns.get("Status") not in (0, 3):
                return None, "External DNS resolver temporarily unavailable"
            addresses = [row["data"] for row in dns.get("Answer", [])
                         if row.get("type") == 1 and ipaddress.ip_address(row["data"]).is_global]
        except (OSError, ValueError):
            return None, "External DNS check unavailable"
        if not addresses:
            return False, "Public DNS record missing"
        for address in addresses:
            connection = PublicConnection(self.host, address)
            try:
                connection.request("GET", "/v1/models", headers={
                    "Authorization": "Bearer " + self.config["apiKey"], "Connection": "close",
                })
                response = connection.getresponse()
                if response.status == 200:
                    return True, "ready"
            except (OSError, http.client.HTTPException):
                pass
            finally:
                connection.close()
        return False, "Public HTTPS ingress unavailable"

    def repair(self, refresh=False, *, startup=False):
        command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden",
                   "-File", str(self.root / "remote-access" / "Start-Remote.ps1")]
        if not startup:
            command.append("-Recovery")
        if refresh:
            command.append("-RefreshFunnel")
        try:
            with (self.runtime / "autostart.log").open("a", encoding="utf-8") as output:
                result = subprocess.run(
                    command, check=False, cwd=str(self.root), stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.STDOUT, timeout=180, creationflags=subprocess.CREATE_NO_WINDOW,
                )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False


class WindowsLease:
    """One monitor per user; optional parent handle binds a standalone monitor to Strata."""
    def __init__(self, runtime, parent_pid=None):
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.parent = self.kernel.OpenProcess(0x100000, False, parent_pid) if parent_pid else None
        self.valid_parent = not parent_pid or bool(self.parent)
        name = "Local\\StrataRemote-" + hashlib.sha256(str(runtime).lower().encode()).hexdigest()[:16]
        self.mutex = self.kernel.CreateMutexW(None, False, name)
        self.owned = False

    def __enter__(self):
        if self.valid_parent and self.mutex:
            while not self.stopped():
                result = self.kernel.WaitForSingleObject(self.mutex, 5000)
                if result in (0, 0x80):
                    self.owned = True
                    break
                if result != 0x102:
                    raise OSError('Remote watchdog singleton could not be acquired')
        return self

    def stopped(self, seconds=0):
        if not self.valid_parent:
            return True
        if self.parent:
            return self.kernel.WaitForSingleObject(self.parent, int(seconds * 1000)) != 0x102
        if seconds:
            time.sleep(seconds)
        return False

    def __exit__(self, *_):
        if self.owned:
            self.kernel.ReleaseMutex(self.mutex)
        for handle in (self.mutex, self.parent):
            if handle:
                self.kernel.CloseHandle(handle)


def monitor_remote_access(root: Path, parent_pid=None):
    if os.name != "nt":
        return
    runtime = Path(os.environ["LOCALAPPDATA"]) / "StrataRemote"
    runtime.mkdir(parents=True, exist_ok=True)

    def say(message):
        log = runtime / "watchdog.log"
        try:
            if log.exists() and log.stat().st_size > 1024 * 1024:
                log.replace(runtime / "watchdog.log.1")
            with log.open("a", encoding="utf-8") as output:
                output.write(time.strftime("%Y-%m-%d %H:%M:%S ") + message + "\n")
        except OSError:
            pass

    with WindowsLease(runtime, parent_pid) as lease:
        if not lease.owned or lease.stopped():
            return
        awake = KeepAwake.create(True, say=lambda *args, **kwargs: say(str(args[0])))
        held = False
        checks = RuntimeChecks(root, runtime)
        monitor = RemoteMonitor(runtime, checks.health, checks.public, checks.repair, say=say)
        try:
            if awake:
                awake.acquire()
                held = True
                say("Idle sleep guard active while the public API is enabled; the display may still sleep.")
            checks.repair()
            say("Remote watchdog started; checking every 30 seconds while Strata runs.")
            while not lease.stopped():
                want_awake = not (runtime / "disabled").exists()
                if awake and want_awake != held:
                    if want_awake:
                        awake.acquire()
                    else:
                        awake.release()
                    held = want_awake
                monitor.tick()
                if lease.stopped(30):
                    break
        finally:
            if awake:
                awake.close()
            say("Remote watchdog stopped; app-scoped sleep guard released.")


def supervise_remote_access(root: Path):
    """Restart a failed parent-bound watchdog; never restart the model or rotate its credentials."""
    runtime = Path(os.environ['LOCALAPPDATA']) / 'StrataRemote'
    try:
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / 'disabled').unlink(missing_ok=True)
    except OSError:
        print('[strata] remote runtime unavailable; local inference remains available', flush=True)
        return
    while True:
        try:
            with (runtime / 'watchdog-console.log').open('a', encoding='utf-8') as output:
                child = subprocess.Popen(
                    [sys.executable, '-m', 'serve.remote_access', '--parent-pid', str(os.getpid())],
                    cwd=str(root), stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                child.wait()
        except (OSError, subprocess.SubprocessError) as error:
            print(f'[strata] remote monitor startup failed ({type(error).__name__}); retrying', flush=True)
        time.sleep(5)


def start_remote_access(root: Path):
    if os.name != "nt" or not (root / "remote-access" / "Start-Remote.ps1").is_file():
        return None
    worker = threading.Thread(target=supervise_remote_access, args=(root,), name="strata-remote-supervisor", daemon=True)
    worker.start()
    return worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    monitor_remote_access(Path(__file__).resolve().parents[1], args.parent_pid)
