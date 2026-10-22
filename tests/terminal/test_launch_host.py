"""Explicit launch bindings: CLI choices, private configs, profiles and client addresses."""

import asyncio
import http.server
import json
import os
import shlex
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from strata_cli.cli import parser
from strata_cli.launch import access, cmd, settings as S, store
from strata_cli.launch.runner import LaunchError, Runner, derive_config
from strata_control import profiles, runtime
from tests.terminal.launch_support import Home, HW_12, catalog, run_cli


def choices(*args):
    a = parser().parse_args(["launch", *args])
    return cmd.settings_from_args(a, catalog(HW_12))[0]


def test_host_reaches_config_profile_status_and_connect_without_editing_install():
    with Home() as home:
        s = choices("--model", "iq3_s", "--host", "127.0.0.2", "--port", "9090",
                    "--context", "131072", "--vision", "gpu", "--kv", "int8")
        base = {"exe": "component:strata-engine", "args": ["--max-context", "131072", "--kv", "int8"],
                "vision": {"exe": "reader", "gpu": True}, "tokenizer": "synthetic-tokenizer", "model_name": "synthetic"}
        installed = home.base / "installed.json"
        before = json.dumps(base).encode()
        installed.write_bytes(before)
        runner = Runner(s, catalog=catalog(HW_12))
        config = runner.build_config(installed)
        runner.ensure_profile(config)
        cfg = json.loads(config.read_text())
        assert cfg["host"] == "127.0.0.2"
        assert cfg["port"] == 9090
        assert cfg["vision"] == base["vision"]
        assert cfg["args"] == base["args"]
        assert installed.read_bytes() == before
        profile, _ = profiles.load(runner.profile)
        assert profile["overrides"]["host"] == "127.0.0.2"
        assert access.urls_for(s) == ["http://127.0.0.2:9090"]
        with mock.patch.object(runtime, "status", return_value={"state": "ready"}):
            assert cmd.running_launches()[0]["url"] == "http://127.0.0.2:9090"
            assert cmd.launch_info()["url"] == "http://127.0.0.2:9090"


@pytest.mark.parametrize("host,kind,key", [("127.0.0.2", "local", False), ("localhost", "local", False),
                                         ("192.0.2.10", "lan", True), ("0.0.0.0", "lan", True)])
def test_host_uses_existing_access_and_key_policy(host, kind, key):
    with Home():
        s = choices("--model", "q2_0", "--host", host)
        s, notes = cmd.apply_defaults(s, catalog(HW_12))
        assert s.access == kind
        assert bool(s.api_key) is key
        assert not [n for n in notes if n.level == "error"]
        assert derive_config({"args": []}, s, bool_flags=frozenset())["host"] == host


def test_explicit_keyless_and_local_access_do_not_bypass_shared_host_checks():
    with Home():
        s = choices("--model", "q2_0", "--host", "192.0.2.10", "--no-api-key")
        s, notes = cmd.apply_defaults(s, catalog(HW_12))
        assert s.api_key is None
        assert derive_config({"args": []}, s, bool_flags=frozenset())["api_key"] == "none"
        assert any(S.NO_KEY_NETWORK in n.text for n in notes)
        s = choices("--model", "q2_0", "--host", "192.0.2.10", "--access", "local")
        s, _ = cmd.apply_defaults(s, catalog(HW_12))
        assert s.api_key


def test_saved_host_overrides_round_trip_and_access_can_reset_it():
    with Home():
        original = S.LaunchSettings(name="Office", family="qwen", model="Q2_0", access="lan", host="192.0.2.10")
        store.save(original)
        s = choices("--preset", "Office", "--host", "127.0.0.2")
        assert (s.host, s.access) == ("127.0.0.2", "local")
        assert store.load("Office").host == "192.0.2.10"
        assert choices(*shlex.split(S.flags_text(s))[2:]).host == s.host
        assert choices(*shlex.split(S.flags_text(s, preset="Office"))[2:]).host == s.host
        assert choices("--preset", "Office", "--access", "local").host is None
        assert cmd.launch_info("Office")["url"] == "http://192.0.2.10:8080"


@pytest.mark.parametrize("value", ["", "http://127.0.0.1", "localhost:8080", "192.0.2.999", "::1", "example.com", "127.0.0.1 "])
def test_bad_host_is_rejected_before_hardware_or_downloads(value):
    with mock.patch.object(Runner, "run", side_effect=AssertionError("must not run")):
        code, out, err = run_cli(["launch", "--host", value, "--no-tui"])
    assert code == 2
    assert "IPv4 address or localhost" in err
    assert "Traceback" not in out + err


@pytest.mark.parametrize("kind,host", [("cloudflare", "192.0.2.10"), ("tailscale", "127.0.0.1")])
def test_host_cannot_make_a_fixed_address_helper_unreachable(kind, host):
    notes = S.check_access(S.LaunchSettings(access=kind, host=host))
    assert notes[0].level == "error"
    assert "--host" in notes[0].line()


def test_cloudflare_can_still_reach_a_wildcard_bind():
    s = S.LaunchSettings(access="cloudflare", host="0.0.0.0", api_key="synthetic-key")
    assert S.check_access(s) == []
    assert access.urls_for(s, tunnel_url="https://example.trycloudflare.com") == [
        "https://example.trycloudflare.com", "http://127.0.0.1:8080"]


def test_wildcard_is_never_advertised_as_a_client_destination():
    s = S.LaunchSettings(access="lan", host="0.0.0.0", port=9090)
    with mock.patch.object(access, "lan_ips", return_value=["192.0.2.10"]):
        assert access.urls_for(s) == ["http://127.0.0.1:9090", "http://192.0.2.10:9090"]


def test_omitted_host_keeps_legacy_serialization_flags_and_bind_defaults():
    for kind, host in (("local", "127.0.0.1"), ("lan", "0.0.0.0"),
                       ("cloudflare", "127.0.0.1"), ("tailscale", "0.0.0.0")):
        s = S.LaunchSettings(access=kind)
        assert "host" not in s.to_dict()
        assert "--host" not in S.flags_text(s)
        assert derive_config({"args": []}, s, bool_flags=frozenset())["host"] == host


def test_same_host_reuses_the_server_but_a_changed_host_requires_restart():
    with Home():
        s = S.LaunchSettings(name="binding", family="qwen", model="Q2_0", host="127.0.0.2")
        runner = Runner(s, catalog=catalog(HW_12))
        row = {"profile": runner.profile, "name": runner.name, "port": s.port, "choices": runner.choices()}
        with mock.patch.object(Runner, "serving_launches", return_value=[row]), \
             mock.patch.object(runtime, "stop") as stop:
            result = runner.check_serving()
            assert result.reused
            assert result.urls == ["http://127.0.0.2:8080"]
            runner.s.host = "127.0.0.3"
            with pytest.raises(LaunchError) as error:
                runner.check_serving()
            assert error.value.code == 6
            assert "--restart" in error.value.next
            stop.assert_not_called()
            runner.restart = True
            assert runner.check_serving() is None


def test_explicit_shared_host_keeps_an_existing_key():
    with Home(), mock.patch.object(access, "kept_key", return_value="synthetic-existing-key"), \
         mock.patch.object(access, "new_key", side_effect=AssertionError("must keep the key")):
        s = choices("--model", "q2_0", "--host", "192.0.2.10")
        s, _ = cmd.apply_defaults(s, catalog(HW_12))
        assert s.api_key == "synthetic-existing-key"


def test_monitor_uses_the_explicit_address():
    from strata_cli.launch import app, monitor_ext
    with mock.patch.object(monitor_ext, "LaunchMonitor") as monitor:
        monitor.return_value.stop_requested = False
        info = {"url": "http://127.0.0.2:9090", "urls": ["http://127.0.0.2:9090"]}
        assert app.run_monitor(info) == 0
        assert monitor.call_args.args[0].root == "http://127.0.0.2:9090"


def test_access_screen_keeps_a_saved_bind_until_the_access_choice_changes():
    from strata_cli.launch.app import Ctx, LaunchApp
    from strata_cli.launch.screens import AccessScreen
    from tests.terminal.launch_support import text_of

    async def exercise():
        cat = catalog(HW_12)
        s = S.LaunchSettings(access="local", host="127.0.0.2")
        ctx = Ctx(catalog=cat, computer=cat.computer(), rows=cat.rows(), settings=s)
        app = LaunchApp(ctx)
        async with app.run_test(size=(110, 34)) as pilot:
            app.show(AccessScreen())
            await pilot.pause()
            assert s.host == "127.0.0.2"
            assert "127.0.0.2" in text_of(app)
            await pilot.press("down")
            await pilot.pause()
            assert s.access == "lan"
            assert s.host is None

    with Home(), mock.patch.object(access, "lan_ips", return_value=[]):
        asyncio.run(exercise())


def test_http_monitor_accepts_owned_interfaces_but_not_other_lan_hosts():
    from strata_cli.client import Endpoint
    with mock.patch("psutil.net_if_addrs", return_value={"test": [SimpleNamespace(address="192.0.2.10")]}):
        assert Endpoint("http://192.0.2.10:9090").root == "http://192.0.2.10:9090"
        for host in ("192.0.2.11", "example.com", "0.0.0.0", "224.0.0.1"):
            with pytest.raises(ValueError, match="requires HTTPS"):
                Endpoint(f"http://{host}:9090")


def test_monitor_reaches_a_nondefault_loopback_bind_without_the_system_proxy():
    from strata_cli.client import Endpoint

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.2", 0), Health)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = access.urls_for(S.LaunchSettings(host="127.0.0.2", port=server.server_port))[0]
        env = {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9", "NO_PROXY": "", "no_proxy": ""}
        with mock.patch.dict(os.environ, env):
            assert Endpoint(url).get("/health") == {"status": "ok"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
