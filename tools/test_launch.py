"""Launch YAML and CLI tests: no GPU, model downloads or installation.

    python -m unittest tools.test_launch
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import yaml

import run
from serve import launchconfig, runconfig


class Launch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "strata-coder-q2_0.json"
        self.cfg = {"exe": "engine/strata", "args": ["--pack", "/data/packs/coder", "--max-context", "65536",
                                                    "--kv", "int8", "--vram-reserve-mib", "1500"],
                    "tokenizer": "tokenizer", "model_name": "coder", "port": 8081, "host": "0.0.0.0",
                    "api_key": "", "gpu": "1,0", "backend": "hip", "env": {"TUNING": "keep"},
                    "sampling": {"temperature": 0.7, "top_p": 0.9}, "open_browser": True}
        self.base.write_text(json.dumps(self.cfg), encoding="utf-8")
        self.path = self.root / "launch.yaml"
        self.environ = mock.patch.dict(os.environ)
        self.environ.start()
        self.addCleanup(self.environ.stop)
        for key in ("STRATA_API_KEY", "OTHER_KEY", "MISSING_KEY"):
            os.environ.pop(key, None)

    def write(self, **settings):
        raw = {"config": self.base.name, **settings}
        self.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return launchconfig.load(self.path, self.root)

    def test_minimal_launch_keeps_paths_and_hardware_but_defaults_to_loopback(self):
        got = self.write()
        for key in ("exe", "args", "gpu", "backend", "env", "sampling", "model_name", "port"):
            self.assertEqual(got[key], self.cfg[key])
        self.assertEqual(got["host"], "127.0.0.1")
        self.assertEqual(got["cwd"], str(self.root))
        self.assertEqual(got["tokenizer"], str(self.root / "tokenizer"))
        self.assertEqual(json.loads(self.base.read_text()), self.cfg)

    def test_model_pick_skips_non_models_and_does_not_pick_the_newest(self):
        (self.root / "strata-zz.json").write_text(json.dumps(self.cfg))
        (self.root / "strata-shared-settings.json").write_text('{"temperature": 1}')
        (self.root / "strata-broken.json").write_text('{broken')
        self.assertEqual(list(launchconfig.installed(self.root)), ["coder-q2_0", "zz"])
        self.path.write_text("model: CODER-Q2_0\n")
        self.assertEqual(launchconfig.load(self.path, self.root)["model_name"], "coder")
        self.path.write_text("model: missing\n")
        with self.assertRaisesRegex(ValueError, "not installed"):
            launchconfig.load(self.path, self.root)

    def test_engine_and_server_overrides_replace_all_old_values(self):
        self.cfg["args"] += ["--max-context=131072", "--max-context", "262144"]
        self.base.write_text(json.dumps(self.cfg))
        got = self.write(context=32768, kv="q4_0", vram_reserve_mib=2048, port=9000,
                         open_browser=False, lazy_load=True, sampling={"temperature": 0.2})
        self.assertEqual(got["args"], ["--pack", "/data/packs/coder", "--max-context", "32768",
                                      "--kv", "q4_0", "--vram-reserve-mib", "2048"])
        self.assertEqual(got["sampling"], {"temperature": 0.2, "top_p": 0.9})
        self.assertEqual((got["port"], got["open_browser"], got["lazy_load"]), (9000, False, True))

    def test_invalid_settings_are_rejected(self):
        cases = [{"context": x} for x in (0, -1, True, 2.5, "32768", 2**31)]
        cases += [{"port": x} for x in (0, -1, 65536, True, "8080")]
        cases += [{"kv": "unknown"}, {"contex": 32768}, {"api_key": 123}, {"api_key_env": "bad name"},
                  {"open_browser": "false"}, {"sampling": [1]}, {"sampling": {"top_k": 65}},
                  {"sampling": {"temperature": float("nan")}}, {"idle_unload_s": float("inf")},
                  {"vram_reserve_mib": -1}, {"model": "coder-q2_0"}]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.write(**settings)
        for text in ("[]", "{}", "config: [secret: broken", "model: ../strata"):
            self.path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                launchconfig.load(self.path, self.root)

    def test_lan_needs_a_key_and_environment_keys_are_not_written(self):
        for host in ("0.0.0.0", "192.168.1.10", "server.lan"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "requires an API key"):
                self.write(host=host)
        with self.assertRaisesRegex(ValueError, "requires an API key"):
            self.write(host="0.0.0.0", api_key="   ")
        os.environ["STRATA_API_KEY"] = "secret-token"
        got = self.write(host="0.0.0.0", api_key="file-key")
        self.assertEqual(got["api_key"], "secret-token")
        self.assertNotIn("secret-token", self.path.read_text())
        os.environ["OTHER_KEY"] = "custom-token"
        self.assertEqual(self.write(host="0.0.0.0", api_key_env="OTHER_KEY")["api_key"], "custom-token")
        with self.assertRaisesRegex(ValueError, "set the MISSING_KEY"):
            self.write(api_key_env="MISSING_KEY")
        os.environ["STRATA_API_KEY"] = ""
        with self.assertRaisesRegex(ValueError, "STRATA_API_KEY is empty"):
            self.write()
        self.assertEqual(self.write(api_key_env="OTHER_KEY")["api_key"], "custom-token")

    def test_ipv6_hosts_are_rejected_with_or_without_a_key(self):
        for host in ("::1", "::", "2001:db8::1", "::ffff:127.0.0.1", "fe80::1%lo", "[::1]"):
            for key in ("", "test-key"):
                with self.subTest(host=host, key=key), self.assertRaisesRegex(ValueError, "IPv6"):
                    self.write(host=host, api_key=key)
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as refused:
            run.main(["check", "--config", str(self.path)])
        self.assertEqual(refused.exception.code, 2)
        self.assertIn("IPv6", error.getvalue())

    def test_extended_context_reuses_only_a_covering_rope_configuration(self):
        with self.assertRaisesRegex(ValueError, "RoPE"):
            self.write(context=393216)
        self.cfg["args"] += ["--rope-scaling", "yarn", "--rope-scale", "2"]
        self.base.write_text(json.dumps(self.cfg))
        self.assertEqual(launchconfig.arg_value(self.write(context=524288)["args"], "--max-context"), "524288")
        with self.assertRaisesRegex(ValueError, "RoPE"):
            self.write(context=600000)

    def test_settings_write_to_yaml_keeps_base_and_secret_reference(self):
        os.environ["STRATA_API_KEY"] = "do-not-save-this-key"
        cfg = self.write(host="0.0.0.0", api_key_env="STRATA_API_KEY", context=32768)
        old_yaml, old_base = self.path.read_bytes(), self.base.read_bytes()
        cfg, _ = runconfig.apply(cfg, {"sampling.temperature": None, "sampling.top_k": 20,
                                      "vram_reserve_mib": None, "open_browser": False})
        bak = runconfig.save(self.path, cfg)
        got = runconfig.load(self.path)
        self.assertEqual(got["sampling"], {"top_p": 0.9, "top_k": 20})
        self.assertNotIn("--vram-reserve-mib", got["args"])
        self.assertFalse(got["open_browser"])
        raw = yaml.safe_load(self.path.read_text())
        self.assertEqual(raw["config"], self.base.name)
        self.assertEqual(raw["api_key_env"], "STRATA_API_KEY")
        self.assertNotIn("do-not-save-this-key", self.path.read_text())
        self.assertEqual(self.base.read_bytes(), old_base)
        self.assertEqual(bak.read_bytes(), old_yaml)
        self.assertEqual(runconfig.load(self.base), self.cfg)          # JSON loading remains unchanged

    def test_check_and_run_use_the_same_launch_file_without_installing(self):
        os.environ["STRATA_API_KEY"] = "hidden-token"
        self.write(context=32768, open_browser=False)
        (self.root / "engine").mkdir()
        (self.root / "engine/strata").touch()
        (self.root / "tokenizer").mkdir()
        for name in ("vocab.json", "merges.txt", "token_type.json"):
            (self.root / "tokenizer" / name).touch()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(run.main(["check", "--config", str(self.path)]), 0)
        self.assertIn("context 32768", stdout.getvalue())
        self.assertNotIn("hidden-token", stdout.getvalue())
        from serve import server
        with mock.patch.object(server, "main", return_value=0) as main, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(["run", "--config", str(self.path)]), 0)
            main.assert_called_once_with(["--engine", "strata", "--config", str(self.path)])

    def test_run_restores_each_models_saved_draft_subset_and_preserves_custom_files(self):
        import setup

        (self.root / "engine").mkdir()
        (self.root / "engine/strata").touch()
        (self.root / "tokenizer").mkdir()
        for name in ("vocab.json", "merges.txt", "token_type.json"):
            (self.root / "tokenizer" / name).touch()
        rt = self.root / "mtp/rt"
        rt.mkdir(parents=True)
        shipped = self.root / "data"
        shipped.mkdir()
        subsets = {"cjk": b"shipped cjk", "en": b"shipped en", "fr": b"shipped fr"}
        for choice, contents in subsets.items():
            (shipped / setup.DRAFT_VOCABS[choice]).write_bytes(contents)
        active = rt / "draft_vocab.bin"
        active.write_bytes(subsets["cjk"])
        self.cfg["args"] += ["--mtp", "mtp/rt"]
        self.cfg["draft_vocab"] = "en"
        self.base.write_text(json.dumps(self.cfg))
        other = self.root / "strata-swift-iq3_s.json"
        second = {**self.cfg, "draft_vocab": "fr", "args": [*self.cfg["args"][:-2], "--mtp=mtp/rt"]}
        other.write_text(json.dumps(second))
        self.write()
        with mock.patch.object(setup, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(["check", "--config", str(self.path)]), 0)
            self.assertEqual(active.read_bytes(), subsets["cjk"])
            from serve import server
            for config, expected in ((self.base, subsets["en"]), (other, subsets["fr"]),
                                     (self.base, subsets["en"])):
                self.path.write_text(f"config: {config.name}\n")

                def start(args):
                    self.assertEqual(active.read_bytes(), expected)  # restored before the engine can start
                    return 0

                with mock.patch.object(server, "main", side_effect=start):
                    self.assertEqual(run.main(["run", "--config", str(self.path)]), 0)
            self.cfg.pop("draft_vocab")
            self.base.write_text(json.dumps(self.cfg))
            with mock.patch.object(server, "main", return_value=0):
                self.assertEqual(run.main(["run", "--config", str(self.path)]), 0)
            self.assertEqual(active.read_bytes(), subsets["cjk"])
            active.write_bytes(b"user supplied custom subset")
            self.path.write_text(f"config: {other.name}\n")
            with mock.patch.object(server, "main", return_value=0):
                self.assertEqual(run.main(["run", "--config", str(self.path)]), 0)
            self.assertEqual(active.read_bytes(), b"user supplied custom subset")

    def test_init_never_overwrites_and_uses_an_installed_name(self):
        with mock.patch.object(launchconfig, "installed", return_value={"coder-q2_0": self.base}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(["init", "--config", str(self.path)]), 0)
        self.assertEqual(yaml.safe_load(self.path.read_text())["model"], "coder-q2_0")
        earlier = self.path.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run.main(["init", "--config", str(self.path)])
        self.assertEqual(self.path.read_bytes(), earlier)

    def test_init_inherits_small_and_large_installed_contexts(self):
        for context in (8192, 131072):
            with self.subTest(context=context):
                self.path.unlink(missing_ok=True)
                self.cfg["args"][self.cfg["args"].index("--max-context") + 1] = str(context)
                self.base.write_text(json.dumps(self.cfg))
                with mock.patch.object(launchconfig, "installed", return_value={"coder-q2_0": self.base}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run.main(["init", "--config", str(self.path)]), 0)
                    cfg = launchconfig.load(self.path)
                self.assertNotIn("context", yaml.safe_load(self.path.read_text()))
                self.assertEqual(cfg["args"], self.cfg["args"])

    def test_cli_host_override_is_checked_before_the_socket_or_engine(self):
        self.write()
        from serve import server
        with mock.patch.object(server, "Server") as socket, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            server.main(["--engine", "mock", "--config", str(self.path), "--host", "0.0.0.0"])
        socket.assert_not_called()

    def test_cli_api_key_can_secure_a_yaml_lan_launch(self):
        self.path.write_text(f"config: {self.base.name}\nhost: 0.0.0.0\n")
        from serve import server
        with mock.patch.object(server, "Server", side_effect=OSError("stop before loading")) as listener, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            server.main(["--engine", "mock", "--config", str(self.path), "--api-key", "cli-secret"])
        self.assertEqual(listener.call_args.args[0], ("0.0.0.0", 8081))

    def test_cli_ipv6_override_is_rejected_before_binding(self):
        self.write()
        from serve import server
        with mock.patch.object(server, "Server", side_effect=AssertionError("IPv6 reached the listener")) as listener, \
                contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            server.main(["--engine", "mock", "--config", str(self.path), "--host", "::1", "--api-key", "test-key"])
        listener.assert_not_called()
        self.assertIn("IPv6", error.getvalue())

    def test_server_uses_yaml_port_and_key_and_saves_settings_over_http(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        os.environ["STRATA_API_KEY"] = ""
        os.environ["OTHER_KEY"] = "http-test-secret"
        self.write(port=port, context=32768, open_browser=False, api_key_env="OTHER_KEY")
        with tempfile.TemporaryFile(mode="w+") as log:
            proc = subprocess.Popen([sys.executable, str(run.ROOT / "serve/server.py"), "--engine", "mock",
                                     "--config", str(self.path)], cwd=run.ROOT, stdout=log, stderr=log)
            try:
                url = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 10
                while True:
                    try:
                        with urllib.request.urlopen(url + "/health", timeout=0.5) as response:
                            self.assertEqual(response.status, 200)
                        break
                    except urllib.error.URLError:
                        if proc.poll() is not None or time.monotonic() > deadline:
                            log.seek(0)
                            self.fail("mock server did not start: " + log.read())
                        time.sleep(0.05)
                with self.assertRaises(urllib.error.HTTPError) as refused:
                    urllib.request.urlopen(url + "/v1/models", timeout=2)
                self.assertEqual(refused.exception.code, 401)
                refused.exception.close()
                headers = {"Authorization": "Bearer http-test-secret", "Content-Type": "application/json",
                           "Origin": url}
                req = urllib.request.Request(url + "/config", headers=headers)
                with urllib.request.urlopen(req, timeout=2) as response:
                    self.assertEqual(json.load(response)["file"], self.path.name)
                req = urllib.request.Request(url + "/config", method="POST", headers=headers,
                                             data=b'{"set": {"sampling.temperature": 0.25}}')
                with urllib.request.urlopen(req, timeout=2) as response:
                    self.assertEqual(json.load(response)["changed"], ["sampling.temperature"])
                self.assertEqual(runconfig.load(self.path)["sampling"]["temperature"], 0.25)
                self.assertEqual(json.loads(self.base.read_text()), self.cfg)
                self.assertNotIn("http-test-secret", self.path.read_text())
                proc.terminate()                         # exercises the server's SIGTERM cleanup
                self.assertEqual(proc.wait(timeout=10), 0)
                log.seek(0)
                self.assertNotIn("http-test-secret", log.read())
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
