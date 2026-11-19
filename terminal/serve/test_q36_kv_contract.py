"""CPU-only Q36 actual-binary negotiation and pre-READY receipts; no model or GPU."""
import ast
import copy
import io
import json
import subprocess
import unittest
from unittest import mock
from pathlib import Path

from serve import server


CAP = {"schema": 1, "family": "qwen35moe", "layoutVersion": 1, "scaleSchema": "none",
       "cacheTypesK": ["f32", "f16"], "cacheTypesV": ["f32", "f16"],
       "flags": ["--cache-type-k", "--cache-type-v"], "quality": "experimental-unqualified"}
SHA = "ab" * 32
VERIFIED = {**CAP, "binarySha256": SHA}
LINE = "Q36_KV_CAPABILITIES " + json.dumps(CAP)


class Q36KvContract(unittest.TestCase):
    @staticmethod
    def _constructor_namespace():
        # Execute the real production constructor through the READY gate, replacing
        # process IO only. Do not start a server, engine, model, or GPU operation.
        path = Path(server.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StrataEngine")
        init = copy.deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"))
        stop = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Attribute) and t.attr == "known_ctx" for t in n.targets))
        init.body = init.body[:stop + 1]
        pump = copy.deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "_ready_pump"))
        verify = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "q36_verify_kv_args"))
        memory_verify = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "q36_verify_memory_args"))
        module = ast.fix_missing_locations(ast.Module(body=[verify, memory_verify, ast.ClassDef(name="ReadyPrefix", bases=[], keywords=[], body=[init, pump], decorator_list=[])], type_ignores=[]))
        namespace = dict(vars(server))
        namespace.update(engine_nice=lambda env: 0, end_leftover_engines=mock.Mock(),
                         q36_probe_kv_capabilities=mock.Mock(return_value=VERIFIED),
                         q36_binary_sha256=lambda exe: SHA)
        exec(compile(module, str(path), "exec"), namespace)
        return namespace

    def test_real_constructor_refuses_wrong_ready_receipt_and_changed_binary(self):
        namespace = self._constructor_namespace()
        for valid in (True, False):
            proc = mock.Mock(pid=1)
            proc.stdin = io.StringIO()
            proc.stdout = io.StringIO("INFO kv_layout_version=1 kv_scale_schema=none kv_cache_k=" + ("f16" if valid else "f32") +
                " kv_cache_v=f32 kv_context=256 kv_sessions=1 kv_heads=2 kv_head_dim=256 kv_attention_states=11 kv_allocation_bytes=8650752\nREADY 256 stop\n")
            namespace["popen"] = mock.Mock(return_value=proc)
            args = ["--max-context", "256", "--cache-type-k", "f16"]
            if valid:
                result = namespace["ReadyPrefix"]("fake", args, env={"STRATA_REQUIRE_Q36_KV": "1"})
                self.assertEqual(result.known_ctx, 256)
                proc.kill.assert_not_called()
            else:
                with self.assertRaisesRegex(server.EngineDied, "KV layout"):
                    namespace["ReadyPrefix"]("fake", args, env={"STRATA_REQUIRE_Q36_KV": "1"})
                proc.kill.assert_called_once()
        namespace["q36_binary_sha256"] = lambda exe: "cd" * 32
        namespace["popen"] = mock.Mock()
        with self.assertRaisesRegex(OSError, "binary changed"):
            namespace["ReadyPrefix"]("fake", ["--cache-type-k", "f16"], env={"STRATA_REQUIRE_Q36_KV": "1"})
        namespace["popen"].assert_not_called()

    def test_negotiated_flag_free_f32_uses_production_negotiation_and_ready_gate(self):
        args = ["--max-context", "256"]
        env = {"STRATA_REQUIRE_Q36_KV": "1", "STRATA_Q36_KV_HOST_SHA256": SHA}
        receipt = ("INFO kv_layout_version=1 kv_scale_schema=none kv_cache_k=f32 kv_cache_v=f32 "
                   "kv_context=256 kv_sessions=1 kv_heads=2 kv_head_dim=256 kv_attention_states=11 "
                   "kv_allocation_bytes=11534336\n")
        for supplied, valid in ((receipt, True), ("", False),
                                (receipt.replace("kv_cache_k=f32", "kv_cache_k=f16"), False),
                                (receipt.replace("11534336", "1"), False),
                                (receipt.replace("kv_context=256", "kv_context=128"), False)):
            with self.subTest(receipt=supplied):
                namespace = self._constructor_namespace()
                proc = mock.Mock(pid=1, stdin=io.StringIO(), stdout=io.StringIO(supplied + "READY 256 stop\n"))
                namespace["popen"] = mock.Mock(return_value=proc)
                original_args = list(args)
                if valid:
                    result = namespace["ReadyPrefix"]("fake", args, env=env)
                    self.assertEqual(result.known_ctx, 256)
                    proc.kill.assert_not_called()
                else:
                    with self.assertRaisesRegex(server.EngineDied, "KV layout"):
                        namespace["ReadyPrefix"]("fake", args, env=env)
                    proc.kill.assert_called_once()
                namespace["q36_probe_kv_capabilities"].assert_called_once_with("fake")
                self.assertEqual(args, original_args)
                self.assertEqual(namespace["popen"].call_args.args[1], ["fake", "--serve", *args])

    def test_negotiated_flag_free_f32_rejects_different_or_changed_binary(self):
        args = ["--max-context", "256"]
        env = {"STRATA_REQUIRE_Q36_KV": "1", "STRATA_Q36_KV_HOST_SHA256": SHA}
        for capability in (None, {**VERIFIED, "binarySha256": "cd" * 32}):
            namespace = self._constructor_namespace()
            namespace["q36_probe_kv_capabilities"].return_value = capability
            namespace["popen"] = mock.Mock(side_effect=AssertionError("Unverified binary reached spawn"))
            with self.assertRaisesRegex(ValueError, "Host-negotiated binary"):
                namespace["ReadyPrefix"]("fake", args, env=env)
            namespace["popen"].assert_not_called()
        namespace = self._constructor_namespace()
        namespace["q36_binary_sha256"] = lambda exe: "cd" * 32
        namespace["popen"] = mock.Mock(side_effect=AssertionError("Changed binary reached spawn"))
        with self.assertRaisesRegex(OSError, "binary changed"):
            namespace["ReadyPrefix"]("fake", args, env=env)
        namespace["popen"].assert_not_called()

    def test_legacy_flag_free_f32_without_host_expectation_keeps_ordinary_argv(self):
        namespace = self._constructor_namespace()
        proc = mock.Mock(pid=1, stdin=io.StringIO(), stdout=io.StringIO("READY 256 stop\n"))
        namespace["popen"] = mock.Mock(return_value=proc)
        args = ["--max-context", "256"]
        result = namespace["ReadyPrefix"]("legacy", args, env={"STRATA_REQUIRE_Q36_KV": "1"})
        self.assertEqual(result.known_ctx, 256)
        self.assertEqual(namespace["popen"].call_args.args[1], ["legacy", "--serve", "--max-context", "256"])
        namespace["q36_probe_kv_capabilities"].assert_not_called()
        proc.kill.assert_not_called()

    def test_schema_is_strict_and_proposals_never_become_capabilities(self):
        self.assertEqual(server.q36_parse_kv_capabilities(LINE), CAP)
        for text in ("strata 99 f16", LINE + "\n" + LINE, LINE.replace('"f16"', '"q8_0"'),
                     LINE.replace('"schema": 1', '"schema": true'), "x" * 65536 + LINE,
                     "Q36_KV_CAPABILITIES junk"):
            self.assertIsNone(server.q36_parse_kv_capabilities(text))

    def test_bounded_help_probe_and_identity_change(self):
        proc = mock.MagicMock()
        proc.__enter__.return_value = proc
        proc.stdout = io.BytesIO((LINE + "\n").encode())
        proc.wait.return_value = 0
        with mock.patch.object(server.subprocess, "Popen", return_value=proc) as spawn, \
                mock.patch.object(server, "q36_binary_sha256", return_value=SHA):
            self.assertEqual(server.q36_probe_kv_capabilities("C:/test/strata.exe"), VERIFIED)
            self.assertEqual(spawn.call_args.args[0], ["C:/test/strata.exe", "--help"])
            self.assertNotIn("STRATA_TOKEN", spawn.call_args.kwargs["env"])
            proc.wait.assert_called_with(timeout=3)
        proc.stdout = io.BytesIO(LINE.encode())
        with mock.patch.object(server.subprocess, "Popen", return_value=proc), \
                mock.patch.object(server, "q36_binary_sha256", side_effect=[SHA, "cd" * 32]):
            self.assertIsNone(server.q36_probe_kv_capabilities("fake"))

    def test_timeout_and_oversize_do_not_authorise_formats(self):
        proc = mock.MagicMock()
        proc.__enter__.return_value = proc
        proc.stdout = io.BytesIO(b"x" * 65537)
        proc.wait.return_value = 0
        with mock.patch.object(server.subprocess, "Popen", return_value=proc), \
                mock.patch.object(server, "q36_binary_sha256", return_value=SHA):
            self.assertIsNone(server.q36_probe_kv_capabilities("fake"))
            proc.kill.assert_called()
        proc.stdout = io.BytesIO(LINE.encode())
        proc.wait.side_effect = [subprocess.TimeoutExpired("fake", 3), 0]
        with mock.patch.object(server.subprocess, "Popen", return_value=proc), \
                mock.patch.object(server, "q36_binary_sha256", return_value=SHA):
            self.assertIsNone(server.q36_probe_kv_capabilities("fake"))

    def test_top_level_and_args_preserve_independent_formats(self):
        for k in ("f32", "f16"):
            for v in ("f32", "f16"):
                args = ["--max-context", "131072", "--cache-type-k", k]
                cfg = {"arch": "qwen35moe", "cache_type_v": v}
                with mock.patch.object(server, "q36_probe_kv_capabilities", return_value=VERIFIED):
                    self.assertEqual(server.native_cache_args(cfg, args, "real-child"), [])
                self.assertEqual(args[-4:], ["--cache-type-k", k, "--cache-type-v", v])
        with self.assertRaises(ValueError):
            server.native_cache_args({"arch": "qwen35moe", "cache_type_k": "f16"}, ["--cache-type-k", "f32"], "fake")

    def test_legacy_defaults_never_receive_new_flags_and_f16_fails_closed(self):
        with mock.patch.object(server, "q36_probe_kv_capabilities", return_value=None):
            args = ["--cache-type-k", "f32", "--cache-type-v", "f32"]
            self.assertIsNone(server.q36_verify_kv_args("old", args))
            self.assertEqual(args, [])
            with self.assertRaises(ValueError):
                server.q36_verify_kv_args("old", ["--cache-type-k", "f16"])
        with mock.patch.object(server, "q36_probe_kv_capabilities", return_value=VERIFIED):
            with self.assertRaises(ValueError):
                server.q36_verify_kv_args("new", ["--cache-type-k", "f16"], "cd" * 32)
        for args in (["--cache-type-k", "q8_0"], ["--cache-type-v", "nvfp4"], ["--cache-type-k"],
                     ["--cache-type-k=f16"], ["-ctk", "f16"], ["--cache-type-k", "f16", "--cache-type-k", "f16"]):
            with self.assertRaises(ValueError):
                server.q36_verify_kv_args("fake", args)

    def test_receipt_requires_exact_format_geometry_and_bytes(self):
        args = ["--max-context", "131072", "--cache-type-k", "f16", "--cache-type-v", "f32"]
        info = {"kv_layout_version": 1, "kv_scale_schema": "none", "kv_cache_k": "f16", "kv_cache_v": "f32",
                "kv_context": 131072, "kv_sessions": 1, "kv_heads": 2, "kv_head_dim": 256,
                "kv_attention_states": 11, "kv_allocation_bytes": 4429185024}
        self.assertTrue(server.q36_kv_receipt_matches(info, args))
        for change in ({"kv_cache_k": "f32"}, {"kv_layout_version": 2}, {"kv_scale_schema": "other"},
                       {"kv_allocation_bytes": 1}, {"kv_context": 65536}, {"kv_heads": True}, {"kv_sessions": 0}):
            self.assertFalse(server.q36_kv_receipt_matches({**info, **change}, args))
        self.assertFalse(server.q36_kv_receipt_matches({}, args))


if __name__ == "__main__":
    unittest.main()
