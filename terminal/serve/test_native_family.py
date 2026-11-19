"""CPU-only Host -> Python family selection and actual pre-READY admission."""

import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest import mock

from serve import private_cache, server
from serve import test_q36_kv_contract as kv_tests

PRIVATE_ARGS = [
    "--max-context",
    "256",
    "--prompt-cache",
    "0",
    "--conversation-cache-mib",
    "0",
    "--private-conversation-cache",
]
KV_RECEIPT = (
    "INFO kv_layout_version=1 kv_scale_schema=none kv_cache_k=f16 kv_cache_v=f32 "
    "kv_context=256 kv_sessions=1 kv_heads=2 kv_head_dim=256 kv_attention_states=11 "
    "kv_allocation_bytes=8650752\n"
)


def ready_consumer(cfg: dict, receipt: str) -> tuple:
    """Real constructor admission, with controlled child IO/OS effects."""
    namespace = kv_tests.Q36KvContract._constructor_namespace()
    proc = mock.Mock(
        pid=1, stdin=io.StringIO(), stdout=io.StringIO(receipt + "READY 256 stop\n")
    )
    namespace["popen"] = mock.Mock(return_value=proc)
    with mock.patch.dict(os.environ, {}, clear=True):
        env = server.child_env(cfg)
    obj = namespace["ReadyPrefix"].__new__(namespace["ReadyPrefix"])
    return namespace, proc, env, obj


class NativeFamilyDecisionTests(unittest.TestCase):
    def test_explicit_non_q36_identity_wins_over_both_name_hints(self) -> None:
        for arch in ("qwen4exp", "deepseek41", "future-architecture", 7, {}):
            with self.subTest(arch=arch):
                cfg = {
                    "arch": arch,
                    "exe": "strata-q36.exe",
                    "model_name": "Qwen3.6-35B-A3B",
                    "args": PRIVATE_ARGS,
                    "backend": "cpu",
                }
                self.assertFalse(server.q36_config(cfg))
                self.assertEqual(private_cache.required_capability(cfg), 0)

    def test_absent_identity_retains_legacy_executable_and_model_hints(self) -> None:
        for identity in ({}, {"arch": None}, {"arch": ""}):
            for hint in ({"exe": "strata-q36.exe"}, {"model_name": "Qwen3.6-35B-A3B"}):
                with self.subTest(identity=identity, hint=hint):
                    cfg = {**identity, **hint, "args": PRIVATE_ARGS}
                    self.assertTrue(server.q36_config(cfg))
                    self.assertEqual(private_cache.required_capability(cfg), 2)
        cfg = {"args": PRIVATE_ARGS, "exe": "neutral.exe", "model_name": "Neutral"}
        self.assertFalse(server.q36_config(cfg))
        self.assertEqual(private_cache.required_capability(cfg), 0)

    def test_genuine_q36_requires_its_own_private_receipt_before_ready(self) -> None:
        for arch in ("qwen35moe", "qwen36"):
            for required in (1, 2):
                for reported in (None, 3 - required, required):
                    with self.subTest(arch=arch, required=required, reported=reported):
                        cfg = {
                            "arch": arch,
                            "exe": "neutral.exe",
                            "args": (
                                PRIVATE_ARGS if required == 2 else PRIVATE_ARGS[:-1]
                            ),
                            "env": {
                                "STRATA_REQUIRE_Q36_PRIVATE_CACHE": "0",
                                "STRATA_REQUIRE_Q36_KV": "0",
                            },
                        }
                        marker = (
                            f"INFO q36_private_cache={reported}\n"
                            if reported is not None
                            else ""
                        )
                        namespace, proc, env, obj = ready_consumer(cfg, marker)
                        self.assertEqual(
                            env["STRATA_REQUIRE_Q36_PRIVATE_CACHE"], str(required)
                        )
                        self.assertEqual(env["STRATA_REQUIRE_Q36_KV"], "1")
                        if reported == required:
                            namespace["ReadyPrefix"].__init__(
                                obj, cfg["exe"], list(cfg["args"]), env=env
                            )
                            self.assertEqual(obj.known_ctx, 256)
                            proc.kill.assert_not_called()
                        else:
                            with self.assertRaisesRegex(
                                server.EngineDied, "private conversation cache"
                            ):
                                namespace["ReadyPrefix"].__init__(
                                    obj, cfg["exe"], list(cfg["args"]), env=env
                                )
                            self.assertFalse(hasattr(obj, "known_ctx"))
                            proc.kill.assert_called_once()
                        namespace["q36_probe_kv_capabilities"].assert_not_called()


class HostPythonFamilyConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        node = os.environ.get("STRATA_TEST_NODE_EXE") or shutil.which("node")
        if not node:
            raise RuntimeError(
                "Node is required for the actual Host builder integration fixture"
            )
        root = Path(__file__).resolve().parents[1]
        bridge = root / "apps/host/tests/helpers/native-family-consumer-fixture.mjs"
        output = subprocess.run(
            [node, str(bridge)],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        cls.configs = json.loads(output.stdout)

    def test_host_q38_lookalike_path_needs_no_q36_markers(self) -> None:
        cfg = copy.deepcopy(self.configs["qwen"])
        self.assertEqual(cfg["arch"], "qwen4exp")
        self.assertIn("strata-q36", cfg["exe"])
        self.assertIn("--prompt-cache", cfg["args"])
        args = list(cfg["args"])
        with mock.patch.object(server, "q36_probe_kv_capabilities") as probe:
            args += server.native_cache_args(cfg, args, cfg["exe"])
            probe.assert_not_called()
        namespace, proc, env, obj = ready_consumer(cfg, "")
        namespace["ReadyPrefix"].__init__(obj, cfg["exe"], args, env=env)
        self.assertEqual(env["STRATA_REQUIRE_Q36_PRIVATE_CACHE"], "0")
        self.assertEqual(env["STRATA_REQUIRE_Q36_KV"], "0")
        self.assertEqual(obj.known_ctx, 256)
        namespace["q36_probe_kv_capabilities"].assert_not_called()
        proc.kill.assert_not_called()

    def test_catalog_q38_runtime_keeps_resolved_identity(self) -> None:
        cfg = copy.deepcopy(self.configs["qwen-catalog"])
        self.assertIn("strata-q36", cfg["exe"])
        args = list(cfg["args"])
        with mock.patch.object(server, "q36_probe_kv_capabilities") as probe:
            args += server.native_cache_args(cfg, args, cfg["exe"])
            probe.assert_not_called()
        namespace, proc, env, obj = ready_consumer(cfg, "")
        error = None
        try:
            namespace["ReadyPrefix"].__init__(obj, cfg["exe"], args, env=env)
        except server.EngineDied as exc:
            error = str(exc)
        self.assertIsNone(error, "catalog Q38 must reach READY without Q36 receipts")
        self.assertEqual(cfg["arch"], "qwen4exp")
        self.assertEqual(env["STRATA_REQUIRE_Q36_PRIVATE_CACHE"], "0")
        self.assertEqual(env["STRATA_REQUIRE_Q36_KV"], "0")
        self.assertEqual(obj.known_ctx, 256)
        namespace["q36_probe_kv_capabilities"].assert_not_called()
        proc.kill.assert_not_called()

    def test_catalog_q36_checks_binary_and_both_ready_receipts(self) -> None:
        cfg = copy.deepcopy(self.configs["qwen36"])
        self.assertEqual(cfg["arch"], "qwen35moe")
        self.assertEqual(cfg["model_name"], "Neutral model")
        self.assertNotIn("q36", cfg["exe"])
        for private, kv, accepted in (
            ("INFO q36_private_cache=2\n", KV_RECEIPT, True),
            ("", KV_RECEIPT, False),
            ("INFO q36_private_cache=2\n", "", False),
        ):
            with self.subTest(private=bool(private), kv=bool(kv)):
                args = list(cfg["args"])
                with mock.patch.object(
                    server, "q36_probe_kv_capabilities", return_value=kv_tests.VERIFIED
                ) as probe:
                    args += server.native_cache_args(cfg, args, cfg["exe"])
                    probe.assert_called_once_with(cfg["exe"])
                namespace, proc, env, obj = ready_consumer(cfg, private + kv)
                self.assertEqual(env["STRATA_REQUIRE_Q36_PRIVATE_CACHE"], "2")
                self.assertEqual(env["STRATA_REQUIRE_Q36_KV"], "1")
                if accepted:
                    namespace["ReadyPrefix"].__init__(obj, cfg["exe"], args, env=env)
                    self.assertEqual(obj.known_ctx, 256)
                    proc.kill.assert_not_called()
                else:
                    with self.assertRaisesRegex(
                        server.EngineDied, "private conversation cache|KV layout"
                    ):
                        namespace["ReadyPrefix"].__init__(
                            obj, cfg["exe"], args, env=env
                        )
                    self.assertFalse(hasattr(obj, "known_ctx"))
                    proc.kill.assert_called_once()
                namespace["q36_probe_kv_capabilities"].assert_called_once_with(
                    cfg["exe"]
                )

    def test_host_claim_cannot_replace_child_verification(self) -> None:
        cfg = copy.deepcopy(self.configs["qwen36"])
        for actual in (None, {**kv_tests.VERIFIED, "binarySha256": "cd" * 32}):
            with self.subTest(actual=actual):
                namespace, proc, env, obj = ready_consumer(
                    cfg, "INFO q36_private_cache=2\n" + KV_RECEIPT
                )
                namespace["q36_probe_kv_capabilities"].return_value = actual
                with self.assertRaisesRegex(ValueError, "Host-negotiated binary"):
                    namespace["ReadyPrefix"].__init__(
                        obj, cfg["exe"], list(cfg["args"]), env=env
                    )
                namespace["popen"].assert_not_called()

    def assert_config_refused_before_spawn(self, cfg: dict) -> None:
        namespace = kv_tests.Q36KvContract._constructor_namespace()
        proc = mock.Mock(
            pid=1, stdin=io.StringIO(), stdout=io.StringIO("READY 256 stop\n")
        )
        namespace["popen"] = mock.Mock(return_value=proc)
        obj = namespace["ReadyPrefix"].__new__(namespace["ReadyPrefix"])
        error = None
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            server, "q36_probe_kv_capabilities", return_value=kv_tests.VERIFIED
        ) as prepare_probe:
            try:
                args = list(cfg["args"])
                args += server.native_cache_args(cfg, args, cfg["exe"])
                env = server.child_env(cfg)
                namespace["ReadyPrefix"].__init__(obj, cfg["exe"], args, env=env)
            except (ValueError, TypeError, AttributeError, server.EngineDied) as exc:
                error = exc
        self.assertEqual(
            (namespace["popen"].call_count, hasattr(obj, "known_ctx")),
            (0, False),
            "a contradictory contract must never reach Popen or accepted READY",
        )
        self.assertIsInstance(error, ValueError)
        self.assertRegex(str(error), "q36_run|adapter|Q36")
        prepare_probe.assert_not_called()
        namespace["q36_probe_kv_capabilities"].assert_not_called()
        namespace["end_leftover_engines"].assert_not_called()

    def test_q38_with_retained_q36_contract_never_reaches_ready(self) -> None:
        cfg = copy.deepcopy(self.configs["qwen"])
        cfg["q36_run"] = copy.deepcopy(self.configs["qwen36"]["q36_run"])
        self.assert_config_refused_before_spawn(cfg)

    def test_conflict_is_rejected_by_every_config_consumer(self) -> None:
        cfg = copy.deepcopy(self.configs["qwen"])
        cfg["q36_run"] = copy.deepcopy(self.configs["qwen36"]["q36_run"])
        consumers = (
            server.q36_config,
            private_cache.required_capability,
            server.child_env,
            lambda c: server.native_cache_args(c, list(c["args"]), c["exe"]),
        )
        for consumer in consumers:
            with self.subTest(consumer=consumer.__name__):
                with self.assertRaisesRegex(ValueError, "arch.*q36_run"):
                    consumer(cfg)

    def test_unknown_or_malformed_arch_cannot_downgrade_retained_contract(self) -> None:
        for arch in ("deepseek41", "future-architecture", 7, {}, [], False):
            with self.subTest(arch=arch):
                cfg = copy.deepcopy(self.configs["qwen"])
                cfg["arch"] = arch
                cfg["q36_run"] = copy.deepcopy(self.configs["qwen36"]["q36_run"])
                self.assert_config_refused_before_spawn(cfg)

    def test_retained_contract_structure_and_digest_binding_fail_closed(self) -> None:
        base = self.configs["qwen36"]["q36_run"]
        invalid = [
            False,
            [],
            {},
            {**base, "backend": None},
            {**base, "schemaVersion": True},
            {**base, "schemaVersion": 2},
        ]
        for changes in (
            {"family": "qwen4exp"},
            {"binarySha256": ""},
            {"binarySha256": "cd" * 32},
            {"kvCapability": {}},
            {"kvCapability": False},
        ):
            invalid.append({**base, "backend": {**base["backend"], **changes}})
        for contract in invalid:
            with self.subTest(contract=contract):
                cfg = copy.deepcopy(self.configs["qwen36"])
                cfg["q36_run"] = contract
                self.assert_config_refused_before_spawn(cfg)

    def test_contract_only_legacy_identity_still_requires_binary_proof(self) -> None:
        for arch in (None, ""):
            cfg = copy.deepcopy(self.configs["qwen36"])
            cfg["arch"] = arch
            for capability in (None, kv_tests.VERIFIED):
                with self.subTest(arch=arch, capability=capability is not None):
                    namespace, proc, env, obj = ready_consumer(
                        cfg, "INFO q36_private_cache=2\n" + KV_RECEIPT
                    )
                    namespace["q36_probe_kv_capabilities"].return_value = capability
                    self.assertEqual(env["STRATA_REQUIRE_Q36_KV"], "1")
                    self.assertEqual(env["STRATA_REQUIRE_Q36_PRIVATE_CACHE"], "2")
                    if capability is None:
                        with self.assertRaisesRegex(
                            ValueError, "Host-negotiated binary"
                        ):
                            namespace["ReadyPrefix"].__init__(
                                obj, cfg["exe"], list(cfg["args"]), env=env
                            )
                        namespace["popen"].assert_not_called()
                        self.assertFalse(hasattr(obj, "known_ctx"))
                    else:
                        namespace["ReadyPrefix"].__init__(
                            obj, cfg["exe"], list(cfg["args"]), env=env
                        )
                        self.assertEqual(obj.known_ctx, 256)
                        proc.kill.assert_not_called()
                    namespace["q36_probe_kv_capabilities"].assert_called_once()

    def test_constructor_rejects_incoherent_environment_before_any_effect(self) -> None:
        invalid = (
            {"STRATA_REQUIRE_Q36_KV": "0", "STRATA_Q36_KV_HOST_SHA256": kv_tests.SHA},
            {"STRATA_REQUIRE_Q36_KV": "0", "STRATA_REQUIRE_Q36_PRIVATE_CACHE": "2"},
            {"STRATA_REQUIRE_Q36_KV": "0", "STRATA_Q36_LAUNCH_GENERATION": "run-1"},
            {"STRATA_REQUIRE_Q36_KV": "bad"},
            {"STRATA_REQUIRE_Q36_KV": "1", "STRATA_REQUIRE_Q36_PRIVATE_CACHE": "bad"},
            {"STRATA_REQUIRE_Q36_KV": "1", "STRATA_Q36_KV_HOST_SHA256": ""},
        )
        for env in invalid:
            with self.subTest(env=env):
                namespace, proc, _, obj = ready_consumer(
                    {"arch": "qwen4exp", "args": []}, ""
                )
                error = None
                try:
                    namespace["ReadyPrefix"].__init__(obj, "neutral.exe", [], env=env)
                except (ValueError, server.EngineDied) as exc:
                    error = exc
                self.assertEqual(namespace["popen"].call_count, 0)
                self.assertFalse(hasattr(obj, "known_ctx"))
                self.assertIsInstance(error, ValueError)
                namespace["q36_probe_kv_capabilities"].assert_not_called()
                namespace["end_leftover_engines"].assert_not_called()

    def test_constructor_captures_one_environment_for_proof_and_ready(self) -> None:
        cfg = copy.deepcopy(self.configs["qwen36"])
        namespace, proc, env, obj = ready_consumer(cfg, KV_RECEIPT)

        def mutate_original_environment(*args, **kwargs):
            env["STRATA_REQUIRE_Q36_KV"] = "0"
            env["STRATA_REQUIRE_Q36_PRIVATE_CACHE"] = "0"
            return proc

        namespace["popen"].side_effect = mutate_original_environment
        with self.assertRaisesRegex(server.EngineDied, "private conversation cache"):
            namespace["ReadyPrefix"].__init__(
                obj, cfg["exe"], list(cfg["args"]), env=env
            )
        proc.kill.assert_called_once()
        self.assertFalse(hasattr(obj, "known_ctx"))

    def test_constructor_validates_inherited_environment_when_unsupplied(self) -> None:
        bad = {
            "STRATA_REQUIRE_Q36_KV": "0",
            "STRATA_Q36_KV_HOST_SHA256": kv_tests.SHA,
        }
        for env, accepted in (({}, True), (bad, False)):
            with self.subTest(accepted=accepted):
                namespace, proc, _, obj = ready_consumer(
                    {"arch": "qwen4exp", "args": []}, ""
                )
                with mock.patch.dict(os.environ, env, clear=True):
                    if accepted:
                        namespace["ReadyPrefix"].__init__(obj, "neutral.exe", [])
                        self.assertEqual(obj.known_ctx, 256)
                    else:
                        with self.assertRaisesRegex(ValueError, "Q36"):
                            namespace["ReadyPrefix"].__init__(obj, "neutral.exe", [])
                        namespace["popen"].assert_not_called()
                        namespace["end_leftover_engines"].assert_not_called()
                        self.assertFalse(hasattr(obj, "known_ctx"))

    def test_constructor_captures_argv_for_binary_proof_and_receipt(self) -> None:
        cfg = copy.deepcopy(self.configs["qwen36"])
        wrong = KV_RECEIPT.replace("kv_cache_k=f16", "kv_cache_k=f32").replace(
            "8650752", "11534336"
        )
        namespace, proc, env, obj = ready_consumer(
            cfg, "INFO q36_private_cache=2\n" + wrong
        )
        args = list(cfg["args"])

        def mutate_original_arguments(*unused, **kwargs):
            args[args.index("--cache-type-k") + 1] = "f32"
            return proc

        namespace["popen"].side_effect = mutate_original_arguments
        with self.assertRaisesRegex(server.EngineDied, "KV layout"):
            namespace["ReadyPrefix"].__init__(obj, cfg["exe"], args, env=env)
        proc.kill.assert_called_once()
        self.assertFalse(hasattr(obj, "known_ctx"))

    def test_legacy_private_only_environment_keeps_receipt_requirement(self) -> None:
        for receipt, accepted in (("", False), ("INFO q36_private_cache=2\n", True)):
            with self.subTest(accepted=accepted):
                namespace, proc, _, obj = ready_consumer(
                    {"arch": "qwen4exp", "args": []}, receipt
                )
                env = {"STRATA_REQUIRE_Q36_PRIVATE_CACHE": "2"}
                if accepted:
                    namespace["ReadyPrefix"].__init__(obj, "neutral.exe", [], env=env)
                    self.assertEqual(obj.known_ctx, 256)
                else:
                    with self.assertRaisesRegex(
                        server.EngineDied, "private conversation cache"
                    ):
                        namespace["ReadyPrefix"].__init__(
                            obj, "neutral.exe", [], env=env
                        )
                    proc.kill.assert_called_once()
                namespace["q36_probe_kv_capabilities"].assert_not_called()

    def test_user_argument_layer_keeps_private_requirement_through_ready(self) -> None:
        cfg = copy.deepcopy(self.configs["qwen36"])
        cfg["user_args"], cfg["args"] = cfg["args"], []
        args = server.user_args.merge(cfg["args"], cfg["user_args"])
        with mock.patch.object(
            server, "q36_probe_kv_capabilities", return_value=kv_tests.VERIFIED
        ):
            args += server.native_cache_args(cfg, args, cfg["exe"])
        namespace, proc, env, obj = ready_consumer(cfg, KV_RECEIPT)
        with self.assertRaisesRegex(server.EngineDied, "private conversation cache"):
            namespace["ReadyPrefix"].__init__(obj, cfg["exe"], args, env=env)
        self.assertEqual(env["STRATA_REQUIRE_Q36_PRIVATE_CACHE"], "2")
        proc.kill.assert_called_once()
        self.assertFalse(hasattr(obj, "known_ctx"))

    def test_constructor_rejects_lost_private_requirement(self) -> None:
        cfg = copy.deepcopy(self.configs["qwen36"])
        namespace, proc, env, obj = ready_consumer(cfg, KV_RECEIPT)
        env["STRATA_REQUIRE_Q36_PRIVATE_CACHE"] = "0"
        with self.assertRaisesRegex(ValueError, "Q36.*private|private.*Q36"):
            namespace["ReadyPrefix"].__init__(
                obj, cfg["exe"], list(cfg["args"]), env=env
            )
        namespace["popen"].assert_not_called()
        namespace["q36_probe_kv_capabilities"].assert_not_called()
        namespace["end_leftover_engines"].assert_not_called()
        self.assertFalse(hasattr(obj, "known_ctx"))


if __name__ == "__main__":
    unittest.main()
