"""CPU-only bounded memory admission through the real pre-READY constructor."""

import copy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from serve import server
from serve import test_q36_kv_contract as kv_tests
from strata_control import private_files

SHA, VERIFIED = kv_tests.SHA, kv_tests.VERIFIED

CAP = {
    "schema": 1,
    "family": "qwen35moe",
    "modes": ["bounded-experts-v1"],
    "flag": "--q36-memory",
    "requiresExpertStore": "mapped-pack",
    "requiresExplicitSlots": True,
    "scope": "expert-working-storage",
    "totalProcessBound": False,
    "qualification": "experimental-unqualified",
}
BOUND = {**CAP, "binarySha256": SHA}
ARGS = [
    "--max-context",
    "256",
    "--q36-expert-store",
    "mapped-pack",
    "--expert-slots",
    "5",
    "--q36-memory",
    "bounded-experts-v1",
]
ENV = {
    "STRATA_REQUIRE_Q36_KV": "1",
    "STRATA_Q36_MEMORY_MODE": "bounded-experts-v1",
    "STRATA_Q36_MEMORY_HOST_SHA256": SHA,
    "STRATA_Q36_MEMORY_CAPABILITY": json.dumps(BOUND),
}
RECEIPT = (
    "INFO q36_memory_mode=bounded-experts-v1 "
    "q36_memory_scope=expert-working-storage q36_memory_total_process_bound=0 "
    "q36_memory_cpu_stage_bytes=18512896 q36_memory_h2d_stage_bytes=67108864 "
    "q36_memory_io_scratch_bytes=1048576\n"
)


def constructor(receipt: str = RECEIPT) -> tuple:
    ns = kv_tests.Q36KvContract._constructor_namespace()
    ns["q36_probe_capabilities"] = mock.Mock(
        return_value={"kvCapability": None, "memoryCapability": BOUND}
    )
    proc = mock.Mock(
        pid=1, stdin=io.StringIO(), stdout=io.StringIO(receipt + "READY 256 stop\n")
    )
    ns["popen"] = mock.Mock(return_value=proc)
    return ns, proc


def config() -> dict:
    return {
        "arch": "qwen35moe",
        "q36_memory": "bounded-experts-v1",
        "args": ARGS,
        "q36_run": {
            "schemaVersion": 1,
            "generation": "launch",
            "backend": {
                "family": "qwen36",
                "binarySha256": SHA,
                "memoryCapability": BOUND,
            },
            "requested": {"q36_memory": "bounded-experts-v1"},
            "planned": {"q36_memory": "bounded-experts-v1"},
            "effective": {"q36_memory": "bounded-experts-v1"},
            "argv": ARGS,
        },
    }


class Q36MemoryContract(unittest.TestCase):
    def test_spec_f2_empty_raw_memory_is_rejected_in_each_argument_layer(self) -> None:
        for layer in ("args", "user_args"):
            for requested in (None, "legacy", "bounded-experts-v1"):
                cfg = {"arch": "qwen35moe", "args": list(ARGS[:-2])}
                cfg[layer] = [
                    *(cfg["args"] if layer == "args" else []),
                    "--q36-memory",
                    "",
                ]
                if requested is not None:
                    cfg["q36_memory"] = requested
                before = copy.deepcopy(cfg)
                with self.subTest(layer=layer, requested=requested), mock.patch.object(
                    server, "q36_probe_capabilities"
                ) as probe:
                    with self.assertRaisesRegex(ValueError, "memory"):
                        server.child_env(cfg)
                    with self.assertRaisesRegex(ValueError, "memory"):
                        server.native_cache_args(cfg, list(cfg["args"]), "fake")
                    probe.assert_not_called()
                    self.assertEqual(cfg, before)
        ns, _ = constructor()
        with self.assertRaisesRegex(ValueError, "memory"):
            ns["ReadyPrefix"](
                "fake", ["--q36-memory", ""], env={"STRATA_REQUIRE_Q36_KV": "1"}
            )
        ns["popen"].assert_not_called()
        ns["end_leftover_engines"].assert_not_called()

    def test_spec_f3_real_cli_preflight_accepts_the_runtime_user_layer(self) -> None:
        class PreflightAccepted(Exception):
            pass

        for base in (
            ["--max-context", "256"],
            [
                "--max-context",
                "256",
                "--q36-expert-store",
                "legacy",
                "--expert-slots",
                "0",
                "--q36-memory",
                "legacy",
            ],
        ):
            cfg = {"arch": "qwen35moe", "args": base, "user_args": ARGS[2:]}
            with self.subTest(base=base), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps(cfg), encoding="utf-8")
                with mock.patch.object(
                    private_files, "adopt_user_defaults"
                ), mock.patch.object(
                    server.lifeline, "install_stop_handlers"
                ), mock.patch.object(
                    server, "rttime_limit_note", return_value=None
                ), mock.patch.object(
                    server, "apply_saved_tuning", side_effect=lambda c: c
                ), mock.patch.object(
                    server, "Server", side_effect=PreflightAccepted
                ) as network, mock.patch.object(
                    server, "popen"
                ) as spawn, mock.patch.object(
                    server, "q36_probe_capabilities"
                ) as probe, mock.patch.dict(
                    os.environ, {}, clear=True
                ), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    with self.assertRaises(PreflightAccepted):
                        server._main(["--engine", "strata", "--config", str(path)])
                    network.assert_called_once()
                    spawn.assert_not_called()
                    probe.assert_not_called()

            # The same user precedence must survive the actual runtime builder,
            # then independently satisfy the actual-child constructor gate.
            with mock.patch.object(
                server.drafting, "detect_hardware", return_value=("cpu", [])
            ), mock.patch.object(
                server, "drafting_vram_mib", return_value=None
            ), mock.patch.object(
                server, "drafting_card_class", return_value="cpu"
            ), mock.patch.object(
                server,
                "q36_probe_capabilities",
                return_value={"kvCapability": None, "memoryCapability": BOUND},
            ), mock.patch.dict(
                os.environ, {}, clear=True
            ):
                runtime = server.engine_args(cfg, "fake", announce=False)
                env = server.child_env(cfg)
            self.assertEqual(runtime, ARGS)
            self.assertEqual(cfg["args"], base)
            self.assertEqual(cfg["user_args"], ARGS[2:])
            ns, _ = constructor()
            self.assertEqual(ns["ReadyPrefix"]("fake", runtime, env=env).known_ctx, 256)

    def test_native_slot_decimal_range_is_checked_before_any_probe(self) -> None:
        ns, _ = constructor()
        args = list(ARGS)
        args[args.index("--expert-slots") + 1] = "2147483648"
        with self.assertRaisesRegex(ValueError, "slots"):
            ns["ReadyPrefix"]("fake", args, env=ENV)
        ns["popen"].assert_not_called()
        ns["q36_probe_capabilities"].assert_not_called()
        args[args.index("--expert-slots") + 1] = "05"
        self.assertEqual(ns["ReadyPrefix"]("fake", args, env=ENV).known_ctx, 256)

    def test_help_schema_is_independent_and_the_bounded_query_is_shared(self) -> None:
        line = "Q36_MEMORY_CAPABILITIES " + json.dumps(CAP)
        self.assertEqual(server.q36_parse_memory_capabilities(line), CAP)
        self.assertEqual(
            server.q36_parse_memory_capabilities(
                line.replace('["bounded-experts-v1"]', "[]")
            ),
            {**CAP, "modes": []},
        )
        for text in (
            line + "\n" + line,
            "x" * 65536 + line,
            line + "junk",
            line.replace('"schema": 1', '"schema": true'),
            line.replace('"requiresExplicitSlots": true', '"requiresExplicitSlots": 1'),
            line.replace('"totalProcessBound": false', '"totalProcessBound": true'),
            line.replace('"scope":', '"extra": 1, "scope":'),
            line.replace("qwen35moe", "qwen4exp"),
        ):
            self.assertIsNone(server.q36_parse_memory_capabilities(text))
        proc = mock.MagicMock()
        proc.__enter__.return_value = proc
        proc.stdout = io.BytesIO((kv_tests.LINE + "\n" + line).encode())
        proc.wait.return_value = 0
        with mock.patch.dict(
            os.environ,
            {"STRATA_BACKTRACE": "0", "STRATA_API_KEY": "synthetic-probe-secret"},
        ), mock.patch.object(
            server.subprocess, "Popen", return_value=proc
        ) as spawn, mock.patch.object(server, "q36_binary_sha256", return_value=SHA):
            self.assertEqual(
                server.q36_probe_capabilities("fake"),
                {"kvCapability": VERIFIED, "memoryCapability": BOUND},
            )
            spawn.assert_called_once()
            self.assertEqual(spawn.call_args.args[0], ["fake", "--help"])
            self.assertTrue(
                set(k.upper() for k in spawn.call_args.kwargs["env"]).issubset(
                    {
                        "PATH", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH",
                        "CUDA_PATH", "GGML_NO_BACKTRACE",
                    }
                )
            )
            self.assertEqual(spawn.call_args.kwargs["env"].get("GGML_NO_BACKTRACE"), "1")
            self.assertNotIn("STRATA_API_KEY", spawn.call_args.kwargs["env"])
        proc.stdout = io.BytesIO(line.encode())
        with mock.patch.object(
            server.subprocess, "Popen", return_value=proc
        ), mock.patch.object(server, "q36_binary_sha256", side_effect=[SHA, "cd" * 32]):
            self.assertEqual(
                server.q36_probe_capabilities("fake"),
                {"kvCapability": None, "memoryCapability": None},
            )

    def test_actual_host_builder_crosses_python_admission_with_exact_mode(self) -> None:
        root = Path(__file__).resolve().parents[1]
        output = subprocess.run(
            [
                os.environ.get("STRATA_TEST_NODE_EXE", "node"),
                str(
                    root / "apps/host/tests/helpers/native-family-consumer-fixture.mjs"
                ),
                "--memory",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        cfg = json.loads(output.stdout)["bounded"]
        self.assertEqual(cfg["arch"], "qwen35moe")
        cap = cfg["q36_run"]["backend"]["memoryCapability"]
        ns, proc = constructor(RECEIPT + "INFO q36_private_cache=2\n")
        ns["q36_probe_capabilities"].return_value["memoryCapability"] = cap
        ns["q36_binary_sha256"] = lambda exe: cap["binarySha256"]
        with mock.patch.dict(os.environ, {}, clear=True):
            env = server.child_env(cfg)
        args = list(cfg["args"])
        with mock.patch.object(
            server,
            "q36_probe_capabilities",
            return_value={
                "memoryCapability": cap,
                "kvCapability": None,
            },
        ) as probe:
            args += server.native_cache_args(cfg, args, "fake")
            probe.assert_called_once_with("fake")
        obj = ns["ReadyPrefix"]("fake", args, env=env)
        self.assertEqual(obj.known_ctx, 256)
        self.assertEqual(env["STRATA_REQUIRE_Q36_PRIVATE_CACHE"], "2")
        self.assertEqual(env["STRATA_Q36_MEMORY_HOST_SHA256"], cap["binarySha256"])
        self.assertEqual(ns["popen"].call_args.args[1], ["fake", "--serve", *args])
        proc.kill.assert_not_called()

    def test_real_constructor_requires_memory_receipt_even_without_kv(self) -> None:
        for receipt in (
            "",
            RECEIPT.replace("bounded-experts-v1", "legacy"),
            RECEIPT.replace("expert-working-storage", "total-process"),
            RECEIPT.replace("bound=0", "bound=1"),
            RECEIPT.replace("bytes=18512896", "bytes=0"),
            RECEIPT.replace(" q36_memory_io_scratch_bytes=1048576", ""),
        ):
            with self.subTest(receipt=receipt):
                ns, proc = constructor(receipt)
                with self.assertRaisesRegex(server.EngineDied, "memory"):
                    ns["ReadyPrefix"]("fake", ARGS, env=ENV)
                proc.kill.assert_called_once()

    def test_valid_receipt_requires_independent_actual_binary_proof(self) -> None:
        ns, proc = constructor()
        obj = ns["ReadyPrefix"]("fake", ARGS, env=ENV)
        self.assertEqual(obj.known_ctx, 256)
        ns["q36_probe_capabilities"].assert_called_once_with("fake")
        proc.kill.assert_not_called()
        self.assertEqual(
            ns["popen"].call_args.kwargs["env"]["STRATA_Q36_MEMORY_MODE"],
            "bounded-experts-v1",
        )
        for actual in (
            None,
            {**BOUND, "binarySha256": "cd" * 32},
            {**BOUND, "modes": []},
        ):
            ns, _ = constructor()
            ns["q36_probe_capabilities"].return_value["memoryCapability"] = actual
            with self.assertRaisesRegex(ValueError, "memory|binary"):
                ns["ReadyPrefix"]("fake", ARGS, env=ENV)
            ns["popen"].assert_not_called()
            ns["end_leftover_engines"].assert_not_called()
        ns, _ = constructor()
        ns["q36_binary_sha256"] = lambda exe: "cd" * 32
        with self.assertRaisesRegex(OSError, "binary changed"):
            ns["ReadyPrefix"]("fake", ARGS, env=ENV)
        ns["popen"].assert_not_called()
        ns["end_leftover_engines"].assert_not_called()

    def test_retained_expectations_cannot_be_downgraded_by_config_or_argv(self) -> None:
        bad = []
        for key, value in (
            ("arch", "qwen4exp"),
            ("q36_memory", "legacy"),
            ("args", ARGS[:-2]),
            ("args", ARGS[:-4] + ARGS[-2:]),
            ("args", ARGS + ["--q36-memory", "legacy"]),
        ):
            cfg = copy.deepcopy(config())
            cfg[key] = value
            bad.append(cfg)
        for section in ("requested", "planned", "effective"):
            cfg = copy.deepcopy(config())
            cfg["q36_run"][section]["q36_memory"] = "legacy"
            bad.append(cfg)
        for cap in (
            None,
            {**BOUND, "binarySha256": "cd" * 32},
            {**BOUND, "totalProcessBound": True},
        ):
            cfg = copy.deepcopy(config())
            cfg["q36_run"]["backend"]["memoryCapability"] = cap
            bad.append(cfg)
        for cfg in bad:
            with self.subTest(cfg=cfg), mock.patch.object(
                server, "q36_probe_kv_capabilities"
            ) as probe:
                with self.assertRaisesRegex(ValueError, "Q36|q36_run|memory"):
                    server.native_cache_args(cfg, list(cfg["args"]), "fake")
                with self.assertRaises(ValueError):
                    server.child_env(cfg)
                probe.assert_not_called()

    def test_config_projects_owned_memory_env_over_ambient_values(self) -> None:
        cfg = config()
        cfg["env"] = {key: "wrong" for key in ENV if "MEMORY" in key}
        with mock.patch.dict(os.environ, cfg["env"], clear=True):
            env = server.child_env(cfg)
        self.assertEqual(env["STRATA_Q36_MEMORY_MODE"], "bounded-experts-v1")
        self.assertEqual(env["STRATA_Q36_MEMORY_HOST_SHA256"], SHA)
        self.assertEqual(json.loads(env["STRATA_Q36_MEMORY_CAPABILITY"]), BOUND)

    def test_environment_and_arguments_must_agree_before_probe_or_spawn(self) -> None:
        for delta, args in (
            ({"STRATA_Q36_MEMORY_MODE": "legacy"}, ARGS),
            ({"STRATA_REQUIRE_Q36_KV": "0"}, ARGS),
            ({"STRATA_Q36_MEMORY_HOST_SHA256": "cd" * 32}, ARGS),
            ({"STRATA_Q36_MEMORY_CAPABILITY": "{}"}, ARGS),
            ({}, ARGS[:-2]),
            ({}, [*ARGS[:-3], "0", *ARGS[-2:]]),
        ):
            ns, _ = constructor()
            with self.assertRaisesRegex(ValueError, "Q36|memory|adapter"):
                ns["ReadyPrefix"]("fake", args, env={**ENV, **delta})
            ns["popen"].assert_not_called()
            ns["q36_probe_capabilities"].assert_not_called()
            ns["end_leftover_engines"].assert_not_called()

    def test_constructor_snapshot_keeps_mode_when_callers_mutate_inputs(self) -> None:
        ns, proc = constructor()
        args, env = list(ARGS), dict(ENV)

        def probe(exe: str) -> dict:
            args[-1] = "legacy"
            env["STRATA_Q36_MEMORY_MODE"] = "legacy"
            return {"kvCapability": None, "memoryCapability": BOUND}

        ns["q36_probe_capabilities"].side_effect = probe
        obj = ns["ReadyPrefix"]("fake", args, env=env)
        self.assertEqual(obj.known_ctx, 256)
        self.assertEqual(ns["popen"].call_args.args[1][-1], "bounded-experts-v1")
        proc.kill.assert_not_called()

    def test_legacy_stays_flag_free_and_needs_no_memory_receipt(self) -> None:
        ns, proc = constructor("")
        obj = ns["ReadyPrefix"](
            "old", ["--q36-memory", "legacy"], env={"STRATA_REQUIRE_Q36_KV": "1"}
        )
        self.assertEqual(obj.known_ctx, 256)
        self.assertEqual(ns["popen"].call_args.args[1], ["old", "--serve"])
        ns["q36_probe_capabilities"].assert_not_called()
        proc.kill.assert_not_called()

    def test_memory_does_not_replace_private_or_typed_kv_requirements(self) -> None:
        for env in (
            {**ENV, "STRATA_REQUIRE_Q36_PRIVATE_CACHE": "2"},
            {**ENV, "STRATA_Q36_KV_HOST_SHA256": SHA},
        ):
            ns, proc = constructor()
            ns["q36_probe_capabilities"].return_value["kvCapability"] = VERIFIED
            with self.assertRaisesRegex(
                server.EngineDied, "private conversation|KV layout"
            ):
                ns["ReadyPrefix"]("fake", ARGS, env=env)
            proc.kill.assert_called_once()
