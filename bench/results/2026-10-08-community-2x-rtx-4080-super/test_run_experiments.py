import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("experiment_runner", ROOT / "run_experiments.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class Plans(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / "main-plan.json").read_text())
        self.cfg = {"args": ["--pack", "p", "--native", "n", "--mtp", "m", "--prefill", "7", "--prefill=9", "--batch", "4",
                             "--remote-expert-opt", "--expert-cache-device1", "100"]}

    def test_controlled_flags_are_unique(self):
        rows = list(RUN.cases(self.cfg, self.plan, [0, 1]))
        self.assertEqual(len(rows), 12)
        for row in rows:
            args = row["args"]
            for flag in ("--prefill", "--pipeline-windows", "--adapt-async", "--layer-split"):
                self.assertEqual(args.count(flag), 1)
            for flag in ("--batch", "--remote-expert-opt", "--expert-cache-device1"):
                self.assertNotIn(flag, args)
            self.assertEqual(args[args.index("--prompt-cache") + 1], "0")
        self.assertEqual([r["arm"] for r in rows[:4]], list(reversed([r["arm"] for r in rows[4:8]])))

    def test_reversed_devices_are_preserved(self):
        self.assertTrue(all(r["devices"] == [1, 0] for r in RUN.cases(self.cfg, self.plan, [1, 0])))

    def test_invalid_device_lists(self):
        for devices in ([0], [0, 0], [-1, 1], [0, 1, 2], [False, 1]):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                list(RUN.cases(self.cfg, self.plan, devices))

    def test_mutable_profile_and_environment_rejected(self):
        for cfg in ({"args": ["--expert-profile-save=x"]},
                    {"args": [], "expert_profile_save": "x"}, {"args": [], "env": {"STRATA_X": "1"}}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                list(RUN.cases(cfg, self.plan, [0, 1]))

    def test_missing_flag_value_rejected(self):
        with self.assertRaises(ValueError):
            RUN.replace_flags(["--prefill", "--pack", "p"], {"--prefill": "8192"}, [], [], [])

    def test_prefill_controls_do_not_enable_pipeline(self):
        plan = json.loads((ROOT / "fixtures/prefill_ima_controls.json").read_text())
        rows = list(RUN.cases(self.cfg, plan, [0, 1]))
        self.assertEqual(len(rows), 5)
        by_arm = {row["arm"]: row for row in rows}
        self.assertEqual(by_arm["large-copy-control"]["environment"]["STRATA_GROUP_COPY"], "1")
        self.assertEqual(by_arm["large-step-sync"]["environment"]["STRATA_PF_STEP_SYNC"], "1")
        for row in rows:
            self.assertEqual("STRATA_GROUP_COPY" in row["environment"], row["arm"] == "large-copy-control")
            self.assertEqual("STRATA_PF_STEP_SYNC" in row["environment"], row["arm"] == "large-step-sync")
            self.assertEqual(row["args"].count("--vram-reserve-mib"), 1)
            self.assertEqual(row["args"][row["args"].index("--vram-reserve-mib") + 1], "690")
        self.assertTrue(all(r["args"][r["args"].index("--pipeline-windows") + 1] == "0" for r in rows))

    def test_prepared_input_integrity(self):
        inputs = {"plan_sha256": RUN.digest(self.plan), "prompts": []}
        for length in [1024, *self.plan["prompt_lengths"]]:
            ids = [1] * length
            inputs["prompts"].append({"length": length, "ids": ids, "ids_sha256": RUN.digest(ids)})
        RUN.verify_inputs(self.plan, inputs)
        changed = copy.deepcopy(inputs)
        changed["prompts"][0]["ids"][0] = 2
        with self.assertRaises(ValueError):
            RUN.verify_inputs(self.plan, changed)
        changed = copy.deepcopy(self.plan)
        changed["max_new"] += 1
        with self.assertRaises(ValueError):
            RUN.verify_inputs(changed, inputs)

    def test_gpu_hidden_refuses_before_subprocess(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}, clear=True), patch.object(RUN.subprocess, "check_output") as proc:
            with self.assertRaises(ValueError):
                RUN.execute(ROOT, {}, self.plan, {}, [0, 1], ROOT / "unused", 1)
            proc.assert_not_called()

    def test_execution_state_machine_with_mock_engine(self):
        plan = copy.deepcopy(self.plan)
        plan.update(repetitions=1, prompt_lengths=[4], requests_per_prompt=2)
        plan["arms"] = plan["arms"][:1]
        inputs = {"plan_sha256": RUN.digest(plan), "tokenizer_sha256": {}, "prompts": []}
        for length in [1024, 4]:
            ids = [1] * length
            inputs["prompts"].append({"length": length, "ids": ids, "ids_sha256": RUN.digest(ids)})
        engines = []

        class Engine:
            def __init__(self, exe, args, cwd, log, env):
                Path(log).write_text("mock engine for runner control-flow test only\n")
                self.info = {}
                self.last = {}
                self.closed = False
                self.calls = 0
                engines.append(self)

            def generate(self, ids, cap, sampling, cancelled):
                self.calls += 1
                self.last = {"prompt_ms": 1, "decode_ms": 1, "generated": 1}
                yield 7

            def close(self):
                self.closed = True

        def response(command, **kwargs):
            if command[0] == "git":
                return plan["source_commit"] if command[-1] == "HEAD" and "rev-parse" in command else ""
            return ""

        server = types.ModuleType("serve.server")
        server.StrataEngine = Engine
        server.child_env = lambda cfg: {}
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / "fake-engine"
            exe.write_bytes(b"not an executable: test never launches it")
            cfg = {"exe": str(exe), "tokenizer": directory, "args": self.cfg["args"]}
            out = Path(directory) / "results"
            with patch.dict(os.environ, {}, clear=True), patch.object(RUN.subprocess, "check_output", side_effect=response), patch.dict("sys.modules", {"serve.server": server}):
                self.assertEqual(RUN.execute(ROOT, cfg, plan, inputs, [0, 1], out, 10), 0)
            rows = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
            self.assertEqual(len(engines), 1)
            self.assertTrue(engines[0].closed)
            self.assertEqual(engines[0].calls, 3)
            self.assertEqual([row["warmup"] for row in rows], [True, False, False])
            self.assertEqual([row["request"] for row in rows[1:]], [0, 1])


if __name__ == "__main__":
    unittest.main()
