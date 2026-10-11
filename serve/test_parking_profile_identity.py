"""Real profile files and parking identities, with mock engines; no GPU/model."""
import os
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest import mock

from serve import server
from serve import test_idle_parking_service as idle_tests
from serve import test_request_parking_service as active_tests
from serve import test_resource_lease_service as lease_tests
from serve.routing_costs import runtime_key


class ParkingProfileIdentityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="strata-profile-identity-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.binary = self.root / "never-executed.bin"
        self.binary.write_bytes(b"identity fixture, never executed")
        self.model = self.root / "model.bin"
        self.model.write_bytes(b"model identity fixture")
        self.profile = self.root / "learned.bin"
        self.save_profile()
        nvml = mock.patch("serve.telemetry._Nvml")
        nvml.start().return_value.name.return_value = "fixture GPU"
        self.addCleanup(nvml.stop)
        cpu = mock.patch("serve.telemetry._cpu_name", return_value="fixture CPU")
        cpu.start()
        self.addCleanup(cpu.stop)

    def save_profile(self, reverse=False):
        pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        if reverse:
            pairs.reverse()
        table = [pairs.index((l, e)) for l in range(2) for e in range(2)]
        data = (b"STRP" + struct.pack("<5I", 1, 2, 2, 4, 4)
                + b"".join(struct.pack("<HH", *p) for p in pairs) + struct.pack("<4i", *table))
        previous = self.profile.stat().st_mtime_ns if self.profile.exists() else 1_000_000_000
        temporary = self.root / "learned.tmp"
        temporary.write_bytes(data)
        os.utime(temporary, ns=(previous + 2_000_000_000, previous + 2_000_000_000))
        os.replace(temporary, self.profile)

    def real_identity(self, svc):
        # Exercise relative/absolute aliases and the production startup selector.
        cfg = {"cwd": str(self.root), "expert_profile_save": str(self.profile),
               "expert_profile_save_every": 0}
        args = server.learned_profile_args(cfg, ["--expert-profile", "./learned.bin",
                                               "--native", str(self.model), "--pcie-frac", ".37"])
        svc.engine.spawn = (str(self.binary), args, str(self.root), None, {})
        del svc._parking_identity  # discard the lifecycle fixture's constant identity stub
        return svc

    def idle_service(self):
        return self.real_identity(idle_tests.IdleServiceTests().service())

    def save_on_unload(self, svc, reverse=False):
        unload = svc.engine.unload
        def saving_unload():
            unload()
            self.save_profile(reverse)
        svc.engine.unload = saving_unload

    def assert_idle_resume(self, reverse):
        helper = idle_tests.IdleServiceTests()
        svc = self.idle_service()
        before = self.profile.read_bytes()
        calibration = runtime_key(svc.engine)
        self.save_on_unload(svc, reverse)
        helper.park(svc)
        helper.ready(svc)
        if not reverse:
            self.assertEqual(before, self.profile.read_bytes())
        self.assertNotEqual(calibration, runtime_key(svc.engine))
        list(svc.load_events(threading.Event()))
        self.assertEqual((svc.engine.unloads, svc.engine.reloads), (1, 1))
        self.assertEqual(svc.idle_parking_status["state"], "ready")
        self.assertFalse(svc.fifo.locked())

    def test_idle_reload_after_same_content_atomic_save(self):
        self.assert_idle_resume(False)

    def test_idle_reload_after_learned_ranking_changes(self):
        self.assert_idle_resume(True)

    def test_active_request_resumes_exact_prefix_after_learned_save(self):
        helper = active_tests.ParkingServiceTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        svc = self.real_identity(helper.service([(list(b"abc"), True), (list(b"def"), False)]))
        self.save_on_unload(svc, True)
        events = list(svc.run([1, 2], False, [], 100, {"temperature": .8}, threading.Event()))
        first, second = svc.engine.calls
        self.assertEqual(second[0], first[0] + list(b"abc"))
        self.assertEqual(second[1], 97)
        self.assertEqual(second[2]["seed"], first[2]["seed"])
        self.assertEqual(sum(kind == "done" for kind, _ in events), 1)
        self.assertEqual((svc.engine.unloads, svc.engine.reloads), (1, 1))
        self.assertFalse(svc.fifo.locked())
        self.assertFalse(list(Path(helper.temp.name).iterdir()))

    def test_tool_lease_return_reloads_after_learned_save(self):
        helper = lease_tests.LeaseServiceTests()
        helper.setUp()
        self.addCleanup(helper.tearDown)
        self.addCleanup(helper.doCleanups)
        svc = self.real_identity(helper.svc)
        self.save_on_unload(svc, True)
        row = helper.acquire()
        self.assertTrue(svc.observe_tool_lease())
        self.assertEqual(helper.state(row), "ready")
        self.assertEqual(svc.engine.reloads, 0)
        helper.release(row)
        svc.idle_parked["admission"].observe = lambda *_: {"ready": True}
        list(svc.load_events(threading.Event()))
        self.assertEqual((svc.engine.unloads, svc.engine.reloads), (1, 1))
        self.assertEqual(svc.idle_parking_status["state"], "ready")

    def test_profile_without_matching_save_target_keeps_strict_identity(self):
        for save in (None, str(self.root / "other.bin")):
            with self.subTest(save=save):
                svc = self.idle_service()
                args = svc.engine.spawn[1]
                i = args.index("--expert-profile-save")
                if save is None:
                    del args[i:i + 2]
                else:
                    args[i + 1] = save
                before = svc._parking_identity()
                self.save_profile(True)
                self.assertNotEqual(before, svc._parking_identity())

    def test_relative_save_target_uses_engine_cwd(self):
        svc = self.idle_service()
        args = svc.engine.spawn[1]
        args[args.index("--expert-profile-save") + 1] = "./learned.bin"
        args[args.index("--expert-profile") + 1] = "./learned.bin"
        before = svc._parking_identity()
        self.save_profile(True)
        self.assertEqual(before, svc._parking_identity())

    def test_learned_profile_inside_asset_directory_is_normalized_once_per_reference(self):
        svc = self.idle_service()
        args = svc.engine.spawn[1]
        args[args.index("--native") + 1] = str(self.root)
        before = svc._parking_identity()
        calibration = runtime_key(svc.engine)
        self.save_profile(True)
        self.assertEqual(before, svc._parking_identity())
        self.assertNotEqual(calibration, runtime_key(svc.engine))
        self.model.write_bytes(b"changed model within the same directory")
        self.assertNotEqual(before, svc._parking_identity())

    def test_slot_count_and_expert_membership_changes_invalidate(self):
        svc = self.idle_service()
        original = self.profile.read_bytes()
        before = svc._parking_identity()
        self.profile.write_bytes(original[:16] + struct.pack("<I", 5) + original[20:])
        self.assertNotEqual(before, svc._parking_identity())
        # Two valid partial rankings with identical format, layout and count.
        header = b"STRP" + struct.pack("<5I", 1, 2, 2, 2, 2)
        self.profile.write_bytes(header + struct.pack("<4H", 0, 0, 0, 1))
        partial = svc._parking_identity()
        self.profile.write_bytes(header + struct.pack("<4H", 0, 0, 1, 1))
        self.assertNotEqual(partial, svc._parking_identity())

    def test_binary_model_config_context_and_vision_changes_still_invalidate(self):
        for changed in ("binary", "model", "config", "environment", "affinity", "context", "vision", "path"):
            with self.subTest(changed=changed):
                svc = self.idle_service()
                before = svc._parking_identity()
                if changed == "binary":
                    self.binary.write_bytes(self.binary.read_bytes() + b"changed")
                elif changed == "model":
                    self.model.write_bytes(self.model.read_bytes() + b"changed")
                elif changed == "config":
                    svc.engine.spawn[1].extend(["--kv", "fp16"])
                elif changed == "environment":
                    svc.engine.spawn[4]["STRATA_KV_GROW"] = "1"
                elif changed == "affinity":
                    svc.engine.spawn += (False, [0, 2])
                elif changed == "context":
                    svc.engine.max_context += 1
                elif changed == "vision":
                    svc.vision = object()
                else:
                    other = self.root / "other-profile.bin"
                    other.write_bytes(self.profile.read_bytes())
                    args = svc.engine.spawn[1]
                    for flag in ("--expert-profile", "--expert-profile-save"):
                        args[args.index(flag) + 1] = str(other)
                self.assertNotEqual(before, svc._parking_identity())

    def test_changed_model_still_prevents_reload_after_legitimate_save(self):
        helper = idle_tests.IdleServiceTests()
        svc = self.idle_service()
        self.save_on_unload(svc, True)
        helper.park(svc)
        helper.ready(svc)
        self.model.write_bytes(b"different model")
        with self.assertRaisesRegex(ValueError, "identity"):
            list(svc.load_events(threading.Event()))
        self.assertEqual(svc.engine.reloads, 0)
        self.assertFalse(svc.fifo.locked())

    def test_profile_layout_change_invalidates_identity(self):
        svc = self.idle_service()
        before = svc._parking_identity()
        # A different model layout with the same number of pairs and file size.
        pairs = [(0, n) for n in range(4)]
        self.profile.write_bytes(b"STRP" + struct.pack("<5I", 1, 1, 4, 4, 4)
                                 + b"".join(struct.pack("<HH", *p) for p in pairs) + bytes(16))
        self.assertNotEqual(before, svc._parking_identity())

    def test_invalid_learned_profile_cannot_authorize_reload(self):
        original = self.profile.read_bytes()
        cases = [b"bad header", original[:27], b"NOPE" + original[4:],
                 original[:4] + struct.pack("<I", 2) + original[8:],
                 original[:24] + struct.pack("<HH", 5, 0) + original[28:],
                 original[:28] + original[24:28] + original[32:]]
        for data in cases:
            with self.subTest(data=data):
                svc = self.idle_service()
                self.profile.write_bytes(data)
                with self.assertRaises((ValueError, OSError)):
                    svc._parking_identity()
                self.profile.write_bytes(original)

    def test_missing_learned_profile_cannot_authorize_reload(self):
        svc = self.idle_service()
        self.profile.unlink()
        with self.assertRaises((ValueError, OSError)):
            svc._parking_identity()


if __name__ == "__main__":
    unittest.main()
