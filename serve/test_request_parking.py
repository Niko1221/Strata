"""Exact request journals and reload admission, without an engine or model."""
import errno
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from serve.request_parking import (GIB, MIB, ParkingError, ParkingJournal,
                                   RequestParking, ResumeAdmission, effective_sampling)


class SeedTests(unittest.TestCase):
    def test_missing_and_zero_seed_selected_once_without_changing_input(self):
        for source in ({"reasoning_effort": "high"}, {"seed": 0}):
            with patch("serve.request_parking.secrets.randbelow", return_value=123):
                selected = effective_sampling(source)
            self.assertEqual(selected["seed"], 124)
            self.assertNotEqual(source.get("seed"), 124)
            with patch("serve.request_parking.secrets.randbelow", side_effect=AssertionError("new seed")):
                self.assertEqual(effective_sampling(selected), selected)

    def test_invalid_seed_or_nonfinite_sampling_rejected(self):
        for seed in (True, False, -1, 1.5, "42", 2**64):
            with self.subTest(seed=seed), self.assertRaises(ParkingError):
                effective_sampling({"seed": seed})
        with self.assertRaises(ParkingError):
            effective_sampling({"temperature": float("nan")})


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.journals = []

    def tearDown(self):
        for journal in self.journals:
            journal.remove()
        self.temp.cleanup()

    def journal(self, identity="model-build-config-a"):
        journal = ParkingJournal(self.base, identity, "request-one")
        self.journals.append(journal)
        return journal

    def write(self, journal, **kwargs):
        params = {"prefix_ids": [1, 203040, 0, 2147483647], "sampling": {"seed": 42, "reasoning_effort": "high"},
                  "initial_budget": 4096, "consumed_output": 31, "sequence": 2}
        params.update(kwargs)
        return journal.write(**params)

    def test_exact_ids_sampling_remaining_and_identity_round_trip(self):
        identity = {"model": "abc", "build": "def", "config": "ghi"}
        journal = self.journal(identity)
        ids = [999, 12, 1000, 0, 12]
        record = self.write(journal, prefix_ids=ids)
        ids.append(888)
        self.assertEqual(record["prefix_ids"], ids[:-1])
        self.assertEqual(record["remaining_budget"], 4065)
        self.assertEqual(record["sampling"], {"seed": 42, "reasoning_effort": "high"})
        self.assertEqual(journal.load(identity)["prefix_ids"], ids[:-1])
        self.assertEqual(record["sequence"], 2)

    def test_failed_atomic_replace_preserves_previous_valid_record(self):
        journal = self.journal()
        before = self.write(journal)
        with patch("serve.request_parking.os.replace", side_effect=OSError(errno.ENOSPC, "disk full")):
            with self.assertRaises(OSError):
                self.write(journal, consumed_output=55, sequence=3)
        self.assertEqual(journal.load(journal.identity), before)
        self.assertEqual([p.name for p in journal.directory.iterdir()], ["request.json"])

    def test_fsync_failure_does_not_create_a_committed_record(self):
        journal = self.journal()
        with patch("serve.request_parking.os.fsync", side_effect=OSError(errno.ENOSPC, "disk full")):
            with self.assertRaises(OSError):
                self.write(journal)
        self.assertFalse(journal.path.exists())
        self.assertEqual(list(journal.directory.iterdir()), [])

    def test_corrupted_or_truncated_record_is_rejected(self):
        journal = self.journal()
        self.write(journal)
        envelope = json.loads(journal.path.read_text())
        envelope["record"]["prefix_ids"][0] += 1
        journal.path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ParkingError, "checksum"):
            journal.load(journal.identity)
        journal.path.write_text('{"broken":')
        with self.assertRaises(ParkingError):
            journal.load(journal.identity)

    def test_identity_mismatch_rejected_before_resume(self):
        journal = self.journal()
        self.write(journal)
        with self.assertRaisesRegex(ParkingError, "identity changed"):
            journal.load("different-build")

    def test_bad_tokens_budgets_and_missing_seed_rejected(self):
        journal = self.journal()
        for changes in ({"prefix_ids": []}, {"prefix_ids": [True]}, {"prefix_ids": [-1]},
                        {"consumed_output": 4097}, {"initial_budget": 0}, {"sequence": -1},
                        {"prefix_ids": None}, {"initial_budget": "4096"}, {"consumed_output": None},
                        {"sampling": {"temperature": 0}}):
            with self.subTest(changes=changes), self.assertRaises(ParkingError):
                self.write(journal, **changes)
        self.assertFalse(journal.path.exists())

    def test_zero_remaining_budget_round_trips_without_reset(self):
        journal = self.journal()
        self.assertEqual(self.write(journal, consumed_output=4096)["remaining_budget"], 0)

    def test_embedding_copied_hashed_and_independent_of_original_path(self):
        journal = self.journal()
        original = self.base / "source.bin"
        original.write_bytes(b"\x00\x01image-embedding\x00" * 20)
        record = self.write(journal, embedding=original)
        saved = Path(record["embedding_path"])
        self.assertEqual(saved.parent, journal.directory)
        self.assertEqual(saved.read_bytes(), original.read_bytes())
        original.unlink()
        self.assertEqual(journal.load(journal.identity)["embedding_path"], str(saved))
        saved.write_bytes(b"x" * saved.stat().st_size)
        with self.assertRaisesRegex(ParkingError, "embedding checksum"):
            journal.load(journal.identity)

    def test_cleanup_only_deletes_owned_files(self):
        first, second = self.journal(), self.journal()
        self.write(first)
        self.write(second)
        unrelated = first.directory / "unowned.txt"
        unrelated.write_text("preserve")
        with self.assertRaises(OSError):
            first.remove()
        self.assertEqual(unrelated.read_text(), "preserve")
        self.assertTrue(second.path.exists())
        unrelated.unlink()
        first.remove()
        first.remove()  # Idempotent only after successful removal.
        self.assertTrue(second.path.exists())

    def test_directory_identity_replacement_is_refused(self):
        journal = self.journal()
        self.write(journal)
        moved = journal.directory.with_name(journal.directory.name + "-moved")
        journal.directory.rename(moved)
        journal.directory.mkdir()
        try:
            with self.assertRaisesRegex(ParkingError, "directory changed"):
                journal.remove()
        finally:
            journal.directory.rmdir()
            moved.rename(journal.directory)

    def test_another_requests_valid_journal_cannot_be_replayed(self):
        first, second = self.journal(), self.journal()
        self.write(first)
        self.write(second)
        second.path.write_bytes(first.path.read_bytes())
        with self.assertRaisesRegex(ParkingError, "active request"):
            second.load(second.identity)

    @unittest.skipUnless(os.name == "nt", "Windows private DACL assertion")
    def test_windows_non_acl_or_unreadable_volume_never_creates_journal(self):
        with patch("serve.request_parking._windows_volume_flags", return_value=0):
            with self.assertRaisesRegex(ParkingError, "persistent ACLs"):
                ParkingJournal(self.base, "test")
        self.assertEqual(list(self.base.iterdir()), [])
        with patch("serve.request_parking._windows_volume_flags", side_effect=OSError("volume unavailable")):
            with self.assertRaisesRegex(OSError, "volume unavailable"):
                ParkingJournal(self.base, "test")
        self.assertEqual(list(self.base.iterdir()), [])

    @unittest.skipUnless(os.name == "nt", "Windows private DACL assertion")
    def test_windows_private_directory_has_protected_two_principal_dacl(self):
        import ctypes
        from ctypes import wintypes
        journal = self.journal()
        self.write(journal)
        adv = ctypes.WinDLL("advapi32", use_last_error=True)
        adv.GetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p,
                                         wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        adv.GetSecurityDescriptorControl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.WORD),
                                                     ctypes.POINTER(wintypes.DWORD)]
        adv.GetSecurityDescriptorDacl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
                                                  ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL)]
        adv.GetAclInformation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.c_int]
        needed = wintypes.DWORD()
        adv.GetFileSecurityW(str(journal.directory), 4, None, 0, ctypes.byref(needed))
        self.assertGreater(needed.value, 0)
        buffer = ctypes.create_string_buffer(needed.value)
        self.assertTrue(adv.GetFileSecurityW(str(journal.directory), 4, buffer, needed, ctypes.byref(needed)))
        control, revision = wintypes.WORD(), wintypes.DWORD()
        self.assertTrue(adv.GetSecurityDescriptorControl(buffer, ctypes.byref(control), ctypes.byref(revision)))
        self.assertTrue(control.value & 0x1000)  # SE_DACL_PROTECTED: no broad parent inheritance.
        present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), ctypes.c_void_p()
        self.assertTrue(adv.GetSecurityDescriptorDacl(buffer, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)))
        self.assertTrue(present.value and dacl.value)
        class AclSize(ctypes.Structure):
            _fields_ = [("ace_count", wintypes.DWORD), ("bytes_used", wintypes.DWORD), ("bytes_free", wintypes.DWORD)]
        info = AclSize()
        self.assertTrue(adv.GetAclInformation(dacl, ctypes.byref(info), ctypes.sizeof(info), 2))
        self.assertEqual(info.ace_count, 2)  # Current user and SYSTEM.

    @unittest.skipUnless(os.name != "nt", "POSIX mode assertion")
    def test_posix_journal_has_private_permissions(self):
        journal = self.journal()
        self.write(journal)
        self.assertEqual(journal.directory.stat().st_mode & 0o077, 0)
        self.assertEqual(journal.path.stat().st_mode & 0o077, 0)


def sample(now, ram=50, gpu=8000, commit=55):
    return {"sampled_at": now, "ram_total": 64 * GIB, "ram_used": (64 - ram) * GIB,
            "gpu_mem_total": 8192 * MIB, "gpu_mem_used": (8192 - gpu) * MIB,
            "ram_commit_required": True, "ram_commit_available": commit * GIB}


class AdmissionTests(unittest.TestCase):
    def admission(self):
        return ResumeAdmission({"ram_bytes": 40 * GIB, "commit_bytes": 40 * GIB, "gpu_bytes": 7000 * MIB},
                               recovery_seconds=5, require_commit=True)

    def test_full_reload_footprint_and_stable_recovery_required(self):
        a = self.admission()
        self.assertFalse(a.observe(sample(1, gpu=500), 1)["ready"])
        for stamp in range(2, 7):
            self.assertFalse(a.observe(sample(stamp), stamp)["ready"])
        self.assertTrue(a.observe(sample(7), 7)["ready"])

    def test_invalid_direct_admission_limits_are_rejected(self):
        footprint = {"ram_bytes": GIB, "commit_bytes": GIB, "gpu_bytes": GIB}
        for changes in ({"ram_headroom_gib": -1}, {"vram_headroom_mib": 0}, {"recovery_seconds": 0},
                        {"retry_seconds": float("nan")}, {"require_commit": 0}):
            with self.subTest(changes=changes), self.assertRaises(ParkingError):
                ResumeAdmission(footprint, **changes)

    def test_physical_or_commit_pressure_blocks_reload_independently(self):
        for reading in (sample(1, ram=42), sample(1, commit=42), sample(1, gpu=7200)):
            self.assertEqual(self.admission().observe(reading, 1)["reason"], "reload_footprint_unavailable")

    def test_missing_stale_native_or_invalid_sensor_never_admits(self):
        for changes in ({"sampled_at": -20}, {"sampled_at": 20}, {"ram_commit_available": None},
                        {"gpu_mem_used": float("nan")}, {"native_capacity_required": True}):
            a = self.admission()
            reading = {**sample(1), **changes}
            self.assertFalse(a.observe(reading, 1)["ready"])
            self.assertIsNone(a.since)

    def test_missing_commit_cannot_hide_behind_physical_ram(self):
        reading = sample(1)
        del reading["ram_commit_available"]
        self.assertEqual(self.admission().observe(reading, 1)["reason"], "commit_unavailable")

    def test_replayed_readings_and_clock_gaps_restart_dwell(self):
        a = self.admission()
        a.observe(sample(1), 1)
        self.assertEqual(a.observe(sample(1), 2)["reason"], "telemetry_not_advancing")
        a.observe(sample(3), 3)
        self.assertFalse(a.observe(sample(30), 30)["ready"])
        self.assertEqual(a.since, 30)

    def test_failed_reload_backoff_is_bounded_and_requires_new_dwell(self):
        a = self.admission()
        a.failed(1)
        self.assertEqual(a.retry_at, 31)
        self.assertEqual(a.observe(sample(2), 2)["reason"], "reload_backoff")
        for stamp in range(31, 36):
            self.assertFalse(a.observe(sample(stamp), stamp)["ready"])
        self.assertTrue(a.observe(sample(36), 36)["ready"])
        for _ in range(20):
            a.failed(100)
        self.assertEqual(a.retry_at, 400)


class ConfigAndPressureTests(unittest.TestCase):
    def config(self):
        return RequestParking({"enabled": True, "directory": str(Path(tempfile.gettempdir()).resolve()),
                               "pressure_seconds": 4})

    def test_disabled_is_default_and_cannot_create_journal(self):
        p = RequestParking()
        self.assertFalse(p.should_park({"action": "wait", "reason": "ram_floor"}, 1))
        with self.assertRaises(ParkingError):
            p.create_journal("identity")

    def test_enabled_requires_explicit_absolute_directory(self):
        for config in ({"enabled": True}, {"enabled": True, "directory": "relative"},
                       {"enabled": True, "directory": str(Path(tempfile.gettempdir()).resolve()), "unknown": 1}):
            with self.assertRaises(ParkingError):
                RequestParking(config)

    def test_only_persistent_critical_pressure_parks(self):
        p = self.config()
        for stamp in range(1, 5):
            self.assertFalse(p.should_park({"action": "wait", "reason": "vram_floor"}, stamp))
        self.assertTrue(p.should_park({"action": "wait", "reason": "vram_floor"}, 5))
        self.assertFalse(p.should_park({"action": "run", "reason": "headroom_available"}, 6))
        for stamp in range(7, 20):
            self.assertFalse(p.should_park({"action": "wait", "reason": "telemetry_unavailable"}, stamp))

    def test_missing_or_replayed_intervals_do_not_earn_park(self):
        p = self.config()
        waiting = {"action": "wait", "reason": "non_evictable_ram_floor"}
        p.should_park(waiting, 1)
        self.assertFalse(p.should_park(waiting, 50))
        self.assertFalse(p.should_park(waiting, 50))
        self.assertFalse(p.should_park(waiting, 49))


if __name__ == "__main__":
    unittest.main()
