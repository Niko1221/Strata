"""Tests for independent expert-profile validation and rollback."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_expert_profile as tool


def result(group, speed):
    return {"group": group, "timings": {"predicted_per_second": speed}}


class Validation(unittest.TestCase):
    def test_group_medians_and_regression_threshold(self):
        baseline = {"code": 50.0, "general": 40.0}
        candidate = {"code": 55.0, "general": 35.0}
        failed = tool.regressions(baseline, candidate, 0.10)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["group"], "general")
        self.assertAlmostEqual(failed[0]["loss"], 0.125)

    def test_accept_keeps_candidate_and_only_restarts_once(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original.bin"
            candidate = Path(directory) / "candidate.bin"
            original.write_bytes(b"original")
            candidate.write_bytes(b"candidate")
            restarts = []

            def evaluate(_tasks):
                return [result("code", 50.0)]

            report = tool.validate(original, candidate, [], evaluate, lambda: restarts.append(None), 0.10)
            self.assertEqual(report["status"], "accepted")
            self.assertEqual(original.read_bytes(), b"candidate")
            self.assertEqual(restarts, [None])
            self.assertTrue((Path(directory) / "original.bin.validation-backup").exists())

    def test_regression_restores_original_and_restarts_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original.bin"
            candidate = Path(directory) / "candidate.bin"
            original.write_bytes(b"original")
            candidate.write_bytes(b"candidate")
            restarts = []
            answers = [[result("general", 50.0)], [result("general", 20.0)]]

            def evaluate(_tasks):
                return answers.pop(0)

            report = tool.validate(original, candidate, [], evaluate, lambda: restarts.append(None), 0.10)
            self.assertEqual(report["status"], "rolled_back")
            self.assertEqual(original.read_bytes(), b"original")
            self.assertEqual(restarts, [None, None])

    def test_evaluation_failure_also_restores_original(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original.bin"
            candidate = Path(directory) / "candidate.bin"
            original.write_bytes(b"original")
            candidate.write_bytes(b"candidate")
            restarts = []
            answers = [[result("general", 50.0)], RuntimeError("server unavailable")]

            def evaluate(_tasks):
                answer = answers.pop(0)
                if isinstance(answer, Exception):
                    raise answer
                return answer

            with self.assertRaisesRegex(RuntimeError, "server unavailable"):
                tool.validate(original, candidate, [], evaluate, lambda: restarts.append(None), 0.10)
            self.assertEqual(original.read_bytes(), b"original")
            self.assertEqual(restarts, [None, None])

    def test_existing_backup_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original.bin"
            candidate = Path(directory) / "candidate.bin"
            backup = Path(directory) / "original.bin.validation-backup"
            original.write_bytes(b"original")
            candidate.write_bytes(b"candidate")
            backup.write_bytes(b"older backup")
            with self.assertRaises(FileExistsError):
                tool.validate(original, candidate, [], lambda _tasks: [], None, 0.10)
            self.assertEqual(backup.read_bytes(), b"older backup")

    def test_task_format_rejects_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            path.write_text(json.dumps({"prompt": "", "group": "code"}) + "\n")
            with self.assertRaisesRegex(ValueError, "prompt"):
                tool.read_tasks(path)
            path.write_text(json.dumps({"prompt": "valid", "max_tokens": 0}) + "\n")
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                tool.read_tasks(path)


if __name__ == "__main__":
    unittest.main()
