import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from windows_telemetry import _Rates, _WindowsIntelGpu


class RatesTest(unittest.TestCase):
    def test_rates_and_idle(self):
        rates = _Rates()
        self.assertNotIn("power", rates.convert(dict(timestamp=10, energy=100, activity=2)))
        sample = rates.convert(dict(timestamp=12, energy=180, activity=3))
        self.assertEqual(sample["power"], 40)
        self.assertEqual(sample["util"], 50)
        sample = rates.convert(dict(timestamp=13, energy=180, activity=3))
        self.assertEqual(sample["power"], 0)
        self.assertEqual(sample["util"], 0)

    def test_reset_missing_stale_and_nonfinite(self):
        rates = _Rates()
        for sample in [dict(timestamp=10, energy=100), dict(timestamp=11, energy=10),
                       dict(timestamp=11, energy=20), dict(timestamp=20, energy=30),
                       dict(timestamp=21, energy=float("nan")), dict(timestamp=22, energy=40),
                       dict(timestamp=23), dict(timestamp=24, energy=60)]:
            self.assertNotIn("power", rates.convert(sample))

    def test_bad_memory_and_sensors(self):
        rates = _Rates()
        sample = rates.convert(dict(mem_used=11, mem_total=10, temp=float("inf"), vram_temp=-1))
        for key in ("mem_used", "mem_total", "temp", "vram_temp"):
            self.assertNotIn(key, sample)
        sample = rates.convert(dict(mem_used=0, mem_total=10, temp=0))
        self.assertEqual(sample["mem_used"], 0)
        self.assertEqual(sample["temp"], 0)


class ReaderTest(unittest.TestCase):
    def reader(self):
        with patch("windows_telemetry.sys.platform", "linux"):
            reader = _WindowsIntelGpu()
        self.addCleanup(reader.close)
        return reader

    def test_stale_dead_and_snapshot_copy(self):
        reader = self.reader()
        process = Mock()
        process.poll.return_value = None
        reader._process = process
        reader._latest = {"temp": 55}
        reader._received = time.monotonic()
        self.assertEqual(reader.read(), {"temp": 55})
        reader.read()["temp"] = 0
        self.assertEqual(reader.read()["temp"], 55)
        reader._received -= 6
        self.assertEqual(reader.read(), {})
        reader._received = time.monotonic()
        process.poll.return_value = 1
        self.assertEqual(reader.read(), {})

    def test_consumer_ignores_bad_lines(self):
        reader = self.reader()
        process = Mock()
        process.stdout = io.StringIO('broken\n[]\n' + json.dumps(dict(name="Arc", temp=50)) + '\n')
        reader._process = process
        reader._consume()
        self.assertTrue(reader.ok())
        self.assertEqual(reader.name(), "Arc")
        self.assertEqual(reader._latest["temp"], 50)

    def test_missing_helper(self):
        with patch("windows_telemetry.sys.platform", "win32"), patch("windows_telemetry.subprocess.Popen", side_effect=OSError), patch("windows_telemetry.subprocess.CREATE_NO_WINDOW", 0, create=True):
            reader = _WindowsIntelGpu()
        self.assertFalse(reader.ok())
        self.assertEqual(reader.read(), {})

    def test_failed_startup_is_cleaned_up_and_bdf_forwarded(self):
        process = Mock()
        process.stdout = io.StringIO('')
        process.poll.return_value = None
        with patch("windows_telemetry.sys.platform", "win32"), patch("windows_telemetry.subprocess.Popen", return_value=process) as popen, patch("windows_telemetry.subprocess.CREATE_NO_WINDOW", 0, create=True), patch.dict("os.environ", {"STRATA_INTEL_PCI_BDF": "3:0.0"}):
            reader = _WindowsIntelGpu(2)
        self.assertFalse(reader.ok())
        self.assertIsNone(reader._process)
        process.terminate.assert_called_once()
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--index") + 1], "2")
        self.assertEqual(command[command.index("--pci-bdf") + 1], "3:0.0")

    def test_close_terminates_child_and_is_repeatable(self):
        reader = self.reader()
        process = Mock()
        process.poll.side_effect = [None, 0]
        reader._process = process
        reader.close()
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=2)
        reader.close()
        self.assertIsNone(reader._process)


if __name__ == "__main__":
    unittest.main()
