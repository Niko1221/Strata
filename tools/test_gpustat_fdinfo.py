"""GPU-free checks for gpustat's opt-in fdinfo-only mode.

    python -m unittest tools.test_gpustat_fdinfo
"""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sycl.tools import gpustat as G


class StopLoop(Exception):
    pass


class FdinfoSampler(unittest.TestCase):
    def test_intervals_first_reset_stalled_and_empty(self):
        previous = {"1": {"cycles": 10, "total": 100, "vram_kb": 2048}}
        current = {"1": {"cycles": 60, "total": 200, "vram_kb": 3072}}
        with mock.patch.object(G, "TOTAL_MB", 32768):
            result = G.fdinfo_sample(previous, current, 1)
        self.assertEqual(result, {"pci": G.PCI, "ts": 1, "busy_pct": 50.0,
                                  "vram_used_mb": 3, "vram_total_mb": 32768})
        self.assertIsNone(G.fdinfo_sample({}, current, 1)["busy_pct"])
        self.assertIsNone(G.fdinfo_sample(current, previous, 1)["busy_pct"])
        self.assertIsNone(G.fdinfo_sample(current, current, 1)["busy_pct"])
        self.assertEqual(G.fdinfo_sample(current, {}, 1)["busy_pct"], 0)
        self.assertEqual(G.fdinfo_sample(current, {}, 1)["vram_used_mb"], 0)
        previous = {str(i): previous["1"] for i in range(3)}
        current = {str(i): current["1"] for i in range(3)}
        self.assertEqual(G.fdinfo_sample(previous, current, 1)["busy_pct"], 100)

    def test_mode_publishes_without_touching_hardware_helpers(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(G, "FDINFO_ONLY", True), \
                mock.patch.object(G, "OUT", str(Path(folder) / "gpustat.json")), \
                mock.patch.object(G, "clients", return_value={}) as clients, \
                mock.patch.object(G, "hwmon_dir", side_effect=AssertionError("hwmon read")), \
                mock.patch.object(G, "pcie_link", side_effect=AssertionError("PCIe read")), \
                mock.patch.object(G, "hw_temps", side_effect=AssertionError("temperature read")), \
                mock.patch.object(G, "card_name", side_effect=AssertionError("lspci call")), \
                mock.patch.object(G, "rd", side_effect=AssertionError("register read")), \
                mock.patch.object(G.time, "sleep", side_effect=StopLoop):
            with self.assertRaises(StopLoop):
                G.main()
            data = json.loads(Path(G.OUT).read_text())
            clients.assert_called_once_with(G.PCI)
            self.assertEqual(set(data), {"pci", "ts", "busy_pct", "vram_used_mb", "vram_total_mb"})
            self.assertEqual(data["busy_pct"], 0)
            self.assertFalse(Path(G.OUT + ".tmp").exists())

    def test_default_mode_keeps_full_sampler_fields(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(G, "FDINFO_ONLY", False), \
                mock.patch.object(G, "OUT", str(Path(folder) / "gpustat.json")), \
                mock.patch.object(G, "hwmon_dir", return_value=None) as hwmon, \
                mock.patch.object(G, "pcie_link", return_value=None) as pcie, \
                mock.patch.object(G, "card_name", return_value="fixture"), \
                mock.patch.object(G, "rd", return_value="0xe223"), \
                mock.patch.object(G, "clients", return_value={}) as clients, \
                mock.patch.object(G.glob, "glob", return_value=[]), \
                mock.patch.object(G.os, "getloadavg", return_value=(0, 0, 0), create=True), \
                mock.patch.object(G.time, "sleep", side_effect=StopLoop):
            with self.assertRaises(StopLoop):
                G.main()
            data = json.loads(Path(G.OUT).read_text())
            hwmon.assert_called_once_with()
            pcie.assert_called_once_with()
            clients.assert_called_once_with()
            for field in ("name", "power_w", "temp_pkg", "freq_mhz", "fan_rpm", "pcie", "energy_j"):
                self.assertIn(field, data)
            self.assertEqual(data["name"], "fixture")

    def test_card_filter_runs_before_client_id_deduplication(self):
        def fdinfo(path):
            pci = "0000:01:00.0" if path.endswith("/3") else G.PCI
            memory = 999999 if path.endswith("/3") else 2048
            return io.StringIO(f"drm-client-id:\t7\ndrm-pdev:\t{pci}\ndrm-resident-vram0:\t{memory} KiB\n"
                               "drm-cycles-ccs:\t20\ndrm-total-cycles-ccs:\t100\n")

        with mock.patch.object(G, "PCI", "0000:02:00.0"), \
                mock.patch.object(G.glob, "glob", return_value=["/proc/1"]), \
                mock.patch.object(G.os, "listdir", return_value=["3", "4", "5"]), \
                mock.patch.object(G.os, "readlink", return_value="/dev/dri/renderD128"), \
                mock.patch("builtins.open", side_effect=fdinfo), mock.patch.object(G, "rd", return_value="fixture"):
            clients = G.clients(G.PCI)
        self.assertEqual(len(clients), 1)
        self.assertEqual(clients["7"]["vram_kb"], 2048)
        self.assertEqual(clients["7"]["cycles"], 20)


if __name__ == "__main__":
    unittest.main()
