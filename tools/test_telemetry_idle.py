"""An unloaded model must not wake the GPU through telemetry. No GPU or downloads.

    python -m unittest tools.test_telemetry_idle
"""
import unittest
from unittest import mock

from serve import telemetry
from sycl.serve.xe_telemetry import _XeGpu


class IdleTelemetry(unittest.TestCase):
    def test_service_enables_pause_only_for_sycl(self):
        from serve.server import Service
        for backend in ("sycl", "cuda", "hip", "metal", None):
            with self.subTest(backend=backend), mock.patch.object(telemetry, "Telemetry") as constructor:
                service = Service.__new__(Service)
                service.backend = backend
                service.engine = mock.Mock()
                service.engine.alive.return_value = True
                service.start_telemetry()
                active = constructor.call_args.kwargs["gpu_active"]
                if backend == "sycl":
                    self.assertTrue(active())
                    service.engine.alive.return_value = False
                    self.assertFalse(active())
                else:
                    self.assertIsNone(active)

    def test_unload_pauses_all_cards_but_keeps_cpu_and_server_metrics(self):
        loaded = True
        gpu = mock.Mock()
        gpu.ok.return_value = True
        gpu.name.return_value = "fixture"
        gpu.read.return_value = {"power": 40, "temp": 50}
        with mock.patch.object(telemetry, "gpu_reader", return_value=gpu), \
                mock.patch.object(telemetry.threading.Thread, "start"):
            sampler = telemetry.Telemetry(gpu_indices=[0, 1], gpu_active=lambda: loaded, extra=lambda: {"tok_s": 0})
        try:
            self.assertEqual(sampler.sample()["gpu_power"], 80)
            gpu.read.reset_mock()
            loaded = False
            paused = sampler.sample()
            gpu.read.assert_not_called()
            self.assertEqual(gpu.pause.call_count, 2)
            self.assertNotIn("gpu_power", paused)  # unknown, not zero watts or the last awake reading
            self.assertNotIn("gpu_temp", paused)
            self.assertNotIn("gpus", paused)
            self.assertIn("cpu", paused)
            self.assertIn("ram_used", paused)
            self.assertEqual(paused["tok_s"], 0)
            loaded = True
            self.assertEqual(sampler.sample()["gpu_power"], 80)
            self.assertEqual(gpu.read.call_count, 2)
        finally:
            sampler.close()

    def test_default_reader_needs_no_pause_method_and_keeps_sampling(self):
        gpu = mock.Mock(spec=["ok", "name", "read"])
        gpu.ok.return_value = True
        gpu.name.return_value = "fixture"
        gpu.read.return_value = {"power": 40}
        with mock.patch.object(telemetry, "gpu_reader", return_value=gpu), \
                mock.patch.object(telemetry.threading.Thread, "start"):
            sampler = telemetry.Telemetry()
            paused = telemetry.Telemetry(gpu_active=lambda: False)
        try:
            self.assertEqual(sampler.sample()["gpu_power"], 40)
            self.assertNotIn("gpu_power", paused.sample())
            self.assertEqual(gpu.read.call_count, 1)
        finally:
            sampler.close()
            paused.close()

    def test_xe_resumes_with_a_fresh_energy_baseline(self):
        gpu = _XeGpu.__new__(_XeGpu)
        gpu.hwmon, gpu._e = "/fixture", (1.0, 1000000)
        values = {"/fixture/energy1_input": "100000000"}
        with mock.patch.object(gpu, "_rd", side_effect=values.get), \
                mock.patch.object(gpu, "_stat", return_value=None), \
                mock.patch.object(gpu, "_link", return_value=(3, 3, 16)), \
                mock.patch.object(telemetry.os, "listdir", return_value=[]):
            gpu.pause()
            self.assertIsNone(gpu._e)
            with mock.patch("sycl.serve.xe_telemetry.time.time", return_value=10.0):
                self.assertNotIn("power", gpu.read())
            values["/fixture/energy1_input"] = "102000000"
            with mock.patch("sycl.serve.xe_telemetry.time.time", return_value=11.0):
                self.assertEqual(gpu.read()["power"], 2.0)


if __name__ == "__main__":
    unittest.main()
