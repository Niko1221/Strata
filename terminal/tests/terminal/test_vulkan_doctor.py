"""The read-only Vulkan probe added to ``strata doctor`` (the Arc B390 path).

Most tests use a recorded probe result, so they run on any machine. The last
test runs the real ctypes probe when this host has an Intel Vulkan device; it
skips cleanly elsewhere.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_cli.cli import main  # noqa: E402
from strata_control import doctor, vulkan  # noqa: E402


def fake_device(*, intel: bool = True, api: str = "1.4.354", cached_type: int = 2,
                group: int = 32, size_control: bool = True, heap_mib: int = 23716,
                device_type: str = "integrated_gpu"):
    return {
        "name": "Intel(R) Arc(tm) B390 (PTL)" if intel else "Some Other GPU",
        "vendor_id": "0x8086" if intel else "0x1002", "device_id": "0xb080",
        "device_type": device_type, "api_version": api, "driver_version": "0x6802006",
        "intel": intel,
        "queue_families": [{"index": 0, "flags": "0x1f", "count": 1, "compute": True}],
        "extensions": {"VK_EXT_subgroup_size_control": size_control,
                       "VK_KHR_cooperative_matrix": True},
        "extension_count": 210,
        "subgroup": {"size": group, "min": 16, "max": 32, "size_control": size_control,
                     "required_subgroup_size_stages": "0xe0"},
        "memory": {"heaps": [{"index": 0, "size_mib": heap_mib, "device_local": True}],
                   "types": [{"index": i, "flags": [], "heap_index": 0,
                              "device_local": True, "host_visible": False,
                              "host_coherent": False, "host_cached": False} for i in range(3)],
                   "device_local_mib": heap_mib, "host_visible_mib": heap_mib,
                   "cached_host_visible_type": cached_type},
    }


def ok_probe(*devices):
    return {"status": "ok", "reason": None, "loader": "libvulkan",
            "instance_version": "1.4.363", "devices": list(devices)}


class VulkanPlan(unittest.TestCase):
    def test_plan_reads_mem_available_and_sizes_the_cache(self):
        with tempfile.TemporaryDirectory() as d:
            meminfo = Path(d) / "meminfo"
            meminfo.write_text("MemTotal:       33000000 kB\nMemAvailable:   20000000 kB\nCached:           500000 kB\n")
            plan = vulkan.plan({"args": ["--slots", "2048"]}, meminfo_path=str(meminfo), dev_root=str(Path(d) / "dev"))
        self.assertEqual(plan["mem_available_mib"], 19531)
        self.assertEqual(plan["configured_slots"], 2048)
        self.assertEqual(plan["cache_bytes"], 2048 * 860160)
        self.assertIsNotNone(plan["recommended_slots"])
        self.assertLessEqual(plan["recommended_slots"], 2048)
        self.assertEqual(plan["render_nodes"], [], "no render nodes in the fake dev root")

    def test_plan_detects_ple_io_ram_and_bad_slots(self):
        with tempfile.TemporaryDirectory() as d:
            meminfo = Path(d) / "meminfo"
            meminfo.write_text("MemTotal: 33000000 kB\nMemAvailable: 20000000 kB\n")
            plan = vulkan.plan({"args": ["--ple-io", "ram", "--slots", "oops"]}, meminfo_path=str(meminfo))
            direct = vulkan.plan({"args": ["--ple-io", "direct"]}, meminfo_path=str(meminfo))
        self.assertTrue(plan["ple_io_ram"])
        self.assertFalse(direct["ple_io_ram"])
        self.assertEqual(plan["configured_slots"], vulkan.DEFAULT_SLOTS, "bad --slots falls back to the default")

    def test_plan_uses_the_smaller_of_mem_available_and_cgroup_room(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            meminfo = root / "meminfo"
            meminfo.write_text("MemTotal: 33000000 kB\nMemAvailable: 20000000 kB\n")
            cgroup = root / "cgroup"
            cgroup.mkdir()
            (cgroup / "memory.max").write_text("2147483648\n")    # 2 GiB limit
            (cgroup / "memory.current").write_text("1610612736\n")  # 1.5 GiB used
            plan = vulkan.plan({"args": []}, meminfo_path=str(meminfo), cgroup_root=str(cgroup))
        self.assertEqual(plan["mem_available_mib"], 512)
        self.assertEqual(plan["memory_source"], "cgroup (memory)")
        self.assertEqual(plan["recommended_slots"], vulkan.MIN_SLOTS)

    def test_plan_defaults_without_a_config(self):
        with mock.patch.object(vulkan, "_meminfo", return_value={"MemTotal": 0, "MemAvailable": 0}):
            plan = vulkan.plan()
        self.assertIsNone(plan["recommended_slots"], "unknown memory must not produce a bogus recommendation")
        self.assertEqual(plan["configured_slots"], 2048)


class VulkanIssues(unittest.TestCase):
    def test_no_loader_is_an_error_only_when_vulkan_is_required(self):
        result = {"status": "unavailable", "reason": "the Vulkan loader is not installed", "devices": []}
        self.assertEqual(vulkan.issues(result)[0]["severity"], "warning")
        self.assertEqual(vulkan.issues(result, required=True)[0]["code"], "VULKAN_UNAVAILABLE")
        self.assertEqual(vulkan.issues(result, required=True)[0]["severity"], "error")

    def test_a_healthy_intel_device_needs_no_fix(self):
        result = ok_probe(fake_device())
        plan = {"mem_available_mib": 19500, "mem_total_mib": 31622, "configured_slots": 2048,
                "recommended_slots": 2048, "cache_mib": 1680, "ple_io_ram": False,
                "render_nodes": [], "render_node_accessible": True}
        codes = [i["code"] for i in vulkan.issues(result, plan, required=True)]
        self.assertNotIn("VULKAN_NO_INTEL_DEVICE", codes)
        self.assertNotIn("VULKAN_NO_CACHED_HOST_MEMORY", codes)
        self.assertIn("VULKAN_SHARED_MEMORY", codes)  # always explain the iGPU/BAR trap

    def test_missing_cached_memory_and_old_api_are_named(self):
        result = ok_probe(fake_device(cached_type=None, api="1.2.100"))
        plan = {"mem_available_mib": 1000, "mem_total_mib": 31622, "configured_slots": 4096,
                "recommended_slots": 256, "cache_mib": 3360, "ple_io_ram": True,
                "render_nodes": ["/dev/dri/renderD128"], "render_node_accessible": False}
        codes = {i["code"]: i for i in vulkan.issues(result, plan)}
        self.assertIn("VULKAN_NO_CACHED_HOST_MEMORY", codes)
        self.assertIn("VULKAN_API_TOO_OLD", codes)
        self.assertIn("VULKAN_CACHE_TOO_LARGE", codes)
        self.assertIn("VULKAN_PLE_IO_RAM", codes)
        self.assertIn("VULKAN_DEVICE_ACCESS", codes)
        self.assertIn("256", codes["VULKAN_CACHE_TOO_LARGE"]["message"])  # the recommended count

    def test_no_intel_device(self):
        self.assertIn("VULKAN_NO_INTEL_DEVICE", [i["code"] for i in vulkan.issues(ok_probe(fake_device(intel=False)))])

    def test_no_subgroup_size_is_a_soft_warning(self):
        codes = [i["code"] for i in vulkan.issues(ok_probe(fake_device(group=None)))]
        self.assertIn("VULKAN_SUBGROUP_UNKNOWN", codes)


class VulkanProbe(unittest.TestCase):
    def test_missing_loader_is_reported_not_raised(self):
        with mock.patch.object(vulkan, "_load_vulkan", return_value=(None, "no loader")), \
             mock.patch.object(vulkan, "sys") as sys_mod:
            sys_mod.platform = "linux"
            result = vulkan.probe()
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("no loader", result["reason"])

    def test_unsupported_os_is_reported(self):
        with mock.patch.object(vulkan, "sys") as sys_mod:
            sys_mod.platform = "win32"
            result = vulkan.probe()
        self.assertIn("Linux only", result["reason"])
        self.assertEqual(result["devices"], [])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Vulkan path is Linux only")
    def test_real_probe_on_this_host(self):
        result = vulkan.probe()
        if result["status"] != "ok":
            self.skipTest("no Vulkan device on this host: " + str(result["reason"]))
        self.assertTrue(any(d["intel"] and d["vendor_id"] == "0x8086" for d in result["devices"]), result)
        device = next(d for d in result["devices"] if d["intel"])
        self.assertIsNotNone(device["memory"]["cached_host_visible_type"], result)
        self.assertGreater(device["memory"]["device_local_mib"], 0)
        self.assertIn(device["subgroup"]["size"], (16, 32, 64))
        # A JSON report must be serialisable with no NaN and no ctypes objects.
        json.dumps(result, allow_nan=False)


class DoctorIntegration(unittest.TestCase):
    def test_inspect_auto_detects_a_vulkan_run_config(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "run.json"
            exe = Path(d) / "strata-vulkan-engine"
            exe.write_bytes(b"x")
            cfg.write_text(json.dumps({"exe": str(exe), "tokenizer": str(Path(d) / "tok"),
                                       "args": [], "cwd": d}))
            full = {"probe": ok_probe(fake_device()), "plan": {}, "issues": []}
            with mock.patch.object(vulkan, "inspect", return_value=full) as m:
                report = doctor.inspect(cfg)
            self.assertTrue(m.called, "a config naming strata-vulkan-engine must be probed without a flag")
            self.assertEqual(report["vulkan"]["probe"]["devices"][0]["intel"], True)

    def test_inspect_does_not_probe_a_normal_config(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "run.json"
            exe = Path(d) / "strata-engine"
            exe.write_bytes(b"x")
            cfg.write_text(json.dumps({"exe": str(exe), "tokenizer": str(Path(d) / "tok"),
                                       "args": [], "cwd": d, "backend": "cuda"}))
            with mock.patch.object(vulkan, "inspect") as m:
                report = doctor.inspect(cfg)
            self.assertFalse(m.called)
            self.assertIsNone(report["vulkan"])

    def test_cli_vulkan_flag_and_human_output(self):
        with mock.patch.object(vulkan, "inspect", return_value={
                "probe": ok_probe(fake_device()),
                "plan": {"mem_available_mib": 19500, "mem_total_mib": 31622,
                         "configured_slots": 2048, "recommended_slots": 2048, "cache_mib": 1680},
                "issues": [{"severity": "info", "code": "VULKAN_SHARED_MEMORY", "message": "shared RAM"}]}):
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["doctor", "--vulkan"])
        text = out.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("Vulkan: ok", text)
        self.assertIn("Arc(tm) B390", text)
        self.assertIn("subgroup: 32 lanes", text)
        self.assertIn("cached host-visible type 2", text)
        self.assertIn("VULKAN_SHARED_MEMORY", text)

    def test_cli_json_includes_vulkan(self):
        with mock.patch.object(vulkan, "inspect", return_value={
                "probe": ok_probe(fake_device()), "plan": {}, "issues": []}):
            out = io.StringIO()
            with redirect_stdout(out):
                main(["doctor", "--vulkan", "--output", "json"])
        report = json.loads(out.getvalue())["report"]
        self.assertEqual(report["vulkan"]["probe"]["devices"][0]["vendor_id"], "0x8086")


if __name__ == "__main__":
    unittest.main()
