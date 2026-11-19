"""``strata launch --backend vulkan`` selects the native engine by default.

``--vulkan-impl bridge`` keeps the faster llama.cpp bridge. Both paths must stay
out of the normal CUDA/HIP/SYCL/Metal launch flow and must not touch anything
with --plan.
"""
import argparse
import io
import json
import contextlib
import unittest

from strata_cli.launch.cmd import run_launch


class VulkanLaunch(unittest.TestCase):
    def args(self, **override):
        a = dict(output="json", backend="vulkan", plan=True,
                 model_path="/tmp/model.gguf", data_dir=None, port=8123, context=None)
        a.update(override)
        return argparse.Namespace(**a)

    def _plan(self, **override):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = run_launch(self.args(**override))
        self.assertEqual(rc, 0)
        return json.loads(out.getvalue())

    def test_plan_is_json_and_changes_nothing(self):
        p = self._plan()
        self.assertEqual(p["backend"], "vulkan")
        self.assertEqual(p["plan"]["port"], 8123)
        self.assertEqual(p["plan"]["gguf"], "/tmp/model.gguf")
        self.assertIn("/v1", p["plan"]["url"])

    def test_native_is_the_default_impl(self):
        p = self._plan()
        self.assertEqual(p["impl"], "native")
        self.assertEqual(p["plan"]["impl"], "native")
        self.assertTrue(p["plan"]["exe"].endswith("engine/vulkan/engine/strata-vulkan-engine"))

    def test_bridge_impl_is_opt_in(self):
        p = self._plan(vulkan_impl="bridge")
        self.assertEqual(p["impl"], "bridge")
        self.assertTrue(p["plan"]["exe"].endswith("engine/vulkan/bridge/strata-llama-vulkan-engine"))

    def test_case_insensitive_backend(self):
        self.assertEqual(self._plan(backend="Vulkan")["backend"], "vulkan")

    def test_unknown_impl_falls_back_to_native(self):
        self.assertEqual(self._plan(vulkan_impl="bogus")["impl"], "native")

    def test_no_vulkan_branch_for_normal_backends(self):
        # A normal backend must not hit the Vulkan path: it goes through the
        # usual flow and fails here only because this Namespace is deliberately
        # tiny.
        with self.assertRaises(Exception):
            run_launch(self.args(backend="intel"))


if __name__ == "__main__":
    unittest.main()
