"""The capability table (docs/data/capabilities.json) against the release's packages and the pages that quote it."""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capability_table as ct  # noqa: E402


class CapabilityTable(unittest.TestCase):
    def setUp(self):
        self.cap = ct.read_json(ct.DATA)
        self.release = ct.read_json(ct.RELEASE)

    def test_pages_and_table_are_what_the_generator_writes(self):
        self.assertEqual(ct.validate(self.cap, self.release), [])
        self.assertEqual(ct.read(ct.OUT).replace("\r\n", "\n"), ct.page(self.cap, self.release))
        for p in ct.PAGES:
            text = ct.read(p)
            self.assertTrue(ct.MARK_RE.search(text), p)
            self.assertEqual(ct.render(self.cap, self.release, text), text, p)

    def test_a_package_contradicts_a_no(self):
        cap = copy.deepcopy(self.cap)
        {f["id"]: f for f in cap["families"]}["qwen38"]["cells"]["mac"]["supported"] = "no"
        problems = ct.validate(cap, self.release)
        self.assertTrue(any("qwen38 on mac is marked 'no'" in p for p in problems), problems)

    def test_a_yes_without_a_package_is_reported(self):
        cap = copy.deepcopy(self.cap)
        {f["id"]: f for f in cap["families"]}["qwen36"]["cells"]["intel_linux"]["supported"] = "yes"
        problems = ct.validate(cap, self.release)
        self.assertTrue(any("no package for it" in p for p in problems), problems)

    def test_qwen36_packages_are_the_cuda_and_hip_engines_only(self):
        fam = {f["id"]: f for f in self.cap["families"]}["qwen36"]
        pk = ct.packages(self.release, fam)
        self.assertNotIn("mac", pk)
        self.assertNotIn("intel_linux", pk)
        self.assertIn("nvidia_win", pk)

    def test_the_version_line_names_carried_packages(self):
        rel = {"product_version": "0.2.14", "control_version": "0.1.60", "engine_version": "0.1.60",
               "assets": ["strata-engine-0.1.60-linux-x86_64-cuda-cuda13.tar.gz", "strata-engine-0.1.59-linux-x86_64-sycl.tar.gz",
                          "strata-control-0.1.60-windows-x86_64.zip", "strata-control-0.1.59-linux-aarch64.tar.gz",
                          "strata-dsv41-0.1.59-macos-arm64-metal.tar.gz", "Strata-0.2.14-macos-arm64.dmg"]}
        self.assertEqual(ct.versions_line(rel), "App and Terminal 0.2.14, Control and engines 0.1.60; these packages stay at 0.1.59: "
                         "Control linux-aarch64; DeepSeek engines macos-arm64-metal; engines linux-x86_64-sycl. "
                         "App, Control and engine versions differ on purpose.")
        hotfix = {"product_version": "0.2.13", "control_version": "0.1.59.1", "engine_version": "0.1.59",
                  "assets": ["strata-control-0.1.59.1-linux-x86_64.tar.gz", "strata-engine-0.1.59-linux-x86_64-sycl.tar.gz"]}
        self.assertEqual(ct.versions_line(hotfix), "App and Terminal 0.2.13, Control 0.1.59.1, engines 0.1.59. "
                         "App, Control and engine versions differ on purpose.")
        carried = {"product_version": "0.2.15", "control_version": "0.1.61", "engine_version": "0.1.60",
                   "assets": ["strata-control-0.1.61-linux-x86_64.tar.gz", "strata-engine-0.1.60-linux-x86_64-cuda-cuda13.tar.gz"]}
        self.assertEqual(ct.versions_line(carried), "App and Terminal 0.2.15, Control 0.1.61, engines 0.1.60. "
                         "App, Control and engine versions differ on purpose.")

    def test_carried_sycl_package_is_present_without_current_hardware_qualification(self):
        family = next(f for f in self.cap['families'] if f['id'] == 'qwen38')
        self.assertEqual(family['cells']['intel_linux']['supported'], 'untested')
        self.assertEqual(ct.packages(self.release, family)['intel_linux'], [('sycl', '0.1.59')])
        self.assertIn('archive was not downloaded or executed', family['cells']['intel_linux']['limits'])
        self.assertEqual(ct.validate(self.cap, self.release), [])

    def test_source_route_does_not_invent_a_current_package_or_remove_the_backend(self):
        cap = copy.deepcopy(self.cap)
        release = copy.deepcopy(self.release)
        release['upstream']['assets'] = [a for a in release['upstream']['assets'] if '-sycl.' not in a]
        family = next(f for f in cap['families'] if f['id'] == 'qwen38')
        family['cells']['intel_linux']['supported'] = 'source-only'
        self.assertEqual(family['cells']['intel_linux']['supported'], 'source-only')
        self.assertNotIn('intel_linux', ct.packages(release, family))
        self.assertEqual(ct.validate(cap, release), [])
        self.assertIn('Source route only, without a package in this release: Intel Arc on Linux', ct.summary(cap, family))
        changed = copy.deepcopy(release)
        changed['upstream']['assets'].append('strata-engine-0.1.60-linux-x86_64-sycl.tar.gz')
        self.assertTrue(any('source-only but the release has a package' in p for p in ct.validate(cap, changed)))

    def test_control_only_hotfix_keeps_all_four_actual_versions(self):
        release = {'product_version': '0.2.14', 'terminal_version': '0.2.14.1',
                   'control_version': '0.1.60.1', 'engine_version': '0.1.60', 'assets': []}
        self.assertEqual(ct.versions_line(release), 'App 0.2.14, Terminal 0.2.14.1, Control 0.1.60.1, engines 0.1.60. '
                         'App, Control and engine versions differ on purpose.')

    def test_no_page_keeps_the_old_mac_sentence(self):
        for p in ct.PAGES:
            self.assertNotIn("no DeepSeek yet", ct.read(p))

    def test_fork_assets_do_not_replace_the_upstream_hardware_reference(self):
        upstream = self.release["upstream"]
        fork = {"tag": "v9.9.9", "assets": [], "upstream": upstream}
        self.assertEqual(ct.validate(self.cap, fork), [])
        self.assertEqual(ct.page(self.cap, fork), ct.page(self.cap, upstream))
        self.assertNotIn("v9.9.9", ct.page(self.cap, fork))


if __name__ == "__main__":
    unittest.main()
