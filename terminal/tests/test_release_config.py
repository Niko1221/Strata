"""Release resources, signed-updater overlays and platform requirements."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

TAURI = Path(__file__).resolve().parents[2].parent / "app" / "desktop" / "src-tauri"


def load(name: str) -> dict:
    return json.loads((TAURI / name).read_text(encoding="utf-8-sig"))


def merge_patch(base, patch):
    """RFC 7396, the way `tauri build --config <file>` layers an overlay: a null removes the key."""
    if not isinstance(patch, dict):
        return patch
    out = dict(base) if isinstance(base, dict) else {}
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = merge_patch(out.get(key), value)
    return out


class ReleaseConfig(unittest.TestCase):
    def test_windows_release_separates_gpu_engines_and_keeps_private_runtimes(self):
        resources = load("tauri.conf.json")["bundle"]["resources"]
        self.assertNotIn("resources/engine-bundle", resources)
        self.assertNotIn("resources/local-backends", resources)
        for name in ("node", "control", "harness", "notices"):
            self.assertEqual(resources["resources/" + name], name)
        # An explicitly selected developer overlay can still bundle a local engine.
        tester = merge_patch(load("tauri.conf.json"), load("tauri.tester.conf.json"))
        self.assertEqual(tester["bundle"]["resources"]["resources/engine-bundle"], "engine-bundle")

    def test_the_release_overlay_makes_signed_updater_artifacts(self):
        self.assertIs(load("tauri.release.conf.json")["bundle"]["createUpdaterArtifacts"], True)
        self.assertIs(load("tauri.conf.json")["bundle"]["createUpdaterArtifacts"], False)       # a development build needs no private key

    def test_the_mac_app_needs_the_macos_the_metal_engine_needs(self):
        self.assertEqual(load("tauri.conf.json")["bundle"]["macOS"]["minimumSystemVersion"], "15.0")

    def test_the_updater_points_at_github_releases(self):
        up = load("tauri.conf.json")["plugins"]["updater"]
        self.assertEqual(up["endpoints"], ["https://github.com/zapabob/Strata/releases/latest/download/latest.json"])
        self.assertNotIn("REPLACE", up["pubkey"])

    def test_the_update_overlay_drops_only_the_engine(self):
        """The app update (latest.json) is built with tauri.update.conf.json: the same app without engine-bundle, so one update
        fits every installer variant (CUDA 13, CUDA 12, AMD) and downloads 0 engine bytes. Engines come per card from the feed."""
        full = load("tauri.conf.json")
        update = merge_patch(full, load("tauri.update.conf.json"))
        self.assertNotIn("resources/engine-bundle", update["bundle"]["resources"])
        self.assertEqual({k: v for k, v in full["bundle"]["resources"].items() if k != "resources/engine-bundle"}, update["bundle"]["resources"])
        self.assertEqual({k: v for k, v in full["bundle"].items() if k != "resources"}, {k: v for k, v in update["bundle"].items() if k != "resources"})
        self.assertEqual(full["plugins"], update["plugins"])        # same key, same endpoint: it is the same app
        self.assertNotIn("endpoints", load("tauri.update.conf.json").get("plugins", {}))

    def test_one_endpoint_and_no_variant_specific_updater_config(self):
        """Three platform keys in one latest.json is the whole design; a variant-specific endpoint would give an AMD or CUDA 12 PC another variant's installer."""
        for name in ("tauri.conf.json", "tauri.release.conf.json", "tauri.tester.conf.json", "tauri.update.conf.json"):
            self.assertNotIn("{{", json.dumps(load(name).get("plugins", {})), name)


if __name__ == "__main__":
    unittest.main()
