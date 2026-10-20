"""The Docker files stay consistent: shell scripts parse, Dockerfiles use only what the legacy builder knows, the compose examples carry what
docs/DOCKER.md says they carry, and every setting the document lists is one the entry point reads."""
from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"
SH = shutil.which("sh")


class Scripts(unittest.TestCase):
    @unittest.skipIf(SH is None, "needs sh")
    def test_shell_scripts_parse(self):
        for name in ("entrypoint.sh", "strata-wrapper.sh", "healthcheck.sh", "build.sh", "smoke-test.sh"):
            r = subprocess.run([SH, "-n", str(DOCKER / name)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{name}: {r.stderr}")

    def test_scripts_use_unix_line_ends(self):
        for name in ("entrypoint.sh", "strata-wrapper.sh", "healthcheck.sh", "build.sh"):
            self.assertNotIn(b"\r", (DOCKER / name).read_bytes(), name)       # build.sh strips CR for a CRLF checkout, the files themselves stay LF


class Dockerfiles(unittest.TestCase):
    FILES = (DOCKER / "Dockerfile", ROOT / "Dockerfile")

    def test_only_legacy_builder_syntax(self):
        for f in self.FILES:
            text = f.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"<<\s*-?['\"]?\w+", f"{f.name}: a heredoc needs BuildKit")
            self.assertNotIn("--mount=", text, f.name)
            self.assertNotRegex(text, r"COPY\s+--chmod", f.name)

    def test_runtime_contract(self):
        for f in self.FILES:
            text = f.read_text(encoding="utf-8")
            self.assertIn("HEALTHCHECK", text, f.name)
            self.assertIn("STOPSIGNAL SIGTERM", text, f.name)
            self.assertIn('ENTRYPOINT ["/usr/bin/tini", "--", "/opt/strata/entrypoint.sh"]', text, f.name)
            self.assertIn("VOLUME [\"/data\"]", text, f.name)
            self.assertIn("STRATA_COMPONENTS_DIR=/opt/strata/components", text, f.name)

    def test_component_image_checks_the_package_before_use(self):
        text = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("package_engine.py", text)
        self.assertIn("binary_deps.py", text)
        self.assertIn("ldd", text)                       # the finished image resolves every library of the package

    def test_binary_deps_and_package_engine_exist_where_the_dockerfile_copies_them(self):
        for name in ("package_engine.py", "binary_deps.py", "package_terminal.py"):
            self.assertTrue((ROOT / "packaging" / name).is_file(), name)


class Compose(unittest.TestCase):
    def load(self, name: str) -> dict:
        data = yaml.safe_load((DOCKER / "compose" / name).read_text(encoding="utf-8"))
        self.assertIn("strata", data["services"], name)
        return data

    def test_common_settings(self):
        for name in ("docker-compose.nvidia.yml", "docker-compose.amd.yml", "docker-compose.intel.yml"):
            svc = self.load(name)["services"]["strata"]
            self.assertIn("strata-data:/data", svc["volumes"], name)
            self.assertEqual(svc["ulimits"]["memlock"], {"soft": -1, "hard": -1}, name)
            self.assertEqual(svc["shm_size"], "2gb", name)
            self.assertEqual(svc["stop_grace_period"], "2m", name)
            self.assertTrue(svc["ports"][0].startswith("${STRATA_BIND:-127.0.0.1}:"), f"{name}: the port must default to this computer only")
            for var in ("MODEL", "MODEL_PATH", "CONTEXT", "PARALLEL", "VISION", "TUNE", "API_KEY"):
                self.assertIn(var, svc["environment"], f"{name}: {var}")
            self.assertIn("strata-data", self.load(name)["volumes"], name)

    def test_gpu_access_per_vendor(self):
        nv = self.load("docker-compose.nvidia.yml")["services"]["strata"]
        dev = nv["deploy"]["resources"]["reservations"]["devices"][0]
        self.assertEqual(dev["driver"], "nvidia")
        self.assertIn("gpu", dev["capabilities"])
        amd = self.load("docker-compose.amd.yml")["services"]["strata"]
        self.assertIn("/dev/kfd:/dev/kfd", amd["devices"])
        self.assertIn("/dev/dri:/dev/dri", amd["devices"])
        self.assertIn("video", amd["group_add"])
        self.assertTrue(any("render" in str(g).lower() or "RENDER" in str(g) for g in amd["group_add"]))
        intel = self.load("docker-compose.intel.yml")["services"]["strata"]
        self.assertIn("/dev/dri:/dev/dri", intel["devices"])
        self.assertNotIn("/dev/kfd:/dev/kfd", intel["devices"])


class Document(unittest.TestCase):
    def test_every_documented_setting_is_read_by_the_entry_point(self):
        doc = (ROOT / "docs" / "DOCKER.md").read_text(encoding="utf-8")
        section = doc.split("## Settings", 1)[1].split("###", 1)[0]
        entry = (DOCKER / "entrypoint.sh").read_text(encoding="utf-8")
        names: set[str] = set()
        for line in section.splitlines():
            if line.startswith("| `"):
                names.update(re.findall(r"`([A-Z][A-Z_]+)`", line.split("|")[1]))
        self.assertTrue({"MODEL", "CONTEXT", "API_KEY", "TUNE", "VISION", "MODEL_PATH", "STRATA_SET"} <= names, names)
        for name in sorted(names):
            self.assertIn(name, entry, f"docs/DOCKER.md lists {name} but docker/entrypoint.sh does not read it")

    def test_tags_in_the_document_match_the_compose_defaults(self):
        doc = (ROOT / "docs" / "DOCKER.md").read_text(encoding="utf-8")
        for tag in ("0.2.0-cuda13", "0.2.0-cuda12", "0.2.0-rocm-gfx120x", "0.2.0-rocm-gfx110x", "0.2.0-sycl"):
            self.assertIn(tag, doc)
        for name, tag in (("docker-compose.nvidia.yml", "0.2.0-cuda13"), ("docker-compose.amd.yml", "0.2.0-rocm-gfx120x"),
                          ("docker-compose.intel.yml", "0.2.0-sycl")):
            self.assertIn(tag, (DOCKER / "compose" / name).read_text(encoding="utf-8"))

    def test_the_user_doc_says_images_are_not_published(self):
        doc = (ROOT / "docs" / "DOCKER.md").read_text(encoding="utf-8")
        self.assertNotIn("TODO(owner)", doc)
        self.assertIn("Ready-made images are not available yet", doc)


if __name__ == "__main__":
    unittest.main()
