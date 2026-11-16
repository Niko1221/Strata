"""Tests for ab_versions: release pairs and version specs.  No network, no GPU."""
import hashlib
import io
import json
import os
import socket
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_versions import (Busy, Packages, ReleasePair, assert_idle, changed, collect_configs, control_command, ensure_packages,  # noqa: E402
                         isolated_env, launch, list_releases, live_roots, pairs_from_releases, parse_sums, provision_home,
                         resolve_versions, served_engine, snapshot, version_problem)


def a(name):
    return {"name": name, "browser_download_url": f"https://example.invalid/dl/{name}"}


FIXTURE = [  # newest first, as the API returns them
    {"tag_name": "v0.2.10", "assets": [a("strata-engine-0.1.56-windows-x86_64-cuda-cuda13.tar.gz"),
                                       a("strata-control-0.1.56-windows-x86_64.zip"), a("SHA256SUMS.txt")]},
    {"tag_name": "v0.2.9", "assets": [a("strata-engine-0.1.55-windows-x86_64-cuda-cuda13.tar.gz"),
                                      a("strata-control-0.1.55-windows-x86_64.zip"), a("SHA256SUMS.txt")]},
    {"tag_name": "v0.2.8", "assets": [a("strata-engine-0.1.54-windows-x86_64-cuda-cuda12.tar.gz"),
                                      a("strata-control-0.1.54-windows-x86_64.zip"), a("SHA256SUMS.txt")]},
]
WIN13 = ("windows-x86_64", "cuda-cuda13")


class PairsTest(unittest.TestCase):
    def test_pairs_need_engine_and_control_for_backend(self):
        p = pairs_from_releases(FIXTURE, *WIN13)
        self.assertEqual(sorted(p), ["0.1.55", "0.1.56"])
        self.assertEqual(p["0.1.55"].tag, "v0.2.9")

    def test_pair_fields(self):
        p = pairs_from_releases(FIXTURE, *WIN13)["0.1.56"]
        self.assertEqual(p.engine_name, "strata-engine-0.1.56-windows-x86_64-cuda-cuda13.tar.gz")
        self.assertTrue(p.engine_url.endswith(p.engine_name))
        self.assertEqual(p.control_name, "strata-control-0.1.56-windows-x86_64.zip")
        self.assertTrue(p.control_url.endswith(p.control_name))
        self.assertTrue(p.sums_url.endswith("SHA256SUMS.txt"))

    def test_control_without_engine_is_dropped(self):
        rel = [{"tag_name": "v1", "assets": [a("strata-control-0.1.60-windows-x86_64.zip"), a("SHA256SUMS.txt")]}]
        self.assertEqual(pairs_from_releases(rel, *WIN13), {})

    def test_other_target_is_dropped(self):
        rel = [{"tag_name": "v1", "assets": [a("strata-engine-0.1.60-linux-x86_64-cuda-cuda13.tar.gz"),
                                             a("strata-control-0.1.60-linux-x86_64.tar.gz"), a("SHA256SUMS.txt")]}]
        self.assertEqual(pairs_from_releases(rel, *WIN13), {})
        self.assertEqual(sorted(pairs_from_releases(rel, "linux-x86_64", "cuda-cuda13")), ["0.1.60"])

    def test_first_occurrence_wins(self):
        rel = [{"tag_name": "v0.2.11", "assets": FIXTURE[0]["assets"]}, FIXTURE[0]]
        self.assertEqual(pairs_from_releases(rel, *WIN13)["0.1.56"].tag, "v0.2.11")


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.p = pairs_from_releases(FIXTURE, *WIN13)

    def test_range_is_inclusive_and_sorted(self):
        self.assertEqual(resolve_versions("0.1.50..0.1.56", self.p), ["0.1.55", "0.1.56"])
        self.assertEqual(resolve_versions("0.1.56,0.1.55", self.p), ["0.1.55", "0.1.56"])

    def test_numeric_not_lexical_order(self):
        rel = [{"tag_name": f"t{n}", "assets": [a(f"strata-engine-0.1.{n}-windows-x86_64-cuda-cuda13.tar.gz"),
                                                a(f"strata-control-0.1.{n}-windows-x86_64.zip"), a("SHA256SUMS.txt")]}
               for n in (9, 10)]
        self.assertEqual(resolve_versions("0.1.9..0.1.10", pairs_from_releases(rel, *WIN13)), ["0.1.9", "0.1.10"])

    def test_unknown_version_in_list_fails_early(self):
        with self.assertRaisesRegex(ValueError, "0.1.54"):
            resolve_versions("0.1.54,0.1.55", self.p)

    def test_empty_range_fails(self):
        with self.assertRaises(ValueError):
            resolve_versions("0.1.60..0.1.70", self.p)


class ListReleasesTest(unittest.TestCase):
    def test_paginates_until_empty_page_without_token(self):
        pages = {1: [{"tag_name": "v2"}], 2: [{"tag_name": "v1"}], 3: []}
        seen = []

        def opener(req, timeout=None):
            seen.append(req)
            page = int(req.full_url.rsplit("page=", 1)[1].split("&")[0])
            return io.BytesIO(json.dumps(pages[page]).encode())

        out = list_releases("o/r", opener=opener)
        self.assertEqual([r["tag_name"] for r in out], ["v2", "v1"])
        self.assertEqual(len(seen), 3)
        self.assertTrue(all("/repos/o/r/releases" in r.full_url for r in seen))
        self.assertTrue(all("authorization" not in {k.lower() for k in r.headers} for r in seen))


def tar_bytes(name, data=b"x"):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def zip_bytes(name, data=b"x"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, data)
    return buf.getvalue()


class FakeResponse:
    """Like a urlopen response: context manager, `.read(n)`; fails mid-stream when `fail_after` bytes were read."""

    def __init__(self, data, fail_after=None):
        self.buf, self.fail_after, self.read_total = io.BytesIO(data), fail_after, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        if self.fail_after is not None and self.read_total >= self.fail_after:
            raise OSError("connection reset")
        chunk = self.buf.read(n if self.fail_after is None else min(n, 4))
        self.read_total += len(chunk)
        return chunk


def make_opener(files, fail_engine=False):
    def opener(url, timeout=None):
        if url not in files:
            raise OSError(f"no network for {url}")
        return FakeResponse(files[url], fail_after=4 if fail_engine and url.endswith("/e") else None)
    return opener


def no_network(url, timeout=None):
    raise AssertionError(f"unexpected download of {url}")


class PackagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name) / "packages"
        self.engine, self.control = tar_bytes("strata-engine.exe", b"engine"), zip_bytes("strata.exe", b"control")
        self.pair = ReleasePair("0.1.56", "v0.2.10", "strata-engine-0.1.56-windows-x86_64-cuda-cuda13.tar.gz", "https://x.invalid/e",
                                "strata-control-0.1.56-windows-x86_64.zip", "https://x.invalid/c", "https://x.invalid/sums")
        self.sums = self.sums_for(self.engine, self.control)

    def sums_for(self, engine, control):
        h = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
        return f"{h(engine)} *{self.pair.engine_name}\n{h(control)}  {self.pair.control_name}\n".encode()

    def files(self, sums=None):
        return {"https://x.invalid/e": self.engine, "https://x.invalid/c": self.control,
                "https://x.invalid/sums": self.sums if sums is None else sums}

    def test_downloads_extracts_and_marks(self):
        pk = ensure_packages(self.pair, self.cache, make_opener(self.files()))
        self.assertEqual(pk.version, "0.1.56")
        self.assertEqual(pk.engine_dir, self.cache / "strata-engine-0.1.56-windows-x86_64-cuda-cuda13")
        self.assertEqual(pk.control_dir, self.cache / "strata-control-0.1.56-windows-x86_64")
        self.assertEqual((pk.engine_dir / "strata-engine.exe").read_bytes(), b"engine")
        self.assertEqual((pk.control_dir / "strata.exe").read_bytes(), b"control")
        self.assertTrue((pk.engine_dir / ".ok").is_file() and (pk.control_dir / ".ok").is_file())
        self.assertEqual(sorted(p.name for p in self.cache.iterdir()), sorted([pk.engine_dir.name, pk.control_dir.name]))

    def test_checksum_mismatch_raises_and_leaves_no_cache(self):
        bad = self.sums_for(b"other", self.control)
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            ensure_packages(self.pair, self.cache, make_opener(self.files(bad)))
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_missing_checksum_entry_raises(self):
        with self.assertRaisesRegex(ValueError, "no sha256"):
            ensure_packages(self.pair, self.cache, make_opener(self.files(b"")))
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_cache_hit_needs_no_network(self):
        first = ensure_packages(self.pair, self.cache, make_opener(self.files()))
        self.assertEqual(ensure_packages(self.pair, self.cache, no_network), first)

    def test_partial_download_not_cached(self):
        with self.assertRaises(OSError):
            ensure_packages(self.pair, self.cache, make_opener(self.files(), fail_engine=True))
        self.assertEqual(list(self.cache.iterdir()), [])
        pk = ensure_packages(self.pair, self.cache, make_opener(self.files()))
        self.assertEqual((pk.engine_dir / "strata-engine.exe").read_bytes(), b"engine")

    def test_cached_engine_is_kept_when_control_fails(self):
        files = self.files()
        del files["https://x.invalid/c"]
        with self.assertRaises(OSError):
            ensure_packages(self.pair, self.cache, make_opener(files))
        self.assertEqual([p.name for p in self.cache.iterdir()], [self.pair.engine_name[:-len(".tar.gz")]])
        ensure_packages(self.pair, self.cache, make_opener(self.files()))

    def test_parse_sums_handles_star_and_spaces(self):
        self.assertEqual(parse_sums("abc *x.tar.gz\ndef  y.zip\n"), {"x.tar.gz": "abc", "y.zip": "def"})


def fake_packages(root: Path, version: str) -> Packages:
    """Flat engine and control package folders with the smallest manifest read_manifest accepts."""
    engine = root / f"strata-engine-{version}-windows-x86_64-cuda-cuda13"
    control = root / f"strata-control-{version}-windows-x86_64"
    engine.mkdir(parents=True)
    control.mkdir(parents=True)
    (engine / "strata-engine.exe").write_bytes(b"engine")
    (control / "strata.py").write_text("print('control')", encoding="utf-8")
    (engine / "engine-component.json").write_text(json.dumps({
        "schema_version": 1, "id": "strata-engine", "version": version, "platform": "windows-x86_64", "backend": "cuda",
        "entrypoint": "strata-engine.exe", "files": [{"path": "strata-engine.exe", "sha256": "0" * 64, "bytes": 6}]}), encoding="utf-8")
    return Packages(version, engine, control)


class HomeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from strata_control.engine_component import components_dir, scan
        self.components_dir, self.scan = components_dir, scan

    def test_env_points_inside_home_and_scrubs_strata(self):
        env = isolated_env(self.home, {"LOCALAPPDATA": "/real", "STRATA_COMPONENTS_DIR": "x", "PATH": "p"})
        for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
            self.assertTrue(Path(env[key]).is_relative_to(self.home), key)
        self.assertNotIn("STRATA_COMPONENTS_DIR", env)
        self.assertEqual(env["STRATA_NO_ACQUIRE"], "1")
        self.assertEqual(env["PATH"], "p")

    def test_components_dir_follows_env_on_every_platform(self):
        env = isolated_env(self.home, {})
        for windows in (True, False):
            self.assertTrue(self.components_dir(env, windows=windows).is_relative_to(self.home))

    def test_provisioned_engine_is_the_only_one_scan_sees(self):
        pk = fake_packages(self.root / "pk", "0.1.55")
        control = provision_home(self.home, pk)
        components = self.components_dir(isolated_env(self.home, {}))
        found, problems = self.scan(components)
        self.assertEqual([c.version for c in found], ["0.1.55"])
        self.assertEqual(problems, [])
        self.assertEqual(control, components / "control" / "0.1.55")
        self.assertTrue((control / "strata.py").is_file())
        self.assertEqual(json.loads((components / "control" / "active.json").read_text(encoding="utf-8")),
                         {"version": "0.1.55", "folder": "0.1.55"})

    def test_provision_twice_reuses_and_deletes_nothing(self):
        pk = fake_packages(self.root / "pk", "0.1.55")
        control = provision_home(self.home, pk)
        (control / "tuning.json").write_text("{}", encoding="utf-8")        # something a run left behind
        self.assertEqual(provision_home(self.home, pk), control)
        self.assertTrue((control / "tuning.json").is_file())
        self.assertEqual(len(self.scan(self.components_dir(isolated_env(self.home, {})))[0]), 1)


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def free_port(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def test_busy_when_port_open(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen()
            port = s.getsockname()[1]
            with self.assertRaisesRegex(Busy, str(port)):
                assert_idle([port], serving=lambda: [])

    def test_busy_when_strata_serving(self):
        with self.assertRaises(Busy):
            assert_idle([], serving=lambda: [{"pid": 1}])

    def test_idle_passes(self):
        assert_idle([self.free_port()], serving=lambda: [])

    def test_changed_lists_new_removed_modified(self):
        self.assertEqual(changed({"a": (1, 1), "b": (1, 1)}, {"a": (2, 1), "c": (1, 1)}), ["+c", "-b", "~a"])

    def test_snapshot_detects_edits_and_keeps_roots_apart(self):
        r1, r2 = self.root / "one", self.root / "two"
        for r in (r1, r2):
            (r / "sub").mkdir(parents=True)
            (r / "sub" / "f.txt").write_text("x", encoding="utf-8")
        roots = [r1, r2, self.root / "missing"]
        before = snapshot(roots)
        self.assertEqual(len(before), 2)                                    # same relative path in two roots, no collision
        (r2 / "sub" / "f.txt").write_text("longer", encoding="utf-8")
        (r1 / "new.txt").write_text("n", encoding="utf-8")
        diff = changed(before, snapshot(roots))
        self.assertEqual(len(diff), 2)
        self.assertTrue(diff[0].startswith("+") and diff[0].endswith("new.txt"))
        self.assertTrue(diff[1].startswith("~") and diff[1].endswith("f.txt"))

    def live_env(self, **extra):
        home = self.root / "u"
        return {"LOCALAPPDATA": str(self.root / "local"), "APPDATA": str(self.root / "roaming"), "XDG_DATA_HOME": str(self.root / "share"),
                "XDG_STATE_HOME": str(self.root / "state"), "XDG_CONFIG_HOME": str(self.root / "config"),
                "USERPROFILE": str(home), "HOME": str(home), **extra}

    def test_live_roots_cover_every_folder_the_control_writes(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from strata_control.engine_component import components_dir
        from strata_control.paths import state_base
        env = self.live_env()
        roots = live_roots(env, self.root / "elsewhere" / "m.gguf")
        home = self.root / "u"
        self.assertIn(home / ".strata", roots)
        self.assertIn(components_dir(env), roots)
        self.assertIn(state_base(env, home=home), roots)
        self.assertIn(self.root / ("roaming" if os.name == "nt" else "config") / ("Strata" if os.name == "nt" else "strata"), roots)
        self.assertEqual(len(roots), len(set(roots)))
        custom = self.root / "custom-components"
        self.assertIn(custom, live_roots(self.live_env(STRATA_COMPONENTS_DIR=str(custom)), self.root / "x.gguf"))

    def test_live_roots_include_a_real_looking_data_folder(self):
        data = self.root / "data"
        (data / "models" / "tag").mkdir(parents=True)
        (data / "packs").mkdir()
        self.assertIn(data, live_roots(self.live_env(), data / "models" / "tag" / "m.gguf"))

    def test_live_roots_never_a_drive_root_or_home(self):
        home = self.root / "u"
        (home / "models").mkdir(parents=True)
        (home / "packs").mkdir()                                     # a marker, but this is the home folder itself
        self.assertNotIn(home, live_roots(self.live_env(), home / "models" / "x.gguf"))
        (self.root / "models").mkdir()                               # a models folder with no marker beside it
        self.assertNotIn(self.root, live_roots(self.live_env(), self.root / "models" / "x.gguf"))
        anchor = Path(self.root.anchor)                              # D:/models/x.gguf style: the drive root
        self.assertNotIn(anchor, live_roots(self.live_env(), anchor / "models" / "x.gguf"))


FAKE = """import json, os, sys
from pathlib import Path
here = Path(__file__).parent
(here / "args.json").write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
log = Path(os.environ["HOME"]) / "AppData/Local/Strata/launch/x.engine.log"
log.parent.mkdir(parents=True, exist_ok=True)
with log.open("a", encoding="utf-8") as f:
    f.write("[strata] 2026-11-16 10:00:00 engine started: " + os.environ["FAKE_EXE"] + " --serve --port 1\\n")
err = here / "stderr.txt"
if err.exists():
    sys.stderr.write(err.read_text(encoding="utf-8"))
code = here / "exit.txt"
sys.exit(int(code.read_text()) if code.exists() else 0)
"""


def fake_health(status="ok"):
    return lambda url, timeout=None: io.BytesIO(json.dumps({"status": status}).encode())


class LaunchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        self.control = self.root / "control"
        (self.control / "bin").mkdir(parents=True)
        (self.control / "fake.py").write_text(FAKE, encoding="utf-8")
        if os.name == "nt":
            (self.control / "bin" / "strata.cmd").write_text(f'@"{sys.executable}" "%~dp0..\\fake.py" %*\r\n', encoding="utf-8")
        else:
            sh = self.control / "bin" / "strata"
            sh.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$(dirname "$0")/../fake.py" "$@"\n', encoding="utf-8")
            sh.chmod(0o755)
        self.exe = self.home / "components" / "engine" / "strata-engine-0.1.55 x" / "strata.exe"
        self.env = isolated_env(self.home, os.environ) | {"FAKE_EXE": str(self.exe)}

    def run_launch(self, tune=None, model=Path("m.gguf"), health=None):
        return launch(self.control, self.env, model, self.root / "data", 18090, tune, opener=health or fake_health())

    def args(self):
        return json.loads((self.control / "args.json").read_text(encoding="utf-8"))

    def test_launch_passes_tune_and_retune(self):
        r = self.run_launch(tune="quick")
        self.assertTrue(r.ok, r.reason)
        args = self.args()
        self.assertEqual(args[:2], ["launch", "--model-path"])
        for flag in ("--yes", "--no-tui", "--port", "18090", "--data-dir"):
            self.assertIn(flag, args)
        i = args.index("--tune")
        self.assertEqual(args[i:i + 3], ["--tune", "quick", "--retune"])
        self.assertEqual(args[args.index("--output") + 1], "json")

    def test_launch_without_tune_has_no_tune_flags(self):
        self.assertTrue(self.run_launch().ok)
        self.assertFalse({"--tune", "--retune"} & set(self.args()))

    def test_launch_args_keep_odd_paths(self):
        odd = r"C:\x\#GIT\Übung dir\m.gguf"
        self.assertTrue(self.run_launch(model=Path(odd)).ok)
        args = self.args()
        self.assertEqual(args[args.index("--model-path") + 1], str(Path(odd)))

    def test_failed_launch_reports_last_line(self):
        (self.control / "exit.txt").write_text("3")
        (self.control / "stderr.txt").write_text("usage: strata launch\nunknown flag --vram-elastic\n\n", encoding="utf-8")
        r = self.run_launch()
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "unknown flag --vram-elastic")
        self.assertGreaterEqual(r.seconds, 0)

    def test_unhealthy_server_is_not_ok(self):
        r = self.run_launch(health=fake_health("loading"))
        self.assertFalse(r.ok)
        self.assertIn("health", r.reason)

    def test_control_command_prefers_bundled_python(self):
        if os.name == "nt":
            self.assertEqual(control_command(self.control), [str(self.control / "bin" / "strata.cmd")])
            py = self.control / "python" / "python.exe"
            py.parent.mkdir()
            py.write_bytes(b"")
            self.assertEqual(control_command(self.control), [str(py), str(self.control / "strata.py")])
        else:
            self.assertEqual(control_command(self.control), [str(self.control / "bin" / "strata")])

    def write_log(self, *exes, name="x.engine.log"):
        log = self.home / "AppData" / "Local" / "Strata" / "launch" / name
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("".join(f"[strata] 2026-11-16 10:00:0{i} engine started: {e} --serve --ctx 4096\nother\n"
                               for i, e in enumerate(exes)), encoding="utf-8")

    def test_served_engine_takes_newest_line(self):
        old = self.home / "components" / "engine" / "old" / "strata.exe"
        self.write_log(old, self.exe)
        self.assertEqual(served_engine(self.home), self.exe.parent)

    def test_served_engine_reads_quoted_exe(self):
        self.write_log(f'"{self.exe}"')
        self.assertEqual(served_engine(self.home), self.exe.parent)

    def test_version_mismatch_skips(self):
        pk = Packages("0.1.55", self.root / "strata-engine-0.1.55-windows-x86_64-cuda-cuda13", self.control)
        self.write_log(self.home / "components" / "engine" / "strata-engine-0.1.56-windows-x86_64-cuda-cuda13" / "strata.exe")
        msg = version_problem(self.home, pk)
        self.assertEqual(msg, "served engine strata-engine-0.1.56-windows-x86_64-cuda-cuda13, expected strata-engine-0.1.55-windows-x86_64-cuda-cuda13")
        self.write_log(self.home / "components" / "engine" / pk.engine_dir.name / "strata.exe")
        self.assertIsNone(version_problem(self.home, pk))

    def test_version_problem_without_log(self):
        pk = Packages("0.1.55", self.root / "e", self.control)
        self.assertIn("expected e", version_problem(self.home, pk))

    def test_collect_configs_keeps_tuned_files_and_never_overwrites(self):
        launch_dir = self.home / "AppData" / "Local" / "Strata" / "launch"
        launch_dir.mkdir(parents=True)
        (launch_dir / "x.json").write_text("{}", encoding="utf-8")
        (launch_dir / "x.tune.json").write_text('{"tuned": 1}', encoding="utf-8")
        (launch_dir / "x.engine.log").write_text("log", encoding="utf-8")
        dest = self.root / "configs"
        got = collect_configs(self.home, dest)
        rel = Path("AppData/Local/Strata/launch")
        self.assertEqual(sorted(p.name for p in got), ["x.json", "x.tune.json"])
        self.assertFalse((dest / rel / "x.engine.log").exists())
        (launch_dir / "x.tune.json").write_text('{"tuned": 2}', encoding="utf-8")
        collect_configs(self.home, dest)
        self.assertEqual((dest / rel / "x.tune.json").read_text(encoding="utf-8"), '{"tuned": 1}')
        self.assertEqual((dest / rel / "x.tune.json.1").read_text(encoding="utf-8"), '{"tuned": 2}')

    def test_collect_configs_finds_posix_lowercase_state(self):
        d = self.home / ".local" / "state" / "strata" / "launch"
        d.mkdir(parents=True)
        (d / "x.choices.json").write_text("{}", encoding="utf-8")
        presets = self.home / ".config" / "strata" / "presets"
        presets.mkdir(parents=True)
        (presets / "p.json").write_text("{}", encoding="utf-8")
        self.assertEqual(sorted(p.name for p in collect_configs(self.home, self.root / "c")), ["p.json", "x.choices.json"])


if __name__ == "__main__":
    unittest.main()
