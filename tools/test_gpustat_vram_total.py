"""CPU-only checks for the Intel GPU sampler and the reader that consumes it.

Three things are covered:

* the capacity is the card's VRAM BAR.  `/sys/bus/pci/devices/<PCI>/resource` lists the six
  BARs and the ROM at indices 0-6, then the bridge windows the card sits behind at 7-12.  A
  bridge window is sized for everything behind it and can be larger than the card itself, so
  reading the largest line reports the bridge rather than the card's memory;
* the sampler covers every Intel card and writes one entry per card, keyed by PCI address;
* the reader uses the entry belonging to its own card.  The flat fields describe the first
  card, so applying them on the second card reports the other card's figures as its own.
"""
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# An address that cannot exist, so importing the sampler does not probe this machine's cards.
os.environ.setdefault("GPUSTAT_PCI", "0000:00:00.0")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gpustat = _load("gpustat", ROOT / "sycl" / "tools" / "gpustat.py")
telemetry = _load("xe_telemetry", ROOT / "sycl" / "serve" / "xe_telemetry.py")


def resource_file(regions):
    return "\n".join(regions) + "\n"


EMPTY = "0x0 0x0 0x0"


class VramTotalTests(unittest.TestCase):
    def test_a_bridge_window_larger_than_the_card_is_not_the_capacity(self):
        # Indices 0-6 are the card's own BARs and ROM; 7-12 are the bridge it sits behind.
        # Here the bridge window spans more than the card's memory, which is the shape that
        # produced a capacity larger than the card on a two-GPU host.
        text = resource_file([
            "0x7607000000 0x7607ffffff 0x0",   # 0: BAR0
            EMPTY,                              # 1
            "0x6000000000 0x67ffffffff 0x0",   # 2: BAR2, 32 GiB - maps all of VRAM
            EMPTY, EMPTY, EMPTY,                # 3-5
            EMPTY,                              # 6: ROM
            "0x7600000000 0x7606ffffff 0x0",   # 7: bridge window
            EMPTY,                              # 8
            "0x6800000000 0x75ffffffff 0x0",   # 9: bridge window, 56 GiB - larger than the card
            EMPTY, EMPTY, EMPTY,                # 10-12
        ])
        with patch("builtins.open", mock_open(read_data=text)):
            self.assertEqual(gpustat.vram_total_mb("0000:0d:00.0"), 32 * 1024)

    def test_the_card_bar_is_used_when_no_bridge_window_is_larger(self):
        # The ordinary shape: the VRAM BAR is the largest of the card's own regions, and the
        # bridge window behind it is no bigger, so the reading is the same either way.
        text = resource_file([
            "0xfa000000 0xfaffffff 0x0",       # 0: BAR0
            EMPTY,                              # 1
            "0x7800000000 0x7bffffffff 0x0",   # 2: BAR2, 16 GiB
            EMPTY, EMPTY, EMPTY,                # 3-5
            EMPTY,                              # 6: ROM
            "0x7800000000 0x7bffffffff 0x0",   # 7: bridge window, the same span
            EMPTY,                              # 8
            EMPTY, EMPTY, EMPTY, EMPTY,         # 9-12
        ])
        with patch("builtins.open", mock_open(read_data=text)):
            self.assertEqual(gpustat.vram_total_mb("0000:11:00.0"), 16 * 1024)

    def test_the_explicit_override_still_wins(self):
        with patch.dict(os.environ, {"GPUSTAT_VRAM_MB": "24576"}):
            self.assertEqual(gpustat.vram_total_mb("0000:0d:00.0"), 24576)


class ClientFilterTests(unittest.TestCase):
    """clients() must count only the card being sampled.

    With two cards fitted, summing every /dev/dri client made the "used" figure larger than
    the total it is drawn against, so the Monitor read over 100 percent.
    """

    def _proc(self, entries):
        """Build fake /proc/<pid> entries: (pid, fd, card, cid, pdev, vram_kb)."""
        listing, links, text = {}, {}, {}
        for pid, fd, card, cid, pdev, vram_kb in entries:
            listing.setdefault(f"/proc/{pid}/fd", []).append(str(fd))
            links[f"/proc/{pid}/fd/{fd}"] = card
            body = [f"drm-client-id:\t{cid}"]
            if pdev is not None:
                body.append(f"drm-pdev:\t{pdev}")
            body += [f"drm-resident-vram0:\t{vram_kb} KiB",
                     "drm-cycles-rcs:\t0", "drm-total-cycles-rcs:\t1"]
            text[f"/proc/{pid}/fdinfo/{fd}"] = "\n".join(body) + "\n"

        real_open, real_listdir, real_readlink = open, os.listdir, os.readlink

        def fake_open(path, *a, **k):
            if str(path) in text:
                return io.StringIO(text[str(path)])
            return real_open(path, *a, **k)

        def fake_listdir(path):
            if str(path) in listing:
                return listing[str(path)]
            return real_listdir(path)

        def fake_readlink(path):
            if str(path) in links:
                return links[str(path)]
            return real_readlink(path)

        pids = [f"/proc/{pid}" for pid, *_ in entries]
        return pids, fake_open, fake_listdir, fake_readlink

    def collect(self, pci, entries):
        pids, fake_open, fake_listdir, fake_readlink = self._proc(entries)
        with patch.object(gpustat.glob, "glob", lambda pattern: pids), \
             patch.object(gpustat, "PCIS", [pci]), \
             patch("builtins.open", fake_open), \
             patch.object(gpustat.os, "listdir", fake_listdir), \
             patch.object(gpustat.os, "readlink", fake_readlink):
            return gpustat.clients(pci)

    def test_another_cards_client_is_not_counted(self):
        got = self.collect("0000:0d:00.0", [
            (100, 3, "/dev/dri/card1", "a", "0000:0d:00.0", 1000),
            (101, 3, "/dev/dri/card0", "b", "0000:11:00.0", 9000),
        ])
        self.assertEqual(sorted(got), ["a"])
        self.assertEqual(got["a"]["vram_kb"], 1000)

    def test_a_client_without_a_pdev_is_still_counted(self):
        # Older kernels may not report drm-pdev; those clients keep the previous behaviour.
        got = self.collect("0000:0d:00.0", [(100, 3, "/dev/dri/card1", "a", None, 500)])
        self.assertEqual(sorted(got), ["a"])
        self.assertEqual(got["a"]["vram_kb"], 500)


class EnumerationTests(unittest.TestCase):
    """find_cards() must return every Intel card on xe, not just the first."""

    def _cards(self, listing, vendors, tiles):
        """listing: {drm path: resolved target}; vendors/tiles: {drm path: value}."""
        def fake_glob(pattern):
            if pattern.startswith("/sys/class/drm/"):
                return listing
            return [t for t in tiles.get(pattern, [])]

        def fake_open(path, *a, **k):
            for d, v in vendors.items():
                if str(path) == f"{d}/vendor":
                    return io.StringIO(v)
            raise OSError(path)

        with patch.object(gpustat.glob, "glob", fake_glob), \
             patch("builtins.open", fake_open), \
             patch.object(gpustat.os.path, "realpath", lambda p: listing.get(p, p)):
            return gpustat.find_cards()

    def test_every_intel_card_is_returned(self):
        listing = {"/sys/class/drm/card0/device": "/sys/devices/a/0000:0d:00.0",
                   "/sys/class/drm/card1/device": "/sys/devices/b/0000:11:00.0"}
        vendors = {"/sys/class/drm/card0/device": "0x8086\n", "/sys/class/drm/card1/device": "0x8086\n"}
        tiles = {"/sys/class/drm/card0/device/tile*/gt*": ["/sys/class/drm/card0/device/tile0/gt0"],
                 "/sys/class/drm/card1/device/tile*/gt*": ["/sys/class/drm/card1/device/tile0/gt0"]}
        self.assertEqual(self._cards(listing, vendors, tiles), ["0000:0d:00.0", "0000:11:00.0"])

    def test_a_non_intel_card_is_skipped(self):
        listing = {"/sys/class/drm/card0/device": "/sys/devices/a/0000:0d:00.0",
                   "/sys/class/drm/card1/device": "/sys/devices/b/0000:2b:00.0"}
        vendors = {"/sys/class/drm/card0/device": "0x8086\n", "/sys/class/drm/card1/device": "0x10de\n"}
        tiles = {"/sys/class/drm/card0/device/tile*/gt*": ["/sys/class/drm/card0/device/tile0/gt0"],
                 "/sys/class/drm/card1/device/tile*/gt*": ["/sys/class/drm/card1/device/tile0/gt0"]}
        self.assertEqual(self._cards(listing, vendors, tiles), ["0000:0d:00.0"])


class HwmonTests(unittest.TestCase):
    """hwmon_dir() must pick the 'xe' entry belonging to the card, not the first one found."""

    def test_the_matching_card_is_chosen(self):
        names = ["/sys/class/hwmon/hwmon10/name", "/sys/class/hwmon/hwmon11/name"]
        bodies = {names[0]: "xe\n", names[1]: "xe\n"}
        devices = {"/sys/class/hwmon/hwmon10/device": "/sys/devices/a/0000:0d:00.0",
                   "/sys/class/hwmon/hwmon11/device": "/sys/devices/b/0000:11:00.0"}

        def fake_open(path, *a, **k):
            if str(path) in bodies:
                return io.StringIO(bodies[str(path)])
            raise OSError(path)

        real = os.path.realpath

        def fake_realpath(p):
            if str(p) in devices:
                return devices[str(p)]
            if str(p) == "/sys/bus/pci/devices/0000:11:00.0":
                return "/sys/devices/b/0000:11:00.0"
            return real(p)

        with patch.object(gpustat.glob, "glob", lambda p: names), \
             patch("builtins.open", fake_open), \
             patch.object(gpustat.os.path, "realpath", fake_realpath):
            # the second card must get hwmon11, not the first 'xe' entry
            self.assertEqual(gpustat.hwmon_dir("0000:11:00.0"), "/sys/class/hwmon/hwmon11")


class NameTests(unittest.TestCase):
    """card_name() must return the device name from lspci -mm without the trailing fields."""

    def test_revision_and_subsystem_fields_are_not_swallowed(self):
        line = ('11:00.0 "VGA compatible controller" "Intel Corporation" "DG2 [Arc A770]" '
                '-r08 -p00 "Sparkle Computer Co., Ltd." "Device 4134"\n')
        out = subprocess_result(line)
        with patch.object(gpustat.subprocess, "run", out):
            self.assertEqual(gpustat.card_name("56a0", "0000:11:00.0"), "DG2 [Arc A770]")

    def test_the_named_card_skips_lspci(self):
        with patch.object(gpustat.subprocess, "run", side_effect=AssertionError("must not run lspci")):
            self.assertEqual(gpustat.card_name("e223", "0000:0d:00.0"),
                             "Intel(R) Arc(TM) Pro B70 Graphics")


def subprocess_result(stdout):
    return lambda *a, **k: types.SimpleNamespace(stdout=stdout, returncode=0)


class DocumentTests(unittest.TestCase):
    """The written document carries every card under "cards" and the first card flat."""

    def test_each_card_is_written_and_the_first_is_flattened(self):
        cards = {"0000:0d:00.0": {"name": "B70", "pci": "0000:0d:00.0", "ts": 1.0,
                                  "vram_total_mb": 32768, "vram_used_mb": 100},
                 "0000:11:00.0": {"name": "A770", "pci": "0000:11:00.0", "ts": 1.0,
                                  "vram_total_mb": 16384, "vram_used_mb": 200}}
        with patch.object(gpustat, "PCIS", list(cards)):
            doc = gpustat.document(cards)
        self.assertEqual(sorted(doc["cards"]), sorted(cards))
        self.assertEqual(doc["vram_total_mb"], 32768)      # flat block is the first card
        self.assertEqual(doc["pci"], "0000:0d:00.0")


class ReaderMatchingTests(unittest.TestCase):
    """The reader must use the sampler entry for its own card, and nothing else."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def _reader(self, pci):
        g = telemetry._XeGpu.__new__(telemetry._XeGpu)     # no probing of this machine
        g.dev = f"/sys/bus/pci/devices/{pci}"
        g.hwmon = None
        g._e = None
        g.STAT = self.tmp.name
        return g

    def _write(self, doc):
        with open(self.tmp.name, "w") as f:
            json.dump(doc, f)

    def test_the_own_card_entry_is_used(self):
        self._write({"ts": time.time(), "cards": {
            "0000:0d:00.0": {"pci": "0000:0d:00.0", "vram_total_mb": 32768, "vram_used_mb": 100},
            "0000:11:00.0": {"pci": "0000:11:00.0", "vram_total_mb": 16384, "vram_used_mb": 200}},
            "pci": "0000:0d:00.0", "vram_total_mb": 32768, "vram_used_mb": 100})
        st = self._reader("0000:11:00.0")._stat()
        self.assertEqual(st["vram_total_mb"], 16384)       # not the flattened first card

    def test_a_card_the_sampler_did_not_read_yields_nothing(self):
        self._write({"ts": time.time(), "cards": {
            "0000:0d:00.0": {"pci": "0000:0d:00.0", "vram_total_mb": 32768}},
            "pci": "0000:0d:00.0", "vram_total_mb": 32768})
        self.assertIsNone(self._reader("0000:11:00.0")._stat())

    def test_a_single_card_document_for_another_card_yields_nothing(self):
        # The shape that would otherwise show one card's capacity on another card.
        self._write({"ts": time.time(), "pci": "0000:11:00.0", "vram_total_mb": 16384, "vram_used_mb": 200})
        self.assertIsNone(self._reader("0000:0d:00.0")._stat())

    def test_a_single_card_document_for_this_card_is_used(self):
        self._write({"ts": time.time(), "pci": "0000:0d:00.0", "vram_total_mb": 32768, "vram_used_mb": 100})
        self.assertEqual(self._reader("0000:0d:00.0")._stat()["vram_total_mb"], 32768)

    def test_a_document_without_a_pci_field_is_used(self):
        # An older sampler wrote no pci field at all; it is assumed to be this card.
        self._write({"ts": time.time(), "vram_total_mb": 32768, "vram_used_mb": 100})
        self.assertEqual(self._reader("0000:0d:00.0")._stat()["vram_total_mb"], 32768)

    def test_a_stale_document_is_ignored(self):
        self._write({"ts": time.time() - 120, "pci": "0000:0d:00.0", "vram_total_mb": 32768})
        self.assertIsNone(self._reader("0000:0d:00.0")._stat())

    def test_read_reports_the_own_cards_memory(self):
        self._write({"ts": time.time(), "cards": {
            "0000:0d:00.0": {"pci": "0000:0d:00.0", "vram_total_mb": 32768, "vram_used_mb": 100},
            "0000:11:00.0": {"pci": "0000:11:00.0", "vram_total_mb": 16384, "vram_used_mb": 200}},
            "pci": "0000:0d:00.0", "vram_total_mb": 32768, "vram_used_mb": 100})
        out = self._reader("0000:11:00.0").read()
        self.assertEqual(out["mem_total"], 16384 * 2**20)
        self.assertEqual(out["mem_used"], 200 * 2**20)


if __name__ == "__main__":
    unittest.main()
