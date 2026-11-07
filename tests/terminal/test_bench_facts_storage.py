"""#4493: `storage_of` gave up on any mount whose source was not a /dev path, so a model copied to /dev/shm (tmpfs), a
btrfs subvolume (`/dev/nvme0n1p2[/@home]`) or an NFS export all reported "the drive holding the model files could not
be identified". The strings below are what `findmnt -n -o SOURCE,FSTYPE --target` and `lsblk -J -s` print."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

from strata_cli import bench_facts as bf

NVME = {"blockdevices": [{"name": "nvme0n1p2", "type": "part", "children": [
    {"name": "nvme0n1", "type": "disk", "rota": False, "model": "Samsung SSD 980 PRO 2TB", "tran": "nvme"}]}]}


@unittest.skipUnless(sys.platform.startswith("linux"), "the findmnt/lsblk branch")
class StorageKinds(unittest.TestCase):
    def storage(self, findmnt_line, lsblk=None):
        calls = []

        def run_cmd(cmd, timeout=15):
            calls.append(cmd)
            if cmd[:1] == ["findmnt"]:
                return findmnt_line + "\n"
            if cmd[:1] == ["lsblk"]:
                return json.dumps(lsblk or {"blockdevices": []})
            return ""

        with mock.patch.object(bf, "run_cmd", run_cmd):
            return bf.storage_of(Path("/mnt/models/x.gguf")), calls

    def test_tmpfs_is_ram(self):
        found, _ = self.storage("tmpfs  tmpfs")
        self.assertEqual(found, [{"kind": "ram", "model": "tmpfs", "bus": "memory"}])

    def test_an_nfs_export_is_network_and_keeps_the_host_out(self):
        found, _ = self.storage("nas.home.local:/volume1/models  nfs4")
        self.assertEqual(found, [{"kind": "network", "model": "nfs4", "bus": "network"}])

    def test_a_btrfs_subvolume_is_looked_up_by_its_device(self):
        found, calls = self.storage("/dev/nvme0n1p2[/@home]  btrfs", NVME)
        self.assertEqual(found, [{"kind": "nvme", "model": "Samsung SSD 980 PRO 2TB", "bus": "nvme"}])
        self.assertEqual([c for c in calls if c[:1] == ["lsblk"]][0][-1], "/dev/nvme0n1p2")

    def test_a_plain_device_is_unchanged(self):
        found, _ = self.storage("/dev/nvme0n1p2  ext4", NVME)
        self.assertEqual(found[0]["kind"], "nvme")

    def test_an_unknown_source_is_still_none(self):
        found, _ = self.storage("overlay  overlay")
        self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
