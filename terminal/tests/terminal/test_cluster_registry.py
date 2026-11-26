"""Opt-in LAN cluster registry: validation, persistence, and secret handling."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from strata_control.cluster import Cluster, ClusterError, Node, Registry, clusters_dir


class ClusterRegistry(unittest.TestCase):
    def setUp(self):
        # A fresh directory per test; no real user profile is touched.
        self.root = Path(tempfile.mkdtemp(prefix="strata-cluster-test-"))
        self.env = patch.dict("os.environ", {"STRATA_HOME": str(self.root)}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.registry = Registry()

    def test_default_directory_follows_strata_home(self):
        self.assertEqual(clusters_dir(), self.root / "clusters")
        self.assertEqual(self.registry.root, self.root / "clusters")

    def test_create_and_round_trip(self):
        cluster = self.registry.create("home-lab", model="hy4/SQ-64", token_env="MY_CLUSTER_TOKEN")
        cluster.add(Node(id="desk", address="192.168.1.10", port=8791, backend="cuda",
                         gpu="RTX 4090", vram_gib=24, ram_gib=64, expert_slots=8192, role="head"))
        cluster.add(Node(id="laptop", address="laptop.local", backend="cuda",
                         vram_gib=8, ram_gib=32, expert_slots=2048, status="healthy"))
        self.registry.save(cluster)
        again = self.registry.load("home-lab")
        self.assertEqual(again.model, "hy4/SQ-64")
        self.assertEqual(set(again.nodes), {"desk", "laptop"})
        self.assertEqual(again.nodes["desk"].gpu, "RTX 4090")
        self.assertEqual(again.capacities()["vram_gib"], 32.0)
        self.assertEqual(self.registry.names(), ["home-lab"])

    def test_no_secret_is_written_and_token_comes_from_environment(self):
        with patch.dict("os.environ", {"MY_CLUSTER_TOKEN": "super-secret"}, clear=False):
            self.registry.create("lab", token_env="MY_CLUSTER_TOKEN")
            raw = json.loads((self.registry.root / "lab.json").read_text(encoding="utf-8"))
            self.assertNotIn("super-secret", json.dumps(raw))
            self.assertEqual(raw["token_env"], "MY_CLUSTER_TOKEN")
            self.assertEqual(self.registry.load("lab").token(), "super-secret")
        with self.assertRaises(ClusterError):
            self.registry.load("lab").token()

    def test_rejects_invalid_names_addresses_ports_and_backends(self):
        for bad in ("Home", "1lab", "a b", "a" * 65):
            with self.assertRaises(ClusterError, msg=bad):
                self.registry.create(bad)
        with self.assertRaises(ClusterError):
            Node(id="x", address="not a host!!")
        with self.assertRaises(ClusterError):
            Node(id="x", address="127.0.0.1", port=0)
        with self.assertRaises(ClusterError):
            Node(id="x", address="127.0.0.1", backend="vulkan")
        with self.assertRaises(ClusterError):
            Node(id="x", address="127.0.0.1", role="master")

    def test_duplicate_remove_and_delete(self):
        self.registry.create("lab")
        node = Node(id="a", address="10.0.0.2")
        self.registry.add_node("lab", node)
        with self.assertRaises(ClusterError):
            self.registry.add_node("lab", Node(id="a", address="10.0.0.3"))
        self.assertNotIn("a", self.registry.remove_node("lab", "a").nodes)
        with self.assertRaises(ClusterError):
            self.registry.remove_node("lab", "a")
        self.registry.delete("lab")
        self.assertEqual(self.registry.names(), [])

    def test_unknown_fields_and_bad_files_are_refused(self):
        self.registry.create("lab")
        path = self.registry.root / "lab.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["extra"] = "unexpected"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(ClusterError):
            self.registry.load("lab")
        path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(ClusterError):
            self.registry.load("lab")

    def test_cluster_object_rejects_inconsistent_node_keys(self):
        with self.assertRaises(ClusterError):
            Cluster(name="lab", nodes={"other": Node(id="a", address="10.0.0.2")})


if __name__ == "__main__":
    unittest.main()
