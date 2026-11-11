"""#4473: `strata bench run --url` has no run config, so the cards come from the server's name for them. On a layer
split over two identical cards the server says "NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090", which matched no
single card, and the report kept the first card only."""
from __future__ import annotations

import unittest
from unittest import mock

from strata_cli import bench_facts as bf


def card(i, name):
    return {"device_index": i, "name": name, "memory_bytes": 25769803776, "memory_kind": "dedicated",
            "power_limit_w": 250.0}


class ServedCards(unittest.TestCase):
    def names(self, local, served_gpu, selected=()):
        with mock.patch.object(bf, "nvidia_gpus", lambda: (local, "580.178.04", "13.0")):
            block, _ = bf.hardware(bf.Why(), selected=list(selected), served_gpu=served_gpu, model_dir=None,
                                   declared_background=[])
        return [(g["device_index"], g["name"]) for g in block["gpus"]]

    def test_two_identical_cards_named_by_the_server_are_both_reported(self):
        local = [card(0, "NVIDIA GeForce RTX 3090"), card(1, "NVIDIA GeForce RTX 3090")]
        self.assertEqual(self.names(local, "NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090"),
                         [(0, "NVIDIA GeForce RTX 3090"), (1, "NVIDIA GeForce RTX 3090")])

    def test_a_card_the_server_does_not_use_is_left_out(self):
        local = [card(0, "NVIDIA GeForce GT 1030"), card(1, "NVIDIA GeForce RTX 3090"), card(2, "NVIDIA GeForce RTX 3090")]
        self.assertEqual(self.names(local, "NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090"),
                         [(1, "NVIDIA GeForce RTX 3090"), (2, "NVIDIA GeForce RTX 3090")])

    def test_one_served_card_of_two_stays_one(self):
        local = [card(0, "NVIDIA GeForce RTX 3090"), card(1, "NVIDIA GeForce RTX 3090")]
        self.assertEqual(self.names(local, "NVIDIA GeForce RTX 3090"), [(0, "NVIDIA GeForce RTX 3090")])

    def test_a_mixed_pair_keeps_both(self):
        local = [card(0, "NVIDIA GeForce RTX 3090"), card(1, "NVIDIA GeForce RTX 4090")]
        self.assertEqual(self.names(local, "NVIDIA GeForce RTX 4090 + NVIDIA GeForce RTX 3090"),
                         [(1, "NVIDIA GeForce RTX 4090"), (0, "NVIDIA GeForce RTX 3090")])

    def test_a_run_config_selection_still_wins(self):
        local = [card(0, "NVIDIA GeForce RTX 3090"), card(1, "NVIDIA GeForce RTX 3090")]
        self.assertEqual(self.names(local, "NVIDIA GeForce RTX 3090", selected=[1]), [(1, "NVIDIA GeForce RTX 3090")])


if __name__ == "__main__":
    unittest.main()
