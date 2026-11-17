import unittest
from serve import server


def cfg(gpus, args=(), layer_split=None):
    c = {"gpu": gpus, "args": list(args)}
    if layer_split is not None:
        c["layer_split"] = layer_split
    return c


class PlacementHelpers(unittest.TestCase):
    # 1. one helper
    def test_one_helper_suppresses_split(self):
        for a in (["--peer-device", "1"], ["--expert-cache-device1", "auto"]):
            self.assertFalse(server.layer_split_of(cfg([0, 1], a)))
            self.assertTrue(server.helper_cache_of(cfg([0, 1], a)))

    # 2/3. two and three helpers
    def test_multi_helper(self):
        a = ["--expert-cache-device1", "auto", "--expert-cache-device2", "auto"]
        self.assertFalse(server.layer_split_of(cfg([0, 1, 2], a)))
        b = a + ["--expert-cache-device3", "auto"]
        self.assertFalse(server.layer_split_of(cfg([0, 1, 2, 3], b)))

    # 4. helper + explicit layer split -> configuration error
    def test_helper_plus_split_is_error(self):
        with self.assertRaises(SystemExit):
            server.check_placement(cfg([0, 1], ["--expert-cache-device1", "auto"], layer_split="24"))
        with self.assertRaises(SystemExit):
            server.check_placement(cfg([0, 1], ["--peer-device", "1", "--layer-split", "24"]))

    # 4b. the default "auto" split value beside helper args is NOT a conflict: helper wins, no injection
    def test_default_auto_value_helper_wins(self):
        c = cfg([0, 1], ["--expert-cache-device1", "auto"], layer_split="auto")
        self.assertFalse(server.layer_split_of(c))
        server.check_placement(c)  # must not raise

    # 5. helper + automatic placement -> no injection
    def test_helper_beats_automatic(self):
        c = cfg([0, 1], ["--peer-device", "1"])
        self.assertFalse(server.layer_split_of(c))
        server.check_placement(c)  # must not raise

    # 6. no helper, multiple GPUs -> split still applies (status quo)
    def test_no_helper_multiple_gpus(self):
        self.assertTrue(server.layer_split_of(cfg([0, 1])))
        server.check_placement(cfg([0, 1]))  # no error

    # 7. single primary GPU alone
    def test_single_gpu(self):
        self.assertFalse(server.layer_split_of(cfg([0])))
        server.check_placement(cfg([0]))

    # 8. reordered runtime ordinals: the primary is whatever is first in the visible list
    def test_reordered_ordinals_peer_still_helper(self):
        c = cfg([2, 0], ["--peer-device", "1"])  # visible slot 1 = physical 0, helper slot of visible order
        self.assertFalse(server.layer_split_of(c))
        server.check_placement(c)

    # 9a. missing helper GPU: deviceN slots with a lone gpu, and peer index outside the visible set
    def test_missing_helper_gpu(self):
        with self.assertRaises(SystemExit):
            server.check_placement(cfg([0], ["--expert-cache-device1", "auto"]))
        with self.assertRaises(SystemExit):
            server.check_placement(cfg([0, 1], ["--peer-device", "2"]))
        with self.assertRaises(SystemExit):
            server.check_placement(cfg([0, 1], ["--peer-device", "0"]))

    # 9b. helper with insufficient free VRAM is caught by the engine (cache allocation), not statically here.
    def test_vram_gate_is_engine_side(self):
        # the serve layer cannot know free VRAM; ensure it does not pretend to: placement stays valid
        c = cfg([0, 1], ["--expert-cache-device1", "4096"])
        self.assertFalse(server.layer_split_of(c))
        server.check_placement(c)

    # 10. #1352 speed reorder must NOT touch helper arms: primary stays first (found live in production:
    # where [0,1,2] with the 9060 primary reordered to 1,2,0 and would have made a 6600 the primary).
    def test_helper_arm_keeps_config_visible_order(self):
        sc = {0: 120, 1: 60, 2: 60}
        vr = {0: 16384, 1: 8192, 2: 8192}
        c = cfg([0, 1, 2], ["--expert-cache-device1", "auto", "--expert-cache-device2", "auto"])
        self.assertEqual(server.ordered_gpus(c, scores=sc, vrams=vr), [0, 1, 2])
        c2 = cfg([0, 1], ["--peer-device", "1"])
        self.assertEqual(server.ordered_gpus(c2, scores={0: 120, 1: 60}, vrams={0: 16384, 1: 8192}), [0, 1])


if __name__ == "__main__":
    unittest.main()
