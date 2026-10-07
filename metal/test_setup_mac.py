"""metal/setup_mac.py's config conversion, without a GPU, a model or the network:  python -m unittest metal.test_setup_mac"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup_mac as M  # noqa: E402

HIP_CFG = {   # what setup.py writes on its HIP path before the overlay turns it into the Metal one
    "exe": "/x/engine/strata", "cwd": "/x", "tokenizer": "/d/packs/q2_0/tokenizer", "model_name": "q", "port": 8080,
    "args": ["--pack", "/d/packs/q2_0", "--native", "/d/models/Q2_0/a-00001-of-00002.gguf", "--expert-cache", "auto",
             "--mtp", "/d/mtp/rt", "--max-context", "65536", "--kv", "q4_0"],
    "lib_dirs": ["/opt/rocm/lib"], "backend": "hip", "gpu": 0, "gpus_asked": True, "draft_vocab": "en",
    "vision": {"exe": "/x/engine-metal/strata-vision", "mmproj": "/d/models/mmproj.gguf", "gpu": True},
    "sampling": {"temperature": 0.6},
}


class ToMetal(unittest.TestCase):
    def setUp(self):
        M.MTP.update(on=False, dir=None)

    def test_plain(self):
        out = M.to_metal(dict(HIP_CFG))
        self.assertEqual(out["backend"], "metal")
        self.assertEqual(out["exe"], str(M.ENGINE / M.EXE))
        self.assertEqual(out["args"], ["--gguf", "/d/models/Q2_0/a-00001-of-00002.gguf", "--max-context", "65536",
                                       "--kv", "q4_0"])                 # the CUDA engine's --mtp rt folder is gone
        for k in ("lib_dirs", "gpu", "gpus_asked", "draft_vocab"):
            self.assertNotIn(k, out)
        self.assertEqual(out["vision"], HIP_CFG["vision"])              # images and hand-set keys stay
        self.assertEqual(out["sampling"], {"temperature": 0.6})

    def test_mtp_brings_its_head_and_a_16_bit_cache(self):
        M.MTP.update(on=True, dir=Path("/d/mtp"))
        with mock.patch.object(M, "mtp_gguf", return_value=Path("/d/mtp/head.gguf")) as made, \
                mock.patch.object(M.S, "ok"):
            out = M.to_metal(dict(HIP_CFG))
        made.assert_called_once_with(Path("/d/mtp"))
        self.assertEqual(out["args"][-6:], ["--kv", "f16", "--mtp", "/d/mtp/head.gguf", "--spec", "3"])

    def test_a_context_past_the_trained_one_is_refused(self):
        cfg = dict(HIP_CFG, args=["--native", "/m.gguf", "--max-context", str(M.TRAINED_CTX * 2)])
        with mock.patch.object(M.S, "fail", side_effect=SystemExit(1)) as fail:
            with self.assertRaises(SystemExit):
                M.to_metal(cfg)
        self.assertIn("rope scaling", fail.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
