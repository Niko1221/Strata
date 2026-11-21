"""The run config setup writes: the KV format it names, on a gfx12 card with a q4u32 engine and everywhere else.

The snapshot machinery drives setup.py's own last stretch (strata_control.setup_config.config_for) on a simulated
machine, so this is the real path, not a helper called on its own.
"""

import unittest
from pathlib import Path

from tests.snapshot.matrix import model_set, machine
from tests.snapshot.setup_driver import drive_setup
from tests.snapshot.world import World

ROOT = Path(__file__).resolve().parents[2]

#: this tree's engine (the Strata release it changed, plus the suffix of what it adds) and an upstream one
Q4U32 = "q4u32"
UPSTREAM = ""


def kv_of(args: list) -> str | None:
    """The format a run config names, or None when it names none."""
    if "--kv" not in args:
        return None
    return args[args.index("--kv") + 1]


class SetupWritesTheKvFormat(unittest.TestCase):
    def drive(self, machine_key: str, flavor: str):
        world = World(machine(machine_key), model_set("iq3_s")).build()
        try:
            rec = drive_setup(world, engine_flavor=flavor)
        finally:
            world.close()
        self.assertNotIn("not_driven", rec, rec)
        return rec["config_file"]["args"]

    def test_a_gfx12_card_with_a_q4u32_engine_leaves_the_format_to_the_engine(self):
        args = self.drive("r9700_linux", Q4U32)                  # the R9700 (gfx1201)
        self.assertIsNone(kv_of(args), "the engine picks the format itself")

    def test_a_gfx12_card_with_an_upstream_engine_names_the_format_as_before(self):
        args = self.drive("r9700_linux", UPSTREAM)
        self.assertEqual(kv_of(args), "int8")

    def test_another_card_with_a_q4u32_engine_names_the_format_as_before(self):
        self.assertEqual(kv_of(self.drive("nv_12gb_streaming", Q4U32)), "int8")
        self.assertEqual(kv_of(self.drive("strix_halo_win", Q4U32)), "int8")


if __name__ == "__main__":
    unittest.main()
