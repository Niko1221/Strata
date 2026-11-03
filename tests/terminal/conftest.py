"""Terminal tests do not depend on the graphics card of the box they run on.

The Drafting step of `strata launch` (and the Drafting line of the summary) exists only where the engine can run DFlash2 (CUDA or HIP), so the number
of steps and the layout of the summary used to differ between an Nvidia box and an Intel one. Every test starts on a computer with no card the engine
knows; a test about drafting names the card it means (test_drafting.py and test_drafting_hardware.py patch `detect_hardware` themselves)."""
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _no_card_unless_a_test_names_one(request):
    from strata_control import drafting
    if request.module.__name__.rsplit(".", 1)[-1].startswith("test_drafting"):
        yield                                    # these name their card themselves (setUpModule / per test)
        return
    with mock.patch.object(drafting, "detect_hardware", return_value=("cpu", [])):
        yield
