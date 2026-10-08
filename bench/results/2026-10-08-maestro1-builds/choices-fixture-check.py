"""Run choices fixtures with the AVX2 desktop host they assume.

CPU detection and older-CPU behavior are tested separately without this override.
"""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from tools import test_setup_choices
print('Choices fixtures only: simulate an AVX2 desktop CPU; no GPU execution.', flush=True)
with patch.object(test_setup_choices.setup, 'cpu_info', return_value=('simulated AVX2 desktop', True, False)):
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(test_setup_choices))
sys.exit(not result.wasSuccessful())
