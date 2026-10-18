"""Guard: the strata_llamacpp FetchContent_Populate() call must pin policy CMP0169 to OLD.

CMake 3.30 deprecated calling FetchContent_Populate(<name>) with the name of a declared
dependency (policy CMP0169); CMake 4.4 warns on every configure while the policy is unset.
Strata keeps the populate-then-add_subdirectory form on purpose - it adds only the checkout's
ggml/ subdirectory into ${CMAKE_BINARY_DIR}/ggml with EXCLUDE_FROM_ALL, the same binary dir the
STRATA_GGML_DIR branch uses - so FetchContent_MakeAvailable() is not a drop-in on the 3.24 floor
(it moves the binary dir and only honours EXCLUDE_FROM_ALL in FetchContent_Declare from 3.28).
The supported compatibility path is to pin CMP0169 OLD, guarded so CMake < 3.30 (no such policy)
still configures.  This test fails if a FetchContent_Populate(strata_llamacpp) call is added or
edited without that pin.  No compiler, no GPU, no build.

    python -m unittest tools.test_fetchcontent_policy
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CMAKE_FILES = [ROOT / "CMakeLists.txt", ROOT / "sycl" / "CMakeLists.txt"]

# a cmake_policy(SET CMP0169 OLD) inside an if(POLICY CMP0169) guard, anywhere before the call
POLICY_PIN = re.compile(
    r"if\s*\(\s*POLICY\s+CMP0169\s*\)\s*"
    r"(?:#[^\n]*\n\s*)*"
    r"cmake_policy\s*\(\s*SET\s+CMP0169\s+OLD\s*\)",
    re.IGNORECASE,
)
POPULATE_CALL = re.compile(r"FetchContent_Populate\s*\(\s*strata_llamacpp\s*\)")


class FetchContentPolicy(unittest.TestCase):
    def test_files_exist(self):
        for path in CMAKE_FILES:
            self.assertTrue(path.is_file(), f"missing {path.relative_to(ROOT)}")

    def test_every_populate_call_is_pinned_old(self):
        for path in CMAKE_FILES:
            text = path.read_text(encoding="utf-8")
            rel = path.relative_to(ROOT)
            for match in POPULATE_CALL.finditer(text):
                before = text[: match.start()]
                self.assertIsNotNone(
                    POLICY_PIN.search(before),
                    f"{rel}: FetchContent_Populate(strata_llamacpp) at offset {match.start()} "
                    "has no guarded 'if(POLICY CMP0169) cmake_policy(SET CMP0169 OLD)' before it - "
                    "CMake 4.4 warns on every configure (CMP0169 deprecation).",
                )

    def test_the_pin_is_guarded_not_bare(self):
        # a bare cmake_policy(SET CMP0169 OLD) breaks CMake < 3.30 (unknown policy name); each pin must sit
        # inside the if(POLICY CMP0169) block opened just before it, with no endif between the two.
        for path in CMAKE_FILES:
            text = path.read_text(encoding="utf-8")
            rel = path.relative_to(ROOT)
            for match in re.finditer(r"cmake_policy\s*\(\s*SET\s+CMP0169\s+OLD\s*\)", text, re.IGNORECASE):
                before = text[: match.start()]
                opened = list(re.finditer(r"if\s*\(\s*POLICY\s+CMP0169\s*\)", before, re.IGNORECASE))
                self.assertTrue(opened, f"{rel}: cmake_policy(SET CMP0169 OLD) at offset {match.start()} "
                                        "has no if(POLICY CMP0169) before it.")
                gap = before[opened[-1].end():]
                self.assertNotIn("endif", gap.lower(),
                                 f"{rel}: cmake_policy(SET CMP0169 OLD) at offset {match.start()} is not inside "
                                 "the if(POLICY CMP0169) block opened just before it (an endif closes it first).")


if __name__ == "__main__":
    unittest.main()
