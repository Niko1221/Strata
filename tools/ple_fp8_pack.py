#!/usr/bin/env python3
"""Kept so existing `tools/ple_fp8_pack.py` invocations keep working.

The tool now handles BF16 as well as F8_E4M3, so `ple_table_pack.py` is its real name. Renaming the file
alone would break every command line already written against the old one, and this tool is the only way to
get a non-IQ4_NL PLE table at all -- so the old name has to keep working. Nothing about the FP8 path
changes: the arguments, the output format and the metadata are the same.
"""
import pathlib
import runpy
import sys

sys.argv[0] = str(pathlib.Path(__file__).with_name("ple_table_pack.py"))
runpy.run_path(sys.argv[0], run_name="__main__")