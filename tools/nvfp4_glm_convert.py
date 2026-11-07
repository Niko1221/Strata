#!/usr/bin/env python3
"""tools/nvfp4_glm_convert.py - dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 (ModelOpt NVFP4) -> GGUF.

Uses /home/simon/work/llama.cpp (mainline, 2026-10-31), whose converter registers
Glm5NextForConditionalGeneration (conversion/qwen3vl.py) -> the "glm5-next" GGUF arch - the same
arch the served UD-Q2_K_XL pack carries (expert_count 288, 8 active).  The converter's generic
ModelOpt-NVFP4 layer (conversion/base.py) repacks the E2M1+E4M3 expert weights losslessly into
ggml block_nvfp4; dense tensors follow --outtype.

    python tools/nvfp4_glm_convert.py --model <dir> --outfile <gguf> [--outtype q8_0] [--dry-run]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

LLAMA = pathlib.Path("/home/simon/work/llama.cpp")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the ModelOpt NVFP4 checkpoint directory")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--outtype", default="q8_0")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(LLAMA / "gguf-py"))
    sys.path.insert(0, str(LLAMA))
    import gguf  # noqa: E402

    # 342B does not fit in 62 GB of RAM: the converter accumulates every tensor in GGUFWriter
    # (add_tensor keeps the data) and flushes only at the end - the process is OOM-killed at
    # ~layer 26 (exit 137, reproduced three times, dry-run and real alike).  Forcing the writer's
    # temp-file mode bounds memory: each tensor goes to a SpooledTemporaryFile (256 MB in RAM,
    # disk-backed beyond) and write_tensors_to_file streams it out.  Peak disk: one extra model
    # copy in $TMPDIR - point TMPDIR at a big filesystem for this run.
    _orig_gw_init = gguf.GGUFWriter.__init__

    def _gw_init(self, *args, **kwargs):
        kwargs["use_temp_file"] = True
        _orig_gw_init(self, *args, **kwargs)

    gguf.GGUFWriter.__init__ = _gw_init

    import convert_hf_to_gguf as C  # noqa: E402

    sys.argv = ["convert_hf_to_gguf.py", a.model, "--outtype", a.outtype, "--outfile", a.outfile] \
        + (["--dry-run"] if a.dry_run else [])
    t0 = time.time()
    rc = C.main()
    print("nvfp4_glm_convert: %.0f s, rc=%s" % (time.time() - t0, rc))
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
