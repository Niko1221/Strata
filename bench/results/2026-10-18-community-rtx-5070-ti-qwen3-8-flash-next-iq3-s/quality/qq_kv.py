#!/usr/bin/env python3
"""tools/quant_quality.py with another K/V type than its fixed `--kv fp16` (the int8 run of this report):

    STRATA_QQ_KV=int8 python qq_kv.py run --config ... --out ... [the quant_quality.py run options]

Run from the Strata folder. Everything else is quant_quality.py unchanged.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import quant_quality as QQ  # noqa: E402

QQ.FIXED[QQ.FIXED.index("--kv") + 1] = os.environ["STRATA_QQ_KV"]
sys.exit(QQ.main())
