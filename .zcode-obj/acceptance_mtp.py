# Phase-2 P1 validation: the acceptance suite with the MTP drafter on. Same suite, same rig, the
# engine arguments gain --mtp <rt> - a grammar request now drafts (the mask chain) instead of being
# refused. Run: python .zcode-obj/acceptance_mtp.py <stamp>
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import grammar_acceptance as ga  # noqa: E402

_base_args = ga.engine_args(16384)


def engine_args(max_context):
    args = _base_args[:]
    args[args.index("--max-context")] = "--max-context"
    args[args.index("--max-context") + 1] = str(max_context)
    return args + ["--mtp", "F:/Strata-data/mtp/rt"]


ga.engine_args = engine_args

if __name__ == "__main__":
    sys.argv[0] = str(ROOT / "tools" / "grammar_acceptance.py")
    sys.exit(ga.main())
