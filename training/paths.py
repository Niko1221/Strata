"""Where the training assets live. Resolved from ``__file__``, never from cwd.

Layout - two sibling top-level trees split by responsibility::

    training/           implementation, tests, bench probes, docs, evaluated evidence
    examples/training/  runnable sample inputs, demo transcript, sample outputs

Upstream Strata has no ``examples/`` tree; this PR introduces it so that sample
data stops being mixed into the implementation directory. Everything else
follows upstream's split-by-responsibility convention: tests sit next to the
code (``training/tests/``, cf. ``serve/test_*.py``), one-off probes live under
``training/bench/`` (cf. upstream ``tools/`` and ``bench/``), and docs follow
the component they describe (cf. ``packaging/README.md``).

Why ``__file__``: ``train.py`` imports these modules in-process and never
chdirs, so a bare relative path depends on the caller's working directory and
fails far away from the actual error. Stdlib-only - the no-torch CI job
(``test_train_core.py``) imports this module.
"""
from __future__ import annotations

from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
EXAMPLES_DIR = REPO_ROOT / "examples" / "training"


def example(name: str) -> str:
    """Absolute path of ``name`` inside ``examples/training/``."""
    return str(EXAMPLES_DIR / name)


def find(name_or_path: str, *extra_dirs: str | Path) -> str:
    """Resolve an asset with one shared search order.

    ``name_or_path`` (absolute, cwd-relative or bare file name) > ``extra_dirs``
    > ``training/`` > ``examples/training/``. Raises ``FileNotFoundError``
    listing every candidate tried, so a missing asset is diagnosable instead of
    surfacing as a confusing error deep inside a loader.
    """
    p = Path(name_or_path)
    if p.is_absolute():
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"文件不存在：{p}")
    candidates = [Path.cwd() / p,
                  *(Path(d) / name_or_path for d in extra_dirs),
                  TRAINING_DIR / name_or_path,
                  EXAMPLES_DIR / name_or_path]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        f"{name_or_path} 不存在（试过 {'、'.join(str(c) for c in candidates)}）")
