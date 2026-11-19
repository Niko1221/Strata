"""Run small CPU contracts and seven targeted mutations without training/install.

Supplied Linux evidence is never overwritten. CUDA visibility is disabled for
these child processes explicitly, since this run qualifies only CPU reference
behaviour and the installed cu121 wheel is not qualified for the PC's sm_120.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MUTANTS = [
    (
        "wrong_half_angle",
        "math.cos(theta/2)",
        "math.cos(theta)",
        "test_same_lift_intertwiner_and_sign",
    ),
    (
        "normalisation_removed",
        "co/torch.sqrt(co.square().sum(-1,keepdim=True))",
        "co",
        "test_query_chart_identity_norm_bound_grad",
    ),
    (
        "wrong_inverse",
        "dequantize(k)@self.views[r]",
        "dequantize(k)@self.views[r].T",
        "test_exact_key_roundtrip_and_noncontiguous",
    ),
    (
        "wrong_bit_order",
        "1<<torch.arange(bits,device=p.device)",
        "1<<torch.arange(bits-1,-1,-1,device=p.device)",
        "test_pack_roundtrip",
    ),
    (
        "beta_ignored",
        "scores+cache.beta*correction",
        "scores+correction",
        "test_qjl_zero_and_formula",
    ),
    (
        "scale_twice",
        "scores/math.sqrt(q.shape[-1])",
        "scores/q.shape[-1]",
        "test_exact_views_preserve_scores",
    ),
    (
        "clifford_sign",
        "torch.cat((l.T,z),1)",
        "torch.cat((-l.T,z),1)",
        "test_clifford_and_chirality",
    ),
]


def run(command: list[str], cwd: Path, log: Path) -> subprocess.CompletedProcess:
    """Capture exact command, return status and both streams in a fresh file."""
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    process = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )
    with log.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"command": command, "cwd": str(cwd), "returncode": process.returncode}
            )
            + "\n"
        )
        stream.write(process.stdout + process.stderr)
    return process


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--math-sdpa-check",
        action="store_true",
        help="Also test the unchanged reference suite under explicit CPU MATH SDPA",
    )
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = [
        (
            "reference_pytest",
            [
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--confcutdir=.",
                "tests/test_contracts.py",
                "tests/test_tiny_lm.py",
            ],
        ),
        (
            "dense_diagnostics",
            [
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--confcutdir=.",
                "tests/test_dense_diagnostics.py",
                "tests/test_intake.py",
            ],
        ),
        (
            "validation",
            [
                "-m",
                "triality_poc.validate",
                "--device",
                "cpu",
                "--out",
                str(output / "validation.json"),
            ],
        ),
        (
            "environment",
            [
                "-c",
                "import json,torch,numpy,pytest,sys,platform; "
                "print(json.dumps(dict(python=sys.version,"
                "executable=sys.executable,platform=platform.platform(),"
                "torch=torch.__version__,numpy=numpy.__version__,"
                "pytest=pytest.__version__,cuda_visible=False,"
                "cuda_build=torch.version.cuda)))",
            ],
        ),
    ]
    if args.math_sdpa_check:
        commands.append(
            (
                "reference_pytest_math_sdpa",
                [
                    "-c",
                    "import torch,pytest\n"
                    "from torch.nn.attention import sdpa_kernel,SDPBackend\n"
                    "torch.set_num_threads(1)\n"
                    "with sdpa_kernel(SDPBackend.MATH):\n"
                    "    result=pytest.main(['-q','-p','no:cacheprovider',"
                    "'--confcutdir=.','tests/test_contracts.py',"
                    "'tests/test_tiny_lm.py'])\n"
                    "raise SystemExit(result)",
                ],
            )
        )
    checks = {}
    for name, tail in commands:
        process = run([sys.executable, *tail], ROOT, output / f"{name}.txt")
        checks[name] = process.returncode == 0
    source = (ROOT / "triality_poc/core.py").read_text(encoding="utf-8")
    scratch = ROOT / ".scratch"
    scratch.mkdir(exist_ok=True)
    mutations = []
    for name, old, new, test in MUTANTS:
        if old not in source:
            raise RuntimeError(f"Mutation anchor absent: {name}")
        with tempfile.TemporaryDirectory(prefix="mutant_", dir=scratch) as directory:
            work = Path(directory)
            shutil.copytree(
                ROOT / "triality_poc",
                work / "triality_poc",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            shutil.copytree(
                ROOT / "tests",
                work / "tests",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            (work / "triality_poc/core.py").write_text(
                source.replace(old, new, 1), encoding="utf-8"
            )
            process = run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--confcutdir=.",
                    f"tests/test_contracts.py::{test}",
                ],
                work,
                output / f"mutation_{name}.txt",
            )
            mutations.append(
                {
                    "name": name,
                    "returncode": process.returncode,
                    "detected": process.returncode == 1 and "failed" in process.stdout,
                }
            )
    checks["seven_targeted_mutations"] = all(row["detected"] for row in mutations)
    result = {
        "scope": (
            "fresh Windows CPU reference verification; "
            "no training or real model inference"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "interpreter": sys.executable,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "cpu_only": True,
        "cuda_visibility": "CUDA_VISIBLE_DEVICES=-1 in child processes",
        "supplied_checkpoints_loaded": False,
        "checks": checks,
        "mutations": mutations,
        "passed": all(checks.values()),
    }
    (output / "reproduction.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
