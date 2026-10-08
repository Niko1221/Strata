"""Q4_0 fast-prefill parity: CPU fixture check, optional CUDA glm_pack_test executable."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

import glm_synth_gguf as synth
import iq_pack
from gguf_reader import GGUFFile


def fixture(root):
    source, pack = root / "model.gguf", root / "pack"
    metadata, tensors = synth.build_prefill()
    synth.write_gguf(source, metadata, tensors, q4_native=True)
    # Release the F32 source before making the pack's mapped views.
    del tensors
    subprocess.run([sys.executable, str(Path(iq_pack.__file__)), "--gguf", str(source),
                    "--out", str(pack), "--compat-bf16"], check=True)
    gguf = GGUFFile(source)
    experts = [t for t in gguf.tensors if "_exps.weight" in t.name]
    assert len(experts) == 15 and all(t.type_name == "Q4_0" for t in experts)
    assert gguf.metadata["glm5-next.expert_used_count"] == 8
    assert gguf.metadata["glm5-next.attention.kv_lora_rank"] == 512
    assert 2 * gguf.metadata["glm5-next.expert_feed_forward_length"][0] == \
        gguf.metadata["glm5-next.embedding_length"]
    lines = [line.split() for line in (pack / "native_experts.txt").read_text().splitlines()
             if line and not line.startswith("#")]
    assert len(lines) == 5 and all(int(line[1]) == 2 and int(line[2]) == 2 for line in lines)
    assert not (pack / "experts.bin").exists()  # actual GGUF disk reads
    print(f"fixture PASS: {len(gguf.tensors)} tensors, 15 Q4_0 expert tensors, 5 disk layers")
    return pack


def cuda_parity(executable, root, pack):
    # Separate processes: the runner caches environment settings in function statics.
    common = {k: v for k, v in os.environ.items()
              if not k.startswith(("STRATA_GLM_", "GLM_TEST_"))}
    common.update(STRATA_GLM_POOL_GB="0.06", STRATA_GLM_RAM_GB="0", STRATA_GLM_WARM="0",
                  STRATA_GLM_PREFILL_CHUNK="16", STRATA_GLM_PREFILL_MIN="1",
                  STRATA_GLM_PREFILL_LIGHT="0", STRATA_GLM_AHEAD_READ="0",
                  STRATA_GLM_PREFILL_PROF="1")

    def run(name, settings, rows):
        output = root / f"{name}.f32"
        result = subprocess.run([str(executable), str(pack), "69", str(output)],
                                env=dict(common, **settings), capture_output=True, text=True)
        assert result.returncode == 0, f"{name}: runner failed\n{result.stdout}\n{result.stderr}"
        logits = np.fromfile(output, dtype=np.float32)
        assert logits.size == rows * 512, f"{name}: missing logits"
        assert np.isfinite(logits).all(), f"{name}: nonfinite logits"
        return logits.reshape(rows, 512), result.stderr

    reference, _ = run("stream", {"STRATA_GLM_NO_PREFILL": "1"}, 69)
    for landing in (12, 64):
        for ahead in (0, 1):
            name = f"land{landing}-ahead{ahead}"
            logits, log = run(name, {"GLM_TEST_PREFILL": "65", "GLM_TEST_REQUIRE_PREFILL": "1",
                                    "STRATA_GLM_PREFILL_LAND": str(landing),
                                    "STRATA_GLM_PREFILL_PRED_T": str(ahead)}, 4)
            assert re.search(rf"disk landing ring {landing}\b", log), f"{name}: wrong landing ring\n{log}"
            counts = re.search(r"disk experts used (\d+), read (\d+)", log)
            assert counts and int(counts[1]) > landing and int(counts[2]) >= int(counts[1]), \
                f"{name}: disk landing ring was not exercised\n{log}"
            error = float(np.max(np.abs(logits - reference[65:])))
            assert error < 2e-3, f"{name}: max tail logit error {error}"
            print(f"{name} PASS: five chunks + four continuation rows, max error {error:.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, help="CUDA glm_pack_test binary; omitted: CPU fixture check")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="glm-prefill-", ignore_cleanup_errors=True) as temporary:
        root = Path(temporary)
        pack = fixture(root)
        if args.runner:
            cuda_parity(args.runner.resolve(), root, pack)


if __name__ == "__main__":
    main()
