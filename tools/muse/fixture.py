"""Generate a synthetic F32 Muse Spark GGUF and full forced-token logits.

Run from the repository root: python -m tools.muse.fixture --output-dir PATH
No downloads, tokenizer, native runtime, or accelerator are required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.gguf_writer import GGUFWriter
from tools.muse.reference import forward, load_f32

SEED = 2762
TOKENS = [1, 5, 8, 3, 7, 2, 9]
ARCH = "muse-spark"


def metadata() -> dict:
    """Literal key spellings from the published GGUF header; no HF mask flags."""
    values = {
        "block_count": 2, "context_length": 32, "embedding_length": 64,
        "feed_forward_length": 32, "attention.head_count": 4,
        "attention.head_count_kv": 2, "attention.key_length": 16,
        "attention.value_length": 16, "rope.freq_base": 500000.0,
        "rope.freq_base_swa": 10000.0,
        "attention.layer_norm_rms_epsilon": 1e-5,
        "post_norm_epsilon": 1e-8, "expert_count": 4, "expert_used_count": 2,
        "attention.sliding_window": 3,
        "attention.sliding_window_pattern": [True, False],
        "moe_latent_size": 16, "expert_feed_forward_length": 32,
        "expert_weights_scale": 1.3, "expert_weights_norm": True,
        "expert_gating_func": 2, "logit_scale": 0.7,
        "final_logit_softcapping": 2.5,
    }
    return {
        "general.architecture": ARCH, "general.type": "model",
        "general.name": "strata synthetic muse oracle seed 2762",
        "general.file_type": 0, "general.alignment": 32,
        **{f"{ARCH}.{key}": value for key, value in values.items()},
        "tokenizer.ggml.model": "no_vocab",
    }


def tensor_shapes() -> dict[str, tuple[int, ...]]:
    """NumPy row-major shapes, the reverse of contiguous-axis-first GGUF."""
    shapes = {
        "token_embd.weight": (64, 64), "output_norm.weight": (64,),
        "output.weight": (64, 64),
    }
    layer = {
        "attn_norm.weight": (64,), "attn_q.weight": (64, 64),
        "attn_k.weight": (32, 64), "attn_v.weight": (32, 64),
        "attn_q_norm.weight": (16,), "attn_k_norm.weight": (16,),
        "attn_gate.weight": (64, 64), "attn_output.weight": (64, 64),
        "attn_resid_alpha.weight": (64,), "attn_resid_beta.weight": (64,),
        "ffn_norm.weight": (64,), "post_ffw_norm.weight": (64,),
        "ffn_resid_alpha.weight": (64,), "ffn_resid_beta.weight": (64,),
        "ffn_gate_inp.weight": (4, 64), "exp_probs_b.bias": (4,),
        "ffn_latent_down.weight": (16, 64), "ffn_latent_up.weight": (64, 16),
        "ffn_pre_exp_norm.weight": (16,), "ffn_post_exp_norm.weight": (16,),
        "ffn_gate_exps.weight": (4, 32, 16),
        "ffn_up_exps.weight": (4, 32, 16),
        "ffn_down_exps.weight": (4, 16, 32),
    }
    for i in range(2):
        shapes.update({f"blk.{i}.{name}": shape for name, shape in layer.items()})
    return shapes


def weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    result = {}
    for name, shape in tensor_shapes().items():
        if name.endswith("exp_probs_b.bias"):
            # Changes top-2 selection without being used in the mixture weights.
            bias = [0.16, -0.105, 0.065, -0.045] if name.startswith("blk.0.") else [-0.04, 0.035, -0.1, 0.12]
            value = np.array(bias, dtype=np.float32)
        elif "resid_" in name:
            # Already-folded, physically consistent alpha=sqrt(1-beta**2).
            beta = rng.uniform(0.22, 0.78, shape).astype(np.float32)
            value = np.sqrt(1 - beta * beta) if "alpha" in name else beta
        elif len(shape) == 1:
            # Distinct nonuniform, already-centered/folded norm weights.
            value = rng.uniform(0.45, 1.65, shape).astype(np.float32)
            if "attn_q_norm" in name:
                value *= np.float32(1.4)  # Q metap scale is already folded.
        else:
            scale = 0.8 / np.sqrt(shape[-1])
            value = (rng.standard_normal(shape) * scale).astype(np.float32)
            if "ffn_gate_inp" in name:
                value *= np.float32(0.7)  # Router output multiplier already folded.
            if "ffn_down_exps" in name:
                # Different expert magnitudes expose postnorm-after-sum bugs.
                value *= np.array([0.3, 0.8, 1.7, 2.6], dtype=np.float32)[:, None, None]
            if "ffn_latent_up" in name:
                # Put the branch near sqrt(post_eps). Otherwise RMS almost
                # cancels a missing common router scale/normalization factor.
                value *= np.float32(1e-4)
        result[name] = np.asarray(value, dtype=np.float32)
    # Pair alpha/beta from the same folded residual gate, independently per branch.
    for i in range(2):
        for branch in ("attn", "ffn"):
            p = f"blk.{i}.{branch}_resid_"
            result[p + "alpha.weight"] = np.sqrt(1 - result[p + "beta.weight"] ** 2)
    return result


def write_model(path: Path, meta: dict, tensors: dict[str, np.ndarray]) -> None:
    writer = GGUFWriter()
    for key, value in meta.items():
        kind = {str: "string", bool: "bool", int: "u32", float: "f32", list: "array:bool"}[type(value)]
        writer.add(key, value, kind)
    for name, value in tensors.items():
        # add_f32 writes C-order bytes but keeps the passed shape unchanged.
        # Reverse the directory dimensions, NOT the data: dim 0 varies fastest.
        writer.add_f32(name, value)
        writer.tensors[-1].shape = list(reversed(value.shape))
    writer.write(path)


def generate(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.gguf"
    write_model(model_path, metadata(), weights())
    # Reference uses actual serialized f32 values, including rounded metadata.
    meta, tensors = load_f32(model_path)
    logits = forward(meta, tensors, TOKENS)
    reference = {
        "tokens": TOKENS, "logits": logits.tolist(), "generated_tokens": [],
        "provenance": {"oracle": "independent NumPy FP32", "seed": SEED,
                       "shape": list(logits.shape), "scales": "already converter-folded"},
    }
    (output_dir / "reference.json").write_text(
        json.dumps(reference, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    generate(args.output_dir)
    print(f"wrote {args.output_dir / 'model.gguf'} and reference.json (7 x 64 logits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
