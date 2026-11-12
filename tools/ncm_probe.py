#!/usr/bin/env python3
"""Standalone TinyStories NCM probe; no Strata runtime imports or dependencies.

Run with python -I and an explicitly reviewed NCM checkout. Isolation mode avoids
implicit imports from the current directory; it does not sandbox upstream code
or the pickle checkpoints loaded by that code. See docs/NCM.md.
"""

import argparse
import contextlib
import datetime
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def break_even(window, baseline_tps, verify_ms, draft_ms):
    """IID conditional draft acceptance; accepted prefix, plus one target token.

    Returns equality threshold; beating the baseline requires a higher q.
    The actual lookup-chain baseline must be measured separately.
    """
    required = baseline_tps * (verify_ms + draft_ms) / 1000
    if required >= window:
        threshold = None
    elif required <= 1:
        threshold = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            expected = sum(mid ** i for i in range(window))
            if expected < required:
                lo = mid
            else:
                hi = mid
        threshold = hi
    return {"window": window, "draft_tokens": window - 1,
            "baseline_tps": baseline_tps, "verify_ms": verify_ms,
            "draft_ms": draft_ms, "required_committed_tokens": required,
            "acceptance_equality_threshold": threshold,
            "can_beat_baseline": required < window,
            "assumption": "constant conditional acceptance q; E[tokens]=sum(q**i, i=0..window-1)"}


def probe(args):
    root, ckpt = args.repo.resolve(), args.ckpt.resolve()
    for relative in ("models/next_concept.py", "tokenization/vqvae_tokenizer.py"):
        if not (root / relative).is_file():
            raise ValueError(f"missing NCM source: {root / relative}")
    for name in ("config.yaml", "generator.pt", "tokenizer.pt",
                 "bpe_vocab_8192.pickle", "bpe_merges_8192.pickle"):
        if not (ckpt / name).is_file():
            raise ValueError(f"missing checkpoint file: {ckpt / name}")

    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ckpt / "config.yaml")
    if cfg.task != "tinystories":
        raise ValueError("this probe implements the released TinyStories evaluation path only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe; no CPU fallback measurements")
    # Explicitly import the reviewed checkout, even under Python isolation mode.
    sys.path.insert(0, str(root))
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cfg.tokenizer.vqvae.checkpoint_path = str(ckpt / cfg.checkpoints.tokenizer)
    cfg.tokenizer.pre_tokenizer.vocab_file_path = str(ckpt / cfg.checkpoints.bpe_vocab)
    cfg.tokenizer.pre_tokenizer.merges_file_path = str(ckpt / cfg.checkpoints.bpe_merges)
    # Upstream constructors print parameter counts. Keep stdout valid JSON.
    with contextlib.redirect_stdout(sys.stderr):
        tokenizer = VQVAETokenizer.from_config(cfg.tokenizer)
        tokenizer.vqvae = tokenizer.vqvae.cuda().eval()
        net = NextConceptTransformer(vqvae_local=tokenizer.vqvae, config=cfg.model).cuda().eval()
    state = torch.load(ckpt / cfg.checkpoints.generator, map_location="cpu", weights_only=False)
    net.load_state_dict(state["model"], strict=True)
    l0_prob = state["inference"]["level0_unigram"].float().cpu()
    del state
    eos, pad = tokenizer.pre_tokenizer.eos_token_id, tokenizer.pre_tokenizer.pad_token_id
    prefix = torch.full((args.batch_size, int(cfg.model.prefix_len)), eos, device="cuda")

    def generate(decode):
        # Match upstream: sample the first level on CPU, then transfer to GPU.
        first = torch.multinomial(l0_prob, args.batch_size * net.levels[0], replacement=True)
        first = first.view(args.batch_size, net.levels[0]).cuda()
        codes = net.generate(first, prefix, temperature=float(cfg.generation.temperature),
                             top_k=int(cfg.generation.top_k), top_p=float(cfg.generation.top_p),
                             cfg_scale=float(cfg.generation.cfg_scale), cfg_null_eos=eos)
        if not decode:
            return codes
        _, levels = tokenizer.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
        ids = tokenizer.vqvae.decoder(levels[-1]).argmax(-1).tolist()
        visible = 0
        for row in ids:
            if eos in row:
                row = row[:row.index(eos)]
            row = [token for token in row if token not in (eos, pad)]
            tokenizer.pre_tokenizer.decode(row, skip_special_tokens=True)
            visible += len(row)
        return len(ids) * len(ids[0]), visible

    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000, result

    with torch.inference_mode():
        for _ in range(args.warmup):
            generate(True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        generation_ms, full_ms, raw_count, visible_count = [], [], 0, 0
        for _ in range(args.iterations):
            elapsed, _ = timed(lambda: generate(False))
            generation_ms.append(elapsed)
            elapsed, (raw, visible) = timed(lambda: generate(True))
            full_ms.append(elapsed)
            raw_count += raw
            visible_count += visible
        # Profile a separate generation, outside the throughput timings. One
        # call covers the full transformer stack and output head for a level.
        forwards = []
        original = net._decode_level_logits

        def profile_level(*positional, **keywords):
            elapsed, logits = timed(lambda: original(*positional, **keywords))
            forwards.append({"code_positions_per_sample": int(logits.shape[1]), "ms": elapsed})
            return logits

        net._decode_level_logits = profile_level
        try:
            generate(False)
        finally:
            net._decode_level_logits = original

    parameters = {id(p): p for model in (net, tokenizer.vqvae) for p in model.parameters()}
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    return {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
        "ncm_revision": revision, "task": cfg.task, "precision": "float32; no autocast",
        "seed": args.seed, "cpu_threads": args.threads, "batch_size": args.batch_size,
        "warmup": args.warmup, "iterations": args.iterations,
        "generator_parameters": sum(p.numel() for p in net.parameters()),
        "tokenizer_parameters": sum(p.numel() for p in tokenizer.vqvae.parameters()),
        "combined_unique_parameters": sum(p.numel() for p in parameters.values()),
        "text_vocab_size": int(cfg.tokenizer.vqvae.in_vocab_size),
        "latent_vocab_sizes": list(cfg.tokenizer.vqvae.vqvae_vocab_size),
        "levels": list(net.levels), "profiled_level_forwards": forwards,
        "raw_text_positions_per_sample": int(cfg.tokenizer.vqvae.context_length),
        "latent_generation_ms": generation_ms, "full_generation_ms": full_ms,
        "full_generation_median_ms": statistics.median(full_ms),
        "raw_text_positions": raw_count, "visible_text_tokens": visible_count,
        "raw_text_positions_per_second": raw_count / (sum(full_ms) / 1000),
        "visible_text_tokens_per_second": visible_count / (sum(full_ms) / 1000),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "limits": "TinyStories BPE tokens, not Qwen tokens; no acceptance or quality measurement",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="explicit reviewed NCM checkout")
    parser.add_argument("--ckpt", type=Path, help="TinyStories checkpoint directory")
    parser.add_argument("--iterations", type=positive_int, default=20)
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="also save JSON here")
    parser.add_argument("--break-even", action="store_true", help="calculate only; no torch needed")
    parser.add_argument("--window", type=positive_int, choices=range(2, 17))
    parser.add_argument("--baseline-tps", type=positive)
    parser.add_argument("--verify-ms", type=positive)
    parser.add_argument("--draft-ms", type=positive)
    args = parser.parse_args()
    required = ("window", "baseline_tps", "verify_ms", "draft_ms") if args.break_even else ("repo", "ckpt")
    if any(getattr(args, key) is None for key in required):
        parser.error("required: " + ", ".join("--" + key.replace("_", "-") for key in required))
    try:
        result = (break_even(args.window, args.baseline_tps, args.verify_ms, args.draft_ms)
                  if args.break_even else probe(args))
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(1, f"ncm_probe: {exc}\n")
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
