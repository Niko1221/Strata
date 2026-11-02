"""训练线 MTP 的验收 probe：greedy accept rate 对齐引擎（0.89/0.86/0.85 @ steps 1-3）
+ decode 提速实测。与引擎 tools/mtp_probe.py 同一判据口径。

用法（prtrain examples/training，本机 Windows）：
  python mtp_probe_tr.py --gguf <shard1.gguf> <shard2.gguf> --pack experts_nf4.bin \
      --tokenizer <tok dir> --mtp-dir D:/.../Strata-data/mtp --new 96
"""

from __future__ import annotations

import argparse
import time

import torch

import platform_check

platform_check.require(platform_only=True)

from loader_qwen4 import load_model
from mtp_draft import MtpDrafter, load_mtp_state, spec_generate
from sft import text_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--mtp-dir", required=True)
    ap.add_argument("--prompt", default="The capital of France is Paris. The president of France is")
    ap.add_argument("--new", type=int, default=96)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mode", default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--pre-fc-norm", default="full", choices=["full", "stream"])
    args = ap.parse_args()

    torch.manual_seed(0)
    t0 = time.time()
    model, _ = load_model(args.gguf, args.pack)
    model.eval()
    model.config.use_cache = True
    print(f"model loaded {time.time() - t0:.0f}s", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    body, head = text_model(model), model.lm_head

    # 组装 drafter
    t0 = time.time()
    cfg = getattr(model.config, "text_config", model.config)
    drafter = MtpDrafter(cfg, pre_fc_norm_mode=args.pre_fc_norm).cuda().eval()
    state = load_mtp_state(args.mtp_dir)
    missing, unexpected = drafter.load_state_dict(state, strict=False)
    missing = [m for m in missing if "_embed" not in m and "_head" not in m and "_rotary" not in m]
    print(f"drafter loaded {time.time() - t0:.0f}s  missing={missing} unexpected={unexpected}", flush=True)
    assert not missing and not unexpected, "MTP 权重键不齐，先对键再谈速度"
    drafter.share_main(model)

    # ---- teacher-forced draft 检验：prompt 的 cell (R_i, x_{i+1}) @ pos i 应预测 x_{i+2}
    captured_tf: dict[str, torch.Tensor] = {}
    mixer_mod = next(m for n, m in body.named_modules() if n.endswith("hyper_connection_mixer"))
    h_tf = mixer_mod.register_forward_pre_hook(lambda m, a, kw: captured_tf.update(R=a[0]), with_kwargs=True)
    with torch.no_grad():
        out_tf = body(input_ids=ids, use_cache=False)
    h_tf.remove()
    R_all_tf = captured_tf["R"]
    T_tf = ids.shape[1]
    with torch.no_grad():
        from transformers.cache_utils import DynamicCache
        dcache_tf = DynamicCache(config=drafter.cache_config)
        _, mixed_tf = drafter.cell(R_all_tf[:, : T_tf - 2], ids[:, 1 : T_tf - 1],
                                   torch.arange(T_tf - 2, device=ids.device).view(1, 1, -1).expand(3, 1, -1).contiguous(),
                                   dcache_tf, T_tf - 2)
        lg_tf = drafter.logits_of(mixed_tf)[0]            # [T-2, V]
        pred = lg_tf.argmax(-1)
        truth = ids[0, 2:T_tf]
        acc = (pred == truth).float().mean().item()
    print(f"teacher-forced draft accuracy over {T_tf - 2} cells: {acc:.3f}", flush=True)
    if acc < 0.3:
        print("  每位对比 pred/truth:", [(int(p), int(t)) for p, t in zip(pred[:8], truth[:8])], flush=True)

    vram = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak VRAM after load: {vram:.1f} GiB", flush=True)

    # 无 MTP 基线：plain decode 速度
    with torch.no_grad():
        out = body(input_ids=ids, use_cache=True)
        past = out.past_key_values
        cur = head(out.last_hidden_state[:, -1:]).argmax(-1)
        for _ in range(4):  # 预热（triton JIT）
            out = body(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cur = head(out.last_hidden_state[:, -1:]).argmax(-1)
        torch.cuda.synchronize()
        t = time.time()
        n_base = 16
        for _ in range(n_base):
            out = body(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cur = head(out.last_hidden_state[:, -1:]).argmax(-1)
        torch.cuda.synchronize()
        base_ms = (time.time() - t) / n_base * 1000
    print(f"plain decode: {base_ms:.0f} ms/token  ({1000 / base_ms:.1f} tok/s)", flush=True)

    # MTP spec decode
    torch.cuda.reset_peak_memory_stats()
    for trial in range(2):
        r = spec_generate(body, head, drafter, ids, max_new=args.new, k=args.k,
                          mode=args.mode, temperature=args.temperature)
        text = tok.decode(r["tokens"])
        ms_tok = r["wall"] / len(r["tokens"]) * 1000
        print(f"[trial {trial}] mode={args.mode} k={args.k} "
              f"accept={r['accept_rate']:.2f} tok/fwd={r['tokens_per_fwd']:.2f} "
              f"target_forwards={r['target_forwards']}  {ms_tok:.0f} ms/token  "
              f"({1000 / ms_tok:.1f} tok/s, plain 的 {base_ms / ms_tok:.2f}×)", flush=True)
        print(f"  text: {text[:120]!r}", flush=True)
        print(f"  accept hist: {r['accept']}", flush=True)
    vram = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak VRAM with MTP: {vram:.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
