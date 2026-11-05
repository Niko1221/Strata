"""Qwen3.8-Flash-Next's complete text forward pass on Ascend CANN.

This is an independent backend inside Strata. It uses the original GGUF model,
Strata's tokenizer and scalar weight decoders, and torch_npu for NPU execution.
It does not call a llama.cpp inference server or a CUDA engine.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch_npu  # noqa: F401
from torch.nn import functional as F

from weights import DEFAULT_SOURCE, DeviceWeights, ROOT, WeightStore

sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402


class Qwen38:
    def __init__(self, store, device="npu:0", budget_gib=16, context=8192, weight_dtype="fp32"):
        self.store = store
        self.w = DeviceWeights(store, device, budget_gib, weight_dtype)
        self.weight_dtype = weight_dtype
        self.device = self.w.device
        md = store.metadata
        self.md = md
        self.n = md["qwen4exp.embedding_length"]
        self.layers = md["qwen4exp.block_count"]
        self.hc = md["qwen4exp.hyper_connection.count"]
        self.eps = md["qwen4exp.attention.layer_norm_rms_epsilon"]
        self.k_heads = md["qwen4exp.ssm.group_count"]
        self.v_heads = md["qwen4exp.ssm.time_step_rank"]
        self.ssm_dim = md["qwen4exp.ssm.state_size"]
        self.conv_k = md["qwen4exp.ssm.conv_kernel"]
        self.heads = md["qwen4exp.attention.head_count"]
        self.kv_heads = md["qwen4exp.attention.head_count_kv"]
        self.head_dim = md["qwen4exp.attention.key_length"]
        self.rot = md["qwen4exp.rope.dimension_count"]
        self.base = md["qwen4exp.rope.freq_base"]
        self.topk = md["qwen4exp.expert_used_count"]
        self.ple_layers = set(md["qwen4exp.ple.layers"])
        self.interval = md["qwen4exp.full_attention_interval"]
        self.idx_dim = md["qwen4exp.attention.indexer.key_length"]
        self.idx_heads = md["qwen4exp.attention.indexer.head_count"]
        self.idx_topk = md["qwen4exp.attention.indexer.top_k"]
        self.context = context
        if not 1 <= context <= md["qwen4exp.context_length"]:
            raise ValueError("context must be within the GGUF's trained context length")
        if self.n != 2560 or self.layers != 48 or self.hc != 4:
            raise ValueError("only the original 48-layer Flash-Next geometry is verified")
        self.freq = self.base ** (
            -torch.arange(self.rot // 2, device=self.device, dtype=torch.float32) * 2 / self.rot
        )
        self.tokenizer = Tokenizer.from_gguf(store.headers[0].path)
        self.reset()

    def reset(self):
        self.position = 0
        self.tokens = []
        self.gdn = {}
        self.attention = {}
        self.ple_history = {}

    def norm(self, x, gamma):
        return x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps) * gamma

    def rope(self, x, position):
        angles = self.freq * position
        c, s = angles.cos(), angles.sin()
        left, right = x[..., :self.rot // 2], x[..., self.rot // 2:self.rot]
        return torch.cat((left * c - right * s, left * s + right * c, x[..., self.rot:]), -1)

    def hc_mix(self, residual, prefix, inject=True):
        normalized = self.norm(residual, self.w.vector(prefix + "norm.weight").reshape(self.hc, self.n))
        flat = normalized.flatten()
        low = F.silu(self.w.linear(prefix + "down.weight", flat) / self.hc)
        gate = self.w.linear(prefix + "up.weight", low).reshape(self.hc, self.n)
        mixed = (normalized * gate.sigmoid()).mean(0)
        scatter = self.w.linear(prefix + "inject.weight", flat) if inject else None
        return mixed, scatter

    def hc_combine(self, residual, output, scatter):
        return residual + output[None, :] * (2 * torch.sigmoid(scatter / self.hc))[:, None]

    def ple_rows(self, token):
        eos = self.md["qwen4exp.ple.eos_token_id"]
        ctx = [token]
        cut = False
        for back in (1, 2):
            previous = self.tokens[-back] if len(self.tokens) >= back else -1
            cut = cut or previous < 0 or previous == eos
            ctx.append(eos if cut else previous)
        multipliers = self.md["qwen4exp.ple.layer_multipliers"]
        offsets = self.md["qwen4exp.ple.head_offsets"]
        vocabs = self.md["qwen4exp.ple.head_vocab_sizes"]
        rows = []
        for length in (2, 3):
            mixed = 0
            for i in range(length):
                mixed ^= (ctx[i] * multipliers[i]) & ((1 << 64) - 1)
            for head in range(8):
                h = (length - 2) * 8 + head
                rows.append(mixed % vocabs[h] + offsets[h])
        return rows

    def ple(self, residual, token, il):
        p = f"blk.{il}."
        rows = self.ple_rows(token)
        width = self.md["qwen4exp.embedding_length_per_layer_input"]
        futures = [
            self.store.pool.submit(self.store.decode, "per_layer_token_embd.weight", row * width, width)
            for row in rows
        ]
        emb = torch.cat([torch.from_numpy(job.result()) for job in futures]).to(self.device)
        key = self.w.linear(p + "ple_key.weight", emb).reshape(self.hc, self.n)
        value = self.w.linear(p + "ple_value.weight", emb)
        key = self.norm(key, self.w.vector(p + "ple_norm_key.weight").reshape(self.hc, self.n))
        query = self.norm(residual, self.w.vector(p + "ple_norm_query.weight").reshape(self.hc, self.n))
        score = (key * query).sum(-1) / math.sqrt(self.n)
        gate = torch.sigmoid(score.sign() * score.abs().clamp_min(1e-6).sqrt())
        gated = value[None, :] * gate[:, None]
        normalized = self.norm(gated, self.w.vector(p + "ple_norm_conv.weight").reshape(self.hc, self.n))
        history = self.ple_history.setdefault(il, [])
        kernel = self.w.vector(p + "ple_conv1d.weight").reshape(self.hc, self.n, -1)
        conv = normalized * kernel[..., -1]
        dilation = self.md["qwen4exp.ple.ngram_size"]
        for tap in range(kernel.shape[-1] - 1):
            back = (kernel.shape[-1] - 1 - tap) * dilation
            if len(history) >= back:
                conv = conv + history[-back] * kernel[..., tap]
        history.append(normalized)
        if len(history) > dilation * (kernel.shape[-1] - 1):
            del history[0]
        return residual + gated + F.silu(conv)

    def gdn_step(self, x, il):
        p = f"blk.{il}."
        qkv = self.w.linear(p + "attn_qkv.weight", x)
        z = self.w.linear(p + "attn_gate.weight", x).reshape(self.v_heads, self.ssm_dim)
        kernel = self.w.vector(p + "ssm_conv1d.weight")
        if il not in self.gdn:
            self.gdn[il] = (
                [],
                torch.zeros((self.v_heads, self.ssm_dim, self.ssm_dim),
                            device=self.device, dtype=torch.float32),
            )
        history, state = self.gdn[il]
        convolved = qkv * kernel[:, -1]
        for back in range(1, self.conv_k):
            if len(history) >= back:
                convolved = convolved + history[-back] * kernel[:, -1 - back]
        history.append(qkv)
        if len(history) >= self.conv_k:
            del history[0]
        convolved = F.silu(convolved)
        key_dim = self.k_heads * self.ssm_dim
        q = convolved[:key_dim].reshape(self.k_heads, self.ssm_dim)
        k = convolved[key_dim:2 * key_dim].reshape(self.k_heads, self.ssm_dim)
        v = convolved[2 * key_dim:].reshape(self.v_heads, self.ssm_dim)
        # llama.cpp's L2 norm uses sqrt(sum(x*x) + eps), without mean.
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + self.eps)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + self.eps)
        mapping = torch.arange(self.v_heads, device=self.device) % self.k_heads
        q, k = q[mapping], k[mapping]
        beta = self.w.linear(p + "ssm_beta.weight", x).sigmoid()
        alpha = self.w.linear(p + "ssm_alpha.weight", x) + self.w.vector(p + "ssm_dt.bias")
        decay = (F.softplus(alpha) * self.w.vector(p + "ssm_a")).exp()
        state.mul_(decay[:, None, None])
        prediction = torch.bmm(k[:, None, :], state).squeeze(1)
        delta = (v - prediction) * beta[:, None]
        state.add_(k[:, :, None] * delta[:, None, :])
        result = torch.bmm(q[:, None, :], state).squeeze(1) / math.sqrt(self.ssm_dim)
        # Flash-Next uses sigmoid here; Qwen3.6 uses SiLU. Keeping that
        # family-specific distinction is required even though both use GDN.
        result = self.norm(result, self.w.vector(p + "ssm_norm.weight")) * z.sigmoid()
        return self.w.linear(p + "ssm_out.weight", result.flatten())

    def attention_step(self, x, il):
        p = f"blk.{il}."
        qg = self.w.linear(p + "attn_q.weight", x).reshape(self.heads, 2, self.head_dim)
        q = self.rope(self.norm(qg[:, 0], self.w.vector(p + "attn_q_norm.weight")), self.position)
        k = self.w.linear(p + "attn_k.weight", x).reshape(self.kv_heads, self.head_dim)
        k = self.rope(self.norm(k, self.w.vector(p + "attn_k_norm.weight")), self.position)
        v = self.w.linear(p + "attn_v.weight", x).reshape(self.kv_heads, self.head_dim)
        idx = self.w.linear(p + "indexer.k_proj.weight", x)
        if il not in self.attention:
            self.attention[il] = (
                torch.empty((self.context, self.kv_heads, self.head_dim), device=self.device,
                            dtype=torch.float16),
                torch.empty((self.context, self.kv_heads, self.head_dim), device=self.device,
                            dtype=torch.float16),
                torch.zeros((self.context + 4, self.idx_dim), device=self.device,
                            dtype=torch.float16),
            )
        keys, values, index_keys = self.attention[il]
        keys[self.position], values[self.position], index_keys[self.position] = k, v, idx
        count = self.position + 1
        selected = None
        if count > self.idx_topk + 3:
            # Text positions are contiguous. Completed four-token blocks are
            # pooled before normalization/rotation; the current tail stays visible.
            ratio = self.md["qwen4exp.attention.compress_ratios"][il]
            full = count // ratio
            pooled = index_keys[:full * ratio].float().reshape(full, ratio, self.idx_dim).mean(1)
            if count % ratio:
                # Unpooled cells point at the spare block. Its four gather
                # indices are cell zero, and its RoPE position is zero.
                pooled = torch.cat((pooled, index_keys[:1].float()), 0)
            blocks = pooled.shape[0]
            pooled = self.norm(pooled, self.w.vector(p + "indexer.k_norm.weight"))
            block_positions = torch.arange(blocks, device=self.device) * ratio
            if count % ratio:
                block_positions[-1] = 0
            pooled = torch.stack([self.rope(pooled[b], block_positions[b]) for b in range(blocks)])
            iq = self.w.linear(p + "indexer.q_proj.weight", x).reshape(self.idx_heads, self.idx_dim)
            iq = self.rope(self.norm(iq, self.w.vector(p + "indexer.q_norm.weight")), self.position)
            score = torch.relu(iq @ pooled.T).sum(0)
            if count % ratio:
                score[-1] += 1e9
            expanded = score.repeat_interleave(ratio)[:count]
            selected = expanded.topk(min(count, self.idx_topk + ratio - 1)).indices
        k_all, v_all = keys[:count], values[:count]
        if selected is not None:
            k_all, v_all = k_all[selected], v_all[selected]
        mapping = torch.arange(self.heads, device=self.device) // (self.heads // self.kv_heads)
        k_all = k_all[:, mapping].permute(1, 0, 2).float()
        v_all = v_all[:, mapping].permute(1, 0, 2).float()
        score = torch.bmm(q[:, None, :], k_all.transpose(1, 2)).squeeze(1) / math.sqrt(self.head_dim)
        mixed = torch.bmm(score.softmax(-1)[:, None, :], v_all).squeeze(1)
        return self.w.linear(p + "attn_output.weight", (mixed * qg[:, 1].sigmoid()).flatten())

    def moe(self, x, il):
        p = f"blk.{il}."
        router = self.w.linear(p + "ffn_gate_inp.weight", x)
        scores, experts = router.softmax(-1).topk(self.topk)
        scores = scores / scores.sum()
        ids = experts.cpu().tolist()
        self.w.prefetch_experts(p, ids)
        result = torch.zeros_like(x)
        for i, expert in enumerate(ids):
            gate = self.w.linear(p + "ffn_gate_exps.weight", x, expert)
            up = self.w.linear(p + "ffn_up_exps.weight", x, expert)
            down = self.w.linear(p + "ffn_down_exps.weight", F.silu(gate) * up, expert)
            result = result + scores[i] * down
        shared_gate = self.w.linear(p + "ffn_gate_shexp.weight", x)
        shared_up = self.w.linear(p + "ffn_up_shexp.weight", x)
        shared = self.w.linear(p + "ffn_down_shexp.weight", F.silu(shared_gate) * shared_up)
        shared_score = self.w.linear(p + "ffn_gate_inp_shexp.weight", x).sigmoid()
        return result + shared * shared_score

    @torch.inference_mode()
    def step(self, token, *, logits=True, trace=None, cancelled=None):
        if self.position >= self.context:
            raise ValueError(f"context limit {self.context} reached")
        vocab = len(self.tokenizer.tokens)
        if not 0 <= token < vocab:
            raise ValueError("token id outside the vocabulary")
        emb = self.store.decode("token_embd.weight", token * self.n, self.n)
        residual = torch.from_numpy(emb).to(self.device).repeat(self.hc, 1)
        for il in range(self.layers):
            if cancelled is not None and cancelled():
                raise InterruptedError("inference cancelled")
            started = time.monotonic()
            if il in self.ple_layers:
                residual = self.ple(residual, token, il)
            x, scatter = self.hc_mix(residual, f"blk.{il}.hc_attn_")
            out = (self.attention_step(x, il) if il % self.interval == self.interval - 1
                   else self.gdn_step(x, il))
            residual = self.hc_combine(residual, out, scatter)
            x, scatter = self.hc_mix(residual, f"blk.{il}.hc_ffn_")
            residual = self.hc_combine(residual, self.moe(x, il), scatter)
            if trace is not None:
                torch.npu.synchronize()
                trace(il, residual, time.monotonic() - started)
        self.tokens.append(token)
        self.position += 1
        if not logits:
            return None
        x, _ = self.hc_mix(residual, "output_hc_", inject=False)
        return self.w.linear("output.weight", x)

    def generate(self, prompt, max_tokens=32, temperature=0, callback=None, trace=None,
                 logits_callback=None, cancelled=None, progress=None):
        tokens = self.tokenizer.encode(prompt, parse_special=True) if isinstance(prompt, str) else list(prompt)
        if not tokens or len(tokens) + max_tokens > self.context:
            raise ValueError("empty prompt or prompt plus generation exceeds context")
        self.reset()
        started = time.monotonic()
        distribution = None
        for index, token in enumerate(tokens):
            if cancelled is not None and cancelled():
                raise InterruptedError("inference cancelled")
            distribution = self.step(token, logits=index == len(tokens) - 1, trace=trace,
                                     cancelled=cancelled)
            if progress:
                torch.npu.synchronize()
                progress(index + 1, time.monotonic() - started)
            print(f"prefill {index + 1}/{len(tokens)} elapsed={time.monotonic() - started:.2f}s",
                  file=sys.stderr, flush=True)
        torch.npu.synchronize()
        prefill_seconds = time.monotonic() - started
        generated = []
        logprobs = []
        token_logprobs = []
        stop_ids = {self.md["tokenizer.ggml.eos_token_id"], self.md["qwen4exp.ple.eos_token_id"]}
        if "<|fim_suffix|>" in self.tokenizer.ids:
            stop_ids.add(self.tokenizer.ids["<|fim_suffix|>"])
        stopped = False
        decode_start = time.monotonic()
        for _ in range(max_tokens):
            if cancelled is not None and cancelled():
                raise InterruptedError("inference cancelled")
            if logits_callback is not None:
                logits_callback(distribution)
            lp = distribution.log_softmax(-1)
            values, indices = lp.topk(5)
            token = (int(distribution.argmax()) if temperature <= 0 else
                     int(torch.multinomial((distribution / temperature).softmax(-1), 1)))
            if token in stop_ids:
                stopped = True
                break
            generated.append(token)
            logprobs.append(list(zip(indices.cpu().tolist(), values.cpu().tolist())))
            token_logprobs.append(float(lp[token]))
            print(f"generated {len(generated)}/{max_tokens} id={token} "
                  f"elapsed={time.monotonic() - decode_start:.2f}s",
                  file=sys.stderr, flush=True)
            if callback:
                callback(token)
            if len(generated) < max_tokens:
                distribution = self.step(token, cancelled=cancelled)
        torch.npu.synchronize()
        decode_seconds = time.monotonic() - decode_start
        return {
            "text": self.tokenizer.decode(generated),
            "prompt_tokens": tokens,
            "tokens": generated,
            "top5_logprobs": logprobs,
            "token_logprobs": token_logprobs,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "decode_steps": max(0, len(generated) - 1),
            "decode_tokens_per_second": max(0, len(generated) - 1) / max(decode_seconds, 1e-9),
            "finish_reason": "stop" if stopped else "length",
            "weights": self.w.report(),
            "backend": "strata-cann-torch",
            "device": str(self.device),
            "weight_dtype": self.weight_dtype,
        }


def arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--decoder", type=Path, default=ROOT / "build-cann/libstrata_cann_decode.so")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--budget-gib", type=float, default=16)
    ap.add_argument("--weight-dtype", choices=("fp32", "fp16"), default="fp32")
    ap.add_argument("--disk-cache-gib", type=float, default=16)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--tokens", help="comma-separated prompt token ids")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--dump-layers", type=Path, help="optional float32 residual dumps for an oracle comparison")
    ap.add_argument("--dump-logits", type=Path, help="optional float32 logits at every generation step")
    return ap


def main():
    args = arguments().parse_args()
    torch.set_num_threads(8)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.matmul.allow_hf32 = False
    store = WeightStore(args.model_dir, args.cache_dir, args.decoder, args.source,
                        disk_cache_gib=args.disk_cache_gib)
    try:
        engine = Qwen38(store, args.device, args.budget_gib, args.context, args.weight_dtype)
        prompt = [int(t) for t in args.tokens.split(",")] if args.tokens else args.prompt
        if args.dump_layers:
            args.dump_layers.mkdir(parents=True, exist_ok=True)
        def trace(il, value, seconds):
            if args.dump_layers:
                np.save(args.dump_layers / f"position-{engine.position:05d}-layer-{il:02d}.npy",
                        value.cpu().numpy())
            print(json.dumps({"layer": il, "seconds": round(seconds, 3),
                              "first": value.flatten()[:6].cpu().tolist(),
                              "network_mb": store.stats["network_bytes"] / 1e6}),
                  file=sys.stderr, flush=True)
        full_logits = []
        result = engine.generate(
            prompt, args.max_tokens, trace=trace if args.trace or args.dump_layers else None,
            logits_callback=(lambda value: full_logits.append(value.cpu().numpy()))
            if args.dump_logits else None,
        )
        if args.dump_logits:
            np.stack(full_logits).astype(np.float32).tofile(args.dump_logits)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        print(text, flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
