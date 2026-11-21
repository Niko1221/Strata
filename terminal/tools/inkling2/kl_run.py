"""Inkling-2-Mini numerics probe: logits of one long text under a given weight dtype / expert activation mode.

usage: kl_run.py --model DIR --dtype float32|bfloat16 --mode a16|w4a4 --text FILE --out FILE.pt [--max-tokens N]

mode a16  : the transformers PR reference as is (experts dequantized, activations at the model dtype)
mode w4a4 : expert GEMM inputs quantized as vLLM does on Hopper/Blackwell (vllm PR #69028, ops/nvfp4.py,
            ops/mxfp8.py, ops/activation.py swiglu_quantize): NVFP4 per-row amax/1536 + per-16 e4m3 block
            scale chosen by the 6-vs-4 "fouroversix_mae" rule (layers 0-19), MXFP8 per-32 ue8m0 (layers 20-23).
            The SwiGLU output is multiplied by the routing weight BEFORE its quantization, as swiglu_quantize does.
"""
import argparse, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration
from transformers.models.inkling2_loop import modeling_inkling2_loop as M

FP32_MAX = 3.4028234663852886e38
MXFP8_FIRST_LAYER = 20  # hf_quant_config.json: layers 20-23 experts are MXFP8, 0-19 NVFP4


def e2m1_round(x):
    """fp32 -> nearest e2m1 value, ties as cvt.rn.satfinite (vllm ops/nvfp4.py _e2m1 thresholds)."""
    a = x.abs()
    v = torch.where(a <= 0.25, 0.0, torch.where(a < 0.75, 0.5, torch.where(a <= 1.25, 1.0, torch.where(
        a < 1.75, 1.5, torch.where(a <= 2.5, 2.0, torch.where(a < 3.5, 3.0, torch.where(a <= 5.0, 4.0, 6.0)))))))
    return torch.where(torch.signbit(x), -v, v)


def nvfp4_qdq(x):
    """Quantize-dequantize rows of x [M, K] (fp32 math) as vllm nvfp4_quantize; returns fp32."""
    x = x.float()
    m, k = x.shape
    amax = x.abs().amax(1, keepdim=True)
    enc = torch.clamp(1536.0 / amax, max=FP32_MAX)
    dec = 1.0 / enc
    outer = amax * (1.0 / 1536.0)
    xb = x.view(m, k // 16, 16)
    bmax = xb.abs().amax(-1, keepdim=True)
    best, best_err = None, None
    for div in (6.0, 4.0):  # order matters: 4 wins only if strictly better
        s = torch.clamp(bmax * (enc.unsqueeze(-1) * (1.0 / div if div == 6.0 else 0.25)), max=448.0)
        s = s.to(torch.float8_e4m3fn).float()
        inv = torch.clamp(1.0 / (s * dec.unsqueeze(-1)), max=FP32_MAX)
        r = e2m1_round(xb * inv)
        rec = (r * s) * outer.unsqueeze(-1)
        err = (rec - xb).abs().sum(-1, keepdim=True)
        if best is None:
            best, best_err = rec, err
        else:
            best = torch.where(err < best_err, rec, best)
    return best.view(m, k)


def mxfp8_qdq(x):
    x = x.float()
    m, n = x.shape
    g = x.view(m, n // 32, 32)
    raw = g.abs().amax(-1, keepdim=True) / 448.0
    e = ((raw.view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    e = torch.where(raw <= 2.0 ** -127, torch.zeros_like(e), torch.clamp(e, max=254))
    scale = torch.clamp((e << 23).view(torch.float32), min=2.0 ** -127)
    inv = torch.where(e == 0, torch.full_like(scale, 2.0 ** 127), 1.0 / scale)
    v = torch.clamp(g * inv, -448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (v * scale).view(m, n)


def w4a4_forward(self, hidden_states, top_k_index, top_k_weights):
    qdq = mxfp8_qdq if self.layer_idx >= MXFP8_FIRST_LAYER else nvfp4_qdq
    out = torch.zeros_like(hidden_states, dtype=torch.float32)
    for e in torch.unique(top_k_index).tolist():
        if e == self.num_experts:
            continue
        pos, tok = torch.where((top_k_index == e).T)
        x = qdq(hidden_states[tok])
        gu = F.linear(x, self.gate_up_proj[e].float())
        gate, up = gu.chunk(2, dim=-1)
        act = (torch.tanh(gate / self.act_fn.gate_softcap.softcap) * self.act_fn.gate_softcap.softcap
               * torch.sigmoid(gate) * torch.tanh(up / self.act_fn.up_softcap.softcap) * self.act_fn.up_softcap.softcap)
        act = act * top_k_weights[tok, pos, None].float()  # gamma before quantization (swiglu_quantize)
        y = F.linear(qdq(act), self.down_proj[e].float())
        out.index_add_(0, tok, y)
    return out.to(hidden_states.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the Inkling-2-Mini checkpoint directory")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--mode", default="a16")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--pos-offset", type=int, default=0, help="position_ids = arange + N (only LogScale reads them)")
    ap.add_argument("--ids-out", help="also write the token ids, comma-separated, for strata-ink2ref --tokens-file")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Inkling2LoopForConditionalGeneration.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="cpu").eval()
    for i, layer in enumerate(model.model.language_model.layers):
        layer.mlp.experts.layer_idx = i
        if a.mode == "w4a4":
            layer.mlp.experts.forward = w4a4_forward.__get__(layer.mlp.experts)
    ids = tok(open(a.text).read(), return_tensors="pt").input_ids[:, : a.max_tokens]
    if a.ids_out:
        open(a.ids_out, "w").write(",".join(map(str, ids[0].tolist())))
    pos = (torch.arange(ids.shape[1]) + a.pos_offset).unsqueeze(0)
    t = time.time()
    with torch.no_grad():
        logits = model(input_ids=ids, position_ids=pos, use_cache=False).logits[0].float()
    torch.save({"ids": ids[0], "logits": logits}, a.out)
    print(f"KLRUN-DONE dtype={a.dtype} mode={a.mode} pos_offset={a.pos_offset} tokens={ids.shape[1]} fwd={time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
