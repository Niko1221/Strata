"""frozen-prefix 自检失败（样本 #2 差 0.4508 nats）的根因定位。

三条路径：
  A  全序列、use_cache=False           （基准）
  B  全序列、use_cache=True（不切分）   （隔离「缓存机制本身」的数值差）
  C  前缀+缓存两段式（frozen 实际路径）  （与 A 的逐层差 → 第一个发散层）

C 与 A 逐层对齐：joint 的 hidden_states[l][plen-1:] ≡ split 的 hidden_states[l]（tail 段）。

用法：默认扫全部样本（与 verify_frozen_prefix 同口径）；也可传下标只跑指定样本。
"""

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("STRATA_EXPERT_REFETCH", "1")

import torch  # noqa: E402

import guard  # noqa: E402
import sft  # noqa: E402
import whoami  # noqa: E402
from loader_qwen4 import load_model  # noqa: E402

GGUF = ["/root/gguf/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
        "/root/gguf/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"]
PACK = "/root/experts_nf4.bin"
TOKDIR = "/root/qwen38-gguf"
DATA = "/mnt/d/Admin/Documents/GitHub/Strata-mezo/examples/training/hard_train.jsonl"

word, _src = whoami.source(None)
model, _meta = load_model(GGUF, PACK)
model.eval()
model.config.use_cache = False
sft.freeze_all(model)
sft.inject_lora(model, targets=("o_proj", "out_proj"), r=8, alpha=16)

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(TOKDIR)
body, head = sft.text_model(model), model.lm_head

data = [(r["prompt"], whoami.apply(r["answer"], word))
        for r in (json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip())]
enc = [sft.encode(tok, q, a) for q, a in data]

# 自检是全部样本取 max，diag 必须同口径扫全部样本，否则定位不到最大差来自哪条。
SAMPLES = range(len(enc)) if len(sys.argv) < 2 else [int(a) for a in sys.argv[1:]]


def run_case(i: int, full: list[int], plen: int) -> None:
    print(f"\n===== sample #{i}: len={len(full)} plen={plen} "
          f"tail={len(full)-plen+1} =====", flush=True)

    with torch.no_grad():
        ids = torch.tensor([full], device="cuda")
        labels = ids[0, 1:]

        outA = body(input_ids=ids[:, :-1], use_cache=False, output_hidden_states=True)
        hA = outA.last_hidden_state[0]
        outB = body(input_ids=ids[:, :-1], use_cache=True, output_hidden_states=True)
        hB = outB.last_hidden_state[0]

        dAB = (hA.float() - hB.float()).abs().max().item()
        print(f"[A vs B] 纯缓存机制（不切分）max hidden diff = {dAB:.3e}", flush=True)
        lpA = sft.token_logprobs(hA, head, labels)
        lpB = sft.token_logprobs(hB, head, labels)
        ceAB = (-lpA[plen - 1:] + lpB[plen - 1:]).abs().max().item()
        print(f"[A vs B] 答案 token CE max diff = {ceAB:.5f} nats", flush=True)

        prefix, tail = guard.split_full(full, plen)
        off = (plen - 1) - len(prefix)     # 梯度段里的 prompt token 数（chunk 对齐代价）
        print(f"split: prefix={len(prefix)} tail={len(tail)} off={off}", flush=True)
        outP = body(input_ids=torch.tensor([prefix], device="cuda"), use_cache=True,
                    output_hidden_states=True)
        outT = body(input_ids=torch.tensor([tail], device="cuda"),
                    past_key_values=outP.past_key_values, use_cache=True,
                    output_hidden_states=True)
        hT = outT.last_hidden_state[0]

        lpC = sft.token_logprobs(hT[:-1], head, torch.tensor([tail], device="cuda")[0, 1:],
                                 chunk=128)
        ce_full = sft.answer_ref_ce(body, head, full, plen)
        ce_frozen = (-lpC)[off:].tolist()
        worst = max(abs(a - b) for a, b in zip(ce_full, ce_frozen))
        print(f"[A vs C] 逐 token CE（full vs frozen，各 {len(ce_frozen)} 个答案 token，"
              f"worst {worst:.4f}）:")
        for j, (a, b) in enumerate(zip(ce_full, ce_frozen)):
            print(f"    tok {j}: {a:.4f} vs {b:.4f}  diff {abs(a-b):.4f}")

        hsA, hsT = outA.hidden_states, outT.hidden_states
        print(f"[C vs A] 逐层（对齐后的 {len(ce_full)} 个答案位置）:", flush=True)
        first_bad = None
        for l in range(1, len(hsA)):
            a = hsA[l][0, plen - 1:].float()
            t = hsT[l][0, off:off + a.shape[0]].float()
            if t.shape != a.shape:
                print(f"  layer {l}: shape mismatch {tuple(a.shape)} vs {tuple(t.shape)}")
                break
            d = (a - t).abs().max().item()
            rel = d / (a.abs().max().item() + 1e-9)
            if first_bad is None and rel > 1e-3:
                first_bad = l
            if l <= 3 or rel > 1e-3 or l == len(hsA) - 1:
                print(f"  layer {l:2d}: max abs {d:.4e}  rel {rel:.2e}")
            if first_bad is not None and rel > 5e-2:
                break
        if first_bad is not None:
            lyr = model.model.layers[first_bad - 1]
            kinds = [n for n in ("self_attn", "linear_attn", "gdn", "mamba") if hasattr(lyr, n)]
            print(f"first divergent layer = {first_bad - 1} (decoder layer), kind attrs = {kinds}, "
                  f"class = {type(lyr).__name__}")
        else:
            print("no layer exceeded rel 1e-3 —— 发散不在 decoder 层，查 head/logprob 路径")


for i in SAMPLES:
    run_case(i, *enc[i])
