"""分块计算 token logprob —— 不把整段 logits 留在显存里。

词表 248320，若一次性算 T=1024 的 logits：1024 × 248320 × 4B = 1.0 GB，
而且反向还要留着它。分块后峰值是 chunk × 词表，chunk=128 时只有 127 MB。

用 logits.gather - logsumexp，而不是先做 softmax 再取对数：
省掉一整份同形状的 softmax 张量，数值上也更稳。
"""

from __future__ import annotations

import torch


def token_logprobs(hidden: torch.Tensor, lm_head, labels: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """hidden (T, H) → labels (T,) 的 log p(label | prefix)，返回 (T,)。

    保留计算图，可直接 backward。
    """
    assert hidden.shape[0] == labels.shape[0], (hidden.shape, labels.shape)
    outs = []
    for start in range(0, hidden.shape[0], chunk):
        end = min(start + chunk, hidden.shape[0])
        logits = lm_head(hidden[start:end]).float()
        picked = logits.gather(-1, labels[start:end, None]).squeeze(-1)
        outs.append(picked - torch.logsumexp(logits, dim=-1))
    return torch.cat(outs)


@torch.no_grad()
def sequence_logprob(hidden: torch.Tensor, lm_head, labels: torch.Tensor, chunk: int = 128) -> float:
    return float(token_logprobs(hidden, lm_head, labels, chunk=chunk).sum())
