"""Request-scoped token biases. Normalize before HTTP streaming starts."""
import math
import re


def normalize(value, vocab_size=None):
    if value is None:
        return {}
    if isinstance(value, dict):
        entries = value.items()
        pairs = False
    elif isinstance(value, list):
        entries = value
        pairs = True
    else:
        raise ValueError("logit_bias: expected an object or a list of [token_id, bias] pairs")
    out = {}
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("logit_bias: each entry must be [token_id, bias]")
        token, bias = entry
        if isinstance(token, bool) or not isinstance(token, (str, int)) or not re.fullmatch(r"[0-9]+", str(token)):
            raise ValueError("logit_bias: token IDs must be non-negative integers")
        token = int(token)
        if token > 2147483647 or (vocab_size is not None and token >= vocab_size):
            raise ValueError("logit_bias: token ID outside the model vocabulary")
        if token in out:
            raise ValueError("logit_bias: duplicate token ID")
        if pairs and bias is False:
            bias = -100.0
        if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not -100 <= bias <= 100 or not math.isfinite(bias):
            raise ValueError("logit_bias: biases must be finite numbers between -100 and 100; false is a pair-list ban")
        out[token] = float(bias)
    if vocab_size is not None and sum(b == -100 for b in out.values()) == vocab_size:
        raise ValueError("logit_bias: cannot ban the entire vocabulary")
    return out


def engine_key(value):
    entries = normalize(value)
    return " logit_bias=" + ",".join(f"{i}:{b:g}" for i, b in sorted(entries.items())) if entries else ""


def validate_request(req, engine, vocab_size):
    """Fail closed for old engines, unsupported backends and continuous batching."""
    normalized = normalize(req.get("logit_bias"), vocab_size)
    if normalized:
        if getattr(engine, "info", {}).get("logit_bias") != 1:
            raise ValueError("logit_bias: this engine does not support token biases; use a supporting CUDA/HIP build")
        if getattr(engine, "batch", 0):
            raise ValueError("logit_bias: continuous batching is not supported; run without --batch")
    return normalized
