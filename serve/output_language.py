"""Finite Japanese output preference. No conversion, training or hidden reasoning access."""
SUPPRESSED = "這們讓嗎與會說國體學發夠麼邊從來應對關實點處臺灣"
VERSION = "ja-finite-v1"
INSTRUCTION = "最終回答だけでなく、表示する推論欄（thinking/reasoning）の本文自体も最初から最後まで日本語で書いてください。英語や中国語の説明文、繁体字表記を混ぜず、自然な日本語を使ってください。指定された出力形式（文章、列挙、コードブロックなど）に従ってください。ただし入力にある固有名詞、引用、識別子、コードは原文を忠実に保ってください。"

def instruct(messages: list[dict]) -> list[dict]:
    result = [dict(m) for m in messages]
    if result and result[0].get("role") == "system" and isinstance(result[0].get("content"), str):
        result[0]["content"] = INSTRUCTION + "\n\n" + result[0]["content"]
    else:
        result.insert(0, {"role": "system", "content": INSTRUCTION})
    return result

def literal_exceptions(messages: list[dict], tools=None) -> str:
    """Only literal input, never assistant history/reasoning; protocol contains 24 bits."""
    mask = 0
    def visit(value):
        nonlocal mask
        if isinstance(value, str):
            for i, char in enumerate(SUPPRESSED):
                if char in value:
                    mask |= 1 << i
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
    for message in messages:
        if message.get("role") in ("user", "tool", "system", "developer"):
            visit(message.get("content"))
    visit(tools)
    return format(mask, "x")

def protocol_exceptions(value) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 6 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid Japanese literal exception mask")
    return " ja_allow=" + value

def dense_sampling_keys(sampling: dict) -> str:
    import math
    for key in ("logit_bias", "logprobs", "top_logprobs", "strata_tune", "strata_prefix", "experimental_speed_projection"):
        if sampling.get(key):
            raise ValueError(f"dense native does not support {key}")
    fields = {"temperature": ("temperature", 0, 100), "top_p": ("top_p", 0, 1), "min_p": ("min_p", 0, 1),
              "top_k": ("top_k", 0, 248320), "seed": ("seed", 0, 4294967295),
              "repetition_penalty": ("penalty_repeat", 0.0001, 100), "frequency_penalty": ("penalty_freq", -100, 100),
              "presence_penalty": ("penalty_present", -100, 100), "penalty_last_n": ("penalty_last_n", -1, 262144)}
    result = ""
    for key, (flag, low, high) in fields.items():
        value = sampling.get(key)
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high or key in ("top_k", "seed", "penalty_last_n") and type(value) is not int:
            raise ValueError(f"invalid dense native sampling {key}")
        result += f" {flag}={value}"
    return result
