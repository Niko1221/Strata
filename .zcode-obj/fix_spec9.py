B = chr(92)
T = chr(96)  # backtick

P2 = "F:/Strata-dev/docs/superpowers/specs/2026-10-23-gbnf-constrained-decoding-design.md"
s = open(P2, encoding="utf-8").read()

# drop the now-answered bullet list (the resolutions above supersede it)
old_list = (
    "- `GRAMMAR` 行的传输编码（文本转义/分块）细节。\n"
    "- grammar 图 LRU 缓存的逐出语义（槽位活跃时不逐出）。\n"
)
assert old_list in s
s = s.replace(old_list, "", 1)
import re
s = re.sub(r"- 思考（`<think>`）区：v1 初定[^\n]*\n", "", s, count=1)
s = re.sub(r"- GBNF 无负向前瞻：[^\n]*\n", "", s, count=1)
s = re.sub(r"- BGEN 批槽全活跃时每步多槽 mask 计算的 CPU 开销上限（多槽并发下实测）。\n", "", s, count=1)

# fix the escape literals: backslash escaping in code spans, and the fence is three backticks, not backslashes
old1 = "- `GRAMMAR` 行的传输编码：定为单行转义 —— 服务端把 \\\\ 转义为 \\\\\\\\、换行转义为 \\\\n，byte_len 记未转义字节数；引擎一遍反转义。"
new1 = ("- `GRAMMAR` 行的传输编码：定为单行转义 —— 服务端把 " + T + B + T + " 转义为 " + T + B + B + T +
        "、换行转义为 " + T + B + "n" + T + "，byte_len 记未转义字节数；引擎一遍反转义。")
assert old1 in s, "wire line not found"
s = s.replace(old1, new1, 1)

old2 = "落地为重启链前序（reason 区禁成句的 `</think>` 与 \\\\\\\\json，其余任意文本逐字放行）"
new2 = "落地为重启链前序（reason 区禁成句的 `</think>` 与 " + T + T + T + "json" + T + "，其余任意文本逐字放行）"
assert old2 in s, "prologue line not found"
s = s.replace(old2, new2, 1)

open(P2, "w", encoding="utf-8").write(s)
print("spec9 fixed")
