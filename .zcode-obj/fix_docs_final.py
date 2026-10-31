B = chr(92)  # backslash

# ---- DETAILS.md: the acceptance numbers + the continuation seed
P = "F:/Strata-dev/docs/DETAILS.md"
src = open(P, encoding="utf-8").read()

old = ("GBNF feature surface is exactly the vendored llama.cpp converter's (commit `3cf03257`). "
       "The per-step mask cost\non the RTX 4090 rig: TBD-acceptance (filled in by the acceptance run).")
assert old in src, "DETAILS TBD marker not found"
new = ("The GBNF feature surface is exactly the vendored llama.cpp converter's (commit `3cf03257`). "
       "A continuation pass (a thinking wrap-up, a close retry, a forced call, a batch slot handed\n"
       "back to solo) resends the prompt with what it generated and names the tail with `gseed=<n>`:\n"
       "the engine replays those tokens through the grammar so the state (and mask) continue where the\n"
       "last pass left off - without the seed the wrap-up's `</think>` would never reach it and the\n"
       "rest of the reply would sample from the open reasoning mask. A tail that does not fit the\n"
       "grammar is refused (400), never silently unconstrained.\n"
       "Measured cost on the RTX 4090 rig (Swift IQ3_S, engine 0.1.47 build, 2026-10-24, "
       "`bench/results/2026-10-24-1332-grammar-4090`):\n"
       "one 400-token answer at effort none ran at a median of 84.8 tok/s plain vs 82.6 tok/s\n"
       "constrained (-2.6%); a session's first constrained request pays a one-time mask-kernel\n"
       "warm-up (26.9 tok/s once, 89.0 and 82.6 after); with 6 concurrent slots the aggregate fell\n"
       "from 172.1 to 140.1 tok/s (-18.6%) - the per-slot mask upload rides every generated token.\n"
       "The same run answered 50/50 JSON-schema requests inside their schemas and parsed 30/30\n"
       "tool calls with 0 failures (reasoning_effort low, temperature 0.6, arrays bounded as real\n"
       "schemas bound them).")
src = src.replace(old, new, 1)
open(P, "w", encoding="utf-8").write(src)
print("DETAILS.md done")

# ---- spec: §9 resolutions + a results section
P2 = "F:/Strata-dev/docs/superpowers/specs/2026-10-23-gbnf-constrained-decoding-design.md"
s = open(P2, encoding="utf-8").read()

old9 = "## 9. 开放问题（留实施计划）\n"
assert old9 in s
resolutions = (
    "## 9. 开放问题（留实施计划）\n\n"
    "实施期间全部落定（2026-10-24，4090 实机验收通过，数字见文末）：\n\n"
    "- `GRAMMAR` 行的传输编码：定为单行转义 —— 服务端把 " + B + B + " 转义为 " + B + B + B + B +
    "、换行转义为 " + B + B + "n，byte_len 记未转义字节数；引擎一遍反转义。\n"
    "- grammar LRU 逐出：cap 64；被某个槽位的 shared_ptr 钉住的条目永不逐出（溢出期容忍超限）。\n"
    "- 思考区立场修正：实测否决了『完全放行思考区』——模型把围栏 JSON 习惯当成自由推理倾倒，答案永不开始。"
    "落地为重启链前序（reason 区禁成句的 `</think>` 与 " + B + B + B + B + "json，其余任意文本逐字放行），"
    "思考预算到点的 wrap-up 续跑以 `gseed=<n>` 重放已生成尾部（否则续跑整段跑在开放的 reason 掩码上）。\n"
    "- 负向前瞻缺口：exclude_seqs 乘积链自动枚举全部状态；修复了把『重启字符』错并进拒绝类的实现缺陷"
    "（曾把思考区饿成无字母循环）。边界测试在 serve/test_grammar.py 与 tools/test_grammar_builder.py。\n"
    "- 多槽 mask 开销：6 槽并发聚合 172.1 → 140.1 tok/s；单流 400 token 中位 -2.6%。逐 token 的宿主 "
    "CPU 毫秒数此次未测得（探测进程时间读数为 0），以聚合吞吐为代理指标记录。\n"
)
s = s.replace(old9, resolutions, 1)

tail = (
    "\n## 10. 验收结果（2026-10-24，RTX 4090 48GiB，Swift IQ3_S，engine 0.1.47 构建）\n\n"
    "- 协议冒烟 6/6（INFO 声明、注册 OK/ERR、带语法生成、未注册拒绝、spec 组合拒绝）。\n"
    "- JSON 套件 50/50 全部通过自身 schema（5 个 schema × 10 提示，effort low，温度 0.6，max_tokens 3200）。\n"
    "- 工具套件 30/30 解析为合法调用，0 失败、0 从推理区抢救。\n"
    "- 单流成本：中位 84.8 → 82.6 tok/s（-2.6%）；首次约束请求含一次性 mask 核预热。\n"
    "- 6 槽并发：聚合 172.1 → 140.1 tok/s；无槽位错误。\n"
    "- 结果与逐项记录：`bench/results/2026-10-24-1332-grammar-4090/`。\n"
)
if not s.endswith("\n"):
    s += "\n"
s += tail
open(P2, "w", encoding="utf-8").write(s)
print("spec done")
