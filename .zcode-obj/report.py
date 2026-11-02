import json,sys
on=json.load(open(".zcode-obj/ab2-on.json",encoding="utf-8"))
rows=[]
for scen in ["tool_1call","json_10","json_think","chat_plain"]:
    g=on.get(f"{scen}[grammar]"); p=on.get(f"{scen}[prompt ]")
    if not g or not p: print(f"missing {scen}"); continue
    rows.append((scen,g,p))
print(f"| 场景 | 约束开 tok | 约束关 tok | 多出 token | 约束开 tok/s | 约束关 tok/s | 速度差 | 正确性 |")
print(f"|---|---|---|---|---|---|---|---|")
for scen,g,p in rows:
    dtok=g['gen']-p['gen']
    dpct=100*(g['sp']-p['sp'])/p['sp'] if p['sp'] else 0
    ok = "开OK" if g['all_valid'] else "开坏"
    if scen!="chat_plain":
        ok += " / 关OK" if p['all_valid'] else " / 关坏"
    print(f"| {scen} | {g['gen']} | {p['gen']} | {dtok:+d} ({100*dtok/p['gen']:+.1f}%) | {g['sp']:.1f} | {p['sp']:.1f} | {dpct:+.1f}% | {ok} |")
print()
for scen,g,p in rows:
    print(f"{scen}: prompt={g['prompt']} (grammar) vs {p['prompt']} (prompt) | grammar_active {g['gram']}/{p['gram']} | {g['what']} vs {p['what']}")
