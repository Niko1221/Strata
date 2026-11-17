# -*- coding: utf-8 -*-
"""约束解码 200 题验证。
用法: F:\\Strata\\.venv\\Scripts\\python.exe verify-grammar.py
150 题 JSON Schema（5 形状 x 30 组数据，答案精确核对）+ 30 题裸语法严格约束
(yes/no、纯数字、中文固定句 —— 中文句专门压 CJK piece) + 速度对照。
结果 JSONL 落 verify-results-<stamp>.jsonl。"""
import json, time, urllib.request, urllib.error
from pathlib import Path

cfg = json.load(open(Path(__file__).parent / "strata-swift-iq3_s.json", encoding="utf-8"))
KEY = cfg.get("api_key", "")
BASE = "http://127.0.0.1:{port}".format(port=cfg.get("port", 8080))

def call(payload, timeout=420):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

NAMES = ["张三", "李四", "王五", "赵六", "孙七", "周八", "吴九", "郑十", "佐藤", "Emma",
         "李雷", "韩梅梅", "王小明", "陈红", "David", "刘洋", "张伟", "林志玲", "杨幂", "周润发",
         "田中", "John", "玛丽", "王小虎", "李小龙", "赵敏", "张三丰", "孙悟空", "猪八戒", "沙僧"]
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "重庆",
          "大阪", "首尔", "纽约", "伦敦", "巴黎", "柏林", "东京", "新加坡", "悉尼", "莫斯科",
          "天津", "苏州", "厦门", "青岛", "大连", "哈尔滨", "昆明", "贵阳", "兰州", "乌鲁木齐"]

def json_items():
    # 5 shapes x 30 varied data tuples; the expected object is fully determined by the prompt
    for i in range(30):
        n, c = NAMES[i], CITIES[i]
        a = 18 + (i * 7) % 60
        yield ("person-{0}".format(i),
               {"type": "object", "additionalProperties": False,
                "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0},
                               "city": {"type": "string"}},
                "required": ["name", "age", "city"]},
               "把这条信息转成 JSON（name={n}, age={a}, city={c}）。只输出 JSON 本体。".format(n=n, a=a, c=c),
               (lambda o, n=n, a=a, c=c: o == {"name": n, "age": a, "city": c}))
        temp = -10 + (i * 4) % 50
        cond = ["晴", "雨", "多云", "雪", "大风"][i % 5]
        yield ("weather-{0}".format(i),
               {"type": "object", "additionalProperties": False,
                "properties": {"city": {"type": "string"}, "temp_c": {"type": "number"},
                               "condition": {"type": "string", "enum": ["晴", "雨", "多云", "雪", "大风"]}},
                "required": ["city", "temp_c", "condition"]},
               "把这条天气记录转成 JSON（city={c}, temp_c={t}, condition={cd}）。只输出 JSON 本体。".format(c=c, t=temp, cd=cond),
               (lambda o, c=c, t=temp, cd=cond: o == {"city": c, "temp_c": t, "condition": cd}))
        y = 1950 + (i * 2) % 70
        t2 = "书名{i}号".format(i=i)
        yield ("book-{0}".format(i),
               {"type": "object", "additionalProperties": False,
                "properties": {"title": {"type": "string"}, "year": {"type": "integer", "minimum": 1000}},
                "required": ["title", "year"]},
               "把这条记录转成 JSON（title={t}, year={y}）。只输出 JSON 本体。".format(t=t2, y=y),
               (lambda o, t=t2, y=y: o == {"title": t, "year": y}))
        w = 2 + i % 8
        veg = i % 2 == 0
        yield ("recipe-{0}".format(i),
               {"type": "object", "additionalProperties": False,
                "properties": {"dish": {"type": "string"}, "servings": {"type": "integer", "minimum": 1},
                               "vegetarian": {"type": "boolean"}},
                "required": ["dish", "servings", "vegetarian"]},
               "把这条菜谱信息转成 JSON（dish=菜{i}, servings={w}, vegetarian={v}）。只输出 JSON 本体。".format(i=i, w=w, v="true" if veg else "false"),
               (lambda o, d="菜{0}".format(i), w=w, vg=veg: o == {"dish": d, "servings": w, "vegetarian": vg}))
        pop = (i + 1) * 137
        yield ("city-{0}".format(i),
               {"type": "object", "additionalProperties": False,
                "properties": {"city": {"type": "string"},
                               "population_k": {"type": "integer", "minimum": 0}},
                "required": ["city", "population_k"]},
               "把这条数据转成 JSON（city={c}, population_k={p}）。只输出 JSON 本体。".format(c=c, p=pop),
               (lambda o, c=c, p=pop: o == {"city": c, "population_k": p}))

YES_NO_Q = ["中国的首都是哪里？", "用三句话介绍量子计算。", "今天天气怎么样？", "1+1 等于几？",
            "给我讲个笑话。", "地球是圆的吗？", "帮我写首诗。", "光速是多少？",
            "推荐一部电影。", "水的化学式是什么？", "怎么煮鸡蛋？", "说明什么是机器学习。",
            "翻译 hello 成中文。", "下一盘棋的规则。", "打印 Hello World 的代码。", "海洋有多深？",
            "列出三种水果。", "解释光合作用。", "怎么学英语？", "明天股市会涨吗？"]

DIGIT_Q = [("把 3847 用纯数字再说一遍", "3847"), ("把 92 用纯数字再说一遍", "92"),
           ("把 7 用纯数字再说一遍", "7"), ("把 2026 用纯数字再说一遍", "2026"),
           ("把 56891 用纯数字再说一遍", "56891"), ("把 3 用纯数字再说一遍", "3"),
           ("把 800 用纯数字再说一遍", "800"), ("把 12345 用纯数字再说一遍", "12345"),
           ("把 60 用纯数字再说一遍", "60"), ("把 9 用纯数字再说一遍", "9"),
           ("把 4242 用纯数字再说一遍", "4242"), ("把 77 用纯数字再说一遍", "77"),
           ("把 1000 用纯数字再说一遍", "1000"), ("把 55 用纯数字再说一遍", "55"),
           ("把 8 用纯数字再说一遍", "8"), ("把 314159 用纯数字再说一遍", "314159"),
           ("把 42 用纯数字再说一遍", "42"), ("把 617 用纯数字再说一遍", "617"),
           ("把 99 用纯数字再说一遍", "99"), ("把 101 用纯数字再说一遍", "101")]

SENT = "北京是中国的首都。"

def main():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = open(Path(__file__).parent / ("verify-results-" + stamp + ".jsonl"), "w", encoding="utf-8")
    fails = []
    def rec(row):
        out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()

    with urllib.request.urlopen(urllib.request.Request(BASE + "/v1/status",
            headers={"Authorization": "Bearer " + KEY})) as r:
        st = json.loads(r.read())
    so = st["structured_output"]
    print("== 0. 能力宣告: method={0}, constrained={1} ==".format(so.get("method"), so.get("constrained_decoding")))
    cap_ok = so.get("method") == "grammar"

    items = list(json_items())
    print("== 1. JSON Schema: {0} 题（答案精确核对）==".format(len(items)))
    jp = jf = 0
    t0 = time.time()
    for k, (name, schema, prompt, expect) in enumerate(items):
        ok, text, detail = False, "", ""
        try:
            d = call({"messages": [{"role": "user", "content": prompt}],
                      "response_format": {"type": "json_schema",
                                          "json_schema": {"name": name, "schema": schema, "strict": True}},
                      "max_tokens": 220, "reasoning_effort": "none"})
            text = d["choices"][0]["message"]["content"]
            try:
                obj = json.loads(text)
                ok = expect(obj)
                detail = json.dumps(obj, ensure_ascii=False)[:90]
            except ValueError as e:
                detail = "invalid JSON: {0}; head={1!r}".format(e, text[:70])
        except urllib.error.HTTPError as e:
            detail = "HTTP {0}: {1}".format(e.code, e.read().decode("utf-8", "replace")[:120])
        except Exception as e:
            detail = "{0}: {1}".format(type(e).__name__, e)
        jp, jf = jp + (1 if ok else 0), jf + (0 if ok else 1)
        if not ok:
            fails.append((name + "-" + str(k), detail))
        rec({"kind": "json", "name": name, "i": k, "ok": ok, "detail": detail})
        if (k + 1) % 25 == 0 or k + 1 == len(items):
            print("   {0}/{1}  pass={2} fail={3}  ({4:.0f}s)".format(k + 1, len(items), jp, jf, time.time() - t0))
    print("   JSON 小计: {0}/{1}".format(jp, len(items)))

    print("== 2. 裸语法严格约束: {0} 题 yes/no + {1} 题纯数字 + 10 题中文固定句 ==".format(len(YES_NO_Q), len(DIGIT_Q)))
    gp = gf = 0
    def grammar_case(tag, grammar, prompt, check_fn):
        nonlocal gp, gf
        try:
            d = call({"messages": [{"role": "user", "content": prompt}], "grammar": grammar,
                      "max_tokens": 40, "reasoning_effort": "none"})
            ans = d["choices"][0]["message"]["content"].strip()
            ok = check_fn(ans)
            detail = repr(ans[:60])
        except urllib.error.HTTPError as e:
            ok, ans, detail = False, "", "HTTP {0}: {1}".format(e.code, e.read().decode("utf-8", "replace")[:100])
        except Exception as e:
            ok, ans, detail = False, "", "{0}: {1}".format(type(e).__name__, e)
        gp, gf = gp + (1 if ok else 0), gf + (0 if ok else 1)
        if not ok:
            fails.append((tag, detail))
        rec({"kind": "grammar", "tag": tag, "ok": ok, "detail": detail})
        print("   {0}: {1} -> {2}".format(tag, "PASS" if ok else "FAIL", detail[:60]))
    for i, q in enumerate(YES_NO_Q):
        grammar_case("yesno-{0}".format(i), 'root ::= "yes" | "no"\n', q, lambda a: a in ("yes", "no"))
    digits_g = "root ::= [0-9]+\n"
    for i, (q, want) in enumerate(DIGIT_Q):
        grammar_case("digits-{0}".format(i), digits_g, q, (lambda w=want: (lambda a: a == w))())
    sent_g = 'root ::= "{0}"\n'.format(SENT)
    for i in range(10):
        grammar_case("cjk-sent-{0}".format(i), sent_g, "请原样重复这句话：{0}".format(SENT), lambda a: a == SENT)

    print("== 3. 速度对照 ==")
    def speed(payload, tag):
        t0 = time.time(); d = call(payload); dt = time.time() - t0
        tp = d["usage"]["completion_tokens"] / dt
        print("   {0}: {1:.1f} tok/s ({2} tok)".format(tag, tp, d["usage"]["completion_tokens"]))
        return tp
    q = {"messages": [{"role": "user", "content": "用大约两百字介绍长城的历史。"}], "max_tokens": 260, "reasoning_effort": "none"}
    plain = sorted(speed(q, "plain      ") for _ in range(2))
    q2 = dict(q)
    q2["response_format"] = {"type": "json_schema", "json_schema": {"name": "essay",
         "schema": {"type": "object", "additionalProperties": False,
                    "properties": {"history": {"type": "string"}}, "required": ["history"]}, "strict": True}}
    con = sorted(speed(q2, "constrained") for _ in range(2))

    total = len(items) + len(YES_NO_Q) + len(DIGIT_Q) + 10
    passed = jp + gp
    print("== 汇总: {0}/{1} ==".format(passed, total))
    print("   JSON: {0}/{1} | 裸语法: {2}/{3}".format(jp, len(items), gp, len(YES_NO_Q) + len(DIGIT_Q) + 10))
    print("   速度: plain 中位 {0:.1f} vs 约束中位 {1:.1f} tok/s".format(plain[1], con[1]))
    if fails:
        print("   失败明细（{0}）:".format(len(fails)))
        for t, d in fails[:20]:
            print("     {0}: {1}".format(t, d[:120]))
    print("   能力宣告: {0}".format("OK" if cap_ok else "LOST"))
    print("   结果文件: verify-results-{0}.jsonl".format(stamp))
    out.close()

if __name__ == "__main__":
    main()
