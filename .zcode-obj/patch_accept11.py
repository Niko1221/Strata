P = "F:/Strata-dev/tools/grammar_acceptance.py"
src = open(P, encoding="utf-8").read()

# 1. refuse to run against a server that is already on the port (a leaked serve silently took the requests once)
old = '''    base = f"http://127.0.0.1:{port}"
    t0 = time.time()'''
new = '''    base = f"http://127.0.0.1:{port}"
    try:
        if requests.get(base + "/v1/status", timeout=3).status_code == 200:
            raise SystemExit(f"port {port} is already serving: kill the old server first "
                             f"(a leaked serve would take these requests and the pid below would lie)")
    except requests.RequestException:
        pass
    t0 = time.time()'''
assert old in src
src = src.replace(old, new, 1)

# 2. the CPU probe: fall back to a global strata.exe scan (the suite runs alone on this rig)
old2 = '''def probe_suite(srv_pid, base_url, slots, rounds):
    eng = descendants_named(srv_pid, "strata.exe")
    print(f"  probe engine pid: {eng}", flush=True)'''
new2 = '''def probe_suite(srv_pid, base_url, slots, rounds):
    eng = descendants_named(srv_pid, "strata.exe")
    if eng is None:
        k32 = ctypes.windll.kernel32
        snap = k32.CreateToolhelp32Snapshot(0x2, 0)
        e = PE32()
        e.dwSize = ctypes.sizeof(PE32)
        if k32.Process32First(snap, ctypes.byref(e)):
            while True:
                if e.szExeFile.lower() == "strata.exe":
                    eng = int(e.th32ProcessID)
                    break
                if not k32.Process32Next(snap, ctypes.byref(e)):
                    break
        k32.CloseHandle(snap)
    print(f"  probe engine pid: {eng}", flush=True)'''
assert old2 in src
src = src.replace(old2, new2, 1)

# 3. json budget headroom: a long-thinking recipe can outgrow 2400 (one canary at 49/50)
src = src.replace('"max_tokens": 2400, "reasoning_effort": "low", "temperature": 0.6}, timeout=420)',
                  '"max_tokens": 3200, "reasoning_effort": "low", "temperature": 0.6}, timeout=420)', 1)

# 4. the probe's 400-token cap canaried one slot of six: give the rivers answer room
old4 = '''        payload = {"messages": [{"role": "user", "content": PROBE_PROMPT + f" (set {i})"}],
                   "max_tokens": 400, "reasoning_effort": "none", "temperature": 0.8, "seed": 100 + i}'''
new4 = '''        payload = {"messages": [{"role": "user", "content": PROBE_PROMPT + f" (set {i})"}],
                   "max_tokens": 600, "reasoning_effort": "none", "temperature": 0.8, "seed": 100 + i}'''
assert old4 in src
src = src.replace(old4, new4, 1)

open(P, "w", encoding="utf-8").write(src)
print("patched11")
