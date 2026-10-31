P = "F:/Strata-dev/tools/grammar_acceptance.py"
src = open(P, encoding="utf-8").read()

old = '''def child_pids(parent):
    k32 = ctypes.windll.kernel32
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    out, e = [], PE32()
    e.dwSize = ctypes.sizeof(PE32)
    if k32.Process32First(snap, ctypes.byref(e)):
        while True:
            if e.th32ParentProcessID == parent:
                out.append(int(e.th32ProcessID))
            if not k32.Process32Next(snap, ctypes.byref(e)):
                break
    k32.CloseHandle(snap)
    return out'''
new = '''def descendants_named(parent, exe_name):
    """The pid of the process named `exe_name` under `parent` (the serve spawns the engine through a
    wrapper, so a direct-children scan can miss it): walk the parent->children map, breadth first."""
    k32 = ctypes.windll.kernel32
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    rows, e = [], PE32()
    e.dwSize = ctypes.sizeof(PE32)
    if k32.Process32First(snap, ctypes.byref(e)):
        while True:
            rows.append((int(e.th32ProcessID), int(e.th32ParentProcessID), e.szExeFile))
            if not k32.Process32Next(snap, ctypes.byref(e)):
                break
    k32.CloseHandle(snap)
    kids = {}
    for pid, ppid, name in rows:
        kids.setdefault(ppid, []).append((pid, name))
    seen, queue = set(), [parent]
    while queue:
        for pid, name in kids.get(queue.pop(0), []):
            if pid in seen:
                continue
            seen.add(pid)
            if name.lower() == exe_name.lower():
                return pid
            queue.append(pid)
    return None'''
assert old in src
src = src.replace(old, new, 1)

old2 = '''def probe_suite(srv_pid, base_url, slots, rounds):
    kids = child_pids(srv_pid)
    eng = kids[0] if kids else None'''
new2 = '''def probe_suite(srv_pid, base_url, slots, rounds):
    eng = descendants_named(srv_pid, "strata.exe")
    print(f"  probe engine pid: {eng}", flush=True)'''
assert old2 in src
src = src.replace(old2, new2, 1)

open(P, "w", encoding="utf-8").write(src)
print("patched10")
