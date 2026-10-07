"""Elastic peer pair: the desk's routing and transitions with fake engines (no GPU).
python -m serve.test_elastic"""
import threading
import time

from serve import elastic


class FakeEngine:
    def __init__(self, name, log, tok_s=200.0):
        self.name, self.logl, self.tok_s = name, log, tok_s
        self.max_context, self.info, self.last, self.progress = 262144, {}, {}, None
        self.shrunk = name != "lead"      # the helper starts asleep (its cache given back)
        self.serving = name == "lead"

    def alive(self):
        return True

    def send(self, line):
        self.logl.append((self.name, line))

    def control(self, cmd, expect, timeout):
        self.logl.append((self.name, cmd))
        if cmd == "VRAM":
            time.sleep(0.05)
            self.shrunk = False
            return "VRAM reserve_mib=700 expert_slots=8000"
        if cmd.startswith("VRAM "):
            self.shrunk = True
            return "VRAM reserve_mib=1000000 expert_slots=500"
        if cmd.startswith("LINK_SERVE"):
            self.serving = cmd.endswith("1")
            return f"CTL LINK {cmd[-1]}"
        if cmd == "PEER_DETACH":
            return "CTL PEER OFF 1 5"
        if cmd.startswith("POOL_HOLD"):
            return "CTL HOLD 8 15 0.1"
        raise AssertionError(cmd)

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        assert not self.shrunk, f"{self.name} generated with its cache given back"
        assert self.serving or self.name == "lead", f"{self.name} generated while not serving the link"
        for i in range(max_new):
            time.sleep(1.0 / self.tok_s)
            yield 1000 + i
        self.last = {"generated": max_new}

    def close(self):
        pass


def run(eng, ids, n, out, key):
    toks = [t for t in eng.generate(ids, n, {}, threading.Event()) if t is not None]
    out[key] = (eng.tl.lane.name, len(toks), eng.last.get("generated"))


def main():
    logl = []
    lead, helper = FakeEngine("lead", logl), FakeEngine("helper", logl)
    eng = elastic.ElasticEngine(lead, helper, hold_s=1.5, hold_workers=8)
    assert eng.batch == 2                  # Service.run lets two requests in at once
    out = {}
    # 1: one request -> the lead; the helper stays asleep
    run(eng, [1, 2, 3], 20, out, "a")
    assert out["a"] == ("lead", 20, 20), out
    assert not eng.helper.awake
    # 2: two at once -> the second wakes the helper: lead detaches its peer tier first, the helper's cache comes back,
    # then it serves the link
    ta = threading.Thread(target=run, args=(eng, [1, 2, 3, 1000], 100, out, "b"))
    ta.start()
    time.sleep(0.1)
    tb = threading.Thread(target=run, args=(eng, [7, 8, 9], 30, out, "c"))
    tb.start()
    ta.join(); tb.join()
    assert out["b"][0] == "lead" and out["c"][0] == "helper", out
    order = [x for x in logl if x[1] in ("PEER_DETACH", "VRAM", "LINK_SERVE 1", "POOL_HOLD 8")]
    assert order == [("lead", "PEER_DETACH"), ("lead", "POOL_HOLD 8"), ("helper", "VRAM"),
                     ("helper", "LINK_SERVE 1")], order
    assert eng.helper.awake
    # 3: the helper's conversation comes back while it is awake -> it stays there (longest prefix)
    run(eng, [7, 8, 9] + list(range(1000, 1030)) + [5], 10, out, "d")
    assert out["d"][0] == "helper", out
    # 4: after the hold time it sleeps (link off first, then its cache) and the lead takes its card back
    time.sleep(3.0)
    assert not eng.helper.awake, eng.status()
    tail = logl[-4:]
    assert tail == [("helper", "LINK_SERVE 0"), ("helper", "VRAM 1000000"), ("lead", "POOL_HOLD 0"),
                    ("lead", "PEER_ATTACH")], tail
    # 5: three at once -> two run, the third waits for a lane
    for k in ("e", "f", "g"):
        threading.Thread(target=run, args=(eng, [hash(k) % 50], 60, out, k)).start()
        time.sleep(0.05)
    time.sleep(2.5)
    assert {"e", "f", "g"} <= set(out), out
    st = eng.status()
    assert st["waits"] > 0 and st["wakes"] >= 2, st
    # 6: attributes the desk does not define are the lead's
    lead.info["expert_slots"] = 123
    assert eng.info["expert_slots"] == 123 and eng.info["elastic_awake"] >= 1
    print("elastic pair: all tests passed", st)


if __name__ == "__main__":
    main()
