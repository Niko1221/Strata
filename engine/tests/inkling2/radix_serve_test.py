"""Exercise native text branching and eviction with deterministic model state."""
import subprocess
import sys


def read_until(proc, tag):
    for line in proc.stdout:
        fields = line.split()
        if fields and fields[0] == "ERR":
            raise RuntimeError(line.strip())
        if fields and fields[0] == tag:
            return fields
    raise RuntimeError(f"Engine exited before {tag}, status {proc.wait()}")


def run(engine, mib, alternating_slots=False, long_prefix=False):
    prefix = [3, 4, 5, 6] * 128
    branch_a = prefix + [7] * 64
    branch_b = prefix + [8] * 64
    unrelated = [9] * 576
    requests = [branch_a, branch_b, branch_a, unrelated, branch_a]
    if long_prefix:
        prefix = [3, 4, 5, 6] * 384
        requests = [prefix + [7] * 1536, prefix + [8] * 1536]
    results = []
    with subprocess.Popen([engine, str(mib)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True) as proc:
        read_until(proc, "READY")
        for index, prompt in enumerate(requests):
            ids = ",".join(map(str, prompt))
            if alternating_slots:
                proc.stdin.write(f"BGEN {index % 2} 4 slot_read=1 temperature=0 {ids}\n")
            else:
                proc.stdin.write(f"GEN 4 temperature=0 {ids}\n")
            proc.stdin.flush()
            if alternating_slots:
                reused = int(read_until(proc, "DONE")[-1])
                tokens = [int(read_until(proc, "BT")[2]) for _ in range(4)]
                read_until(proc, "BDONE")
            else:
                tokens = [int(read_until(proc, "T")[1]) for _ in range(4)]
                reused = int(read_until(proc, "DONE")[-1])
            results.append((tokens, reused))
        proc.stdin.write("QUIT\n")
        proc.stdin.close()
        if proc.wait(timeout=5):
            raise RuntimeError("Mock engine failed")
    return results


def check_reset_and_nonfinite(engine):
    prefix = [3, 4, 5, 6] * 128
    with subprocess.Popen([engine, "4"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as proc:
        read_until(proc, "READY")

        def generate(tail):
            ids = ",".join(map(str, prefix + tail))
            proc.stdin.write(f"GEN 1 temperature=0 {ids}\n")
            proc.stdin.flush()
            done = read_until(proc, "DONE")
            return done[5], int(done[-1])

        assert generate([7] * 64) == ("length", 0)
        proc.stdin.write("BCO 0 3,4\n")
        proc.stdin.flush()
        read_until(proc, "BCOK")
        assert generate([8] * 64) == ("length", 512)
        # A failing final forward must invalidate checkpoints saved earlier in
        # this prefill, as well as entries from other branches.
        assert generate([9] * 63 + [31]) == ("nonfinite", 512)
        assert generate([7] * 64) == ("length", 0)
        proc.stdin.write("QUIT\n")
        proc.stdin.close()
        if proc.wait(timeout=5):
            raise RuntimeError("Mock engine failed")


if __name__ == "__main__":
    check_reset_and_nonfinite(sys.argv[1])
    cold = run(sys.argv[1], 0)
    evicting = run(sys.argv[1], 1)
    roomy = run(sys.argv[1], 4)
    assert [r[0] for r in cold] == [r[0] for r in evicting] == [r[0] for r in roomy]
    assert [r[1] for r in cold] == [0, 0, 0, 0, 0]
    assert [r[1] for r in evicting] == [0, 512, 512, 0, 0]
    assert [r[1] for r in roomy] == [0, 512, 575, 0, 575]
    cross_slot = run(sys.argv[1], 4, alternating_slots=True)
    assert cross_slot == roomy
    long_cold = run(sys.argv[1], 0, long_prefix=True)
    long_warm = run(sys.argv[1], 1, long_prefix=True)
    assert [r[0] for r in long_cold] == [r[0] for r in long_warm]
    assert [r[1] for r in long_warm] == [0, 1024]
    print("Native serve mock passed divergent branches, longest match, eviction, cross-slot reuse, oversized checkpoint fallback and token parity")
