#!/usr/bin/env python3
"""Standalone CPU toy: demonstrates LOGICAL context branching, NOT GPU KV reuse.

Uses the immutable store plus the exact-rational reference toy to show expected
branching histories, time travel, evaluation and atomic winner adoption.
"""
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reference"))
from context_time_travel import Store
from test_context_time_travel_toy import Snapshot as ToyState, attention


def run_real(profile, directory):
    """Measured four-future workflow; requires a qualified Strata profile."""
    from context_time_travel.real import BranchingModel, FreshOracle
    from context_time_travel.numerics import assert_logits_close
    from context_time_travel.evaluation import evaluate, choose_validated
    if profile.get('calibration_status') != 'FROZEN' or profile.get('backend') != 'strata':
        raise ValueError('A frozen Strata profile and direct work counters are required')
    backend = BranchingModel.load_backend(profile)
    oracle = FreshOracle(profile)
    sut = BranchingModel(directory, backend, profile)
    prefix = tuple(profile['fixtures']['prefix'])
    a, b = tuple(profile['fixtures']['suffix_a']), tuple(profile['fixtures']['suffix_b'])
    suffixes = dict(A=a, B=b, C=a+b, D=b+a)
    if len(set(suffixes.values())) != 4:
        raise ValueError('Four distinct suffixes are required')
    errors, checked, candidates, histories = [], [], {}, {}
    def check(snapshot, history, label):
        expected = oracle.forward(history)
        assert oracle.last['computed_tokens'] == len(history)
        numerical = assert_logits_close(sut.next_logits(snapshot), expected,
                                        atol=profile['atol'], rtol=profile['rtol'])
        assert sut.history(snapshot) == history
        assert sut.consumed(snapshot) == len(history)
        sut.measure(numerical, snapshot, history)
        errors.append(numerical.max_absolute_error)
        checked.append(label)
    try:
        root = sut.create(prefix)
        main_head = sut.store.fork('main', root)
        for name, suffix in suffixes.items():
            if name == 'D':
                old = sut.store.head('A')
                rewound = sut.store.move('A', old.revision, root, 'rewind-A')
                assert sut.history(candidates['A']) == histories['A']
                head = rewound
            else:
                head = sut.store.fork(name, root)
            tip = root
            check(tip, prefix, name+':0')
            for index, token in enumerate(suffix, 1):
                tip = sut.feed(tip, (token,))
                check(tip, prefix+suffix[:index], name+':'+str(index))
            candidates[name], histories[name] = tip, prefix+suffix
            sut.store.move('A' if name == 'D' else name, head.revision, tip, 'publish-'+name)
        saved = {name:sut.snapshot_bytes(tip) for name,tip in candidates.items()}
        controllers = {name:sut.controller(tip) for name,tip in candidates.items()}
        source = """
import json
row = next(row for row in json.loads(fixtures['checks']) if tuple(row['tokens']) == candidate['tokens'])
result = {'mandatory': {'valid': row['status']}, 'score': row['score'], 'unresolved': []}
"""
        fixture = json.dumps([dict(tokens=histories[n],score=s,status=v)
                    for n,s,v in zip('ABCD',(7,10,8,6),('PASS','FAIL','PASS','PASS'))]).encode()
        ids = []
        for name in 'ABCD':
            record = evaluate(sut.store,candidates[name],dict(task_root=root,
                evaluator={'name':'four-future-fixture','version':'1','source':source},
                fixtures={'checks':fixture},metric={'name':'verified-score','order':'maximize'},
                seed=17,mandatory=['valid'],timeout_seconds=5))
            ids.append(sut.store.put_evaluation(record))
        for order in (ids, list(reversed(ids)), ids[1:]+ids[:1]):
            assert choose_validated(sut.store,order,mandatory=('valid',))['snapshot_id'] == candidates['C']
        assert sut.store.get_evaluation(ids[1])['outcomes']['mandatory']['valid'] == 'FAIL'
        adopted = sut.store.move('main',main_head.revision,candidates['C'],'adopt-C')
        for tip in candidates.values():
            sut.checkpoint(tip)
        assert list(sut.checkpoint_directory.iterdir())
        sut.evict_all()
        assert not list(sut.checkpoint_directory.iterdir())
        backend.close()
        backend = BranchingModel.load_backend(profile)
        sut = BranchingModel(directory,backend,profile)
        assert sut.store.head('main') == adopted
        replayed = 0
        for name in 'CABD':
            sut.evict_all()
            before = backend._refresh_stats()
            check(candidates[name],histories[name],'cold:'+name)
            after = backend._refresh_stats()
            work = after['forward_tokens']-before['forward_tokens']
            assert work == len(histories[name])
            replayed += work
            assert after['loaded_file_bytes'] == before['loaded_file_bytes']
            assert sut.snapshot_bytes(candidates[name]) == saved[name]
            assert sut.controller(candidates[name]) == controllers[name]
        expected = [n+':'+str(i) for n,s in suffixes.items() for i in range(len(s)+1)]
        expected += ['cold:'+n for n in 'CABD']
        return dict(winner='C',candidates=candidates,failed_high_score='B',
            all_checkpoint_files_evicted=not list(sut.checkpoint_directory.iterdir()),
            cold_reopened=True,snapshots_unchanged=all(sut.snapshot_bytes(t)==saved[n] for n,t in candidates.items()),
            replayed_tokens=replayed,expected_replay_tokens=sum(map(len,histories.values())),
            cold_checked=list('CABD'),checked_boundaries=checked,expected_boundaries=expected,
            controller_before=controllers,controller_after={n:sut.controller(t) for n,t in candidates.items()},
            max_absolute_error=max(errors),execution_profile_id=profile['execution_profile_id'])
    finally:
        sut.evict_all()
        backend.close()
        oracle.close()


def main():
    with tempfile.TemporaryDirectory(prefix="ctt-demo-") as tmp:
        store = Store(tmp)
        p = (1, 3)
        contract = "toy-mean-v1;integer-inputs;exact-fractions"
        controller = {"stream_key_hex": b"demo-key".hex(), "next_draw": 0,
                      "generation_status": "idle"}
        old_states = {}
        # Toy values can be negative, whereas real token IDs are nonnegative.
        # Zigzag encode toy values at the fixture boundary only, not in the store.
        def enc(x): return 2*x if x >= 0 else -2*x-1
        def dec(x): return x//2 if x%2 == 0 else -(x+1)//2
        root = store.create_root(contract, tuple(map(enc, p)), controller)
        store.fork("main", root)
        for name, suffix in (("A", (5,)), ("B", (9,)), ("C", (-1,))):
            store.fork(name, root)
            tip = store.append(root, tuple(map(enc, suffix)), controller)
            store.move(name, 0, tip, "append"+name)
            old_states[name] = tip
            inputs = tuple(map(dec, store.lineage(tip)[1]))
            observed = attention(ToyState(inputs, b"demo-key"))
            expected = Fraction(sum(p + suffix), len(p + suffix))
            assert observed == expected
            print(f"{name}: {inputs} -> exact toy attention {observed}")
        # Rewind A and explore D while preserving original A.
        store.move("A", 1, root, "rewindA")
        new_a = store.append(root, tuple(map(enc, (7,))), controller)
        store.move("A", 2, new_a, "replaceA")
        assert tuple(map(dec, store.lineage(old_states["A"])[1])) == p+(5,)
        assert tuple(map(dec, store.lineage(new_a)[1])) == p+(7,)
        print("Rewind A -> D: old A still retained; D has exact history", p+(7,))
        # Toy evaluator deliberately does not mutate the store.
        candidates = {**old_states, "D": new_a}
        before = {name: store.head(name) for name in ("A", "B", "C", "main")}
        scores = {}
        for name, snapshot_id in candidates.items():
            x = tuple(map(dec, store.lineage(snapshot_id)[1]))
            scores[name] = -abs(attention(ToyState(x, b"demo-key")) - Fraction(2))
        assert {name: store.head(name) for name in before} == before
        winner = max(sorted(scores), key=lambda k: scores[k])
        store.move("main", 0, candidates[winner], "adoptWinner")
        print("Independent toy evaluation scores:", {k:str(v) for k,v in sorted(scores.items())})
        print(f"Winner = {winner}; main branch revision = {store.head('main').revision}")
        reopened = Store(tmp)
        history = tuple(map(dec, reopened.lineage(reopened.head("main").tip)[1]))
        print("Cold reopen preserved winning exact token history:", history)
        assert store.head("main") == reopened.head("main")
        print("TOY DEMO PASS (not a real-model, durability or COW certification)")


if __name__ == "__main__":
    main()
