"""Real process termination around reference publication and format migration."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

from context_time_travel.store import InvalidOperation, NotFound, Store, _json_bytes


WORKER = r'''
import os, pathlib, signal, sys, time
from context_time_travel.store import InvalidOperation, Store
directory, action, target, point = sys.argv[1:]
def kill():
    os.kill(os.getpid(), signal.SIGKILL)
def fault(step):
    if step == point:
        kill()
if action == 'migration':
    replace = os.replace
    def interrupted_replace(source, destination):
        selected = pathlib.Path(destination).name == target
        if selected and point == 'before':
            kill()
        replace(source, destination)
        if selected and point == 'after':
            kill()
    os.replace = interrupted_replace
store = Store(directory, fault_hook=fault)
if action == 'fork':
    store.fork('crashed', target)
elif action == 'move':
    store.move('main', 0, target, 'attempt')
elif action == 'race':
    while not (pathlib.Path(directory) / 'go').exists():
        time.sleep(0.001)
    try:
        store.move(point, 0, target, 'global_operation')
    except InvalidOperation:
        sys.exit(74)
'''


class ReferenceCrashTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def record(self, **row):
        path = os.environ.get('CTT_LOGICAL_WORK_MEASUREMENTS')
        if path:
            with open(path, 'a') as output:
                output.write(json.dumps(dict(test=self.id(), **row)) + '\n')

    def objects(self, store):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in store.objects.glob('*.json')}

    def fixture(self, name):
        store = Store(Path(self.tmp.name) / name)
        root = store.create_root('crash-only', [1, 2], {'draw': 0})
        child = store.append(root, [3], {'draw': 1})
        return store, root, child

    def terminate(self, store, action, target, point):
        result = subprocess.run([sys.executable, '-c', WORKER, str(store.root),
                                 action, target, point], capture_output=True, text=True, timeout=30)
        self.record(action=action, target=target, point=point, returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr)
        self.assertEqual(result.returncode, -signal.SIGKILL, result.stderr)

    def test_fork_kill_keeps_head_and_event_together(self):
        for point, committed in [('after_refs_stage_fsync', False),
                                 ('after_refs_rename_before_dir_fsync', True)]:
            with self.subTest(point=point):
                store, root, _ = self.fixture(point)
                objects = self.objects(store)
                self.terminate(store, 'fork', root, point)
                reopened = Store(store.root)
                if committed:
                    head = reopened.head('crashed')
                    self.assertEqual((head.tip, head.root, head.revision), (root, root, 0))
                else:
                    with self.assertRaises(NotFound):
                        reopened.head('crashed')
                expected = [dict(type='fork', branch='crashed', tip=root)] if committed else []
                self.assertEqual(reopened.events(), expected)
                reopened.fork('later', root)
                self.assertEqual(reopened.events(), expected + [dict(type='fork', branch='later', tip=root)])
                self.assertEqual(self.objects(reopened), objects)

    def test_move_kill_keeps_receipt_head_and_event_together(self):
        for point, committed in [('before_refs_stage', False), ('after_refs_stage_fsync', False),
                                 ('after_refs_rename_before_dir_fsync', True)]:
            with self.subTest(point=point):
                store, root, child = self.fixture(point)
                old = store.fork('main', root)
                sibling = store.fork('sibling', root)
                objects = self.objects(store)
                self.terminate(store, 'move', child, point)
                reopened = Store(store.root)
                head = reopened.head('main')
                self.assertEqual((head.tip, head.revision), (child, 1) if committed else (root, 0))
                self.assertEqual(reopened.head('sibling'), sibling)
                self.assertEqual(len(reopened.events()), 3 if committed else 2)
                receipt = reopened.move('main', old.revision, child, 'attempt')
                self.assertEqual((receipt.tip, receipt.revision), (child, 1))
                self.assertEqual(reopened.move('main', old.revision, child, 'attempt'), receipt)
                self.assertEqual(len(reopened.events()), 3)
                reopened.move('main', 1, root, 'rewind')
                self.assertEqual(Store(store.root).move('main', 0, child, 'attempt'), receipt)
                self.assertEqual(Store(store.root).head('main').revision, 2)
                self.assertEqual(self.objects(reopened), objects)

    def test_migration_kill_restarts_without_losing_history_or_receipts(self):
        for target in ('alpha.json', 'ref-order.json', 'refs.json'):
            for point in ('before', 'after'):
                with self.subTest(target=target, point=point):
                    store, root, child = self.fixture(target + '-' + point)
                    request = dict(type='move', name='alpha', expected_revision=0, target=child)
                    result = dict(name='alpha', revision=1, tip=child, root=root)
                    events = [dict(type='fork', branch='alpha', tip=root),
                              dict(type='fork', branch='beta', tip=root),
                              dict(type='move', branch='alpha', operation_id='legacy_op',
                                   **{'from': root, 'to': child, 'from_revision': 0, 'to_revision': 1})]
                    legacy = dict(schema=1,
                                  branches={'alpha': dict(revision=1, tip=child, root=root),
                                            'beta': dict(revision=0, tip=root, root=root)},
                                  receipts={'legacy_op': dict(digest=hashlib.sha256(_json_bytes(request)).hexdigest(),
                                                              result=result)}, events=events)
                    store.refs_path.write_bytes(_json_bytes(legacy))
                    objects = self.objects(store)
                    self.terminate(store, 'migration', target, point)
                    reopened = Store(store.root)
                    head = reopened.head('alpha')
                    self.assertEqual(head.__dict__, result)
                    self.assertEqual(reopened.move('alpha', 0, child, 'legacy_op'), head)
                    self.assertEqual(reopened.head('beta').tip, root)
                    self.assertEqual(reopened.events(), events)
                    reopened.fork('later', child)
                    self.assertEqual(reopened.events(), events + [dict(type='fork', branch='later', tip=child)])
                    self.assertEqual(self.objects(reopened), objects)

    def test_operation_id_is_global_across_racing_branch_records(self):
        store, root, child = self.fixture('race')
        for name in ('alpha', 'beta'):
            store.fork(name, root)
        workers = [subprocess.Popen([sys.executable, '-c', WORKER, str(store.root), 'race', child, name],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                   for name in ('alpha', 'beta')]
        (store.root / 'go').touch()
        try:
            outputs = [worker.communicate(timeout=30) for worker in workers]
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait()
        codes = [worker.returncode for worker in workers]
        self.record(action='global-operation-race', returncodes=codes, outputs=outputs)
        self.assertEqual(sorted(codes), [0, 74], outputs)
        reopened = Store(store.root)
        winner = ('alpha', 'beta')[codes.index(0)]
        loser = ('alpha', 'beta')[codes.index(74)]
        committed = reopened.head(winner)
        self.assertEqual((committed.tip, committed.revision), (child, 1))
        self.assertEqual(reopened.move(winner, 0, child, 'global_operation'), committed)
        with self.assertRaises(InvalidOperation):
            reopened.move(loser, 0, child, 'global_operation')
        self.assertEqual((reopened.head(loser).tip, reopened.head(loser).revision), (root, 0))
        self.assertEqual(len(reopened.events()), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
