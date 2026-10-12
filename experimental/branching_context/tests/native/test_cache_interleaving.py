"""Real host threads/cache/store only; no backend, GPU or model certification."""
import hashlib
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
import unittest

from context_time_travel.checkpoint_cache import CheckpointCache
from context_time_travel.real import BranchingModel, Checkpoint
from context_time_travel.store import Store


class PausingCache(CheckpointCache):
    """Pause after one real lookup retained its value and released the lock."""
    def __init__(self, source, target):
        super().__init__(source.byte_limit, source.entry_limit)
        for key in source:
            self[key] = source[key]
        self.target = target
        self.read = threading.Event()
        self.resume = threading.Event()
        self.armed = True

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if self.armed and key == self.target:
            self.armed = False
            self.read.set()
            if not self.resume.wait(30):
                raise TimeoutError('Concurrent eviction did not release the lookup')
        return value


def evict_during_transition(sut, snapshot, decoder):
    """The same schedule is used with real images in registered G3-013."""
    cache = PausingCache(sut.cache, snapshot)
    sut.cache = cache
    results = queue.Queue()

    def transition():
        try:
            results.put(('ok', sut._controller_transition(snapshot, decoder)))
        except BaseException as error:
            results.put(('error', type(error).__name__, str(error)))

    worker = threading.Thread(target=transition, name='controller-transition', daemon=True)
    worker.start()
    try:
        if not cache.read.wait(30):
            raise AssertionError('The real checkpoint lookup did not execute')
        # This is another thread, after the value has been read and before the
        # lookup returns. A separate contains/get pair loses this race.
        cache.clear()
    finally:
        cache.resume.set()
        worker.join(30)
    if worker.is_alive():
        raise AssertionError('Controller transition did not terminate')
    return results.get_nowait()


class CacheInterleavingTests(unittest.TestCase):
    def test_controller_transition_survives_concurrent_cache_eviction(self):
        with tempfile.TemporaryDirectory() as directory:
            # Exercise the actual controller-only publication method. It has
            # no numerical/backend dependency; do not construct a fake engine.
            sut = BranchingModel.__new__(BranchingModel)
            sut.store = Store(directory)
            sut.cache = CheckpointCache(4096, 16)
            root = sut.store.create_root('host-support-only', [11, 13], {'draw': 0})
            image = b'actual-owned-host-bytes'
            saved = Checkpoint('host-support-only', 'host-support-only', 2, image,
                               (0.0,), Checkpoint.digest(image, (0.0,)))
            sut.cache[root] = saved
            original = (sut.store.objects / (root + '.json')).read_bytes()
            decoder = {'draw': 37, 'scope': 'host-support-only'}
            result = evict_during_transition(sut, root, decoder)
            path = os.environ.get('CTT_INTERLEAVING_MEASUREMENTS')
            if path:
                with Path(path).open('a') as output:
                    output.write(json.dumps(dict(test=self.id(), result=result,
                        root_sha256=hashlib.sha256(original).hexdigest(),
                        root_unchanged=(sut.store.objects / (root + '.json')).read_bytes() == original,
                        remaining_cache_entries=len(sut.cache), model_forward=False,
                        gpu_used=False, real_threads=2)) + '\n')
            self.assertEqual(result[0], 'ok', 'Concurrent cache eviction broke controller publication: ' + repr(result))
            tip = result[1]
            self.assertEqual(sut.store.lineage(tip)[1], [11, 13])
            self.assertEqual(sut.store.get_snapshot(tip)['controller'], decoder)
            self.assertEqual((sut.store.objects / (root + '.json')).read_bytes(), original)
            self.assertIs(sut.cache[tip], saved)
            self.assertNotIn(root, sut.cache)


if __name__ == '__main__':
    unittest.main()
