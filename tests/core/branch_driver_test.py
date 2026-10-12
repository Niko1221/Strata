"""Real model protocol test. Requires an actual built Strata executable and model."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class BranchDriverTest(unittest.TestCase):
    def test_consumed_frontier_and_direct_work(self):
        command = json.loads(os.environ['STRATA_BRANCH_COMMAND'])
        with open(os.environ['STRATA_BRANCH_STDERR'], 'w') as log:
            transcript = []
            p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=log, text=True, bufsize=1)
            def request(line):
                p.stdin.write(line + '\n')
                p.stdin.flush()
                result = p.stdout.readline()
                transcript.append(dict(command=line, response=result))
                self.assertTrue(result, 'native process ended; inspect stderr')
                return json.loads(result)
            try:
                ready = json.loads(p.stdout.readline())
                transcript.append(dict(command='startup', response=ready))
                self.assertEqual(ready['status'], 'ready')
                self.assertEqual(request('capture')['status'], 'error')
                self.assertEqual(request('feed 1 2 3')['consumed'], 3)
                root = request('capture')['handle']
                work = request('stats')
                self.assertEqual(work['forward_tokens'], 3)
                self.assertEqual(request('feed 4')['consumed'], 4)
                a = request('capture')['handle']
                self.assertEqual(request('restore ' + str(root))['consumed'], 3)
                restored = request('stats')
                self.assertEqual(restored['forward_tokens'], 4, 'restore recomputed a prefix')
                self.assertEqual(request('feed 5')['consumed'], 4)
                self.assertEqual(request('restore ' + str(a))['consumed'], 4)
                self.assertEqual(request('stats')['forward_tokens'], 5)
                self.assertEqual(request('restore 999999')['status'], 'error')
                self.assertEqual(request('stats')['consumed'], 4)
                before_reset = request('stats')['forward_tokens']
                self.assertEqual(request('reset')['consumed'], 0)
                self.assertEqual(request('capture')['status'], 'error')
                self.assertEqual(request('feed 8 9')['consumed'], 2)
                self.assertEqual(request('restore ' + str(a))['consumed'], 4)
                self.assertEqual(request('stats')['forward_tokens'], before_reset + 2)
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory)/'native.state'
                    frontier = Path(directory)/'frontier.f32'
                    ids = Path(directory)/'ids.txt'
                    ids.write_text('1 2 3 4')
                    self.assertEqual(request('logits '+str(frontier))['status'], 'ok')
                    original_logits = frontier.read_bytes()
                    saved = request('write '+str(checkpoint))
                    self.assertEqual(saved['status'], 'ok')
                    self.assertEqual(saved['bytes'], checkpoint.stat().st_size)
                    self.assertGreater(saved['bytes'], 1024*1024)
                    self.assertEqual(request('reset')['consumed'], 0)
                    load = 'load '+str(checkpoint)+' '+str(ids)+' '+str(frontier)
                    self.assertEqual(request(load)['consumed'], 4)
                    self.assertEqual(request('stats')['forward_tokens'], before_reset + 2)
                    self.assertEqual(request('feed 6')['consumed'], 5)
                    self.assertEqual(request('restore '+str(a))['consumed'], 4)
                    self.assertEqual(request('logits '+str(frontier))['status'], 'ok')
                    self.assertEqual(frontier.read_bytes(), original_logits)
                    raw = bytearray(checkpoint.read_bytes())
                    raw[-1] ^= 1
                    checkpoint.write_bytes(raw)
                    self.assertEqual(request(load)['status'], 'error')
                    self.assertEqual(request('stats')['consumed'], 4)
                    self.assertEqual(request('logits '+str(frontier))['status'], 'ok')
                    self.assertEqual(frontier.read_bytes(), original_logits)
                    self.assertGreater(request('stats')['written_file_bytes'], 0)
                    self.assertGreater(request('stats')['loaded_file_bytes'], 0)
                self.assertEqual(request('quit')['status'], 'closed')
                self.assertEqual(p.wait(timeout=30), 0)
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait()
                p.stdin.close()
                p.stdout.close()
                Path(os.environ['STRATA_BRANCH_STDERR'] + '.transcript.json').write_text(json.dumps(transcript, indent=2))


if __name__ == '__main__':
    unittest.main()
