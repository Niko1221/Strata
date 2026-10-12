"""Real POSIX permission probes with an independent OS principal."""
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest

import test_e01
from context_time_travel.store import Store, _stage


class SensitiveStorageTests(unittest.TestCase):
    def test_p02_invariant(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory)
            parent.chmod(0o755)  # The store itself must establish privacy.
            old=os.umask(0)
            try:
                store=Store(parent/'private')
                self.assertEqual(stat.S_IMODE(store.root.stat().st_mode),0o700,
                                 'Private storage must not depend on caller umask')
                root=store.create_root('sensitive',[101,102],{'secret':'controller'})
                store.fork('main',root)
                store.grant(root,'reader',('read',))
                identity=store.put_evaluation(test_e01.record(root,root))
                stage=_stage(store.objects,b'sensitive interrupted stage')
                for path in store.root.rglob('*'):
                    expected=0o700 if path.is_dir() else 0o600
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode),expected,str(path))
                self.assertEqual(store.persistence_policy,
                                 {'kind':'owner-only-plaintext','encrypted':False,'directories':0o700,'files':0o600})
                denied=[store.objects/(root+'.json'),store.root/'evaluations'/(identity+'.json'),stage]
                uid=subprocess.run(['sudo','-n','-u','nobody','id','-u'],capture_output=True,text=True,check=True)
                self.assertNotEqual(int(uid.stdout),os.geteuid())
                for path in denied:
                    probe='from pathlib import Path; Path('+repr(str(path))+').read_bytes()'
                    result=subprocess.run(['sudo','-n','-u','nobody','/usr/bin/python3','-c',probe],capture_output=True,text=True)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('PermissionError',result.stderr)
                stage.unlink()
                self.assertEqual(Store(store.root).get_snapshot(root),store.get_snapshot(root))
            finally:
                os.umask(old)


if __name__ == '__main__':
    unittest.main(verbosity=2)
