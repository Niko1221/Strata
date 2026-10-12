"""G5-007 checks the declared owner-only plaintext policy, not encryption."""
import os
import stat
import subprocess
import unittest

import test_access_control as access


class SensitiveStorageTests(unittest.TestCase):
    setUpClass=classmethod(access.AccessControlTests.setUpClass.__func__)
    tearDownClass=classmethod(access.AccessControlTests.tearDownClass.__func__)
    setUp=access.AccessControlTests.setUp
    check=access.AccessControlTests.check

    def test_g5_007_encrypted_sensitive_persistence(self):
        candidate=self.sut.feed(self.root,self.A)
        saved=self.sut.snapshot_bytes(candidate)
        checkpoint=self.sut.checkpoint(candidate)
        self.assertGreater(checkpoint['payload_path'].stat().st_size,0)
        policy=self.sut.store.persistence_policy
        self.assertEqual(policy['kind'],'owner-only-plaintext')
        self.assertFalse(policy['encrypted'])
        paths=[self.sut.store.root,*self.sut.store.root.rglob('*')]
        for path in paths:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o700 if path.is_dir() else 0o600,str(path))
        uid=subprocess.run(['sudo','-n','-u','nobody','id','-u'],capture_output=True,text=True,check=True)
        self.assertNotEqual(int(uid.stdout),os.geteuid())
        for path in (checkpoint['payload_path'],checkpoint['manifest_path'],
                     self.sut.store.objects/(candidate+'.json')):
            code='from pathlib import Path; Path('+repr(str(path))+').read_bytes()'
            denied=subprocess.run(['sudo','-n','-u','nobody','/usr/bin/python3','-c',code],
                                  capture_output=True,text=True)
            self.assertNotEqual(denied.returncode,0)
            self.assertIn('PermissionError',denied.stderr)
        self.assertEqual(self.sut.snapshot_bytes(candidate),saved)
        self.check(candidate,self.P+self.A)


if __name__ == '__main__':
    unittest.main(verbosity=2)
