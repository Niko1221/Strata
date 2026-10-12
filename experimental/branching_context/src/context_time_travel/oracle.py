"""Fresh full-history oracle subprocess, with no checkpoint or store interface."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def FreshOracle(profile):
    kind = profile.get('backend','llama.cpp')
    if kind == 'strata':
        from .strata_backend import StrataFreshOracle
        return StrataFreshOracle(profile)
    if kind != 'llama.cpp':
        raise ValueError('Unsupported fresh oracle profile')
    return LlamaFreshOracle(profile)


class LlamaFreshOracle:
    def __init__(self, profile):
        self.profile = profile
        self.log = tempfile.TemporaryFile()
        self.process = subprocess.Popen([profile['oracle_binary'], profile['model_path']],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        self.requests = 0
        self.tokens_computed = 0

    def _read(self, n):
        data = bytearray()
        while len(data) < n:
            part = self.process.stdout.read(n-len(data))
            if not part:
                self.log.seek(0)
                raise RuntimeError('Oracle terminated: '+self.log.read().decode(errors='replace')[-5000:])
            data.extend(part)
        return bytes(data)

    def tokenize(self, text):
        if '\n' in text:
            raise ValueError('Token fixture text must be one line')
        self.process.stdin.write(('T '+text+'\n').encode())
        self.process.stdin.flush()
        n, = struct.unpack('<i', self._read(4))
        return struct.unpack('<'+str(n)+'i', self._read(n*4))

    def forward(self, tokens, *, chunk=2048):
        ids = tuple(tokens)
        if not ids or len(ids)>2048 or any(type(t) is not int or t<0 for t in ids):
            raise ValueError('Invalid oracle token history')
        self.process.stdin.write(('F '+str(chunk)+' '+str(len(ids))+' '+' '.join(map(str,ids))+'\n').encode())
        self.process.stdin.flush()
        n, work, position = struct.unpack('<iQi', self._read(16))
        if work != len(ids) or position != len(ids)-1:
            raise AssertionError('Independent oracle did not compute every input')
        raw = self._read(4*n)
        self.requests += 1
        self.tokens_computed += work
        self.last = dict(tokens=ids, computed_tokens=work, position_max=position,
                         logits_sha256=hashlib.sha256(raw).hexdigest())
        return struct.unpack('<'+str(n)+'f', raw)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=30)
        log_path = self.profile.get('oracle_log')
        if log_path:
            self.log.seek(0)
            Path(log_path).write_bytes(self.log.read())
        self.process.stdout.close()
        self.log.close()
