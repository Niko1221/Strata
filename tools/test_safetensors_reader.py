"""Adversarial fixtures for the native C++ reader; no GPU/model/download needed."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

EXE = None


def tensor(dtype="U8", shape=None, begin=0, end=4):
    return {"dtype": dtype, "shape": [4] if shape is None else shape, "data_offsets": [begin, end]}


class Reader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="safetensors-测试-")
        self.root = Path(self.tmp.name)
        self.shard = self.root / "权重.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, header=None, data=b"\x00\x01\x80\xff", raw=None, index=None):
        header = {"x": tensor()} if header is None else header
        raw = json.dumps(header).encode() if raw is None else raw
        self.shard.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
        index = {"weight_map": {k: self.shard.name for k in header if k != "__metadata__"}} if index is None else index
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    def run_reader(self, mode="headers", *args, ok=True, error=None):
        p = subprocess.run([str(EXE), mode, str(self.root), *map(str,args)], capture_output=True, encoding="utf-8")
        self.assertEqual(p.returncode == 0, ok, p.stdout + p.stderr)
        j = json.loads(p.stdout if ok else p.stderr)
        if error:
            self.assertIn(error, j["error"])
        return j

    def test_unicode_scalar_empty_and_unaligned(self):
        self.fixture({"x": tensor("F32", [], 0, 4), "empty": tensor("BF16", [0,7], 4, 4),
                      "__metadata__": {"format": "pt"}})
        j = self.run_reader()
        self.assertEqual(j["tensor_count"], 2)
        self.assertEqual(sum(j["io"]["source_read_bytes"].values()), 0)
        self.assertEqual(self.run_reader("read", "x", 1, 3)["hex"], "0180ff")

    def test_malformed_descriptors(self):
        cases = [
            (tensor(shape=[-1]), "unsigned integer"),
            (tensor(shape=[4.0]), "unsigned integer"),
            (tensor(shape=[True]), "unsigned integer"),
            (tensor(shape=[2**63,8]), "overflow"),
            (tensor(shape=[2**64]), "unsigned integer"),
            (tensor(shape=[4]*17), "rank"),
            (tensor(dtype="FP4_UNKNOWN"), "dtype"),
            (tensor(dtype="BF16"), "byte mismatch"),
            (tensor(begin=3,end=2), "bounds"),
            (tensor(end=5), "bounds"),
            (tensor(begin=-1), "unsigned integer"),
            (tensor(end=4.0), "unsigned integer"),
        ]
        for t, err in cases:
            with self.subTest(t=t):
                self.fixture({"x":t})
                self.run_reader(ok=False,error=err)

    def test_duplicate_keys_and_bad_json(self):
        t = json.dumps(tensor())
        cases = [
            ('{"x":'+t+',"x":'+t+'}', "duplicate JSON key"),
            ('{"x":{"dtype":"U8","dtype":"U8","shape":[4],"data_offsets":[0,4]}}', "duplicate JSON key"),
            ('{"x":'+t+',"__metadata__":{"a":"x","a":"y"}}', "duplicate JSON key"),
            (' '+json.dumps({"x":tensor()}), "begin with object"),
            (json.dumps({"x":tensor()})+'x', "parse error"),
            ('{"x":'+t+',"__metadata__":{"a":2}}', "non-string"),
            ('{"x":'+t+',"__metadata__":[]}', "string map"),
        ]
        for text, err in cases:
            with self.subTest(text=text):
                self.fixture(raw=text.encode())
                self.run_reader(ok=False,error=err)
        self.fixture(raw=b'{"x":"\xff"}')
        self.run_reader(ok=False,error="UTF-8")

    def test_header_limit_and_truncation(self):
        for raw in [b"", b"1234567", struct.pack("<Q",2**64-1), struct.pack("<Q",100000001),
                    struct.pack("<Q",0), struct.pack("<Q",100)+b"{}"]:
            with self.subTest(raw=raw):
                self.fixture()
                self.shard.write_bytes(raw)
                self.run_reader(ok=False)

    def test_nesting_limit(self):
        self.fixture(raw=b'{"x":'+b'['*65+b'0'+b']'*65+b'}')
        self.run_reader(ok=False,error="nesting exceeds")

    def test_overlap_hole_trailing(self):
        for header, data in [
            ({"x":tensor(),"y":tensor()}, b"1234"),
            ({"x":tensor(begin=1,end=5)}, b"12345"),
            ({"x":tensor()}, b"12345"),
        ]:
            with self.subTest(header=header):
                self.fixture(header,data)
                self.run_reader(ok=False)

    def test_index_mismatch(self):
        self.fixture(index={"weight_map":{"missing":self.shard.name}})
        self.run_reader(ok=False,error="mismatch")
        self.fixture(index={"weight_map":{"x":self.shard.name,"missing":self.shard.name}})
        self.run_reader(ok=False,error="missing tensors")
        self.fixture()
        index = self.root / "model.safetensors.index.json"
        index.write_text('{"weight_map":{"x":"权重.safetensors","x":"权重.safetensors"}}',encoding="utf-8")
        self.run_reader(ok=False,error="duplicate JSON key")

    def test_path_traversal_absolute_and_ads(self):
        for name in ["../权重.safetensors", "..\\权重.safetensors", "/x.safetensors", "C:\\x.safetensors",
                     "x:y.safetensors", "\\\\host\\x.safetensors", "x\0.safetensors"]:
            with self.subTest(name=name):
                self.fixture(index={"weight_map":{"x":name}})
                self.run_reader(ok=False,error="unsafe model filename")

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as external:
            self.fixture()
            target = Path(external)/"outside.safetensors"
            target.write_bytes(self.shard.read_bytes())
            self.shard.unlink()
            try:
                self.shard.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlink privilege unavailable: {exc}")
            self.run_reader(ok=False,error="escapes root")

    def test_unindexed_file(self):
        self.fixture()
        (self.root/"extra.safetensors").write_bytes(self.shard.read_bytes())
        self.run_reader(ok=False,error="unindexed safetensors file")

    def test_read_range(self):
        self.fixture()
        self.run_reader("read","x",3,2,ok=False,error="out of bounds")
        self.run_reader("read","x",2**64-1,1,ok=False,error="out of bounds")
        self.assertEqual(self.run_reader("read","x",4,0)["hex"],"")

    def test_64_bit_seek(self):
        size=2**32+257
        self.fixture({"x":tensor(shape=[size],end=size)},data=b"")
        with self.shard.open("r+b") as f:
            if os.name == "nt":
                import ctypes
                import msvcrt
                from ctypes import wintypes
                fn=ctypes.windll.kernel32.DeviceIoControl
                fn.argtypes=[wintypes.HANDLE,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,
                             ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD),ctypes.c_void_p]
                returned=wintypes.DWORD()
                self.assertTrue(fn(msvcrt.get_osfhandle(f.fileno()),0x900C4,None,0,None,0,ctypes.byref(returned),None))
            header_bytes=struct.unpack("<Q",f.read(8))[0]
            f.truncate(8+header_bytes+size)
            f.seek(8+header_bytes+size-4)
            f.write(b"ABCD")
        self.assertEqual(self.run_reader("read","x",size-4,4)["hex"],"41424344")

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--exe",required=True,type=Path)
    args,rest=p.parse_known_args()
    EXE=args.exe.resolve()
    unittest.main(argv=[__file__,*rest])
