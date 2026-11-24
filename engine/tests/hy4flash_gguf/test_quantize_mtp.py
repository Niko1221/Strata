"""The converter must not mistake a metadata array for a tensor directory entry."""
import ctypes
import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import sys
import json

import numpy as np
import pytest


def test_directory_offsets_ignore_metadata_array_headers(tmp_path):
    path=Path(__file__).resolve().parents[3]/'tools/hy4_flash/quantize_mtp.py'
    spec=importlib.util.spec_from_file_location('quantize_mtp',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def string(s):
        raw=s.encode();return struct.pack('<Q',len(raw))+raw
    header=b'GGUF'+struct.pack('<IQQ',3,1,2)
    header+=string('general.architecture')+struct.pack('<I',8)+string('hy_v4_flash')
    header+=string('tokenizer.ggml.token_type')+struct.pack('<IIQ',9,4,3)+struct.pack('<III',1,3,4)
    header+=string('blk.42.nextn.eh_proj.weight')+struct.pack('<IQQ',2,32,2)
    expected=len(header)
    header+=struct.pack('<IQ',30,0)
    file=tmp_path/'tiny.gguf'
    file.write_bytes(header+b'\0'*(-len(header)%32)+b'\0'*128)
    directory=module.Directory(file)
    assert directory.directory_positions==[expected]
    assert directory.tensors[0].type_id==30
    assert directory.metadata['tokenizer.ggml.token_type']==[1,3,4]


def test_conversion_preserves_split_and_non_mtp_bytes(tmp_path):
    root=Path(__file__).resolve().parents[3]
    library=Path(os.environ.get('GGML_BASE_LIBRARY',root.parent/'llama.cpp/build/bin/libggml-base.so'))
    if not library.is_file():pytest.skip('llama.cpp quantizer library unavailable')
    sys.path.insert(0,str(root/'terminal'/'tools'))
    from gguf_reader import GGUFFile
    def string(s):
        raw=s.encode();return struct.pack('<Q',len(raw))+raw
    def write(path,number,tensors):
        kv=[('split.no',2,struct.pack('<H',number)),('split.count',2,struct.pack('<H',2)),
            ('split.tensors.count',5,struct.pack('<i',4))]
        if not number:
            kv += [('general.architecture',8,string('hy_v4_flash')),('hy_v4_flash.block_count',4,struct.pack('<I',43)),
                   ('hy_v4_flash.nextn_predict_layers',4,struct.pack('<I',1)),
                   ('tokenizer.ggml.token_type',9,struct.pack('<IQIII',4,3,1,3,4))]
        header=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(kv))
        header+=b''.join(string(k)+struct.pack('<I',kind)+value for k,kind,value in kv)
        payload=b''
        for name,shape,kind,data in tensors:
            header+=string(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*shape)+struct.pack('<IQ',kind,len(payload))
            payload+=data;payload+=b'\0'*(-len(payload)%32)
        path.write_bytes(header+b'\0'*(-len(header)%32)+payload)
    source=tmp_path/'source';source.mkdir()
    first=source/'tiny-00001-of-00002.gguf';second=source/'tiny-00002-of-00002.gguf'
    floats=np.linspace(-1,1,64,dtype=np.float32)
    bf16=(floats.view(np.uint32)>>16).astype('<u2').tobytes()
    write(first,0,[('blk.0.weight',[32,2],30,bf16),('blk.42.nextn.eh_proj.weight',[32,2],30,bf16),
                   ('blk.42.nextn.enorm.weight',[32],0,b'\0'*128)])
    write(second,1,[('per_layer_token_embd.weight',[96,2],43,bytes(range(96))*2)])
    out=tmp_path/'output'
    command=[sys.executable,str(root/'tools/hy4_flash/quantize_mtp.py'),'--gguf',str(first),
             '--out-dir',str(out),'--ggml-library',str(library),'--threads','2']
    subprocess.run(command,check=True,capture_output=True)
    original,converted=GGUFFile(first),GGUFFile(out/first.name)
    assert original.metadata==converted.metadata
    assert [(t.name,t.type_id) for t in converted.tensors]==[
        ('blk.0.weight',30),('blk.42.nextn.eh_proj.weight',8),('blk.42.nextn.enorm.weight',0)]
    quantize = ctypes.CDLL(str(library.resolve())).quantize_row_q8_0_ref
    quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    quantize.restype = None
    source_values = (np.frombuffer(bf16, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
    expected = np.empty(2 * 34, dtype=np.uint8)
    quantize(source_values.ctypes.data, expected.ctypes.data, source_values.size)
    with (out / first.name).open('rb') as fh:
        fh.seek(converted.data_start + converted.tensors[1].offset)
        assert fh.read(expected.nbytes) == expected.tobytes()
    assert second.read_bytes()==(out/second.name).read_bytes()
    manifest=json.loads((out/'mtp-quantization.json').read_text())
    unchanged=manifest['tensors'][0]
    assert unchanged['sha256']==unchanged['source_sha256']
    before=(out/first.name).read_bytes()
    assert subprocess.run(command,capture_output=True).returncode!=0
    assert (out/first.name).read_bytes()==before
