"""CPU-only real headers and malformed admission; sparse fixtures are never loaded."""
import ctypes
import json
import msvcrt
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from gguf_reader import GGUFFile

exe, evidence = Path(sys.argv[1]), Path(sys.argv[2])
models = [
    Path('H:/CDriveEvacuation/2026-11-06-SO8T-073236/Hikari-Bonsai-27B-Nimble-LoRA-GGUF/Hikari-LoRA-fixed-scale-PQ2_0.gguf'),
    Path('H:/Strata/models/qwen38-27b-orcarouter-IQ4_XS/Qwen3.8-27B-Uncensored-IQ4_XS.gguf'),
]
results = []
def run(*args):
    return subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=30)
def check(label, args, expected, contains=''):
    p = run(*args)
    assert (p.returncode == 0) == expected and contains in (p.stdout + p.stderr), (label, p.returncode, p.stdout, p.stderr)
    results.append({'case': label, 'exit': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr})
    print('PASS', label)
check('help', ['--help'], True, 'own-native')
check('version', ['--version'], True, 'own-native')
for opt in ('--vision', '--shift', '--snapshot', '--graph', '--prompt'):
    check('unsupported '+opt, [opt], False, 'unsupported option')
check('invalid context', ['--ctx-size', '0'], False)
check('missing file', ['--model', 'missing.gguf', '--inspect'], False)
for mode in ('turbo2','q8_turbo3','spin8_turbo4','automatic'):
    check('invalid KV '+mode,['--model','missing.gguf','--kv',mode,'--inspect'],False,'unsupported KV')
check('profile incompatible with auto',['--model','missing.gguf','--kv','auto','--spin8-view','plus','--inspect'],False,'requires spin8_k8v4')
for model in models:
    check('complete file inspect '+model.name, ['--model', model, '--inspect'], True, 'layers=64')
    for mode in ('auto','f16','k8v4','spin8_k8v4','f16_turbo2','f16_turbo3','f16_turbo4'):
        check('KV CLI '+mode+' '+model.name,['--model',model,'--kv',mode,'--inspect'],True,'layers=64')

g = GGUFFile(models[0])
with models[0].open('rb') as f:
    header = f.read(g.data_start)
with tempfile.TemporaryDirectory(prefix='dense-own-cpu-', dir=evidence) as temp:
    def string(value):
        raw=value.encode();return struct.pack('<Q',len(raw))+raw
    def entry(key, ty, value):
        return string(key)+struct.pack('<I',ty)+value
    for array in (False,True):
        key='prism.hadamard.sign_values' if array else 'prism.hadamard.version'
        for reverse in (False,True):
            values=[1,2] if not reverse else [2,1]
            data=b'GGUF'+struct.pack('<IQQ',3,0,2)
            for value in values:
                payload=struct.pack('<IQi',5,1,value) if array else struct.pack('<I',value)
                data+=entry(key,9 if array else 4,payload)
            duplicate=Path(temp)/'duplicate.gguf';duplicate.write_bytes(data)
            check(f'duplicate {key} reverse={reverse}', ['--model',duplicate,'--inspect'],False,'duplicate metadata key')
    # Append malformed metadata before the existing tensor directory. No
    # payload is needed: each case must reject before tensor extent admission.
    first=string(g.tensors[0].name)
    directory=header.rindex(first)
    count=struct.unpack_from('<Q',header,16)[0]
    block_key=string('qwen35.block_count')
    block_value=header.index(block_key)+len(block_key)+4
    for nextn, blocks, entries, expected in ((0xffffffff,63,63,'zero or one'),(2,66,64,'zero or one'),(0,64,63,'bad recurrent map'),(1,65,64,'bad recurrent map')):
        prefix=bytearray(header[:directory]);struct.pack_into('<Q',prefix,16,count+2);struct.pack_into('<I',prefix,block_value,blocks)
        extra=entry('qwen35.nextn_predict_layers',4,struct.pack('<I',nextn))
        mapping=bytes(int((i+1)%4!=0) for i in range(entries))
        extra+=entry('qwen35.attention.recurrent_layers',9,struct.pack('<IQ',7,entries)+mapping)
        raw=prefix+extra+header[directory:];raw+=bytes((-len(raw))%g.alignment)
        malformed=Path(temp)/'malformed-map.gguf';malformed.write_bytes(raw)
        check(f'MTP {nextn} recurrent length {entries}', ['--model',malformed,'--inspect'],False,expected)
    path = Path(temp) / 'synthetic-sparse.gguf'
    with path.open('w+b') as f:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        call = kernel.DeviceIoControl
        call.argtypes = [ctypes.c_void_p,ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong,ctypes.POINTER(ctypes.c_ulong),ctypes.c_void_p]
        call.restype = ctypes.c_int
        returned = ctypes.c_ulong()
        if not call(msvcrt.get_osfhandle(f.fileno()), 0x900C4, None, 0, None, 0, ctypes.byref(returned), None):
            raise ctypes.WinError(ctypes.get_last_error())
        f.write(header)
        f.truncate(models[0].stat().st_size)
    check('synthetic metadata only', ['--model', path, '--inspect'], True, 'folded=401')
    def mutate(label, off, replacement, contains=''):
        with path.open('r+b') as f:
            f.seek(off); original=f.read(len(replacement));f.seek(off);f.write(replacement)
        try:
            check(label, ['--model', path, '--inspect'], False, contains)
        finally:
            with path.open('r+b') as f:
                f.seek(off);f.write(original)
    def marker(name):
        raw=name.encode();needle=struct.pack('<Q',len(raw))+raw
        # folded metadata string arrays can contain tensor names as well: the
        # directory is the final occurrence after all metadata arrays.
        return header.rindex(needle)+len(needle)
    for name, value in [('qwen35.block_count',65),('qwen35.embedding_length',5121),('qwen35.ssm.group_count',15),('prism.hadamard.block_size',512)]:
        off=marker(name);assert struct.unpack_from('<I',header,off)[0]==4
        mutate('wrong '+name,off+4,struct.pack('<I',value))
    off=marker('prism.hadamard.sign_values');assert struct.unpack_from('<II',header,off)==(9,5)
    mutate('invalid explicit sign',off+16,struct.pack('<i',0),'sign must')
    off=marker('blk.0.ssm_conv1d.weight');rank=struct.unpack_from('<I',header,off)[0]
    mutate('conv compressed forbidden',off+4+rank*8,struct.pack('<I',142),'unsupported operation')
    off=marker('blk.0.ffn_gate.weight');rank=struct.unpack_from('<I',header,off)[0]
    mutate('wrong FFN dimensions',off+4,struct.pack('<Q',5121),'geometry')
    mutate('unknown matrix type',off+4+rank*8,struct.pack('<I',999),'unsupported operation')
    mutate('overlapping tensor extent',off+8+rank*8,struct.pack('<Q',0),'overlapping')
    mutate('out of file extent',off+8+rank*8,struct.pack('<Q',models[0].stat().st_size),'extent')
    # Stop before weights: cannot claim sparse bytes are an inference model.
    with path.open('r+b') as f:f.truncate(len(header))
    check('truncated full header', ['--model',path,'--inspect'],False,'extent')
evidence.mkdir(parents=True,exist_ok=True)
(evidence/'cpu-tests.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(f'PASS {len(results)} CPU CLI/admission cases; no CUDA model loading')
