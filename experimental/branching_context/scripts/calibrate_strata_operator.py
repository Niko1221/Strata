"""Fresh-process repeatability of the pinned, single-token Q8 operator.

This is a separate execution profile. The rejected batched-prefill measurements
remain untouched; they are not a source of tolerances for this profile.
"""
import argparse
import array
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from calibration_inputs import execution_settings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-root', type=Path, required=True)
parser.add_argument('--model-hash-receipt', type=Path)
parser.add_argument('--oracle-source-revision', required=True)
parser.add_argument('--admission', type=Path)
parser.add_argument('--destination', type=Path)
args = parser.parse_args()
ROOT = args.native_root.resolve()
DESTINATION = (args.destination or ROOT/'build/strata-q8-operator-calibration.json').resolve()
OUT = ROOT/('build/q8-operator-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S'))
OUT.mkdir()
if DESTINATION.exists():
    raise SystemExit('Frozen operator profile exists; refusing to overwrite')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


admission_path = (args.admission or ROOT/'build/q8-dual-engine-admission.json').resolve()
admission = json.loads(admission_path.read_text())
settings = execution_settings(admission)
if settings['capacity'] != 4096:
    raise SystemExit('The predeclared calibration matrix requires capacity 4096')
command = list(admission['oracle_command'])
ORACLE = Path(command[0]).resolve(strict=True)
if sha(ORACLE) != admission.get('oracle_binary_sha256'):
    raise SystemExit('Oracle differs from the executed admission receipt')
record = dict(status='RUNNING', pid=os.getpid(), started=time.time(), command=command,
    oracle_sha256=sha(ORACLE), oracle_source_revision=args.oracle_source_revision,
    operator='Verifier.run(T=1); successful recurrent commit; next-token frontier',
    oracle_scope='new process and empty session per full history; no checkpoint/handle/store input',
    capacity=settings['capacity'], weight_format='Q8_0', kv_format=settings['kv_format'], model_files={}, commands=[], comparisons=[],
    expert_cache_slots=settings['expert_cache_slots'], resource_admission=str(admission_path),
    resource_admission_sha256=sha(admission_path),
    atol=None, rtol=0, tolerance_rule='2 * worst repeated fresh-process absolute error; rtol=0',
    calibration_lengths=[1,31,127,255,256,257,511,1024,4096], repetitions=3,
    rejected_batched_profile=str(ROOT/'build/q8-batched-calibration-rejected.json'),
    environment={k:v for k,v in os.environ.items() if k.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_'))},
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total',
                                      '--format=csv,noheader'],text=True).strip())


def progress():
    temporary = OUT/'progress.tmp'
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(OUT/'progress.json')


def forward(ids, repeat):
    # A simple serial GPU queue: a hold is observed only BETWEEN finished
    # requests. Thus a test job never changes measured request latency.
    if (ROOT/'build/operator-calibration.hold').exists():
        record['status'] = 'WAITING_FOR_GPU'
        progress()
        while (ROOT/'build/operator-calibration.hold').exists():
            time.sleep(1)
        record['status'] = 'RUNNING'
    prefix = OUT/f'n{len(ids)}-repeat{repeat}'
    token_path = prefix.with_suffix('.tokens')
    logits_path = prefix.with_suffix('.f32')
    token_path.write_text(','.join(map(str,ids)))
    actual_command = command + ['--tokens-file',str(token_path)]
    env = dict(os.environ, STRATA_DUMP_FIRST_LOGITS=str(logits_path))
    if sha(ORACLE) != record['oracle_sha256']:
        raise RuntimeError('Pinned oracle changed')
    for name, item in record['model_files'].items():
        stat = Path(name).stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
                item['bytes'], item['mtime_ns'], item['ctime_ns']):
            raise RuntimeError('Model or pack changed during calibration: ' + name)
    environment = {k: v for k, v in os.environ.items()
                   if k.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_'))}
    if environment != record['environment']:
        raise RuntimeError('Execution environment changed during calibration')
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                        '--format=csv,noheader'], text=True).strip()
    if hardware != record['hardware']:
        raise RuntimeError('Execution hardware changed during calibration')
    start = time.monotonic()
    with prefix.with_suffix('.stdout').open('w') as out, prefix.with_suffix('.stderr').open('w') as err:
        result = subprocess.run(actual_command,cwd=ROOT,env=env,stdout=out,stderr=err,timeout=18000)
    row = dict(command=actual_command,length=len(ids),repeat=repeat,exit_code=result.returncode,
               seconds=time.monotonic()-start,stdout=str(prefix.with_suffix('.stdout')),
               stderr=str(prefix.with_suffix('.stderr')))
    record['commands'].append(row)
    progress()
    if result.returncode:
        raise RuntimeError('Fresh process failed: '+str(prefix))
    work = json.loads(prefix.with_suffix('.stdout').read_text())
    row['native_counters'] = work
    if work['forward_tokens'] != len(ids) or work['consumed'] != len(ids) or work['position_max'] != len(ids)-1:
        raise RuntimeError('Incomplete or repeated prefix computation')
    raw = logits_path.read_bytes()
    values = array.array('f',raw)
    if len(values) != 248320 or not all(map(math.isfinite,values)) or work['logits_bytes'] != len(raw):
        raise RuntimeError('Invalid full-vocabulary frontier')
    row.update(logits_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    progress()
    print(json.dumps(row),flush=True)
    return values


try:
    prior_path = args.model_hash_receipt or ROOT/'build/q8-calibration/progress.json'
    prior = json.loads(prior_path.read_text())
    record['prior_executed_hash_run'] = str(prior_path.resolve())
    for name, previous in prior['model_files'].items():
        path = Path(name)
        stat = path.stat()
        unchanged = previous['bytes'] == stat.st_size and max(stat.st_mtime,stat.st_ctime) < prior['started']
        actual = previous['sha256'] if unchanged else sha(path)
        if actual != previous['sha256']:
            raise RuntimeError('Model changed since inspected Q8 candidate: '+name)
        record['model_files'][name] = dict(bytes=stat.st_size,sha256=actual,mtime_ns=stat.st_mtime_ns,
            ctime_ns=stat.st_ctime_ns,hash_reused_from_executed_run=unchanged)
    progress()
    for length in record['calibration_lengths']:
        ids = [100+i*17 % 137 for i in range(length)]
        reference = forward(ids,0)
        for repeat in range(1,record['repetitions']):
            actual = forward(ids,repeat)
            record['comparisons'].append(dict(length=length,repeat=repeat,exact_token_ids=ids,
                max_absolute_error=max(abs(a-b) for a,b in zip(reference,actual))))
            progress()
    worst = max(c['max_absolute_error'] for c in record['comparisons'])
    record.update(status='FROZEN',baseline_max_absolute_error=worst,atol=2*worst,finished=time.time())
    with DESTINATION.open('x') as output:
        json.dump(record,output,indent=2)
        output.write('\n')
    progress()
except BaseException as error:
    record.update(status='FAIL',error=repr(error),finished=time.time())
    progress()
    raise
