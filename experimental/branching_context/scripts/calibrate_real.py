"""Freeze disjoint fresh-context baseline tolerances before branch tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from context_time_travel.oracle import FreshOracle


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    args = p.parse_args()
    destination = ROOT/'build/real-profile.json'
    if destination.exists():
        raise SystemExit('Frozen profile exists; refusing to overwrite/rebaseline')
    build = json.loads((ROOT/'build/build.json').read_text())
    profile = dict(model_path=args.model, model_weights_sha256=sha(args.model),
        backend_revision=build['backend_revision'],
        oracle_binary=str(ROOT/'build/fresh-oracle'),bridge_library=str(ROOT/'build/libctt.so'),
        oracle_log=str(ROOT/'evidence/runs/calibration-oracle.log'),
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
        platform=platform.platform(), native_build=build,
        execution=dict(n_ctx=2048,n_batch=2048,n_ubatch=2048,n_threads=2,n_gpu_layers=99,
                       flash_attention=False,kv_dtype='F32',positions='absolute zero-based',adapters=[],
                       cuda_cublas_compute_type=os.environ.get('GGML_CUDA_CUBLAS_COMPUTE_TYPE','auto'),
                       nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE','unset')))
    oracle = FreshOracle(profile)
    rows = []
    try:
        base = oracle.tokenize('Independent numerical calibration: weather, arithmetic, music, and ordinary language vary across these fixed input tokens.')
        for n in (1,31,127,255,256,257,511,1024):
            ids = (base*((n+len(base)-1)//len(base)))[:n]
            fresh = oracle.forward(ids)
            for chunk in (1,31,2048):
                other = oracle.forward(ids,chunk=chunk)
                maximum = max(abs(a-b) for a,b in zip(other,fresh))
                rows.append(dict(length=n,chunk=chunk,max_absolute_error=maximum,
                    tokens=list(ids),reference_sha256=hashlib.sha256(__import__('struct').pack('<'+str(len(fresh))+'f',*fresh)).hexdigest()))
        worst = max(row['max_absolute_error'] for row in rows)
        # A fixed 2x baseline margin, selected before any branch result exists.
        profile.update(atol=2*worst,rtol=0,baseline_max_absolute_error=worst,
                       tolerance_rule='atol=2*max fresh full-batch versus fresh chunked baseline error; rtol=0',
                       baseline=rows,oracle_baseline_requests=oracle.requests,
                       oracle_baseline_tokens_computed=oracle.tokens_computed)
        prefix = oracle.tokenize('A saved conversation describes a small garden with red flowers and a wooden gate. Two visitors choose different routes through the garden.')
        suffix_a = oracle.tokenize(' The first visitor turns left.')
        suffix_b = oracle.tokenize(' The second visitor turns right.')
        profile['fixtures'] = dict(prefix=list(prefix),suffix_a=list(suffix_a),suffix_b=list(suffix_b))
        profile['token_fixtures_sha256'] = hashlib.sha256(json.dumps(profile['fixtures'],sort_keys=True).encode()).hexdigest()
        profile['model_contract_id'] = hashlib.sha256(json.dumps(dict(model=profile['model_weights_sha256'],positions='absolute zero-based',adapters=[]),sort_keys=True).encode()).hexdigest()
        profile['execution_profile_id'] = hashlib.sha256(json.dumps(dict(build=build,execution=profile['execution'],hardware=profile['hardware']),sort_keys=True).encode()).hexdigest()
    finally:
        oracle.close()
    destination.write_text(json.dumps(profile,indent=2)+'\n')
    print(json.dumps(dict(profile=str(destination),atol=profile['atol'],baseline_max=worst,
        requests=oracle.requests,computed_tokens=oracle.tokens_computed)))


if __name__ == '__main__':
    main()
