"""Build unchanged upstream main and PR 4347 with the same native toolchain."""
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[2]
ROOT=REPO/'build-native-baseline'
ZONE=timezone(timedelta(hours=8))
MAIN='5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d'
CANDIDATE='7be0a3a17aeea0a7c0bf7bc63ae56c70e53311d4'
ARMS={
    'main':{'revision':MAIN,'source':ROOT/'src-main-5e8a4cca','build':ROOT/'build-main-5e8a4cca-cuda132-sm120'},
    'pr4347':{'revision':CANDIDATE,'source':ROOT/'src-pr4347-7be0a3a1','build':ROOT/'build-pr4347-7be0a3a1-cuda132-sm120'},
}
MODEL = Path(os.environ["STRATA_REPORT_MODEL"])

def stamp():return datetime.now(ZONE).isoformat()
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(record):
    p=OUT/'build-summary.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,indent=2),encoding='utf-8');tmp.replace(p)

def main():
    if (OUT/'build-summary.json').exists():raise RuntimeError('Build evidence exists; do not overwrite')
    spec=importlib.util.spec_from_file_location('native_tools',REPO/'reports/tobeo-20261022/windows-native-baseline/build_native.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    python=Path(sys.executable).parent
    os.environ['PATH']=os.pathsep.join([str(python/'Scripts'),str(python/'Library/bin'),os.environ['PATH']])
    env,cmake,vs=helper.compiler_environment()
    cl=shutil.which('cl.exe',path=env['PATH']);ctest=str(Path(cmake).with_name('ctest.exe'))
    if not cl:raise RuntimeError('No MSVC compiler')
    record={'started':stamp(),'passed':False,'main':MAIN,'candidate':CANDIDATE,'author':'rynwang',
            'toolchain':{'cuda':str(helper.CUDA),'msvc':cl,'vs':vs,'cmake':cmake,'ggml_revision':helper.GGML_REVISION},
            'model':{'path':str(MODEL),'bytes':MODEL.stat().st_size,'sha256':digest(MODEL)},'arms':{}}
    save(record)
    tests=[*helper.HOST_TESTS,'model_family_test','qwen35_sampling_test']
    try:
      for name,arm in ARMS.items():
        src,build=arm['source'],arm['build']
        revision=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
        changes=subprocess.check_output(['git','-C',str(src),'status','--porcelain','--untracked-files=no'],text=True)
        if revision!=arm['revision'] or changes:raise RuntimeError('Unexpected source revision or edits')
        row={'revision':revision,'source':str(src),'build':str(build),'commands':[],'passed':False}
        record['arms'][name]=row;save(record)
        commands=[
          [cmake,'-S',str(src),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
           '-DSTRATA_ENABLE_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=120',
           '-DCUDAToolkit_ROOT='+helper.CUDA.as_posix(),'-DCMAKE_CUDA_COMPILER='+(helper.CUDA/'bin/nvcc.exe').as_posix(),
           '-DCMAKE_C_COMPILER='+Path(cl).as_posix(),'-DCMAKE_CXX_COMPILER='+Path(cl).as_posix(),
           '-DCMAKE_CUDA_HOST_COMPILER='+Path(cl).as_posix(),
           '-DSTRATA_GGML_DIR='+(helper.SOURCE/'third_party/llama.cpp').as_posix(),
           '-DSTRATA_BUILD_TESTS=ON','-DSTRATA_BUILD_CONVERSATION_TESTS=ON','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
          [cmake,'--build',str(build),'--target','strata-q36','strata_kq',*tests,'--parallel','8'],
          [ctest,'--test-dir',str(build),'--output-on-failure','-R','^('+'|'.join(tests)+')$'],
        ]
        with (OUT/(name+'.build.log')).open('w',encoding='utf-8') as log:
          for command in commands:
            print('RUN',name,subprocess.list2cmdline(command),flush=True)
            t=time.monotonic();run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            log.flush();row['commands'].append({'command':command,'exit_code':run.returncode,'seconds':time.monotonic()-t});save(record)
            if run.returncode:raise RuntimeError('Build/check failed: '+name)
        row['binaries']={f:digest(build/f) for f in ['strata-q36.exe','strata_kq.dll']}
        row['tracked_changes_after']=subprocess.check_output(['git','-C',str(src),'status','--porcelain','--untracked-files=no'],text=True)
        if row['tracked_changes_after']:raise RuntimeError('Tracked source changed during build')
        row['passed']=True;save(record)
      record['passed']=True
    except BaseException as exc:
      record['error']=repr(exc);raise
    finally:
      record['finished']=stamp();save(record)

if __name__=='__main__':main()
