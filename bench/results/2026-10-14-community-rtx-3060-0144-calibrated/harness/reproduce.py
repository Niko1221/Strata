"""Run the frozen reduced suite in a new directory using existing model assets."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime',type=Path,required=True)
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
assert not args.out.exists(),'Choose a new output directory; prior results are preserved'
args.out.mkdir(parents=True)
(args.out/'fixed-configs').mkdir()
def restore(value):
    if isinstance(value,str):
        return value.replace('<STRATA>',str(args.runtime.resolve())).replace('<DATA>',str(args.data.resolve())).replace('<LOCAL_ROOT>',str(args.runtime.resolve().parent))
    if isinstance(value,list):return [restore(item) for item in value]
    if isinstance(value,dict):return {key:restore(item) for key,item in value.items()}
    return value
for file in (HERE/'fixed-configs').glob('*.json'):
    obj=restore(json.loads(file.read_text(encoding='utf-8')))
    (args.out/'fixed-configs'/file.name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
shutil.copy2(HERE/'previous-runner.py',args.out/'previous-runner.py')
subprocess.run([sys.executable,str(HERE/'community_bench_0144.py'),'--runtime',str(args.runtime),
    '--data',str(args.data),'--out',str(args.out),'--soak-seconds','900'],check=True)
