"""Task2 locale key merges and reviewed overlapping test/type edits."""
from resolve_host import ROOT,PAT,resolve
import json

def native_types(i,o,b,t):
    return o.replace('installed:boolean;',"context_fit?:import('./native-fit').ContextFit|null;installed:boolean;").replace('visionPrepared?:boolean;','visionPrepared?:boolean;visionChosen?:boolean;').replace('uma?:boolean;','uma?:boolean;dedicated_gb?:number;')

resolve('apps/host/src/lib/native.ts',native_types)
for path in (ROOT/'apps/host/src/locales').glob('*.json'):
    text=path.read_text(encoding='utf-8')
    if not PAT.search(text):continue
    variants=[json.loads(PAT.sub(lambda m:m[j],text)) for j in [1,2,3]]
    o,b,t=variants
    merged={**t,**o}
    for k in merged:
        if k in o and k in b and o[k]==b[k] and k in t:merged[k]=t[k]
    path.write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(path.name,len(merged))
for path in ['helpers/assert-private.mjs','model-directory.test.mjs','qa1-storage.test.mjs','request-flow.test.mjs','ui-do.test.mjs']:
    # Retain stricter NTFS ACE checks, restored-timestamp integrity regression,
    # dynamic fork VERSION and already normalized source/hardware test fixtures.
    resolve('apps/host/tests/'+path,lambda i,o,b,t:o)
resolve('apps/host/tests/overlay-place.test.mjs',lambda i,o,b,t:t)
