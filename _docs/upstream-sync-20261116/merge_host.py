"""Task2: normalized diff3 with only the explicit DeepSeek retirement reversed."""
from pathlib import Path
import subprocess
import tempfile
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('H:/Strata/evidence/upstream-sync-20261116/host-merge')
BASE = 'f33c8c30588ff00e24b6b63bad0bf6a14da19081'
FORK = '6f13c7555a7d50289f0e3587ab780627c65a8000'
UP = '09b603886925994d0d4461857b01321ef3aa431c'
RETIRE = ['b47591e74', 'efa9d6dbe']

def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, timeout=60)

def blob(ref, path):
    r = git('show', f'{ref}:{path}')
    return r.stdout.replace(b'\r\n', b'\n') if r.returncode == 0 else b''

def merge(ours, base, theirs):
    if ours == base: return theirs, 0
    if theirs == base or ours == theirs: return ours, 0
    with tempfile.TemporaryDirectory(dir=OUT) as folder:
        paths = [Path(folder) / x for x in ['fork', 'base', 'upstream-retained']]
        for p, data in zip(paths, [ours, base, theirs]): p.write_bytes(data)
        r = git('merge-file', '-p', '--diff3', '-L', 'fork', '-L', 'base', '-L', 'upstream-retained', *map(str, paths))
        if not 0 <= r.returncode <= 127: raise RuntimeError(r.stderr)
        return r.stdout, r.returncode

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    changed = set(git('diff', '--name-only', BASE, UP, '--', 'apps/host', 'apps/desktop/src-tauri/src').stdout.decode().splitlines())
    changed |= set(git('diff', '--name-only', '--diff-filter=U', '--', 'apps/host').stdout.decode().splitlines())
    changed -= {'apps/host/package.json', 'apps/host/package-lock.json'}
    retirement = {c:set(git('diff', '--name-only', c+'^', c, '--', 'apps/host').stdout.decode().splitlines()) for c in RETIRE}
    rows=[]
    for path in sorted(changed):
        if len(sys.argv)>1 and path<sys.argv[1]: continue
        upstream=blob(UP,path)
        for c in RETIRE:
            if path in retirement[c]:
                current=upstream
                upstream, conflicts=merge(current,blob(c,path),blob(c+'^',path))
                if conflicts and path == 'apps/host/src/components/GuidedSetup.tsx':
                    # Retirement removed an adjacent import; retain the later own-path fit fix.
                    import re
                    upstream=re.sub(rb'<<<<<<< fork\n.*?\n\|\|\|\|\|\|\| base\n.*?\n=======\n.*?\n>>>>>>> upstream-retained\n',b"import { deepseekContexts } from '../lib/deepseek-context';\nimport { contextCacheGb } from '../lib/native-fit';\n",upstream,flags=re.S)
                    conflicts=0
                if conflicts and path == 'apps/host/src/components/NativeModels.tsx':
                    import re
                    upstream=re.sub(rb'<<<<<<< fork\n.*?\n\|\|\|\|\|\|\| base\n.*?\n=======\n.*?\n>>>>>>> upstream-retained\n',b"import { cardRoomGb, contextCacheGb, fitOn, memoryNeeded, versionFit } from '../lib/native-fit';\nimport { contextIsTight, defaultContext, DEFAULT_CONTEXT, OLD_DEEPSEEK_CONTEXT, openingContext } from '../lib/default-context';\n",upstream,flags=re.S)
                    conflicts=0
                if conflicts and path == 'apps/host/src/lib/native-fit.ts':
                    import re
                    upstream=re.sub(rb'<<<<<<< fork\n(.*?)\|\|\|\|\|\|\| base\n.*?\n=======\n.*?\n>>>>>>> upstream-retained\n',lambda m:m[1],upstream,flags=re.S)
                    conflicts=0
                if conflicts and path == 'apps/host/src/lib/native.ts':
                    import re
                    upstream=re.sub(rb'<<<<<<< fork\n.*?\|\|\|\|\|\|\| base\n.*?\n=======\n(.*?)>>>>>>> upstream-retained\n',lambda m:m[1].replace(b'installed:boolean;',b"context_fit?:import('./native-fit').ContextFit|null;installed:boolean;"),upstream,flags=re.S)
                    conflicts=0
                if conflicts and path == 'apps/host/tests/ui-r3.test.mjs':
                    import re
                    upstream=re.sub(rb'<<<<<<< fork\n(.*?)\|\|\|\|\|\|\| base\n.*?\n=======\n(.*?)>>>>>>> upstream-retained\n',lambda m:m[2].splitlines(keepends=True)[0]+m[1].splitlines(keepends=True)[1],upstream,flags=re.S)
                    conflicts=0
                if conflicts:
                    conflict_path=OUT/('reverse-'+path.replace('/','__'))
                    conflict_path.write_bytes(upstream)
                    raise RuntimeError(f'reverse retirement conflict {c} {path}')
        result, conflicts=merge(blob(FORK,path),blob(BASE,path),upstream)
        review=OUT / path
        review.parent.mkdir(parents=True,exist_ok=True)
        review.write_bytes(result)
        target=ROOT/path
        if result:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(result)
        elif target.exists():
            target.unlink()
        rows.append({'path':path,'conflicts':conflicts,'bytes':len(result)})
    (OUT/'manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps([r for r in rows if r['conflicts']],indent=2))
    print('files',len(rows))

if __name__ == '__main__':
    try: main()
    except Exception:
        import traceback
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
