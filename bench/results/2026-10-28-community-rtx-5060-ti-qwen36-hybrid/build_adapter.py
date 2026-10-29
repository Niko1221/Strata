"""Link the file-argument entry against a previously built MSVC/Ninja q36.

Run from an x64 VS2022 Developer Command Prompt. Does not rebuild engine code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess

HERE=Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--cuda',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True,help='New directory for object, build log and hashes')
    a=p.parse_args();a.evidence.mkdir(parents=True,exist_ok=False)
    build=a.build_dir.resolve();source=a.source_dir.resolve()
    ninja=(build/'build.ninja').read_text(encoding='utf-8')
    block=ninja.split('build strata-q36.exe:',1)[1].split('\n\n',1)[0]
    libs=[s.strip('"') for s in shlex.split(re.search(r'^  LINK_LIBRARIES = (.*)$',block,re.M)[1],posix=False)]
    exe=build/'strata-q36-report-file.exe'
    if exe.exists():raise RuntimeError('Report adapter already exists; preserve it before another build')
    paths=[build/'strata-q36.exe',build/'strata_kq.dll']
    paths += [((build/Path(s)).resolve()) for s in libs if Path(s).suffix=='.lib' and (build/Path(s)).exists()]
    before={str(path):digest(path) for path in paths}
    cl=shutil.which('cl.exe');link=shutil.which('link.exe')
    if not cl or not link:raise RuntimeError('Use an x64 VS2022 Developer Command Prompt')
    obj=a.evidence.resolve()/'file_entry.obj'
    commands=[
        [cl,'/nologo','/c','/O2','/EHsc','/std:c++20','/MD','/I'+str(source/'include'),str(HERE/'file_entry.cpp'),'/Fo'+str(obj)],
        [link,'/nologo','/OUT:'+str(exe),'/MACHINE:X64','/INCREMENTAL:NO','/SUBSYSTEM:CONSOLE','/OPT:REF','/OPT:ICF',
         '/LIBPATH:'+str(a.cuda.resolve()/'lib/x64'),'/MANIFEST:EMBED',
         '/MANIFESTINPUT:'+str(source/'cmake/strata-utf8.manifest'),str(obj),*libs],
    ]
    with (a.evidence/'build.log').open('w',encoding='utf-8') as log:
        for command in commands:subprocess.run(command,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert all(digest(Path(path))==sha for path,sha in before.items()),'Original engine artifact changed'
    result={'adapter':str(exe),'adapter_sha256':digest(exe),'original_artifacts_unchanged':True,
            'original_artifacts_sha256':before,'entry_sha256':digest(HERE/'file_entry.cpp')}
    (a.evidence/'build.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(exe)


if __name__=='__main__':main()
