"""Preserve only the authorized main-tree PowerShell module-analysis cache."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import json
import tempfile
from common import *

SOURCE=ROOT/'Microsoft'
DEST=Path('H:/Strata/private-backups/main-powershell-cache-20261107')
EXPECTED='Windows/PowerShell/ModuleAnalysisCache'
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
kernel.CreateFileW.restype=wintypes.HANDLE
kernel.CloseHandle.argtypes=[wintypes.HANDLE]

def sharing_check(path: Path) -> None:
    # Allow readers, refuse any current writer. No process enumeration/open-file
    # crawl and no attempt to terminate a process holding the cache.
    handle=kernel.CreateFileW(str(path),0x80000000,1,None,3,0x80,None)
    require(handle not in (None,ctypes.c_void_p(-1).value),'Cache has an active writer or is inaccessible')
    kernel.CloseHandle(handle)

def preserve(source: Path,target: Path) -> dict:
    before=inventory(source);require(set(before)=={EXPECTED},'Unexpected content in cache tree')
    file=source/EXPECTED;require(file.read_bytes().startswith(b'PSMODULECACHE'),'Not a PowerShell module cache')
    sharing_check(file);mtime=file.stat().st_mtime_ns
    require(not target.exists(),'Preserve prior cache archive')
    shutil.copytree(source,target,copy_function=shutil.copy2)
    require(inventory(target)==before and inventory(source)==before and file.stat().st_mtime_ns==mtime,'Cache changed during preservation')
    sharing_check(file)
    # Only this recorded file is removed; unknown newly appearing entries survive.
    file.unlink()
    for directory in [file.parent,file.parent.parent,source]:directory.rmdir()
    return {'ok':True,'files':1,'bytes':(target/EXPECTED).stat().st_size,'sha256':before[EXPECTED],'sourceRemoved':not source.exists()}

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        root=Path(tempfile.mkdtemp(prefix='cache-sharing-test-',dir=OUT/'temp'))
        source=root/'source';(source/'Windows/PowerShell').mkdir(parents=True)
        file=source/EXPECTED;file.write_bytes(b'PSMODULECACHE synthetic fixture')
        with file.open('r+b'):
            try:sharing_check(file)
            except RuntimeError:pass
            else:raise RuntimeError('Sharing check did not refuse a writer')
        result=preserve(source,root/'preserved');require(result['ok'],'Cache fixture preservation failed')
        print(json.dumps({'ok':True,'sharingRefusal':True,'copyHashAndRemoval':True,'realCacheTouched':False}));return
    if not args.execute:print(json.dumps({'wouldPreserve':False,'cacheExists':SOURCE.exists()}));return
    require_resources();require(not DEST.exists(),'Fresh protected cache destination required')
    os.environ['PSModuleAnalysisCachePath']=str(OUT/'temp/PSModuleAnalysisCache')
    DEST.mkdir();sys.path.insert(0,str(ROOT));from terminal.strata_control import private_files
    private_files.restrict_to_current_user(DEST,directory=True)
    result=preserve(SOURCE,DEST/'Microsoft')
    write_new(DEST/'receipt.json',dict(result,source=str(SOURCE),destination=str(DEST/'Microsoft')))
    write_new(OUT/'main-cache-preserved.json',dict(result,receiptSHA256=sha(DEST/'receipt.json')))
    print(json.dumps(result))

if __name__=='__main__':main()
