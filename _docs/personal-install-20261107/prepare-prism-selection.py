"""Hash the moved GGUF and prepare a private, non-executable single-profile migration plan."""
from __future__ import annotations
import argparse
import json
import subprocess
import winreg
from common import *

PRIVATE=Path('H:/Strata/private-backups/prism041-user-selection-20261107')

def automatic_environment() -> dict:
    values={'Process':os.environ.get('STRATA_LLAMA_SERVER'),'User':None,'Machine':None}
    for label,hive,key in [('User',winreg.HKEY_CURRENT_USER,'Environment'),('Machine',winreg.HKEY_LOCAL_MACHINE,r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment')]:
        try:
            with winreg.OpenKey(hive,key) as handle:values[label]=winreg.QueryValueEx(handle,'STRATA_LLAMA_SERVER')[0]
        except FileNotFoundError:pass
    return values

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',type=Path,default=USER_GGUF);args=parser.parse_args()
    model=args.model.resolve()
    require_resources();require(not PRIVATE.exists(),'Fresh private selection-plan directory required')
    before_env=automatic_environment();before_model=model.stat()
    require(before_model.st_size==5629111584 and sha(model)==USER_GGUF_SHA,'Moved GGUF identity differs')
    require(model.stat().st_mtime_ns==before_model.st_mtime_ns,'GGUF changed while hashing')
    preview=json.loads(subprocess.check_output([str(NODE),str(HERE/'inspect-prism-model.mjs'),str(model)],cwd=ROOT,text=True,encoding='utf-8',timeout=90))
    require(preview['architecture']=='qwen35' and preview['layerCount']==33 and preview['tensorBlockCount']==32
        and preview['minTensorBlock']==0 and preview['maxTensorBlock']==31 and preview['recurrentMapLength']==33,'Expected unmodified Qwen3.5 trunk/map metadata')
    sys.path.insert(0,str(ROOT));from terminal.strata_control import private_files
    PRIVATE.mkdir();private_files.restrict_to_current_user(PRIVATE,directory=True)
    records=[];registry=None
    data=Path('C:/Users/downl/.strata/data')
    for name in ['state.json','local-models.json']:
        path=data/name;content=path.read_bytes();value=hashlib.sha256(content).hexdigest()
        target=PRIVATE/name
        with target.open('xb') as stream:stream.write(content)
        require(sha(path)==value and sha(target)==value,'User state changed during read-only snapshot')
        records.append({'source':str(path),'backup':str(target),'sha256':value})
        if name=='local-models.json':registry=json.loads(content)
    matching=[row for row in registry.get('models',[]) if Path(row.get('local',{}).get('path','')).name.casefold()==USER_GGUF.name.casefold()]
    plan={'preparedOnly':True,'executable':False,'parentReviewRequired':True,'applyAfterInstallOnly':True,
        'target':dict(preview,sha256=USER_GGUF_SHA),'handoffPath':str(USER_GGUF),'observedPathDiffers':model!=USER_GGUF.resolve(),
        'targetCandidates':matching,'ambiguousPreviousReference':len(matching)>1,
        'snapshot':records,'automaticEnvironmentBefore':before_env,'automaticEnvironmentMustRemainExact':True,
        'globalEnvironmentMutationAllowed':False,'genericPointerMutationAllowed':False,
        'allowedNormalProfileChanges':['only the selected target file/path-derived variant mapping','only that target profile runtime.backend to PrismML'],
        'preserveOtherProfileFields':True,'preserveOtherProfiles':True,'preserveModelBytes':True,
        'beforeApply':['fresh idle/user-data backup','capture strata.run-profiles.v1 privately from the actual WebView','record the chosen target profile IDs and previous path/backend/default mapping','hash the moved GGUF again','validate exact signed composite/runtime hashes'],
        'normalApi':{'inspect':'POST /api/local-models/inspect','registerReference':'POST /api/local-models','validate':'POST /api/engine/validate','applyCurrentConfig':'PATCH /api/engine','runtimePatch':{'expert':[{'id':'runtime.backend','value':'PrismML'}]}},
        'profilePersistence':{'kind':'normal Saved settings UI','storageKey':'strata.run-profiles.v1','serverProfileEndpointExists':False,
            'note':'PATCH /api/engine changes current engine config, not an arbitrary per-model profile. Use the normal target-model Saved settings flow and preserve all unrelated entries.'},
        'restoreMappingRequired':['old target variant/path','old target profile config and default association','prior current engine config'],
        'restoreCondition':'parent-reviewed rollback only; do not overwrite concurrent user edits',
        'e2eOnlySettings':{'context':98304,'k':'q8_0','v':'q8_0','flashAttention':True,'parallel':1},
        'qualificationLimits':{'ordinaryModelUseQualified':True,'fullWindowQualified':False,'generalQualityQualified':False,'explicitRealFileMtpNegative':'blocked in inherited handoff'},
        'runtime':{'choice':'PrismML','version':'0.4.1','manifestSHA256':PRISM_MANIFEST_SHA,'entrypointSHA256':PRISM_EXE_SHA,'sourceCommit':PRISM_SOURCE}}
    require(automatic_environment()==before_env,'Automatic environment changed during preparation')
    write_new(PRIVATE/'selection-plan.json',plan)
    write_new(OUT/'prism041-r4/selection-plan.json',{'ok':True,'preparedOnly':True,'applied':False,'modelSHA256':USER_GGUF_SHA,'modelBytes':before_model.st_size,
        'runtimeVersion':'0.4.1','manifestSHA256':PRISM_MANIFEST_SHA,'candidateReferenceCount':len(matching),'ambiguousPreviousReference':len(matching)>1,'modelPathChangedSinceHandoff':model!=USER_GGUF.resolve(),
        'privatePlan':str(PRIVATE/'selection-plan.json'),'privatePlanSHA256':sha(PRIVATE/'selection-plan.json'),
        'userSnapshots':len(records),'automaticEnvironmentUnchanged':True,'automaticUserOverridePresent':before_env['User'] is not None,'normalProfileFieldsChanged':0,'modelFilesCopied':0})
    print(json.dumps({'ok':True,'applied':False,'modelSHA256':USER_GGUF_SHA,'candidateReferenceCount':len(matching),'automaticEnvironmentUnchanged':True,'normalProfileFieldsChanged':0}))

if __name__=='__main__':main()
