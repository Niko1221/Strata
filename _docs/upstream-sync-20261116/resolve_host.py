"""Explicit semantic resolutions of Task2's normalized three-way conflicts."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]
PAT=re.compile(r'^<<<<<<< fork\n(.*?)^\|\|\|\|\|\|\| base\n(.*?)^=======\n(.*?)^>>>>>>> upstream-retained\n',re.M|re.S)

def resolve(path, fn):
    p=ROOT/path
    text=p.read_text(encoding='utf-8')
    count=0
    def sub(m):
        nonlocal count
        result=fn(count,*m.groups());count+=1
        return result
    text=PAT.sub(sub,text)
    p.write_text(text,encoding='utf-8',newline='\n')
    print(path,count)

def native(i,o,b,t):
    if i==0:return o.replace('sourceServerEnvironment }','sourceServerEnvironment, UPDATE_LOCK_HELD }')
    if i==1:return o.replace('appUpdater=null,','appUpdater=null,lockWaitMs=60000,lockPollMs=2000,')
    if i==2:return o.replace("j=>j.state==='running'&&j.kind==='update-strata'","j=>closedForAppUpdate(j)&&j.kind==='update-strata'")
    if i==3:return o.replace('visionEnabled:cfg.ui_vision_enabled','visionChosen:cfg.ui_vision_chosen===true,visionEnabled:cfg.ui_vision_enabled')
    if i==4:return o.replace('arch:config.arch??null,backend:', 'arch:config.arch??null,vision:!!config.vision,backend:')
    if i==5:return o.replace('await preflightInstall(args);const result=', 'await preflightInstall(args);args=await controlDrafter(args);const result=')
    if i==6:return o.replace("'yes':args.vision}","'yes':args.vision,ui_vision_chosen:true}")
    if i==7:
        # captureStart validates the untouched source before any repair or mutation.
        # Existing fork admission later in this function verifies the actual selected engine.
        return o
    if i==8:return o.replace("if(job.state==='running')job.state='cancelled';", "if(job.state==='running'&&!(job.kind==='update-strata'&&typeof job.appUpdateFrom==='string'))job.state='cancelled';")
    raise ValueError(i)

def runner(i,o,b,t):
    fix=t[t.index('/* Run from PowerShell'):t.index('/* A test that never')]
    return o.replace('const requested=process.env.',fix+'const requested=process.env.',1)

def settings(i,o,b,t):
    o=o.replace("${tr('Speed')}","${tr(switchGroup(sw))}")
    row=t[t.index('    const switchRow ='):t.index('    return <details')]
    o=o.replace('    const memoryOffered',row+'    const memoryOffered')
    o=o.replace('rows.some(f => f.group === group) ?', 'rows.some(f => f.group === group) || switchRows.some(sw => switchGroup(sw) === group) ?')
    o=o.replace('</label>))}</section> : null))}', '</label>))}{tr(switchRows.filter(sw => switchGroup(sw) === group).map(switchRow))}</section> : null))}')
    start=o.index('{tr(switchRows.length ?')
    end=o.index('<section className="native-settings-group"><h3 className="h4">{tr("Sampling")}',start)
    ts=t.index('{tr(switchRows.some(')
    te=t.index('<section className="native-settings-group"><h3 className="h4">{tr("Sampling")}',ts)
    return o[:start]+t[ts:te]+o[end:]

if __name__=='__main__':
    resolve('apps/host/server/native-control.mjs',native)
    resolve('apps/host/scripts/test.mjs',runner)
    resolve('apps/host/server/run-config-check.mjs',lambda i,o,b,t:o.replace("'ui_vision_enabled',","'ui_vision_enabled', 'ui_vision_chosen',"))
    resolve('apps/host/src/components/NativeModels.tsx',lambda i,o,b,t:t.replace('contextIsTight(', 'ordinaryKv.length > 0 && ordinaryKv.includes(kv) && contextIsTight('))
    resolve('apps/host/src/components/NativeSettings.tsx',settings)
