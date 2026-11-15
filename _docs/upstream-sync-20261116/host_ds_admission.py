"""Apply the reviewed Host dedicated-runtime boundary after the baseline suite."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
def edit(name,old,new,count=1):
 p=root/name
 s=p.read_text(encoding='utf-8')
 assert s.count(old)==count,(name,old[:100],s.count(old))
 p.write_text(s.replace(old,new),encoding='utf-8',newline='\n')
p='apps/host/server/native-control.mjs'
edit(p,"import {missingFilesError,missingNames}","import {deepseekRuntime,DEEPSEEK_RUNTIME_UNAVAILABLE} from './deepseek-runtime.mjs';\nimport {missingFilesError,missingNames}")
edit(p,'return {...entry,servingPolicy,','const support=deepseekRuntime(cfg,entry.configPath??join(repo(),entry.config),{family:servingPolicy.id,missing:entry.engineIssues??[]});\n  const runtimeSupport=support?{available:support.available,reason:support.reason,executionCompatibility:support.executionCompatibility}:undefined;\n  return {...entry,servingPolicy,runtimeSupport,')
edit(p,"picturesMissing:m.component_status!=='present'&&onlyPictures(m),missing:","picturesMissing:m.component_status!=='present'&&onlyPictures(m),engineIssues:(m.missing??[]).filter(p=>p?.part==='engine'),missing:")
edit(p,' function captureStart(args,runPath=null){',' function admitDedicatedRuntime(model,config,path=null){\n  const entry=snapshot.installed?.models?.find(m=>m.model===model);\n  const support=deepseekRuntime(config,path??entry?.configPath??join(repo(),entry?.config??`strata-${model}.json`),{family:policyFor(model,config).id,missing:entry?.engineIssues??[]});\n  if(support&&!support.available)throw new Error(support.reason);\n  return support;\n }\n function captureStart(args,runPath=null){')
edit(p,'  return {...found,text,config};','  admitDedicatedRuntime(args.model,next,found.path);\n  return {...found,text,config};')
edit(p,"  if(packaged){const engine=await control.engine();if(!engine.usable)","  const dedicated=admitDedicatedRuntime(args.model,config);\n  if(dedicated)exe=dedicated.exe;\n  else if(packaged){const engine=await control.engine();if(!engine.usable)")
edit(p,'  const next=applyOrdinaryKv(applyServingPatch(config,validateNativeSettings(args.values??{})),args.kv,policy);','  const next=applyOrdinaryKv(applyServingPatch(config,validateNativeSettings(args.values??{})),args.kv,policy);')
edit(p,'  return {model:existing?.model,policy};',"  if(policy.id==='deepseek41'){\n   if(existing)admitDedicatedRuntime(existing.model,next);\n   else if(packaged&&size?.runtimeUnavailableReason)throw new Error(size.runtimeUnavailableReason);\n  }\n  return {model:existing?.model,policy};")
edit(p,'const markImages=c=>markPictures(c,visionReader);snapshot=', 'const runtimeCatalog=await dedicatedCatalog(catalog,status.installed);const markImages=c=>markPictures(c,visionReader);snapshot=')
edit(p,'catalog:markEngine(packaged?markImages(catalog):catalog,status.engine)','catalog:markEngine(packaged?markImages(runtimeCatalog):runtimeCatalog,status.engine)')
edit(p,' function installedInfo(info,catalog){'," async function dedicatedCatalog(catalog,installed){\n  if(!packaged||!catalog?.families?.some(f=>f.family==='deepseek41'))return catalog;\n  let library=[];try{library=(await control.engines?.())?.families?.find(f=>f.family==='deepseek')?.engines??[];}catch{/* Unavailable information cannot establish a dedicated runtime. */}\n  const available=library.some(e=>e.fits===true&&typeof e.path==='string'&&['strata-dsv41','strata-dsv41.exe'].some(name=>deepseekRuntime({arch:'deepseek41',exe:join(e.path,name)},join(e.path,'config.json'))?.available));\n  return {...catalog,families:catalog.families.map(f=>f.family!=='deepseek41'?f:{...f,sizes:(f.sizes??[]).map(s=>{\n   const entry=installed?.models?.find(m=>m.model===s.id);\n   const configPath=entry?.configPath;\n   const own=configPath?deepseekRuntime(readJson(configPath),configPath,{family:f.family,missing:entry.engineIssues??[]}):null;\n   return {...s,runtimeUnavailableReason:(own?own.available:available)?null:DEEPSEEK_RUNTIME_UNAVAILABLE};\n  })})};\n }\n function installedInfo(info,catalog){")
p='apps/host/src/lib/native.ts'
edit(p,'needsNewerEngine?:boolean;id:string;','needsNewerEngine?:boolean;runtimeUnavailableReason?:string|null;id:string;')
edit(p,'model_name:string;ready:boolean;','model_name:string;ready:boolean;runtimeSupport?:{available:boolean;reason:string|null;executionCompatibility:\'unverified\'};')
p='apps/host/src/components/NativeModels.tsx'
edit(p,'    async function load() { if (!selectedSize)','    const runtimeReason = installed?.runtimeSupport?.reason ?? selectedSize?.runtimeUnavailableReason;\n    async function load() { if (!selectedSize || runtimeReason)')
edit(p,'     <div className="native-model-actions">','     {runtimeReason ? <p className="small inline-error">{tr(runtimeReason)}</p> : null}\n     <div className="native-model-actions">')
edit(p,'disabled={listPending || !installed?.ready','disabled={!!runtimeReason || listPending || !installed?.ready')
edit(p,'disabled={pending || listPending || !native?.available','disabled={!!runtimeReason || pending || listPending || !native?.available')
p='apps/host/src/components/GuidedSetup.tsx'
edit(p,'     <OnboardingMemory hardware={hardware}','     {version?.runtimeUnavailableReason ? <p className="small inline-error">{tr(version.runtimeUnavailableReason)}</p> : null}\n     <OnboardingMemory hardware={hardware}')
edit(p,'!!version.needsNewerEngine','!!version.needsNewerEngine || !!version.runtimeUnavailableReason',2)
edit(p,'disabled={pending || !!active || !installed?.ready}','disabled={pending || !!active || !installed?.ready || installed.runtimeSupport?.available === false}')
# Fixtures use inert files and mocked Control, never execute these files.
for file,version in [('native-ds-args.test.mjs','0.1.52'),('native-ds-parallel.test.mjs','0.1.55')]:
 p='apps/host/tests/'+file
 edit(p,'  writeFileSync(path, JSON.stringify({',f"  const dedicatedExe=join(root,'engine-dsv41','{version}-windows-x86_64-cuda-cuda13','strata-dsv41.exe');\n  mkdirSync(join(dedicatedExe,'..'),{{recursive:true}});writeFileSync(dedicatedExe,'inert fixture');\n  writeFileSync(path, JSON.stringify({")
 edit(p,f"exe: 'C:\\\\Users\\\\x\\\\AppData\\\\Local\\\\Strata\\\\engine-dsv41\\\\{version}-windows-x86_64-cuda-cuda13\\\\strata-dsv41.exe'","exe: dedicatedExe")
p='apps/host/tests/packaged-control.test.mjs'
for token,count in [("join(f.install, 'engine-dsv41', 'strata-dsv41')",2),("join(f.install, 'engine-dsv41', engine, 'strata-dsv41')",1)]:
 edit(p,'exe: '+token,'exe: inertDedicated('+token+')',count)
edit(p,"const isWin = process.platform === 'win32';","const isWin = process.platform === 'win32';\nfunction inertDedicated(path){mkdirSync(join(path,'..'),{recursive:true});writeFileSync(path,'inert fixture');return path;}")
p='apps/host/tests/load-pictures.test.mjs'
edit(p,"  const config = { exe: 'component:strata-engine'","  const dedicatedExe=join(root,'strata-dsv41');if(arch==='deepseek41')writeFileSync(dedicatedExe,'inert fixture');\n  const config = { exe: arch==='deepseek41'?dedicatedExe:'component:strata-engine'")
p='apps/host/tests/qa1-storage.test.mjs'
edit(p,"new URL('../../../VERSION', import.meta.url)","new URL('../../../../VERSION', import.meta.url)")
print('Applied dedicated runtime admission, UI availability, inert test fixtures, and corrected VERSION fixture path.')
