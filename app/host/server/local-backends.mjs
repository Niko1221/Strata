/** Explicit, self-contained llama-server runtimes. Each fork keeps its own libraries. */
import {createHash} from 'node:crypto';
import {readFileSync,existsSync,realpathSync,createReadStream,readdirSync} from 'node:fs';
import {dirname,join,resolve,relative,isAbsolute} from 'node:path';
import {fileURLToPath} from 'node:url';

const PRISM_BASE='73bfe6cc485554822f80c473a3c0c4f25e80f41b';
const PRISM_PATCH='a291a22cd8915edee441e39b08b1c948e5ecc18b41060529c120701bfcf26ddd';
const PRISM_041=Object.freeze({
 commit:'1965755ceee6738b9a04ceaf5d30e6305d6ccd00',baseCommit:PRISM_BASE,tree:'ae33fd73e8811fd0def3f13488c6942848ac5138',
 manifestSHA256:'9c729c73eb050dc466203fe149a27acd5be48763a33ea2c5d2b89f78e4ddcbaa',
 sourcePatchSHA256:'f95abdc644fe6f4de7e3ea20ff5dea8aa2cfe998fb88cd0fac11ff8f95c66bf8',
 patches:Object.freeze([
  Object.freeze({path:'qwen35_absent_mtp.patch',id:'qwen35-absent-mtp-v1',sha256:'d40d21c784c411a316dd33ea01665bcbb56511a3641073088c128edba83ddbe9'}),
  Object.freeze({path:'spin8_backend_sm120.patch',id:'prism-standard-sm120-v1',sha256:PRISM_PATCH}),
 ]),
});
export const LOCAL_BACKENDS=Object.freeze({
 PrismML:Object.freeze({id:'prism',packageVersion:'0.4.1',repository:'https://github.com/PrismML-Eng/llama.cpp.git',commit:PRISM_041.commit,
  versions:Object.freeze({'0.4.0':Object.freeze({commit:PRISM_BASE,patchSHA256:PRISM_PATCH}),'0.4.1':PRISM_041})}),
 TurboQuant:Object.freeze({id:'turboquant',packageVersion:'0.4.0',repository:'https://github.com/TheTom/llama-cpp-turboquant.git',commit:'643956891dfdf51aecdbd05ca508dd21e21e8195',
  versions:Object.freeze({'0.4.0':Object.freeze({commit:'643956891dfdf51aecdbd05ca508dd21e21e8195',patchSHA256:'41301159b543700425038d569926ed68b131d5f0692d89ab8e1efd1488399144'})})}),
});

/** Version-specific provenance. Integrity approval is not general model/release qualification. */
export function verifyLocalBackendSource(manifest,choice) {
 const expected=Object.hasOwn(LOCAL_BACKENDS,choice)?LOCAL_BACKENDS[choice]:null,source=manifest?.source;
 const approved=expected&&typeof manifest?.version==='string'&&Object.hasOwn(expected.versions,manifest.version)?expected.versions[manifest.version]:null;
 if(!approved)throw new Error('The selected local runtime version is not approved.');
 if(manifest.schema!==1||manifest.backend!==expected.id||source?.repository!==expected.repository||source.commit!==approved.commit)throw new Error('The selected local runtime has an unknown source identity.');
 const patches=source.patches??[];
 if(!Array.isArray(patches))throw new Error('The local runtime has unreviewed source changes.');
 if(approved.manifestSHA256) {
  // This exact private artifact has raw-byte evidence for Git's dirty-status
  // false positive. It grants no exception to any other source or package.
  const proof=source.byteVerification;
  if(source.dirty!==false||source.baseCommit!==approved.baseCommit||source.tree!==approved.tree||source.gitStatusReportedDirty!==true||
     proof?.verified!==true||proof.sourceCommit!==approved.commit||proof.sourceTree!==approved.tree||proof.trackedFiles!==3590||proof.exactRawOverrides!==4||
     source.sourcePatchSHA256!==approved.sourcePatchSHA256||patches.length!==approved.patches.length||
     patches.some((patch,i)=>!patch||patch.path!==approved.patches[i].path||patch.id!==approved.patches[i].id||patch.sha256!==approved.patches[i].sha256))throw new Error('The local runtime has unreviewed version-specific source changes.');
  if(manifest.privateCandidate!==true||manifest.releaseQualified!==false)throw new Error('The private runtime qualification record changed.');
  return {approved,patchFiles:[...approved.patches,{path:'source-patch.diff',sha256:approved.sourcePatchSHA256}]};
 }
 const patched=source.dirty===true&&patches.length===1&&patches[0]?.id===expected.id+'-standard-sm120-v1'&&patches[0].sha256===approved.patchSHA256;
 if(!(source.dirty===false&&patches.length===0)&&!patched)throw new Error('The local runtime has unreviewed source changes.');
 return {approved,patchFiles:patched?[{path:'source-patch.diff',sha256:approved.patchSHA256}]:[]};
}
export function localBackendChoice(expert=[]) {
 const selected=expert.find(item=>item.id==='runtime.backend')?.value??'Automatic';
 if(selected!=='Automatic'&&!Object.hasOwn(LOCAL_BACKENDS,selected))throw new Error('Unknown local llama-server backend: '+selected);
 return selected;
}
/** Installed signed engine components may carry the separate llama-server runtimes. */
function componentBackendRoots(componentsDir,definition,platform) {
 if(!componentsDir)return [];
 const engine=join(componentsDir,'engine');let entries,pinned;
 try {entries=readdirSync(engine,{withFileTypes:true});}catch{return [];}
 try {const value=JSON.parse(readFileSync(join(componentsDir,'.updates','state.json'),'utf8').replace(/^\uFEFF/,'')).engine?.pinned;pinned=typeof value==='string'&&value&&!/[\\/]/.test(value)?value:null;}catch{}
 const prefix=platform==='win32'?'windows-':platform==='darwin'?'macos-':'linux-';
 const roots=[];
 for(const entry of entries) {
  if(!entry.isDirectory()||entry.isSymbolicLink()||entry.name.startsWith('.')||(pinned&&entry.name!==pinned))continue;
  try {
   const component=join(engine,entry.name),bytes=readFileSync(join(component,'engine-component.json')),manifest=JSON.parse(bytes.toString('utf8'));
   if(manifest.id!=='strata-engine'||!/^\d+\.\d+\.\d+$/.test(manifest.version??'')||!manifest.platform?.startsWith(prefix)||!Array.isArray(manifest.files))continue;
   const name=manifest.version+'-'+manifest.platform+'-'+manifest.backend+(manifest.variant?'-'+manifest.variant:'');
   // Control adds the signed manifest digest when reinstalling an existing folder.
   if(name!==entry.name&&name+'-'+createHash('sha256').update(bytes).digest('hex').slice(0,8)!==entry.name)continue;
   const runtime='local-backends/'+definition.id+'/runtime.json',declared=manifest.files.find(file=>file.path===runtime);
   if(!/^[a-f0-9]{64}$/.test(declared?.sha256??''))continue;
   const dir=join(component,'local-backends',definition.id),rel=relative(realpathSync(component),realpathSync(dir));
   if(isAbsolute(rel)||rel==='..'||rel.startsWith('..'+(process.platform==='win32'?'\\':'/')))continue;
   roots.push({dir,componentRuntimeSha256:declared.sha256,componentVersion:manifest.version});
  }catch{ /* A missing, linked or incomplete component is not a runtime. */ }
 }
 return roots.sort((a,b)=>b.componentVersion.localeCompare(a.componentVersion,undefined,{numeric:true}));
}
export function findLocalBackend(choice,{componentsDir,installRoot=resolve(dirname(fileURLToPath(import.meta.url)),'../..'),platform=process.platform}={}) {
 const definition=Object.hasOwn(LOCAL_BACKENDS,choice)?LOCAL_BACKENDS[choice]:null;
 if(!definition)throw new Error('Choose PrismML or TurboQuant explicitly.');
 const roots=[{dir:join(installRoot,'local-backends',definition.id)},...(componentsDir?[{dir:join(componentsDir,'local-backends',definition.id)}]:[]),...componentBackendRoots(componentsDir,definition,platform)];
 const candidates=[];
 for(const {dir:root,componentRuntimeSha256} of roots) {
  const binary=join(root,platform==='win32'?'llama-server.exe':'llama-server');
  if(existsSync(binary)&&existsSync(join(root,'runtime.json'))) {
   try {const version=JSON.parse(readFileSync(join(root,'runtime.json'),'utf8')).version;
     if(Object.hasOwn(definition.versions,version))candidates.push({path:binary,source:choice,manifestPath:join(root,'runtime.json'),backend:definition.id,version,...(componentRuntimeSha256?{componentRuntimeSha256}:{})});
   } catch { /* an incomplete package is not usable */ }
  }
 }
 candidates.sort((a,b)=>{const left=a.version.split('.').map(Number),right=b.version.split('.').map(Number);for(let i=0;i<3;i++)if(left[i]!==right[i])return right[i]-left[i];return 0;});
 return candidates[0]??null;
}
async function hash(path) {const digest=createHash('sha256');for await(const chunk of createReadStream(path))digest.update(chunk);return digest.digest('hex');}
export async function verifyLocalBackend(binary,choice) {
 const expected=Object.hasOwn(LOCAL_BACKENDS,choice)?LOCAL_BACKENDS[choice]:null;
 if(!expected||!binary?.manifestPath)throw new Error('A verified local '+choice+' runtime is required.');
 if(binary.componentRuntimeSha256&&await hash(binary.manifestPath)!==binary.componentRuntimeSha256)throw new Error('Local runtime component manifest changed. Reinstall the engine package.');
 const root=dirname(resolve(binary.manifestPath));
  const bytes=readFileSync(binary.manifestPath),manifest=JSON.parse(bytes.toString('utf8'));
  const {approved,patchFiles}=verifyLocalBackendSource(manifest,choice);
  if(approved.manifestSHA256&&createHash('sha256').update(bytes).digest('hex')!==approved.manifestSHA256)throw new Error('The local runtime manifest differs from the exact approved package.');
 if(!Array.isArray(manifest.files)||!manifest.files.length||!Array.isArray(manifest.cacheTypes))throw new Error('The local runtime manifest is incomplete.');
 let entry=false;const seen=new Set();
 for(const file of manifest.files) {
  if(typeof file?.path!=='string'||isAbsolute(file.path)||file.path.split(/[\\/]/).some(part=>!part||part==='.'||part==='..'))throw new Error('Unsafe local runtime payload path');
  const target=resolve(root,file.path),rel=relative(root,realpathSync(target));
  if(isAbsolute(rel)||rel==='..'||rel.startsWith('..'+(process.platform==='win32'?'\\':'/'))||seen.has(file.path.toLowerCase()))throw new Error('Invalid local runtime payload membership');
  seen.add(file.path.toLowerCase());
  if(await hash(target)!==file.sha256)throw new Error('Local runtime payload changed: '+file.path);
  if(target===resolve(binary.path))entry=true;
 }
 if(!entry)throw new Error('The selected llama-server is not in its runtime manifest.');
  for(const patch of patchFiles)if(!manifest.files.some(file=>file.path===patch.path&&file.sha256===patch.sha256))throw new Error('The approved runtime source patch is missing or changed.');
 const inventory=(folder,prefix='')=>readdirSync(folder,{withFileTypes:true}).flatMap(item=>{
  if(item.isSymbolicLink())throw new Error('Linked local runtime payload');
  const name=prefix+item.name;return item.isDirectory()?inventory(join(folder,item.name),name+'/'):[name];
 });
 const actual=inventory(root).filter(name=>name!=='runtime.json').sort();
 if(JSON.stringify(actual)!==JSON.stringify(manifest.files.map(item=>item.path).sort()))throw new Error('Unexpected local runtime package membership');
 return manifest;
}
export function requireRuntimeKv(manifest,{k,v}) {
 for(const [side,type] of [['K',k],['V',v]])if(!(manifest['cacheTypes'+side]??manifest.cacheTypes).includes(type))throw new Error('This '+manifest.backend+' runtime cannot use '+type+' '+side+' cache. Choose a supported precision or another backend.');
}
