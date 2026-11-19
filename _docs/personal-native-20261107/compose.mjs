/** Task-local external staging, using the existing component recipe's validators. */
import {readFileSync,writeFileSync,existsSync,mkdirSync,copyFileSync,statSync} from 'node:fs';
import {resolve,join,dirname,relative} from 'node:path';
import {files,sha256,verifyEngine,verifyNativeBuild} from '../../app/host/scripts/release-integrity.mjs';
import {LOCAL_BACKENDS,findLocalBackend,verifyLocalBackend} from '../../app/host/server/local-backends.mjs';
const root=resolve(import.meta.dirname,'../..');
const [native,input,folder]=process.argv.slice(2).map(p=>resolve(p));
const owned=resolve('H:/Strata/releases/personal-052-local2-20261107');
for(const path of [native,input,folder])if(!path.startsWith(owned+'\\'))throw Error('Only task-local output allowed');
if(existsSync(folder))throw Error('Fresh composite output required');
const base=verifyEngine(native,'0.5.2');
verifyNativeBuild(native,root,base);
for(const choice of Object.keys(LOCAL_BACKENDS))await verifyLocalBackend(findLocalBackend(choice,{installRoot:input}),choice);
const notices=join(input,'notices');
const record=JSON.parse(readFileSync(join(notices,'NOTICES.json'),'utf8'));
const names=files(notices).map(p=>relative(notices,p).replaceAll('\\','/')).filter(n=>n!=='NOTICES.json').sort();
if(JSON.stringify(names)!==JSON.stringify(Object.keys(record.files).sort()))throw Error('Notice membership changed');
for(const [name,hash] of Object.entries(record.files))if(sha256(join(notices,name))!==hash)throw Error('Notice bytes changed');
mkdirSync(folder,{recursive:true});
function copy(source,target){mkdirSync(dirname(target),{recursive:true});copyFileSync(source,target);if(sha256(source)!==sha256(target))throw Error('Copy changed');}
for(const entry of base.files)copy(join(native,entry.path),join(folder,entry.path));
for(const name of ['local-backends','notices'])for(const path of files(join(input,name)))copy(path,join(folder,name,relative(join(input,name),path)));
const manifest={...base,build_id:base.build_id+'-hakua-runtimes-v1',signature:null,
  files:files(folder).map(path=>({path:relative(folder,path).replaceAll('\\','/'),bytes:statSync(path).size,sha256:sha256(path)}))};
writeFileSync(join(folder,'engine-component.json'),JSON.stringify(manifest,null,2)+'\n');
verifyEngine(folder,'0.5.2');verifyNativeBuild(folder,root,manifest);
for(const choice of Object.keys(LOCAL_BACKENDS))await verifyLocalBackend(findLocalBackend(choice,{installRoot:folder}),choice);
console.log(JSON.stringify({passed:true,folder,files:manifest.files.length,manifestSHA256:sha256(join(folder,'engine-component.json')),signed:false}));
