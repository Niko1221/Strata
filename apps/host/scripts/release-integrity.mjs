import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {existsSync, readFileSync, readdirSync, statSync} from 'node:fs';
import {join, relative, resolve, isAbsolute} from 'node:path';

export const sha256 = path => createHash('sha256').update(readFileSync(path)).digest('hex');
export function files(root) {
  if (!existsSync(root)) throw new Error(`Missing build input: ${root}`);
  return readdirSync(root, {withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name)).flatMap(entry => {
    if (entry.name === '__pycache__' || entry.name.endsWith('.pyc')) return [];
    const path=join(root,entry.name);
    if (entry.isSymbolicLink()) throw new Error(`Linked build input: ${path}`);
    return entry.isDirectory() ? files(path) : [path];
  });
}
export function hostSources(root) {
  const paths=['src','public','server'].flatMap(name=>files(join(root,name)));
  for (const name of ['package.json','package-lock.json','vite.config.ts','vite.config.mjs','index.html','scripts/build.mjs','scripts/release-integrity.mjs']) {
    if (existsSync(join(root,name))) paths.push(join(root,name));
  }
  return Object.fromEntries(paths.map(path=>[relative(root,path).replaceAll('\\','/'),sha256(path)]).sort(([a],[b])=>a.localeCompare(b)));
}
export function nativeSources(root) {
  const paths=['src','include','cmake','spin8/src','spin8/include','spin8/third_party','tqcodec/src','tqcodec/include','third_party/strata-ggml'].filter(name=>existsSync(join(root,name))).flatMap(name=>files(join(root,name)));
  for(const name of ['CMakeLists.txt','spin8/CMakeLists.txt','tqcodec/CMakeLists.txt','third_party/ggml/VERSION.txt']) if(existsSync(join(root,name)))paths.push(join(root,name));
  return Object.fromEntries(paths.map(path=>[relative(root,path).replaceAll('\\','/'),sha256(path)]).sort(([a],[b])=>a.localeCompare(b)));
}
/** The owned Q38 closure excludes unused llama/ggml runtime trees. */
export function q38Sources(root) {
  const paths=['include','src/qwen38','spin8/src','spin8/include','spin8/third_party','tqcodec/src','tqcodec/include'].flatMap(name=>files(join(root,name)));
  paths.push(join(root,'data/spin8/reference.profile'),join(root,'src/kernels/cuda/fused_gdn.cu'));
  return Object.fromEntries(paths.map(path=>[relative(root,path).replaceAll('\\','/'),sha256(path)]).sort(([a],[b])=>a<b?-1:a>b?1:0));
}
/** A distinct portable record binds the already compiled helper, never the app commit. */
export function verifyQ38Build(folder,root,manifest) {
  const record=JSON.parse(readFileSync(join(folder,'q38-build.json'),'utf8'));
  if(record.schema!==1||record.version!==manifest.version||record.entrypoint!=='strata-q38.exe'||record.sourceCommit!=='6ee425434c98277555332c7934e1ac51436eb198'||record.sha256!==sha256(join(folder,record.entrypoint))||record.releaseQualified!==false)throw new Error('Stale Q38 build identity');
  if(JSON.stringify(record.sources)!==JSON.stringify(q38Sources(root)))throw new Error('Q38 compiled source bytes changed');
  const recipe=record.recipe;
  if(recipe?.cudaArchs!=='120'||recipe.cudaRuntime!=='static'||recipe.runtimeImplementation!=='Strata-owned'||recipe.embeddedProfileSHA256!==sha256(join(root,'data/spin8/reference.profile'))||recipe.compilerManifestSHA256!=='40851d465307d73a8c3844222112be5bfab87ca23c070923539c3bab53be4fcb')throw new Error('Changed Q38 build recipe/profile');
  const members=new Set(manifest.files?.map(entry=>entry.path.toLowerCase())??[]);
  for(const name of ['strata-q38.exe','q38-build.json',...Object.keys(record.notices??{})])if(!members.has(name.toLowerCase()))throw new Error('Uncovered Q38 package member: '+name);
  if(!Array.isArray(record.requiredLibraries)||!record.requiredLibraries.length||!Array.isArray(record.imports)||record.imports.some(name=>/llama|ggml/i.test(name)))throw new Error('Invalid Q38 runtime closure');
  for(const name of record.requiredLibraries)if(typeof name!=='string'||/[\\/]/.test(name)||!members.has(name.toLowerCase()))throw new Error('Missing Q38 runtime library: '+name);
  if(JSON.stringify(Object.keys(record.runtimeLibraries??{}).sort())!==JSON.stringify([...record.requiredLibraries].sort()))throw new Error('Incomplete Q38 runtime hashes');
  for(const [name,hash] of Object.entries(record.runtimeLibraries))if(sha256(join(folder,name))!==hash)throw new Error('Changed Q38 runtime library: '+name);
  for(const [name,hash] of Object.entries(record.notices??{}))if(!/^LICENSE-[a-z0-9-]+\.txt$/.test(name)||sha256(join(folder,name))!==hash)throw new Error('Changed Q38 format notice');
  for(const [source,name] of [['src/qwen38/LICENSE-formats.txt','LICENSE-q38-formats.txt'],['tqcodec/DIALECT_LICENSE.txt','LICENSE-turbo-dialects.txt']])if(record.notices?.[name]!==sha256(join(root,source)))throw new Error('Missing Q38 format notice');
  return record;
}
/** Identify actual vendored bytes, or an explicitly selected clean external ggml checkout. */
export function ggmlIdentity(root, dependency) {
  const vendor=join(resolve(root),'third_party','strata-ggml');
  const selected=dependency?.trim()?resolve(dependency):vendor;
  if(selected===vendor){
    const pin=/^([a-f0-9]{40})(?:\s|$)/.exec(readFileSync(join(root,'third_party','ggml','VERSION.txt'),'utf8'))?.[1];
    if(!pin)throw new Error('The vendored ggml upstream pin is missing or invalid');
    const content=Object.fromEntries(files(vendor).map(path=>[relative(vendor,path).replaceAll('\\','/'),sha256(path)]).sort(([a],[b])=>a.localeCompare(b)));
    if(!Object.keys(content).length)throw new Error('The vendored ggml tree is empty');
    return {mode:'vendored',commit:pin,sha256:createHash('sha256').update(JSON.stringify(content)).digest('hex')};
  }
  const git=spawnSync('git',['-C',selected,'rev-parse','HEAD'],{encoding:'utf8',windowsHide:true,timeout:15000});
  if(git.error||git.status!==0||!/^[a-f0-9]{40}$/.test(git.stdout.trim()))throw new Error('Cannot identify the ggml source revision');
  const dirty=spawnSync('git',['-C',selected,'status','--porcelain'],{encoding:'utf8',windowsHide:true,timeout:15000});
  if(dirty.error||dirty.status!==0||dirty.stdout.trim())throw new Error('The ggml dependency has unrecorded changes');
  return {mode:'external',commit:git.stdout.trim()};
}
export function verifyTree(source, staged, {pythonOnly=false}={}) {
  const select=root=>files(root).filter(path=>!pythonOnly || path.endsWith('.py'));
  const expected=select(source).map(path=>relative(source,path).replaceAll('\\','/')).sort();
  const actual=select(staged).map(path=>relative(staged,path).replaceAll('\\','/')).sort();
  if (JSON.stringify(expected)!==JSON.stringify(actual)) throw new Error(`Stale staged membership: ${staged}`);
  for (const name of expected) if (sha256(join(source,name))!==sha256(join(staged,name))) throw new Error(`Stale staged source: ${staged}/${name}`);
}
/** Check the package against its manifest before probing the actual executable. */
export function verifyEngine(folder, version) {
  const manifest=JSON.parse(readFileSync(join(folder,'engine-component.json'),'utf8'));
  if(manifest.id!=='strata-engine' || manifest.version!==version || !Array.isArray(manifest.files) || !manifest.files.length) throw new Error('Stale or invalid bundled engine');
  if(!manifest.files.some(entry=>entry.path===manifest.entrypoint)) throw new Error('Engine entrypoint is not in the manifest');
  for(const entry of manifest.files) {
    if(typeof entry.path!=='string' || isAbsolute(entry.path) || entry.path.split(/[\\\\/]/).some(part=>!part || part==='..' || part==='.')) throw new Error('Unsafe engine manifest path');
    const target=resolve(folder,entry.path);
    const rel=relative(resolve(folder),target);
    if(isAbsolute(rel) || rel.startsWith('..')) throw new Error('Unsafe engine manifest path');
    if(statSync(target).size!==entry.bytes || sha256(target)!==entry.sha256) throw new Error('Changed engine payload: '+entry.path);
  }
  const actual=files(folder).map(path=>relative(folder,path).replaceAll('\\','/')).filter(name=>name!=='engine-component.json').sort();
  const expected=manifest.files.map(entry=>entry.path).sort();
  if(JSON.stringify(actual)!==JSON.stringify(expected))throw new Error('Unexpected engine package membership');
  return manifest;
}
/** A separately shipped engine may bind to a pinned source checkout, never silently to another tree. */
export function verifyNativeBuild(folder,root,manifest,{sourceRoot,sourceCommit,manifestSHA256}={}) {
  const separate=sourceRoot!==undefined || sourceCommit!==undefined || manifestSHA256!==undefined;
  if(separate) {
    if(typeof sourceRoot!=='string'||!isAbsolute(sourceRoot)||!/^[a-f0-9]{40}$/.test(sourceCommit??'')||!/^[a-f0-9]{64}$/.test(manifestSHA256??''))throw new Error('Separate native source requires an absolute root, full commit and manifest SHA-256');
    if(sha256(join(folder,'engine-component.json'))!==manifestSHA256)throw new Error('Separate native manifest changed');
    const head=spawnSync('git',['-C',sourceRoot,'rev-parse','HEAD'],{encoding:'utf8',timeout:15000,windowsHide:true});
    if(head.error||head.status!==0||head.stdout.trim()!==sourceCommit)throw new Error('Separate native source commit changed');
    root=sourceRoot;
  }
  const record=JSON.parse(readFileSync(join(folder,'native-build.json'),'utf8'));
  if(record.version!==manifest.version || record.entrypoint!==manifest.entrypoint || record.sha256!==sha256(join(folder,manifest.entrypoint)))throw new Error('Stale native build identity');
  if(JSON.stringify(record.sources)!==JSON.stringify(nativeSources(root)))throw new Error('Native engine sources changed: rebuild the engine');
  if(record.recipe?.ggml?.mode==='vendored'){
    const current=ggmlIdentity(root);
    if(record.recipe.ggml.commit!==current.commit||record.recipe.ggml.sha256!==current.sha256||record.recipe.ggmlCommit!==current.commit)throw new Error('Vendored ggml identity changed: rebuild the engine');
  }
  const baseline=record.recipe?.cudaSpillBaseline;
  if(baseline!==undefined) {
    if(!baseline||typeof baseline.path!=='string'||isAbsolute(baseline.path)||baseline.path.split(/[\\/]/).some(part=>!part||part==='.'||part==='..')||!/^[a-f0-9]{64}$/.test(baseline.sha256??''))throw new Error('Unsafe CUDA resource baseline identity');
    const path=resolve(root,baseline.path),name=relative(resolve(root),path);
    if(isAbsolute(name)||name.startsWith('..')||sha256(path)!==baseline.sha256)throw new Error('CUDA resource baseline changed: rebuild and qualify the engine');
  }
  if(separate) {
    const paths=['src','include','cmake','spin8','tqcodec','CMakeLists.txt',...(baseline?[baseline.path]:[])];
    const status=spawnSync('git',['-C',root,'status','--porcelain','--untracked-files=all','--',...paths],{encoding:'utf8',timeout:15000,windowsHide:true});
    if(status.error||status.status!==0||status.stdout.trim())throw new Error('Separate native source checkout is not frozen');
  }
  if(existsSync(join(folder,'strata-q38.exe'))||existsSync(join(folder,'q38-build.json')))verifyQ38Build(folder,root,manifest);
  return {sourceRoot:resolve(root),sourceCommit:sourceCommit??null,manifestSHA256:sha256(join(folder,'engine-component.json')),engineSHA256:record.sha256,buildId:manifest.build_id??null};
}
/** The one model-derived fixture allowed in the base runtime, with its full licence. */
export function verifyControlData(source,staged) {
 const dir='data/experimental-speed-projection',name='Qwen3.8-Flash-Next-experimental-speed-projection.gguf';
 const provenance=JSON.parse(readFileSync(join(source,dir,'PROVENANCE.json'),'utf8'));
 const vectorHash='ef0724c5b79297e481017be85769832bc26eb220c984ecba2b0e1b0c8426312b',licenseHash='a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08';
 if(provenance.vector?.sha256!==vectorHash||provenance.vector?.kind!=='controlvector'||provenance.vector?.default!=='off'||provenance.license?.sha256!==licenseHash)throw new Error('Unreviewed control-vector provenance');
 for(const [file,hash] of [[name,vectorHash],['LICENSE',licenseHash]])for(const root of [source,staged])if(sha256(join(root,dir,file))!==hash)throw new Error('Missing or changed control-vector notice/data: '+file);
 if(sha256(join(source,dir,'PROVENANCE.json'))!==sha256(join(staged,dir,'PROVENANCE.json')))throw new Error('Stale control-vector provenance');
 const names=files(staged).map(path=>relative(staged,path).replaceAll('\\','/'));
 const ggufs=names.filter(path=>/\.gguf$/i.test(path));
 if(JSON.stringify(ggufs)!==JSON.stringify([dir+'/'+name]))throw new Error('Unreviewed model data in Control');
 if(names.some(path=>/\.(?:db|sqlite|sqlite3|log)$/i.test(path)||/(?:^|\/)\.env(?:\.|$)/i.test(path)))throw new Error('Live profile data in Control');
 return provenance;
}
export const PUBLIC_ASSET_COMMIT='03f11e2d6adc319ea21e9a1d5820a99cd3767257';
export function verifyPublicAssets(root,record) {
 if(record?.lucide_commit!==PUBLIC_ASSET_COMMIT||!record.files||!Object.keys(record.files).length)throw new Error('Prepare the pinned public assets first');
 const actual=files(root).map(path=>relative(root,path).replaceAll('\\','/')).sort();
 const expected=Object.keys(record.files).sort();
 if(JSON.stringify(actual)!==JSON.stringify(expected))throw new Error('Public asset membership changed');
 for(const name of expected) {
  if(/gtwalsheim/i.test(name)||name.split('/').some(part=>!part||part==='.'||part==='..')||isAbsolute(name))throw new Error('Unapproved public asset');
  if(sha256(join(root,name))!==record.files[name])throw new Error('Public asset payload changed: '+name);
 }
}
export function verifyHost(host) {
  const dist=join(host,'dist');
  const record=JSON.parse(readFileSync(join(dist,'release-build.json'),'utf8'));
  if (JSON.stringify(record.sources)!==JSON.stringify(hostSources(host))) throw new Error('Stale GUI/Host: rebuild apps/host');
  const actual=files(dist).filter(path=>relative(dist,path)!=='release-build.json');
  const assets=Object.fromEntries(actual.map(path=>[relative(dist,path).replaceAll('\\','/'),sha256(path)]).sort(([a],[b])=>a.localeCompare(b)));
  if (JSON.stringify(record.assets)!==JSON.stringify(assets)) throw new Error('Changed GUI build assets');
}
