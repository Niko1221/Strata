/** Build the UI and bind its source, recipe and exact output inventory. */
import {spawnSync} from 'node:child_process';
import {writeFileSync,readFileSync,cpSync} from 'node:fs';
import {resolve,join,relative} from 'node:path';
import {files,hostSources,sha256,verifyPublicAssets,PUBLIC_ASSET_COMMIT} from './release-integrity.mjs';
const root=resolve(import.meta.dirname,'..');
const publicPackage=process.env.STRATA_PUBLIC_PACKAGE==='1';
let publicAssets,publicReceipt;
if(publicPackage) {
 if(!process.env.STRATA_PUBLIC_ASSETS||!process.env.STRATA_PUBLIC_ASSETS_RECEIPT)throw new Error('Set the prepared public asset directory and provenance receipt');
 publicAssets=resolve(process.env.STRATA_PUBLIC_ASSETS);
 publicReceipt=JSON.parse(readFileSync(process.env.STRATA_PUBLIC_ASSETS_RECEIPT,'utf8'));
 verifyPublicAssets(publicAssets,publicReceipt);
}
const sources=hostSources(root);
const result=spawnSync(process.execPath,[join(root,'node_modules/vite/bin/vite.js'),'build'],{cwd:root,stdio:'inherit'});
if(result.error)throw result.error;
if(result.status!==0)process.exit(result.status??1);
if(JSON.stringify(sources)!==JSON.stringify(hostSources(root)))throw new Error('GUI sources changed while building; rebuild');
const dist=join(root,'dist');
if(publicPackage) {
 verifyPublicAssets(publicAssets,publicReceipt);
 cpSync(publicAssets,dist,{recursive:true});
 for(const [name,hash] of Object.entries(publicReceipt.files))if(sha256(join(dist,name))!==hash)throw new Error('Public asset changed while staging');
}
const assets=Object.fromEntries(files(dist).map(path=>[relative(dist,path).replaceAll('\\','/'),sha256(path)]).sort(([a],[b])=>a.localeCompare(b)));
const recipe={publicPackage,nodeVersion:process.versions.node,...(publicPackage?{publicAssetCommit:PUBLIC_ASSET_COMMIT,publicAssets:publicReceipt.files}:{})};
writeFileSync(join(dist,'release-build.json'),JSON.stringify({schema:1,sources,assets,recipe},null,2)+'\n');
