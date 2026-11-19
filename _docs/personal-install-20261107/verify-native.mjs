/** Current-root source binding only. This does not run a model or a native test. */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {verifyEngine,verifyNativeBuild,nativeSources,sha256,verifyPublicAssets} from '../../app/host/scripts/release-integrity.mjs';
const root=resolve(import.meta.dirname,'../..');
const folder=process.argv[2];assert.ok(folder);
for(const key of ['STRATA_NATIVE_SOURCE_ROOT','STRATA_NATIVE_SOURCE_COMMIT','STRATA_NATIVE_MANIFEST_SHA256'])assert.equal(process.env[key],undefined,'Old frozen-source override must not be present');
const manifest=verifyEngine(folder,'0.5.2');
assert.equal(sha256(join(folder,'strata.exe')),'13fc3b96fe71afcb98990d828d2c743f85e8b59b16174a1f7da3eb4ef9b4f8e3');
assert.equal(sha256(join(folder,'strata-q36.exe')),'0789a8583c4409bd517357675b96f9507b7d88ae725474a70e05997cc2be5ede');
const binding=verifyNativeBuild(folder,root,manifest);
const sources=nativeSources(root);assert.equal(Object.keys(sources).length,751);
verifyPublicAssets('H:/Strata/releases/hakua033-public-assets',JSON.parse(readFileSync('H:/Strata/releases/hakua033-public-assets.receipt.json','utf8')));
console.log(JSON.stringify({ok:true,currentRootVerified:true,sourcesCount:751,sourcesSHA256:createHash('sha256').update(JSON.stringify(sources)).digest('hex'),binding,releaseQualified:false}));
