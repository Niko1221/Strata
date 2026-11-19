/** Production integrity/source/signature guards; no native executable is launched. */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import {verifyEngine,verifyNativeBuild,sha256,verifyPublicAssets} from '../../../app/host/scripts/release-integrity.mjs';
const root=resolve(import.meta.dirname,'../../..'),out='H:/Strata/releases/local-053-20261109-opencode/app-files';
if(!process.argv.includes('--execute')){console.log(JSON.stringify({wouldVerify:false,executesEngine:false,receipt:join(out,'staged-native.json')}));process.exit(0);}
const signed=JSON.parse(readFileSync(join(out,'staged-native.json'))),folder=join(root,'apps/desktop/src-tauri/resources/engine-bundle/0.5.3-windows-x86_64-cuda-cuda13');
assert.equal(signed.ok,true);assert.equal(signed.signatureVerified,true);assert.equal(signed.releaseQualified,false);
assert.equal(sha256(join(folder,'strata.exe')),signed.engineSHA256);
assert.equal(sha256(join(folder,'strata-q36.exe')),signed.q36SHA256);
assert.equal(sha256(join(folder,'engine-component.json')),signed.signedManifestSHA256);
assert.equal(signed.prismManifestSHA256,'9c729c73eb050dc466203fe149a27acd5be48763a33ea2c5d2b89f78e4ddcbaa');
assert.equal(signed.prismExeSHA256,'5ca5d8bfdfcb6fe46a96a06b71eaf45aedfbd1c6673bda2a10eb6412613be777');
const manifest=verifyEngine(folder,'0.5.3');
const binding=verifyNativeBuild(folder,root,manifest,{sourceRoot:process.env.STRATA_NATIVE_SOURCE_ROOT,sourceCommit:signed.nativeSourceCommit,manifestSHA256:signed.signedManifestSHA256});
verifyPublicAssets('H:/Strata/releases/hakua033-public-assets',JSON.parse(readFileSync('H:/Strata/releases/hakua033-public-assets.receipt.json')));
const signature=spawnSync('C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe',['-I','-B','-c',"import sys,json;sys.path.insert(0,sys.argv[1]);from terminal.strata_control import signing,engine_component;m=json.load(open(sys.argv[2],encoding='utf-8'));signing.verify_bytes(signing.load_trusted_keys(environ={}),engine_component.signing_payload(m),m.get('signature'))",root,join(folder,'engine-component.json')],{encoding:'utf8',windowsHide:true,timeout:30000});
assert.equal(signature.status,0,'Built-in trusted-key signature verification failed');
console.log(JSON.stringify({ok:true,binding,releaseQualified:false,quality:signed.quality}));
