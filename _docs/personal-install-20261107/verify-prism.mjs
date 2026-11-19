/** Read-only exact runtime/payload verification and real --version probe, never model loading. */
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import {verifyLocalBackend,verifyLocalBackendSource} from '../../app/host/server/local-backends.mjs';
import {sha256} from '../../app/host/scripts/release-integrity.mjs';
const [candidate,legacy,receipt]=process.argv.slice(2);assert.ok(candidate&&legacy&&receipt);
const report={ok:false,readOnlyPayload:true,modelLoaded:false,runtimes:[]};
for(const [folder,version,exeHash] of [[candidate,'0.4.1','5ca5d8bfdfcb6fe46a96a06b71eaf45aedfbd1c6673bda2a10eb6412613be777'],[legacy,'0.4.0','90428627933e0c4120fe0bcaa6a3cc7803dd5fe5846102eeacf4cf1298807a8d']]){
 const binary={path:join(folder,'llama-server.exe'),manifestPath:join(folder,'runtime.json')};
 const manifest=await verifyLocalBackend(binary,'PrismML');assert.equal(manifest.version,version);assert.equal(sha256(binary.path),exeHash);
 const run=spawnSync(binary.path,['--version'],{encoding:'utf8',windowsHide:true,timeout:30000,maxBuffer:65536,env:{SystemRoot:process.env.SystemRoot??'C:/Windows',PATH:join(process.env.SystemRoot??'C:/Windows','System32')}});
 assert.equal(run.status,0);const output=run.stdout+'\n'+run.stderr;
 assert.ok((output.match(/\b[0-9a-f]{7,40}\b/gi)??[]).some(commit=>manifest.source.commit.startsWith(commit.toLowerCase())));
 assert.equal(verifyLocalBackendSource(manifest,'PrismML').approved.commit,manifest.source.commit);
 report.runtimes.push({version,files:manifest.files.length,manifestSHA256:sha256(binary.manifestPath),entrypointSHA256:exeHash,versionCommitMatches:true,source:manifest.source});
}
report.ok=true;writeFileSync(resolve(receipt),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({ok:true,runtimeCount:2,exact041:true,legacy040:true,actualVersionProbes:2,modelLoaded:false}));
