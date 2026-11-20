/** Verify identical CUDA source inputs while preserving distinct reviewed source commits. */
import {execFileSync} from 'node:child_process';
import {resolve} from 'node:path';
import {nativeSources} from '../../app/host/scripts/release-integrity.mjs';
const root=resolve(import.meta.dirname,'../..'), frozen=resolve(process.argv[2]), commit=process.argv[3];
const git=(...args)=>execFileSync('git',['-C',frozen,...args],{encoding:'utf8',windowsHide:true}).trim();
if(!/^[a-f0-9]{40}$/.test(commit)||git('rev-parse','HEAD')!==commit||git('status','--porcelain'))throw Error('Frozen native identity changed');
const source=nativeSources(frozen);
if(JSON.stringify(source)!==JSON.stringify(nativeSources(root)))throw Error('Current CUDA source differs from the reviewed native build');
console.log(JSON.stringify({nativeSourceCommit:commit,identicalNativeFiles:Object.keys(source).length}));
