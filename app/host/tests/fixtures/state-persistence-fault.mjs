/** Deterministic OS-error injection at real fs calls in a scratch Host process.
 * No store, route, or persistence callback is mocked. */
import fs from 'node:fs';
import {syncBuiltinESMExports} from 'node:module';
import {basename,dirname,join,resolve} from 'node:path';
import {assertScratchPath} from '../../scripts/test-fs-guard.mjs';
const home=assertScratchPath(process.env.STRATA_HOME),data=join(home,'data');
const marker=join(home,'fs-fault.json'),read=fs.readFileSync;
const fault=()=>{try{return JSON.parse(read(marker,'utf8'));}catch{return {};}};
const statePath=path=>typeof path==='string'&&dirname(resolve(path))===data&&/^state\.json(?:\.|$)/.test(basename(path));
const denied=()=>Object.assign(new Error('Injected filesystem permission failure'),{code:'EACCES'});
const write=fs.writeFileSync,rename=fs.renameSync,remove=fs.rmSync;
let statePublishes=0;
fs.writeFileSync=function(path,...args){const mode=fault();if(mode.stateWrite&&statePath(path))throw denied();if(mode.partialWrite&&statePath(path)){write.call(this,path,'partial write');throw Object.assign(new Error('Injected filesystem full failure'),{code:'ENOSPC'});}return write.call(this,path,...args);};
fs.renameSync=function(from,to,...args){const mode=fault();if(statePath(to)&&(mode.statePublish||mode.statePublishAfter&&++statePublishes>mode.statePublishAfter)||mode.skillPublish&&dirname(resolve(to))===join(data,'skills'))throw denied();return rename.call(this,from,to,...args);};
fs.rmSync=function(path,...args){if(fault().cleanup&&typeof path==='string'&&dirname(resolve(path))===data&&basename(path).startsWith('.skill-'))throw denied();return remove.call(this,path,...args);};
syncBuiltinESMExports();
