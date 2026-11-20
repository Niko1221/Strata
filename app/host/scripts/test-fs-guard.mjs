/** Refuse recursive test cleanup outside the suite's dedicated scratch tree. */
import fs from 'node:fs';
import {syncBuiltinESMExports} from 'node:module';
import {resolve,relative,isAbsolute,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';

export function assertScratchPath(target,scratch=process.env.STRATA_TEST_SCRATCH) {
 if(!scratch)throw new Error('Recursive test cleanup requires STRATA_TEST_SCRATCH');
 const value=target instanceof URL?fileURLToPath(target):Buffer.isBuffer(target)?target.toString():target;
 const base=resolve(scratch),path=resolve(value);
 const within=p=>{const rel=relative(base,p);return rel!==''&&!isAbsolute(rel)&&rel!=='..'&&!rel.startsWith('..\\')&&!rel.startsWith('../');};
 if(!within(path))throw new Error('Refusing recursive test cleanup outside scratch: '+path);
 // A parent junction must not redirect an apparently safe cleanup elsewhere.
 let parent=path;
 while(!fs.existsSync(parent)&&dirname(parent)!==parent)parent=dirname(parent);
 const actual=fs.realpathSync(parent),realBase=fs.realpathSync(base);
 const rel=relative(realBase,actual);
 if(isAbsolute(rel)||rel==='..'||rel.startsWith('..\\')||rel.startsWith('../'))throw new Error('Refusing linked test cleanup outside scratch: '+path);
 return path;
}
if(process.env.STRATA_TEST_SCRATCH) {
 for(const name of ['rmSync','rmdirSync','rm','rmdir']) {
  const original=fs[name];
  fs[name]=function(path,options,...rest){if(options?.recursive)assertScratchPath(path);return original.call(this,path,options,...rest);};
 }
 for(const name of ['rm','rmdir']) {
  const original=fs.promises[name];
  fs.promises[name]=function(path,options){if(options?.recursive)assertScratchPath(path);return original.call(this,path,options);};
 }
 syncBuiltinESMExports();
}
