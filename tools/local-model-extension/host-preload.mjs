/** Explicit launcher preload; failures never replace the official host listener. */
import http from 'node:http';
import {syncBuiltinESMExports} from 'node:module';
import {readFileSync,lstatSync} from 'node:fs';
import {dirname,resolve} from 'node:path';

const path=process.env.STRATA_LOCAL_MODELS_CONFIG,marker=Symbol.for('strata.local-models.bridge');
if(path&&process.argv[1]&&!http.createServer[marker]){
 try{
  const info=lstatSync(path);if(!info.isFile()||info.isSymbolicLink()||info.size>65536)throw new Error();
  const config=JSON.parse(readFileSync(path,'utf8').replace(/^\uFEFF/,''));
  const entry=resolve(dirname(config.app_executable),'host/server/index.mjs');
  const same=process.platform==='win32'?resolve(process.argv[1]).toLowerCase()===entry.toLowerCase():resolve(process.argv[1])===entry;
  if(same){
   const port=Number(process.env.STRATA_PORT);
   if(!Number.isInteger(port)||port<1||port>65535)throw new Error();
   const {createExtensionBridge}=await import('./extension_bridge.mjs');
   const bridge=createExtensionBridge({extensionRoot:config.extension_root,ownerDirectory:config.owner_directory,
    baseUrl:config.base_url,modelLabels:config.model_labels,apiKeyEnv:config.api_key_env,
    appVersion:process.env.STRATA_APP_VERSION,officialWebRoot:resolve(dirname(entry),'../dist'),port});
   const original=http.createServer;
   const wrapped=function(...args){
    const index=args.findLastIndex(a=>typeof a==='function');
    if(index>=0){const listener=args[index];args[index]=async function(req,res){
     try{if(await bridge.handle(req,res))return;}
     catch{if(res.headersSent||res.writableEnded){if(!res.writableEnded)res.destroy();return;}}
     return listener.call(this,req,res);
    };}
    return original.apply(this,args);
   };
   wrapped[marker]=true;http.createServer=wrapped;syncBuiltinESMExports();
  }
 }catch{console.error('[strata-local-models] Extension Disabled; Starting The Official Host');}
}
