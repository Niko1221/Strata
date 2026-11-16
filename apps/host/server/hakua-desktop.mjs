/** Single-user memory for the Hakua fork; no persona import or implicit turn consent. */
import {randomBytes} from 'node:crypto';
import {resolve,join} from 'node:path';

export const DESKTOP_MEMORY_KEY_ENV='STRATA_HAKUA_LOCAL_KEY';

/** Preserve explicit operator configuration, including an explicit disabled harness. */
export function configureDesktopMemory(cfg,{enabled=false,directory,env=process.env}={}) {
  if(!enabled || cfg.hakua_harness!==undefined) return null;
  const identity=desktopMemoryIdentity({directory,env});
  if(cfg.arch!=='qwen38_dense')privateMemoryArgs(cfg);
  cfg.hakua_harness={enabled:true,users:[{principal:'local',key_env:DESKTOP_MEMORY_KEY_ENV,root:identity.profileRoot}]};
  return identity;
}

/** Shared local identity across native, PrismML and TurboQuant; no store is opened. */
export function desktopMemoryIdentity({directory,env=process.env}={}) {
  if(env.STRATA_BATCH_TRACE) {
    if(!['0','false','off'].includes(String(env.STRATA_BATCH_TRACE).trim().toLowerCase())) {
      throw new Error('Unset STRATA_BATCH_TRACE before using Hakua memory.');
    }
    // The Python protocol logger treats every nonempty value, including "0",
    // as enabled. Clear an explicit off value before starting private memory.
    delete env.STRATA_BATCH_TRACE;
  }
  if(!directory) throw new Error('Hakua memory needs an application data folder.');
  const key=env[DESKTOP_MEMORY_KEY_ENV]??randomBytes(32).toString('hex');
  env[DESKTOP_MEMORY_KEY_ENV]=key;
  const root=join(resolve(directory),'hakua-memory','local');
  return {profileKey:key,profileRoot:root};
}

/** Tune and load must measure the same cache policy. */
export function privateMemoryArgs(cfg) {
  const flags=['--prompt-cache','--conversation-cache-mib'];
  const args=[];
  for(let i=0;i<(cfg.args??[]).length;i++) {
    const arg=cfg.args[i];
    if(flags.includes(arg)){i++;continue;}
    if(flags.some(flag=>arg.startsWith(flag+'=')))continue;
    args.push(arg);
  }
  cfg.args=[...args,'--prompt-cache','0','--conversation-cache-mib','0'];
  delete cfg.slot_save_path;
}
