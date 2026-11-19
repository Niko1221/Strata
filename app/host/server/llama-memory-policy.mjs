/** Only explicit memory mode changes the supervised llama-server cache policy. */
export const PRIVATE_LLAMA_FLAGS=Object.freeze([
  '--cache-ram','0','--cache-reuse','0','--no-cache-prompt',
  '--no-cache-idle-slots','--no-slots','--log-disable',
]);
export function privateLlamaArgs(args) {
  const values=new Set(['--api-key','--api-key-file','--cache-ram','-cram','--cache-reuse','--slot-save-path','--log-file']);
  const toggles=new Set(['--cache-prompt','--no-cache-prompt','--cache-idle-slots','--no-cache-idle-slots','--slots','--no-slots','--log-disable','--verbose','-v']);
  const result=[];
  for(let i=0;i<args.length;i++){
    const arg=args[i],name=arg.split('=')[0];
    if(values.has(name)){if(!arg.includes('='))i++;continue;}
    if(toggles.has(name))continue;
    result.push(arg);
  }
  return [...result,...PRIVATE_LLAMA_FLAGS];
}
