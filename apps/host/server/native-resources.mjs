/** Headroom is a worker/memory budget, not a hard utilisation throttle. */
export function resourceSettings(config,hardware,profile,model={}){
 if(!['work','performance'].includes(profile))throw new Error('Choose a resource preference.');
 const args=[...(config.args??[])];
 const remove=flag=>{for(let at=args.indexOf(flag);at>=0;at=args.indexOf(flag))args.splice(at,2);};
 const value=flag=>{const at=args.indexOf(flag);return at>=0?Number(args[at+1])||0:0;};
 const baseline=config.ui_resource_baseline??{workers:value('--pool-workers'),reserve:Math.max(Number(config.vram_reserve_mib)||0,value('--vram-reserve-mib'))};
 if(profile==='performance'&&config.ui_resource_profile==='work'){
  remove('--pool-workers');if(baseline.workers>0)args.push('--pool-workers',String(baseline.workers));
 }
 const cores=Number(hardware?.cpu?.cores)||0;
 if(profile==='work'&&cores>1){
  const at=args.indexOf('--pool-workers'),previous=at>=0?Number(args[at+1]):0;
  const workers=Math.max(1,Math.floor(cores*.8));remove('--pool-workers');
  args.push('--pool-workers',String(previous>0?Math.min(previous,workers):workers));
 }
 // Preserve calibrated worker counts and the installation's safety/vision reserve.
 const placement=Array.isArray(config.gpu)?config.gpu:[config.gpu??0];
 const gpuMemory=(hardware?.gpus??[]).filter(g=>placement.includes(g.index)).map(g=>g.vram_gb).filter(n=>Number.isFinite(n)&&n>0);
 /* The engine warns below ~1-1.5 GiB of free VRAM (the driver then pages GPU memory to system RAM and decoding falls to a few tokens a
    second, #2065), so neither profile may reserve less: 1536 MiB on a card of 16 GB or less, 1024 above; with pictures 2048. */
 const floor=gpuMemory.length&&Math.min(...gpuMemory)>16?1024:1536;
 const safety=Math.max(config.vision?2048:floor,baseline.reserve);
 const reserve=profile==='work'&&gpuMemory.length?Math.max(safety,Math.floor(Math.min(...gpuMemory)*1024*.2)):safety;
 remove('--vram-reserve-mib');args.push('--vram-reserve-mib',String(reserve));
 // RAM: "leave 20% for my work" keeps the model's RAM within 80% of the PC. A model whose experts do not fit there (the
 // setup rule: experts plus 10 GB) keeps only that many GiB of them in RAM and reads the rest from the model files
 // (--resident-budget-gib, which implies --mmap-experts). Full performance takes the budget out again.
 if(baseline.ramBudget)remove('--resident-budget-gib');
 let env=null;   // the engine's own "leave RAM free" setting travels in the run config's env
 if(baseline.ramHeadroom&&config.env){env={...config.env};delete env.STRATA_RESIDENT_HEADROOM_GIB;}
 const ram=Number(hardware?.ram_gb)||0,arena=Number(model?.arenaGb)||0;let ramBudget=false;
 if(profile==='work'&&ram>0&&arena>0&&!['--resident-experts','--mmap-experts','--resident-budget-gib'].some(f=>args.includes(f))&&ram*.8<arena+10){
  args.push('--resident-budget-gib',String(Math.max(8,Math.min(Math.floor(ram*.8)-10,Math.floor(arena/1.073741824)))));ramBudget=true;
 }
 // The budget caps what the engine keeps in RAM. "Leave 20%" is also asked of the engine itself: at its start it leaves 20% of the RAM
 // available for other programs (it keeps fewer experts if the PC is busy), and it said nothing about this before.
 let ramHeadroom=false;
 if(ramBudget&&!(config.env?.STRATA_RESIDENT_HEADROOM_GIB)){env={...(env??config.env??{}),STRATA_RESIDENT_HEADROOM_GIB:String(Math.ceil(ram*.2))};ramHeadroom=true;}
 return {args,vram_reserve_mib:reserve,...(env?{env}:{}),ui_resource_baseline:{...baseline,ramBudget,ramHeadroom}};
}

/* One automatic second try after a start that died. Returns null when there is nothing safer to try (the caller then reports the first
   failure as it was) or {config, words, detail}: the same run config with ONE safer memory setting, a sentence for the user and the
   technical line for Details. The caller tries once and never again, and does not save this config. */
export function saferStart(config,hardware,model,failure){
 const text=String(failure??'');
 // A start that died for a reason that is not memory (a missing or bad DLL, a crash, a blocked file) is not helped by a memory setting, and must not be called a memory failure.
 if(/0xC0000135|0xC000007B|0xC0000142|0xC000001D|0xC0000005|0xC0000409|0xC0000022|error while loading shared libraries|cannot open shared object file|\bENOENT\b|exit code 127|exit code 139|illegal instruction/i.test(text))return null;
 const ranOut=/out of memory|memory guard|\boom\b|\bkilled\b|exit code 137|cannot allocate|not enough memory|bad_alloc|insufficient memory/i.test(text),first=ranOut?'The first start ran out of memory':'The first start did not finish';
 if(!/server_exited|exited before|died before|engine stopped|pinned-memory-exhausted|mapped staging|out of memory|memory guard|\bkilled\b/i.test(text))return null;
 const args=[...(config.args??[])],env={...(config.env??{})};
 const ram=Number(hardware?.ram_gb)||0,arena=Number(model?.arenaGb)||0;
 // Windows lets a program lock only about half the RAM for the graphics card; the engine pins the model's experts up to that.
 // When it ran out of locked memory, pin less of them: they stay in RAM, a little slower to reach.
 if(/pinned-memory-exhausted|mapped staging/i.test(text)&&!env.STRATA_ARENA_PIN_GIB){
  const hint=Number((text.match(/STRATA_ARENA_PIN_GIB=(\d+)/)??[])[1]);
  const gib=hint>0?hint:ram>0?Math.max(4,Math.floor(ram/2)-8):16;
  env.STRATA_ARENA_PIN_GIB=String(gib);
  return {config:{...config,env},words:`Windows would not lock as much memory as the model asked for, so Strata started again and locks ${gib} GB of the model instead of all of it. Replies may start a little slower.`,detail:`STRATA_ARENA_PIN_GIB=${gib}`};
 }
 if(['--resident-experts','--mmap-experts'].some(f=>args.includes(f)))return null;
 const at=args.indexOf('--resident-budget-gib');
 if(at>=0){
  const next=Math.floor((Number(args[at+1])||0)*.7);if(next<8)return null;
  args[at+1]=String(next);
  return {config:{...config,args},words:`${first}, so Strata started again and keeps less of the model in RAM (${next} GB); more of it is read from the SSD, which is slower.`,detail:`--resident-budget-gib ${next}`};
 }
 if(!(ram>0&&arena>0))return null;
 const budget=Math.max(8,Math.floor(ram*.5)-4);if(budget>=arena)return null;
 args.push('--resident-budget-gib',String(budget));
 return {config:{...config,args},words:`${first}, so Strata started again in low-memory mode: ${budget} GB of the model stays in RAM and the rest is read from the SSD. It is slower, but it should load.`,detail:`--resident-budget-gib ${budget}`};
}
