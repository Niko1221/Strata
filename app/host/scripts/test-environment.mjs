/** Keep test processes separate from the owner's live Strata and Harness settings. */
export function isolatedTestEnvironment(inherited, profile) {
 const env=Object.fromEntries(Object.entries(inherited).filter(([name])=>!/^(STRATA_|DSH_)/i.test(name)));
 const harness=Object.entries(inherited).find(([name])=>name.toUpperCase()==='STRATA_HARNESS_DIR')?.[1];
 if(harness)env.STRATA_HARNESS_DIR=harness;
 return {...env,...profile};
}
