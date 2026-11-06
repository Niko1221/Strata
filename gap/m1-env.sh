# M1 Max speed profile (this fork). Source it; it does NOT change the default path.
# Measured 10/16/26 on an M1 Max 64 GB, IQ3_XXS pack, ctx 262144, t2048/t357 prompts,
# server stopped (gap/results/1016-11{18,35},1016-12{02,20,29,46}).
# pp2048: 164-169 -> 194-199 tok/s (+18%); tg128: 15.4 -> 19.5-20.6 tok/s (+27%).
# WARNING: mpp and scan_v change the greedy continuation vs the default path (they round
# differently); overlap and CTAB are token-identical and overlap is already on by default here.
export STRATA_ADAPT_OVERLAP=1   # default on in this build; set 0 to opt out
export STRATA_METAL_CTAB=1      # already the default; set 0 to opt out
export STRATA_METAL_PROMPT_MOE=mpp     # fp16-slab expert path for prompt chunks
export STRATA_METAL_GDN_SCAN=scan_v    # <8,16> packed-row prompt scan
# Never set STRATA_METAL_PROMPT_EXPERTS=v3 on an M1: it hits the GPU watchdog.
