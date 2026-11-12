# Sanitized configuration (paths shortened)

Engine: `strata --serve --pack <sdk>/packs/iq3_s --native <models>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf <models>/...-00002-of-00002.gguf --mtp <mtp>/rt --expert-profile <sdk>/data/expert-profile.bin --expert-cache auto --spec 5 --kv int8 --max-context 131072 --kv-resident 49152 --prefill auto --prompt-cache 6 --vram-reserve-mib 3072 --lookup-chain 2 --spec-min-p 0.70`

env (deployment-level, after the 0026/0033 deployments):
`STRATA_HIPBLASLT_TUNING=<sdk>/gfx1100-hipblaslt-100500.txt STRATA_PREFILL_RING=512 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 STRATA_ROUTE_TAIL_SKIP=7 STRATA_ROUTE_PRIOR=0.5 STRATA_GDN_CHUNKED=1 STRATA_PF_GEMM=1 STRATA_PF_PAD=1 STRATA_KV_RING_ADAPT=1 STRATA_PF_FUSED=1 STRATA_PA_FAST=1`

Measurements that changed arms ran the same engine with per-run env/config overrides; the deep
workload was a single 93,821-token request (`max_tokens 8`, temperature 0), warm pass measured.
