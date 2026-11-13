# Recording crosswalk

Hardware/software: parent README records physical versus guest capacity, selected GPU, PCIe/power, exact installed component/runner/library identities and background work. Runtime observations were made directly; no PolyStrata SDK version is assigned to Strata.

Model/preparation: MODEL_IDENTITIES.json pins the exact main GGUF repositories/revisions/LFS identities. Per-suite predeclared manifests include draft/profile hashes; native-pack-identities.json includes generated dense tensor/index/tokenizer hashes and conversion metadata. No weights or packs are submitted. Flash changes only the separately attached template, with custom track explicit.

Settings/workload: complete source config/command per suite, actual engine startup/info and full 40-line timing logs. Thinking off, temperature 0, default 3 repetitions and --long; 100/24/64 output caps for short/prompt/agent, 36 stored records and 4 preparation probes. Actual counts/timing/draft/reuse are retained. Unknown Qwen KV format is marked unavailable. Projection/calibration stages were not requested; native Q5_K head observation is recorded.

Measurement/limits: throughput uses separate engine phase times; client TTFT/total are retained. Fresh prompts have 0 reuse; follow-ups retain actual reuse. Expert state evolves. Sampled resources include loading/preparation/shutdown; they are not process RSS or continuous peaks. Failed 128K startup is preserved. Each new suite retained all 36 outcomes and no retries. No quality, isolated speedup or maintainer approval claim.
