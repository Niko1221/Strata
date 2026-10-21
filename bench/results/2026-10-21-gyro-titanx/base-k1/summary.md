# base-k1

- started: 2026-10-21T13:17:41.262285+00:00
- ready in: 40.0 s
- page cache at the start: cold (4.1 s to evict)
- phase timers: on
- machine: {"cpu": "x86_64", "logical_cpus": 12, "platform": "Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44", "gpu": "NVIDIA GeForce GTX TITAN X", "driver": "580.178.04", "gpu_memory_total_mib": 12288, "gpu_memory_used_mib": 14, "ram_total_mib": 15722, "ram_available_mib": 14074, "disk": ["/dev/sda2", "475G", "433G", "40G"]}
- git: 9ae93f7b
- engine: 0.1.46.1
- config: `config.json` in this folder

## what the engine said when it started

```
{
  "expert_cache_slots": 6160,
  "expert_cache_gib": 6.02,
  "resident_ram_gib": 6.0,
  "pcie_gbs": 12.9,
  "file_cache_hold_pct": 30,
  "direct_read_gbs": 0.56,
  "cached_read_gbs": 0.15,
  "page_cache_gib": 4.9,
  "draft_vram_mib": 802,
  "engine_version": "0.1.46.1",
  "weights_mib": 1466,
  "weights_load_s": 2.6,
  "prefill_chunk": 8192,
  "prefill_file_set_gib": 15.7,
  "prefill_page_cache_gib": 4.9,
  "prefill_drive_gb": 16.8
}
```

| workload | n | prompt_tok_s | decode_tok_s | ttft_s | generated | expert_hit_pct | os_read_mb | expert_read_mb | decode_wait_ms | decode_fetches | draft_accept_pct | ms_per_window | stage_ms | gpu_reach_wait_ms | cpu_pool_ms | commit_emit_ms | draft_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 3 | 50.8 | 6.3 | 41.38 | 209 | 90.3 | 25803.6 | 19881.5 | 24701.1 | 4948 | 83.0 | 445.16 | 0.02 | 74.1 | 43.96 | 0.06 | 8.22 |
| prompt | 3 | 89.9 | 11.4 | 46.25 | 16 | 93.3 | 15482.0 | 17263.8 | 8958.3 | 2211 | 85.7 | 234.26 | 0.01 | 78.1 | 18.52 | 0.02 | 7.46 |

## every run

| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | stage ms/window | cpu ms/window | draft ms/window |
|---|---|---|---|---|---|---|---|---|---|---|
| prompt | 1 | 89.9 | 12.2 | 46.254 | 16804.5 | 8994.2 | 82.9 | 0.01 | 40.41 | 7.77 |
| decode | 1 | 43.0 | 4.3 | 59.953 | 34719.6 | 39727.5 | 82.4 | 0.02 | 75.83 | 7.89 |
| prompt | 2 | 93.1 | 11.4 | 44.723 | 15389.9 | 8020.1 | 94.4 | 0.01 | 17.42 | 7.46 |
| decode | 2 | 51.1 | 6.3 | 41.384 | 25803.6 | 24701.1 | 90.3 | 0.02 | 43.96 | 8.22 |
| prompt | 3 | 89.9 | 11.1 | 46.309 | 15482.0 | 8958.3 | 93.3 | 0.01 | 18.52 | 7.06 |
| decode | 3 | 50.8 | 7.3 | 30.791 | 17385.9 | 12114.0 | 92.5 | 0.02 | 35.5 | 8.43 |
