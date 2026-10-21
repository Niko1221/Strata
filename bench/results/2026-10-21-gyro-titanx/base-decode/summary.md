# base-decode

- started: 2026-10-21T11:51:47.591895+00:00
- ready in: 44.0 s
- page cache at the start: cold (3.7 s to evict)
- phase timers: on
- machine: {"cpu": "x86_64", "logical_cpus": 12, "platform": "Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44", "gpu": "NVIDIA GeForce GTX TITAN X", "driver": "580.178.04", "gpu_memory_total_mib": 12288, "gpu_memory_used_mib": 14, "ram_total_mib": 15722, "ram_available_mib": 14122, "disk": ["/dev/sda2", "475G", "433G", "41G"]}
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
  "file_cache_hold_pct": 29,
  "direct_read_gbs": 0.56,
  "cached_read_gbs": 0.15,
  "page_cache_gib": 4.6,
  "draft_vram_mib": 802,
  "engine_version": "0.1.46.1",
  "weights_mib": 1466,
  "weights_load_s": 2.5,
  "prefill_chunk": 8192,
  "prefill_file_set_gib": 15.7,
  "prefill_page_cache_gib": 4.6,
  "prefill_drive_gb": 16.8
}
```

| workload | n | prompt_tok_s | decode_tok_s | ttft_s | generated | expert_hit_pct | os_read_mb | expert_read_mb | decode_wait_ms | decode_fetches | draft_accept_pct | ms_per_window | stage_ms | gpu_reach_wait_ms | cpu_pool_ms | commit_emit_ms | draft_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 3 | 46.7 | 6.7 | 36.76 | 162 | 90.8 | 22439.5 | 18234.7 | 18170.8 | 3720 | 81.9 | 396.19 | 0.02 | 70.54 | 54.64 | 0.05 | 7.71 |

## every run

| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | stage ms/window | cpu ms/window | draft ms/window |
|---|---|---|---|---|---|---|---|---|---|---|
| decode | 1 | 32.2 | 3.8 | 54.179 | 31719.9 | 33957.4 | 82.7 | 0.02 | 102.27 | 7.71 |
| decode | 2 | 46.7 | 6.7 | 36.76 | 22439.5 | 18170.8 | 90.8 | 0.01 | 54.64 | 7.58 |
| decode | 3 | 47.6 | 10.0 | 33.885 | 18470.8 | 10981.2 | 91.5 | 0.07 | 38.65 | 7.86 |
