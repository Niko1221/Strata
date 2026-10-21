# noprefetch

- started: 2026-10-21T12:13:22.425005+00:00
- ready in: 42.0 s
- page cache at the start: cold (3.8 s to evict)
- phase timers: on
- machine: {"cpu": "x86_64", "logical_cpus": 12, "platform": "Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44", "gpu": "NVIDIA GeForce GTX TITAN X", "driver": "580.178.04", "gpu_memory_total_mib": 12288, "gpu_memory_used_mib": 14, "ram_total_mib": 15722, "ram_available_mib": 14035, "disk": ["/dev/sda2", "475G", "433G", "41G"]}
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
  "direct_read_gbs": 0.55,
  "cached_read_gbs": 0.15,
  "page_cache_gib": 5.0,
  "draft_vram_mib": 802,
  "engine_version": "0.1.46.1",
  "weights_mib": 1466,
  "weights_load_s": 2.6,
  "prefill_chunk": 8192,
  "prefill_file_set_gib": 15.7,
  "prefill_page_cache_gib": 5.0,
  "prefill_drive_gb": 16.8
}
```

| workload | n | prompt_tok_s | decode_tok_s | ttft_s | generated | expert_hit_pct | os_read_mb | expert_read_mb | decode_wait_ms | decode_fetches | draft_accept_pct | ms_per_window | stage_ms | gpu_reach_wait_ms | cpu_pool_ms | commit_emit_ms | draft_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 3 | 17.1 | 4.4 | 97.21 | 209 | 90.6 | None | None | None | None | 78.8 | 627.23 | 0.04 | 69.07 | 47.73 | 0.09 | 8.29 |
| prompt | 3 | 45.1 | 4.1 | 92.33 | 16 | 92.4 | None | None | None | None | 66.7 | 551.06 | 0.06 | 77.67 | 35.37 | 0.03 | 8.17 |

## every run

| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | stage ms/window | cpu ms/window | draft ms/window |
|---|---|---|---|---|---|---|---|---|---|---|
| prompt | 1 | 43.8 | 4.1 | 95.045 | None | None | 84.7 | 0.07 | 47.51 | 7.66 |
| decode | 1 | 17.1 | 4.4 | 97.972 | None | None | 83.7 | 0.24 | 69.95 | 7.86 |
| prompt | 2 | 47.4 | 3.8 | 87.636 | None | None | 94.4 | 0.05 | 16.67 | 8.17 |
| decode | 2 | 17.5 | 4.4 | 97.213 | None | None | 90.6 | 0.04 | 47.73 | 9.81 |
| prompt | 3 | 45.1 | 4.7 | 92.334 | None | None | 92.4 | 0.06 | 35.37 | 8.2 |
| decode | 3 | 16.3 | 4.8 | 85.461 | None | None | 91.0 | 0.04 | 46.08 | 8.29 |
