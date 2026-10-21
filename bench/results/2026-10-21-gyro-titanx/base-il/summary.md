# base-il

- started: 2026-10-21T15:29:12.252559+00:00
- ready in: 44.0 s
- page cache at the start: cold (3.6 s to evict)
- phase timers: on
- machine: {"cpu": "x86_64", "logical_cpus": 12, "platform": "Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44", "gpu": "NVIDIA GeForce GTX TITAN X", "driver": "580.178.04", "gpu_memory_total_mib": 12288, "gpu_memory_used_mib": 14, "ram_total_mib": 15722, "ram_available_mib": 13671, "disk": ["/dev/sda2", "475G", "433G", "40G"]}
- git: f6829122
- engine: 0.1.46.1
- config: `config.json` in this folder

## what the engine said when it started

```
{
  "expert_cache_slots": 6160,
  "expert_cache_gib": 6.02,
  "resident_ram_gib": 6.0,
  "pcie_gbs": 12.9,
  "file_cache_hold_pct": 26,
  "direct_read_gbs": 0.56,
  "cached_read_gbs": 0.14,
  "page_cache_gib": 4.5,
  "draft_vram_mib": 802,
  "engine_version": "0.1.46.1",
  "weights_mib": 1466,
  "weights_load_s": 2.5,
  "prefill_chunk": 8192,
  "prefill_file_set_gib": 15.7,
  "prefill_page_cache_gib": 4.5,
  "prefill_drive_gb": 16.8
}
```

| workload | n | prompt_tok_s | decode_tok_s | ttft_s | generated | expert_hit_pct | os_read_mb | expert_read_mb | decode_wait_ms | decode_fetches | draft_accept_pct | ms_per_window | stage_ms | gpu_reach_wait_ms | cpu_pool_ms | commit_emit_ms | draft_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 3 | 44.0 | 6.6 | 39.17 | 155 | 90.8 | 23320.5 | 18448.8 | 18664.6 | 3821 | 80.0 | 402.89 | 0.02 | 75.22 | 30.6 | 0.07 | 7.95 |

## every run

| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | stage ms/window | cpu ms/window | draft ms/window |
|---|---|---|---|---|---|---|---|---|---|---|
| decode | 1 | 43.2 | 6.6 | 39.169 | 23320.5 | 18664.6 | 89.5 | 0.02 | 30.6 | 7.82 |
| decode | 2 | 47.2 | 5.2 | 47.161 | 29448.4 | 32830.7 | 90.8 | 0.5 | 36.09 | 7.95 |
| decode | 3 | 44.0 | 7.3 | 34.939 | 19737.0 | 13789.0 | 92.1 | 0.01 | 28.17 | 8.31 |
