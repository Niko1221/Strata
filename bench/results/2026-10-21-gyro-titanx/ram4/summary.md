# ram4

- started: 2026-10-21T12:47:51.545643+00:00
- ready in: 38.0 s
- page cache at the start: cold (3.9 s to evict)
- phase timers: on
- machine: {"cpu": "x86_64", "logical_cpus": 12, "platform": "Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44", "gpu": "NVIDIA GeForce GTX TITAN X", "driver": "580.178.04", "gpu_memory_total_mib": 12288, "gpu_memory_used_mib": 14, "ram_total_mib": 15722, "ram_available_mib": 14013, "disk": ["/dev/sda2", "475G", "433G", "41G"]}
- git: 9ae93f7b
- engine: 0.1.46.1
- config: `config.json` in this folder

## what the engine said when it started

```
{
  "expert_cache_slots": 6160,
  "expert_cache_gib": 6.02,
  "resident_ram_gib": 4.0,
  "pcie_gbs": 12.9,
  "file_cache_hold_pct": 37,
  "direct_read_gbs": 0.56,
  "cached_read_gbs": 0.14,
  "page_cache_gib": 7.0,
  "draft_vram_mib": 802,
  "engine_version": "0.1.46.1",
  "weights_mib": 1466,
  "weights_load_s": 2.6,
  "prefill_chunk": 8192,
  "prefill_file_set_gib": 17.7,
  "prefill_page_cache_gib": 7.0,
  "prefill_drive_gb": 19.0
}
```

| workload | n | prompt_tok_s | decode_tok_s | ttft_s | generated | expert_hit_pct | os_read_mb | expert_read_mb | decode_wait_ms | decode_fetches | draft_accept_pct | ms_per_window | stage_ms | gpu_reach_wait_ms | cpu_pool_ms | commit_emit_ms | draft_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 3 | 49.5 | 5.4 | 38.65 | 156 | 90.1 | 26788.6 | 22926.4 | 27672.8 | 5648 | 72.7 | 480.24 | 0.09 | 71.51 | 31.39 | 0.06 | 7.85 |
| prompt | 3 | 92.4 | 8.9 | 45.03 | 16 | 92.6 | 16371.6 | 19601.3 | 8133.1 | 2318 | 66.7 | 284.94 | 0.01 | 87.3 | 20.08 | 0.03 | 7.87 |

## every run

| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | stage ms/window | cpu ms/window | draft ms/window |
|---|---|---|---|---|---|---|---|---|---|---|
| prompt | 1 | 90.5 | 8.9 | 45.944 | 18761.4 | 8975.6 | 84.4 | 0.01 | 32.84 | 8.24 |
| decode | 1 | 44.6 | 4.3 | 45.53 | 29048.3 | 29766.9 | 82.5 | 0.15 | 58.66 | 7.94 |
| prompt | 2 | 92.4 | 8.0 | 45.03 | 16371.6 | 8133.1 | 92.6 | 0.01 | 20.08 | 7.74 |
| decode | 2 | 49.5 | 7.5 | 33.566 | 20092.9 | 14399.9 | 90.8 | 0.09 | 27.2 | 7.85 |
| prompt | 3 | 95.2 | 8.9 | 43.696 | 13999.5 | 6832.1 | 93.4 | 0.01 | 19.37 | 7.87 |
| decode | 3 | 54.2 | 5.4 | 38.654 | 26788.6 | 27672.8 | 90.1 | 0.03 | 31.39 | 7.66 |
