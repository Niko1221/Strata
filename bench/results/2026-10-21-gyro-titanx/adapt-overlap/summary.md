# adapt-overlap

- started: 2026-10-21T12:25:33.815983+00:00
- ready in: 42.0 s
- page cache at the start: cold (4.0 s to evict)
- phase timers: on
- machine: {"cpu": "x86_64", "logical_cpus": 12, "platform": "Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44", "gpu": "NVIDIA GeForce GTX TITAN X", "driver": "580.178.04", "gpu_memory_total_mib": 12288, "gpu_memory_used_mib": 14, "ram_total_mib": 15722, "ram_available_mib": 14024, "disk": ["/dev/sda2", "475G", "433G", "41G"]}
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
  "direct_read_gbs": 0.55,
  "cached_read_gbs": 0.14,
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
| decode | 3 | 48.2 | 8.0 | 34.05 | 157 | 90.5 | 20359.0 | 18509.7 | 14827.0 | 3315 | 76.6 | 337.05 | 0.01 | 81.14 | 28.64 | 0.03 | 7.85 |
| prompt | 3 | 91.7 | 10.5 | 45.35 | 16 | 93.3 | 14762.1 | 17342.5 | 8229.4 | 2221 | 66.7 | 217.37 | 0.01 | 98.91 | 18.0 | 0.03 | 7.77 |

## every run

| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | stage ms/window | cpu ms/window | draft ms/window |
|---|---|---|---|---|---|---|---|---|---|---|
| prompt | 1 | 88.2 | 6.1 | 47.158 | 17955.0 | 11512.6 | 79.8 | 0.01 | 52.2 | 7.64 |
| decode | 1 | 41.4 | 5.3 | 41.501 | 26741.7 | 24434.3 | 83.3 | 0.01 | 54.41 | 7.85 |
| prompt | 2 | 91.7 | 11.6 | 45.35 | 14762.1 | 8229.4 | 93.5 | 0.01 | 17.58 | 9.15 |
| decode | 2 | 48.2 | 8.0 | 34.046 | 20359.0 | 14827.0 | 90.5 | 0.01 | 28.64 | 7.78 |
| prompt | 3 | 92.0 | 10.5 | 45.216 | 13856.8 | 7798.2 | 93.3 | 0.01 | 18.0 | 7.77 |
| decode | 3 | 51.1 | 8.8 | 32.359 | 18251.4 | 11590.5 | 91.2 | 0.01 | 23.96 | 7.87 |
