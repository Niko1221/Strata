# il-rows

- started: 2026-10-21T15:33:59.186839+00:00
- ready in: 42.0 s
- page cache at the start: cold (4.0 s to evict)
- phase timers: on
- machine: {"cpu": "x86_64", "logical_cpus": 12, "platform": "Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44", "gpu": "NVIDIA GeForce GTX TITAN X", "driver": "580.178.04", "gpu_memory_total_mib": 12288, "gpu_memory_used_mib": 14, "ram_total_mib": 15722, "ram_available_mib": 13868, "disk": ["/dev/sda2", "475G", "433G", "40G"]}
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
  "file_cache_hold_pct": 27,
  "direct_read_gbs": 0.56,
  "cached_read_gbs": 0.15,
  "page_cache_gib": 4.6,
  "draft_vram_mib": 802,
  "engine_version": "0.1.46.1",
  "weights_mib": 1466,
  "weights_load_s": 2.6,
  "prefill_chunk": 8192,
  "prefill_file_set_gib": 15.7,
  "prefill_page_cache_gib": 4.6,
  "prefill_drive_gb": 16.8
}
```

| workload | n | prompt_tok_s | decode_tok_s | ttft_s | generated | expert_hit_pct | os_read_mb | expert_read_mb | decode_wait_ms | decode_fetches | draft_accept_pct | ms_per_window | stage_ms | gpu_reach_wait_ms | cpu_pool_ms | commit_emit_ms | draft_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 3 | 47.0 | 5.3 | 44.5 | 208 | 90.9 | 25990.1 | 19329.4 | 24718.1 | 4868 | 77.5 | 474.3 | 0.05 | 72.18 | 32.25 | 0.06 | 8.0 |

## every run

| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | stage ms/window | cpu ms/window | draft ms/window |
|---|---|---|---|---|---|---|---|---|---|---|
| decode | 1 | 38.7 | 5.3 | 53.938 | 30485.1 | 32011.9 | 88.6 | 0.02 | 42.47 | 8.32 |
| decode | 2 | 47.4 | 5.3 | 42.014 | 23947.0 | 24718.1 | 90.9 | 0.13 | 32.25 | 7.53 |
| decode | 3 | 47.0 | 6.8 | 44.499 | 25990.1 | 23595.2 | 91.1 | 0.05 | 28.3 | 8.0 |
