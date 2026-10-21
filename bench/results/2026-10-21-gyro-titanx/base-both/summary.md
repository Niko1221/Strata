# base-both

- started: 2026-10-21T12:07:00.424581+00:00
- ready in: 46.0 s
- page cache at the start: cold (3.9 s to evict)
- phase timers: on
- machine: {"cpu": "x86_64", "logical_cpus": 12, "platform": "Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44", "gpu": "NVIDIA GeForce GTX TITAN X", "driver": "580.178.04", "gpu_memory_total_mib": 12288, "gpu_memory_used_mib": 14, "ram_total_mib": 15722, "ram_available_mib": 14108, "disk": ["/dev/sda2", "475G", "433G", "41G"]}
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
  "cached_read_gbs": 0.14,
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
| decode | 3 | 45.5 | 5.8 | 34.47 | 137 | 90.3 | 20688.6 | 18106.6 | 18740.2 | 3880 | 82.1 | 529.3 | 0.03 | 76.44 | 44.94 | 0.06 | 8.24 |
| prompt | 3 | 86.7 | 8.5 | 47.98 | 16 | 93.7 | 14539.1 | 17232.3 | 8136.8 | 2213 | 78.6 | 269.74 | 0.01 | 90.86 | 18.28 | 0.05 | 7.63 |

## every run

| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | stage ms/window | cpu ms/window | draft ms/window |
|---|---|---|---|---|---|---|---|---|---|---|
| prompt | 1 | 84.9 | 4.9 | 48.964 | 16939.9 | 10997.2 | 85.4 | 0.92 | 83.11 | 7.56 |
| decode | 1 | 41.5 | 4.4 | 46.161 | 24588.4 | 26537.3 | 83.0 | 0.93 | 51.67 | 7.41 |
| prompt | 2 | 86.7 | 11.6 | 47.98 | 14539.1 | 8136.8 | 94.6 | 0.01 | 18.28 | 7.63 |
| decode | 2 | 45.5 | 5.8 | 34.466 | 19226.2 | 16322.6 | 90.9 | 0.03 | 44.94 | 8.67 |
| prompt | 3 | 93.2 | 8.5 | 44.636 | 14182.3 | 8054.2 | 93.7 | 0.01 | 17.26 | 7.82 |
| decode | 3 | 56.1 | 6.4 | 31.703 | 20688.6 | 18740.2 | 90.3 | 0.01 | 35.33 | 8.24 |
