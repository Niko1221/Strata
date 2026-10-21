# prefill2048

- started: 2026-10-21T12:35:14.670777+00:00
- ready in: 44.0 s
- page cache at the start: cold (3.8 s to evict)
- phase timers: on
- machine: {"cpu": "x86_64", "logical_cpus": 12, "platform": "Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44", "gpu": "NVIDIA GeForce GTX TITAN X", "driver": "580.178.04", "gpu_memory_total_mib": 12288, "gpu_memory_used_mib": 14, "ram_total_mib": 15722, "ram_available_mib": 14010, "disk": ["/dev/sda2", "475G", "433G", "41G"]}
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
  "draft_vram_mib": 802,
  "engine_version": "0.1.46.1",
  "weights_mib": 1466,
  "weights_load_s": 2.6
}
```

| workload | n | prompt_tok_s | decode_tok_s | ttft_s | generated | expert_hit_pct | os_read_mb | expert_read_mb | decode_wait_ms | decode_fetches | draft_accept_pct | ms_per_window | stage_ms | gpu_reach_wait_ms | cpu_pool_ms | commit_emit_ms | draft_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 3 | 54.9 | 6.3 | 32.16 | 158 | 91.1 | 20604.1 | 18076.2 | 18534.7 | 3794 | 75.5 | 417.47 | 0.02 | 77.94 | 45.09 | 0.05 | 8.13 |
| prompt | 3 | 77.5 | 9.2 | 53.74 | 16 | 94.2 | 27484.8 | 33162.1 | 4225.9 | 1297 | 75.0 | 249.28 | 0.01 | 85.44 | 20.36 | 0.03 | 6.74 |

## every run

| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | stage ms/window | cpu ms/window | draft ms/window |
|---|---|---|---|---|---|---|---|---|---|---|
| prompt | 1 | 72.2 | 5.7 | 57.606 | 29666.4 | 6368.6 | 84.2 | 0.01 | 51.2 | 6.55 |
| decode | 1 | 42.9 | 4.8 | 42.849 | 26146.5 | 26347.1 | 83.2 | 0.02 | 85.04 | 8.07 |
| prompt | 2 | 80.6 | 12.6 | 51.673 | 24008.7 | 4225.9 | 94.5 | 0.01 | 19.82 | 8.51 |
| decode | 2 | 57.1 | 6.3 | 32.156 | 20604.1 | 18534.7 | 91.1 | 0.03 | 45.09 | 8.4 |
| prompt | 3 | 77.5 | 9.2 | 53.744 | 27484.8 | 3476.3 | 94.2 | 0.01 | 20.36 | 6.74 |
| decode | 3 | 54.9 | 7.8 | 28.784 | 16469.1 | 10193.3 | 92.3 | 0.01 | 36.23 | 8.13 |
