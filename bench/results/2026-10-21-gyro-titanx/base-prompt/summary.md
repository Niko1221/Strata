# base-prompt

- started: 2026-10-21T11:56:02.182013+00:00
- ready in: 42.0 s
- page cache at the start: cold (3.9 s to evict)
- phase timers: on
- machine: {"cpu": "x86_64", "logical_cpus": 12, "platform": "Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44", "gpu": "NVIDIA GeForce GTX TITAN X", "driver": "580.178.04", "gpu_memory_total_mib": 12288, "gpu_memory_used_mib": 14, "ram_total_mib": 15722, "ram_available_mib": 14119, "disk": ["/dev/sda2", "475G", "433G", "41G"]}
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
  "page_cache_gib": 4.8,
  "draft_vram_mib": 802,
  "engine_version": "0.1.46.1",
  "weights_mib": 1466,
  "weights_load_s": 2.6,
  "prefill_chunk": 8192,
  "prefill_file_set_gib": 15.7,
  "prefill_page_cache_gib": 4.8,
  "prefill_drive_gb": 16.8
}
```

| workload | n | prompt_tok_s | decode_tok_s | ttft_s | generated | expert_hit_pct | os_read_mb | expert_read_mb | decode_wait_ms | decode_fetches | draft_accept_pct | ms_per_window | stage_ms | gpu_reach_wait_ms | cpu_pool_ms | commit_emit_ms | draft_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt | 3 | 87.4 | 14.7 | 47.59 | 16 | 89.1 | 14763.9 | 17265.9 | 7983.2 | 2145 | 78.6 | 180.87 | 0.01 | 87.64 | 23.55 | 0.03 | 8.15 |

## every run

| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | stage ms/window | cpu ms/window | draft ms/window |
|---|---|---|---|---|---|---|---|---|---|---|
| prompt | 1 | 76.0 | 9.9 | 54.726 | 17085.7 | 9556.7 | 85.6 | 0.01 | 36.16 | 7.73 |
| prompt | 2 | 87.4 | 14.7 | 47.585 | 14763.9 | 7983.2 | 89.1 | 0.01 | 23.55 | 8.15 |
| prompt | 3 | 92.1 | 16.7 | 45.13 | 14082.3 | 7705.6 | 92.1 | 0.01 | 19.43 | 9.02 |
